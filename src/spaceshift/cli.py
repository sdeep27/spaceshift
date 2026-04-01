import argparse
import json
from .viewer import view


def _get_providers():
    """Detect which API providers have keys configured."""
    from litellm.utils import _infer_valid_provider_from_env_vars
    providers = _infer_valid_provider_from_env_vars(None)
    return [p.value for p in providers]


def _get_model_choices_by_category(per_category=8):
    """Build deduplicated model lists per category from rankings."""
    from .LLM import model_rankings, _get_model_name, _get_reasoning_effort
    categories = ["optimal", "best", "fast", "cheap", "open"]
    result = {}
    for cat in categories:
        entries = model_rankings.get(cat, [])
        if entries:
            models = []
            seen = set()
            for e in entries:
                name = _get_model_name(e)
                if name in seen:
                    continue
                seen.add(name)
                effort = _get_reasoning_effort(e)
                suffix = f"  (reasoning: {effort})" if effort and effort != "default" else ""
                models.append((name, suffix))
                if len(models) >= per_category:
                    break
            result[cat] = models
    return result


def _select_model(model_arg):
    """Resolve --model flag or show interactive picker. Returns model string."""
    if model_arg:
        from .LLM import resolve_model
        import io, contextlib
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            resolved = resolve_model(str(model_arg))
        if resolved:
            return resolved
        if str(model_arg).isdigit():
            print(f"Invalid model rank: {model_arg}")
            raise SystemExit(1)
        return str(model_arg)

    import questionary
    from questionary import Separator, Choice
    from rich.console import Console
    console = Console()

    # Show provider info
    providers = _get_providers()
    console.print(f"  [dim]{len(providers)} provider(s): {', '.join(providers)}[/dim]\n")

    # Build categorized choices
    by_cat = _get_model_choices_by_category()
    if not by_cat:
        console.print("[red]No models found in rankings.[/red]")
        raise SystemExit(1)

    choices = []
    model_map = {}
    for cat, models in by_cat.items():
        choices.append(Separator(f"── {cat} ──"))
        for i, (name, suffix) in enumerate(models):
            label = f"  {name}{suffix}"
            choices.append(Choice(label, value=name))
            model_map[name] = name

    # Default to first model in first category
    first_cat = next(iter(by_cat))
    default_model = by_cat[first_cat][0][0]

    answer = questionary.select(
        "Select a model:",
        choices=choices,
        default=default_model,
    ).ask()

    if answer is None:
        raise SystemExit(0)

    console.print(f"\n  Using [bold]{answer}[/bold]\n")
    return answer


def _build_tree_dict(prompt, agent_result):
    """Convert agent's flat {subprompts, superprompts, sideprompts} to prompt_tree format."""
    tree = {"prompt": prompt}
    for direction, key in [("sub", "subprompts"), ("super", "superprompts"), ("side", "sideprompts")]:
        prompts = agent_result.get(key, [])
        if prompts:
            tree[direction] = [
                {"prompt": p, "depth": 0, "parent": prompt, "id": f"{direction}_{i}", "direction": direction}
                for i, p in enumerate(prompts)
            ]
    return tree


def _run_research(prompt, model, save_dir, no_view):
    """Execute the full research pipeline."""
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

    from rich.console import Console
    from .LLM import LLM
    from .prompt_space import subprompt, superprompt, sideprompt, research_tree

    console = Console()

    # Step 1: Agentic prompt tree generation
    console.print(f"\n[bold]Building prompt tree...[/bold]")

    def explore_paths(subject: str, sub_n: int = 5, super_n: int = 5, side_n: int = 3) -> dict:
        """Generate exploration paths for a research subject.

        Args:
            subject: The topic to explore.
            sub_n: Number of sub-prompts to generate (drill deeper into specifics).
            super_n: Number of super-prompts to generate (zoom out to broader context).
            side_n: Number of side-prompts to generate (lateral/parallel perspectives).

        Returns:
            dict with subprompts, superprompts, and sideprompts arrays.
        """
        console.print(f"  [dim]explore_paths: generating {sub_n} sub, {super_n} super, {side_n} side prompts...[/dim]")
        subs = subprompt(subject, n=sub_n, model=model, v=False)
        supers = superprompt(subject, n=super_n, model=model, v=False)
        sides = sideprompt(subject, n=side_n, model=model, v=False)
        console.print(f"  [green]Got {len(subs)} sub, {len(supers)} super, {len(sides)} side prompts[/green]")
        return {"subprompts": subs, "superprompts": supers, "sideprompts": sides}

    agent = LLM(model=model, v=False)
    agent.tools(fns=[explore_paths], tool_choice="required_then_auto", max_turns=5)
    agent.sys(
        "You are a research planner with access to a prompt exploration tool. "
        "Given a subject, use the explore_paths tool to generate exploration paths, "
        "then analyze the results and add any paths the tool missed. "
        "Return your final comprehensive map as JSON with keys: subprompts, superprompts, sideprompts — each an array of prompt strings."
    )
    tree_prompts = agent.user(f"Map all exploration paths for: {prompt}").result_json()

    sub_count = len(tree_prompts.get("subprompts", []))
    super_count = len(tree_prompts.get("superprompts", []))
    side_count = len(tree_prompts.get("sideprompts", []))
    console.print(f"  [bold green]Final tree: {sub_count} sub, {super_count} super, {side_count} side prompts[/bold green]\n")

    # Show the prompts
    for direction, key in [("sub", "subprompts"), ("super", "superprompts"), ("side", "sideprompts")]:
        prompts = tree_prompts.get(key, [])
        if prompts:
            console.print(f"  [bold]{direction}[/bold] ({len(prompts)}):")
            for p in prompts:
                console.print(f"    [dim]{p[:90]}{'...' if len(p) > 90 else ''}[/dim]")

    # Step 2: Build tree dict and run research_tree
    tree_dict = _build_tree_dict(prompt, tree_prompts)

    console.print(f"\n[bold]Generating research outputs...[/bold]")
    result = research_tree(
        tree_dict,
        output_model=model,
        search="auto",
        save=save_dir if save_dir else True,
        v=True,
    )

    save_path = result.get("saved", [])
    if save_path:
        # saved is a list of file paths — get the directory
        import os
        output_dir = os.path.dirname(save_path[0]) if save_path else None
    else:
        output_dir = None

    total = len(result.get("outputs", []))
    console.print(f"\n[bold green]Done! {total} research outputs generated.[/bold green]")
    if output_dir:
        console.print(f"[dim]Saved to: {output_dir}/[/dim]")

    # Step 3: Open viewer
    if not no_view and output_dir:
        console.print(f"\n[bold]Opening viewer...[/bold]\n")
        view(output_dir)


def main():
    parser = argparse.ArgumentParser(prog="spaceshift", description="spaceshift CLI")
    sub = parser.add_subparsers(dest="command")

    # view subcommand
    v = sub.add_parser("view", help="Browse markdown results in the browser")
    v.add_argument("path", nargs="?", default=".", help="Directory to serve (default: current)")
    v.add_argument("--port", type=int, default=8383, help="Port (default: 8383)")
    v.add_argument("--no-open", action="store_true", help="Don't auto-open browser")

    # research subcommand
    r = sub.add_parser("research", help="Run a full research pipeline on a topic")
    r.add_argument("prompt", help="Research topic or question")
    r.add_argument("--model", "-m", default=None, help="Model to use (name, shorthand, or rank number). Interactive picker if omitted.")
    r.add_argument("--no-view", action="store_true", help="Don't auto-open viewer when done")
    r.add_argument("--save", "-s", default=None, help="Output directory (auto-named from prompt if omitted)")

    args = parser.parse_args()

    if args.command == "view":
        view(args.path, args.port, args.no_open)
    elif args.command == "research":
        model = _select_model(args.model)
        _run_research(args.prompt, model, args.save, args.no_view)
    else:
        parser.print_help()
