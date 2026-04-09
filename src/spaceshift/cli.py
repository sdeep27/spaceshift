import argparse
import json
import os
from pathlib import Path
from .viewer import view


# Supported API providers
SUPPORTED_PROVIDERS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "togetherai": "TOGETHERAI_API_KEY",
    "xai": "XAI_API_KEY",
}


def _get_config_path():
    """Get path to config file in user's home directory."""
    return Path.home() / ".spaceshift" / "config.json"


def _load_config():
    """Load config from file and inject API keys into environment."""
    config_path = _get_config_path()
    if not config_path.exists():
        return {}

    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Inject API keys into environment
        api_keys = config.get("api_keys", {})
        for key, value in api_keys.items():
            if value:  # Only set if value exists
                os.environ[key] = value

        return config
    except Exception as e:
        print(f"Warning: Failed to load config from {config_path}: {e}")
        return {}


def _save_config(config):
    """Save config to file with secure permissions."""
    config_path = _get_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Write config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Set secure permissions (user read/write only)
    os.chmod(config_path, 0o600)


def _get_providers():
    """Detect which API providers have keys configured."""
    from litellm.utils import _infer_valid_provider_from_env_vars
    providers = _infer_valid_provider_from_env_vars(None)
    return [p.value for p in providers]


def _ensure_api_keys():
    """Check if any API keys are configured. If not, run first-time setup."""
    providers = _get_providers()
    if not providers:
        from rich.console import Console
        console = Console()
        console.print("\n[yellow]No API keys found. Let's set up your providers.[/yellow]\n")
        _manage_api_keys(first_time=True)
        # Re-check after setup
        providers = _get_providers()
        if not providers:
            console.print("[red]No API keys configured. Exiting.[/red]")
            raise SystemExit(1)


def _obscure_key(key):
    """Obscure an API key for display (show prefix...suffix)."""
    if not key or len(key) < 12:
        return "[invalid]"
    # Show first 7 chars (e.g., "sk-proj") + ... + last 4 chars
    return f"{key[:7]}...{key[-4:]}"


def _manage_api_keys(first_time=False):
    """Interactive API key management wizard."""
    import questionary
    from rich.console import Console
    console = Console()

    # Load current config
    config = _load_config()
    if "api_keys" not in config:
        config["api_keys"] = {}

    if not first_time:
        console.print("\n[bold]API Key Management[/bold]\n")
        providers = _get_providers()
        if providers:
            console.print(f"  [dim]Current providers: {', '.join(providers)}[/dim]\n")

    # Prompt for each supported provider
    provider_names = {
        "openai": "OpenAI",
        "anthropic": "Anthropic",
        "gemini": "Google Gemini",
        "togetherai": "Together.AI",
        "xai": "xAI",
    }

    changes_made = False

    # Build provider list for progress indicator
    all_providers = list(SUPPORTED_PROVIDERS.keys())

    for idx, (provider_key, env_var) in enumerate(SUPPORTED_PROVIDERS.items()):
        # Show progress indicator with current provider highlighted
        progress_parts = []
        for i, p in enumerate(all_providers):
            if i == idx:
                progress_parts.append(f"[bold cyan]{provider_names[p]}[/bold cyan]")
            else:
                progress_parts.append(f"[dim]{provider_names[p]}[/dim]")
        progress = "  " + "  ".join(progress_parts)
        console.print(f"\n{progress}")

        # Check both config and environment
        config_key = config["api_keys"].get(env_var, "")
        env_key = os.environ.get(env_var, "")

        # Determine current state
        has_config_key = bool(config_key)
        has_env_key = bool(env_key) and not has_config_key
        current_key = config_key or env_key

        # Build hint text
        if has_config_key:
            obscured = _obscure_key(config_key)
            status = f"({obscured})"
            hint = "Enter/Tab to keep, paste new key to replace, 'remove' to delete"
        elif has_env_key:
            obscured = _obscure_key(env_key)
            status = f"({obscured} from environment)"
            hint = "Enter/Tab to save to config, paste new key, or 'skip' to ignore"
        else:
            status = ""
            hint = "Enter/Tab to skip, or paste key to add"

        if status:
            console.print(f"    {status}")
        console.print(f"    [dim]{hint}[/dim]")

        key = questionary.text("    ").ask()

        if key is None:  # User cancelled
            if first_time:
                console.print("\n[red]Setup cancelled. Exiting.[/red]")
                raise SystemExit(1)
            return

        key = key.strip()
        key_lower = key.lower()

        # Handle removal
        if key_lower == "remove":
            if has_config_key:
                del config["api_keys"][env_var]
                if env_var in os.environ:
                    del os.environ[env_var]
                console.print(f"  [yellow]✓ {provider_names[provider_key]} key removed[/yellow]\n")
                changes_made = True
            else:
                console.print(f"  [dim]No key to remove[/dim]\n")
            continue

        # Handle skip (Enter, Tab, or explicit "skip")
        if key == "" or key == "\t" or key_lower == "skip":
            if has_env_key and key == "":
                # Empty on env key = save it to config
                config["api_keys"][env_var] = env_key
                console.print(f"  [green]✓ {provider_names[provider_key]} key saved to config[/green]\n")
                changes_made = True
            else:
                # Skip or keep existing
                console.print(f"  [dim]Keeping current configuration[/dim]\n")
            continue

        # Save new key
        config["api_keys"][env_var] = key
        os.environ[env_var] = key
        console.print(f"  [green]✓ {provider_names[provider_key]} key {'updated' if current_key else 'added'}[/green]\n")
        changes_made = True

    # Save config if changes were made
    if changes_made:
        _save_config(config)
        config_path = _get_config_path()
        console.print(f"[green]✓ Configuration saved to {config_path}[/green]\n")
    else:
        console.print(f"[dim]No changes made[/dim]\n")


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
        return None

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
        import os
        output_dir = os.path.dirname(save_path[0]) if save_path else None
    else:
        output_dir = None

    total = len(result.get("outputs", []))
    console.print(f"\n[bold green]Done! {total} research outputs generated.[/bold green]")
    if output_dir:
        console.print(f"[dim]Saved to: {output_dir}/[/dim]")

    # Step 3: Post-process research outputs
    if output_dir:
        _post_process_research(output_dir, model, verbose=True)

    # Step 4: Open viewer
    if not no_view and output_dir:
        console.print(f"\n[bold]Opening viewer...[/bold]\n")
        view(output_dir)


def _post_process_research(output_dir, model, verbose=True):
    """Post-process research outputs to generate synthesis documents."""
    from rich.console import Console
    from .LLM import LLM
    from .tools import ResearchTools

    console = Console()

    console.print("\n[bold cyan]Post-processing research outputs...[/bold cyan]")

    # Create research tools scoped to output directory
    rt = ResearchTools(output_dir)

    # Create agent with research tools
    agent = LLM(model=model, v=verbose)
    agent.tools(fns=rt.all_tools(), tool_choice="required_then_auto", max_turns=20)

    # Clear deliverables, autonomous approach
    agent.sys(
        "You are a research synthesizer. Analyze all research outputs in this directory and create comprehensive synthesis documents.\n\n"
        "The directory contains markdown files with YAML frontmatter. Each file has metadata like direction (root/sub/super/side), "
        "depth, prompt, and parent - forming a research tree structure.\n\n"
        "TOOL USAGE:\n"
        "- Use read() to read markdown files (read one file at a time)\n"
        "- Use ls() and find() to discover files\n"
        "- Use grep() to search file contents\n"
        "- Use write() to create new synthesis documents\n"
        "- Use bash() ONLY for text processing (wc, sort, uniq, diff, etc.) - NOT for reading files\n"
        "- bash() blocks shell operators (&&, ||, ;) for security\n\n"
        "Create these synthesis documents:\n\n"
        "1. COMPREHENSIVE REPORT (_REPORT.md) - A long, detailed synthesis:\n"
        "   - Synthesize ALL findings across the research tree\n"
        "   - Identify key themes, patterns, and insights\n"
        "   - Create narrative flow connecting different research nodes\n"
        "   - Include specific examples and quotes from source files\n"
        "   - Should be substantial - aim for depth and completeness\n\n"
        "2. EXECUTIVE SUMMARY (_SUMMARY.md):\n"
        "   - High-level overview\n"
        "   - Most important takeaways\n"
        "   - For someone who wants the essence without details\n\n"
        "3. GLOSSARY (_GLOSSARY.md):\n"
        "   - Key terms, concepts, and acronyms from the research\n"
        "   - Alphabetically organized\n"
        "   - Brief definitions with context\n\n"
        "4. THEMATIC INDEX (_THEMES.md):\n"
        "   - Organize all research files by theme/topic\n"
        "   - Include markdown links to relevant files\n"
        "   - Help readers navigate to specific areas of interest\n\n"
        "All outputs should use markdown with YAML frontmatter."
    )

    # Execute synthesis
    agent.user(f"Synthesize the research outputs in {output_dir}.").chat()

    console.print("[bold green]✓ Post-processing complete[/bold green]")


def _run_agent(prompt, model, save_dir, no_view):
    """Execute the autonomous research agent."""
    from rich.console import Console
    from .research_agent import research_agent

    console = Console()
    console.print(f"\n[bold]Starting research agent...[/bold]\n")

    result = research_agent(prompt, model=model, save=save_dir or True, v=True)

    output_dir = result.get("output_dir")
    if output_dir:
        console.print(f"\n[dim]Output: {output_dir}/[/dim]")
    if not no_view and output_dir:
        console.print(f"\n[bold]Opening viewer...[/bold]\n")
        view(output_dir)


def _interactive_main():
    """Interactive mode — shown when user runs `spaceshift` with no args."""
    import questionary
    from questionary import Choice, Separator
    from rich.console import Console
    console = Console()

    console.print("\n[bold]spaceshift[/bold] [dim]— open research toolkit[/dim]\n")

    # Show current providers
    providers = _get_providers()
    if providers:
        console.print(f"  [dim]{len(providers)} provider(s): {', '.join(providers)}[/dim]\n")

    while True:
        mode = questionary.select(
            "What would you like to do?",
            choices=[
                Choice("Deep Research", value="research"),
                Choice("Prompt Manipulate", value="prompt"),
                Choice("Compare Models", value="compare"),
                Choice("Grid Search", value="grid"),
                Choice("Prompt Tree", value="tree"),
                Separator(),
                Choice("Manage API Keys", value="keys"),
            ],
        ).ask()

        if mode is None:
            break

        if mode == "research":
            prompt = questionary.text(
                "Research topic:",
                validate=lambda t: len(t.strip()) > 0 or "Enter a topic",
            ).ask()
            if prompt is None:
                continue
            model = _select_model(None)
            if model is None:
                continue
            _run_research(prompt.strip(), model, save_dir=None, no_view=False)
            break
        elif mode == "keys":
            _manage_api_keys(first_time=False)
            # Reload provider display
            providers = _get_providers()
            if providers:
                console.print(f"  [dim]{len(providers)} provider(s): {', '.join(providers)}[/dim]\n")
        else:
            console.print(f"\n[dim]{mode} — coming soon[/dim]\n")


def main():
    # Load config and inject API keys into environment
    _load_config()

    parser = argparse.ArgumentParser(prog="spaceshift", description="spaceshift CLI")
    sub = parser.add_subparsers(dest="command")

    # view subcommand
    v = sub.add_parser("view", help="Browse markdown results in the browser")
    v.add_argument("path", nargs="?", default=".", help="Directory to serve (default: current)")
    v.add_argument("--port", type=int, default=8383, help="Port (default: 8383)")
    v.add_argument("--no-open", action="store_true", help="Don't auto-open browser")

    # research subcommand (direct shortcut)
    r = sub.add_parser("research", help="Run a full research pipeline on a topic")
    r.add_argument("prompt", help="Research topic or question")
    r.add_argument("--model", "-m", default=None, help="Model to use (name, shorthand, or rank number). Interactive picker if omitted.")
    r.add_argument("--no-view", action="store_true", help="Don't auto-open viewer when done")
    r.add_argument("--save", "-s", default=None, help="Output directory (auto-named from prompt if omitted)")

    # synthesize subcommand
    s = sub.add_parser("synthesize", help="Run synthesis agent on research outputs")
    s.add_argument("directory", nargs="?", default=".", help="Directory containing markdown files (default: current)")
    s.add_argument("--model", "-m", default=None, help="Model to use (name, shorthand, or rank number). Uses rank 1 if omitted.")
    s.add_argument("--view", action="store_true", help="Open viewer after synthesis")

    # agent subcommand (dev/testing — not in interactive menu)
    a = sub.add_parser("agent", help="Run autonomous research agent on a topic")
    a.add_argument("prompt", help="Research topic or question")
    a.add_argument("--model", "-m", default=None, help="Model to use (name, shorthand, or rank number). Interactive picker if omitted.")
    a.add_argument("--no-view", action="store_true", help="Don't auto-open viewer when done")
    a.add_argument("--save", "-s", default=None, help="Output directory (auto-named from prompt if omitted)")

    args = parser.parse_args()

    if args.command == "view":
        view(args.path, args.port, args.no_open)
    elif args.command == "research":
        # Ensure API keys before running research
        _ensure_api_keys()
        model = _select_model(args.model)
        if model is None:
            raise SystemExit(0)
        _run_research(args.prompt, model, args.save, args.no_view)
    elif args.command == "agent":
        _ensure_api_keys()
        model = _select_model(args.model)
        if model is None:
            raise SystemExit(0)
        _run_agent(args.prompt, model, args.save, args.no_view)
    elif args.command == "synthesize":
        # Ensure API keys before running synthesis
        _ensure_api_keys()
        model = _select_model(args.model) if args.model else _select_model("1")
        if model is None:
            raise SystemExit(0)
        _post_process_research(args.directory, model, verbose=True)
        if args.view:
            view(args.directory)
    else:
        # Ensure API keys before interactive mode
        _ensure_api_keys()
        _interactive_main()
