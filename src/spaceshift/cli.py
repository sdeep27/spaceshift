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


def _prompt_to_slug(prompt, max_len=40):
    """Convert a prompt to a filesystem-safe slug for directory naming."""
    import re
    slug = re.sub(r'[^\w\s-]', '', prompt.lower().strip())
    slug = re.sub(r'[\s_]+', '_', slug)
    return slug[:max_len].rstrip('_')


def _run_prompt_manipulate(prompt, model, transforms=None, output_model=None,
                           save_dir=None, concurrency=5):
    """Execute the prompt manipulation pipeline."""
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

    from concurrent.futures import ThreadPoolExecutor, as_completed
    from rich.console import Console
    from .prompt_probe import prompt_transform, list_transforms
    from .utils import _write_consolidated_md

    console = Console()

    # Resolve transforms
    if transforms is None:
        transforms = list_transforms(v=False)

    if not transforms:
        console.print("[yellow]No transforms selected.[/yellow]")
        return

    # Resolve save path early so user sees it
    if not save_dir:
        save_dir = os.path.join("output", _prompt_to_slug(prompt))

    console.print(f"\n[bold]Running {len(transforms)} transforms...[/bold]")
    console.print(f"  [dim]Manipulation model: {model}[/dim]")
    if output_model:
        console.print(f"  [dim]Output model: {output_model}[/dim]")
    console.print(f"  [dim]Output folder: {save_dir}/[/dim]")

    # Phase 1: Run transforms concurrently
    transformed_prompts = [None] * len(transforms)
    failed = []

    def _run_single(idx, name):
        return idx, name, prompt_transform(prompt, name, model=model)

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(_run_single, i, name): i
            for i, name in enumerate(transforms)
        }
        done_count = 0
        for future in as_completed(futures):
            try:
                idx, name, result = future.result()
                transformed_prompts[idx] = result
                done_count += 1
                console.print(f"  [{done_count}/{len(transforms)}] [green]{name}[/green]")
            except Exception as e:
                idx = futures[future]
                failed.append((transforms[idx], str(e)))
                done_count += 1
                console.print(f"  [{done_count}/{len(transforms)}] [red]{transforms[idx]} — failed: {e}[/red]")

    # Remove failed transforms
    if failed:
        good_indices = [i for i in range(len(transforms)) if transformed_prompts[i] is not None]
        transforms = [transforms[i] for i in good_indices]
        transformed_prompts = [transformed_prompts[i] for i in good_indices]
        console.print(f"\n[yellow]{len(failed)} transform(s) failed, continuing with {len(transforms)}[/yellow]")

    if not transforms:
        console.print("[red]All transforms failed.[/red]")
        return

    # Save manipulated prompts immediately
    prompts_path = os.path.join(save_dir, "manipulations.md")
    saved = _write_consolidated_md(
        prompts_path, prompt, transforms, transformed_prompts,
        manipulation_model=model,
    )
    console.print(f"\n[bold green]{len(transforms)} transforms saved.[/bold green]")
    console.print(f"  [dim]{saved}[/dim]")

    # Phase 2: Generate outputs and save consolidated file
    if output_model:
        console.print(f"\n[bold]Generating {len(transforms)} outputs with {output_model}...[/bold]")
        from .LLM import LLM

        def _generate_single(idx, name, transformed_prompt):
            llm = LLM(model=output_model, v=False)
            response = llm.user(transformed_prompt).result()
            return idx, name, response

        responses = [None] * len(transforms)
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(_generate_single, i, name, tp): i
                for i, (name, tp) in enumerate(zip(transforms, transformed_prompts))
            }
            done_count = 0
            for future in as_completed(futures):
                try:
                    idx, name, response = future.result()
                    responses[idx] = response
                    done_count += 1
                    console.print(f"  [{done_count}/{len(transforms)}] [green]{name}[/green]")
                except Exception as e:
                    done_count += 1
                    console.print(f"  [{done_count}/{len(transforms)}] [red]{transforms[futures[future]]} — failed: {e}[/red]")

        outputs_path = os.path.join(save_dir, "outputs.md")
        saved_outputs = _write_consolidated_md(
            outputs_path, prompt, transforms, transformed_prompts,
            manipulation_model=model, output_model=output_model, responses=responses,
        )
        output_count = sum(1 for r in responses if r is not None)
        console.print(f"\n[bold green]{output_count} outputs saved.[/bold green]")
        console.print(f"  [dim]{saved_outputs}[/dim]")


def _model_to_provider(model_name):
    """Infer provider from model name string."""
    if model_name.startswith("gemini/"):
        return "gemini"
    if model_name.startswith("xai/"):
        return "xai"
    if model_name.startswith("together_ai/"):
        return "togetherai"
    if "claude" in model_name:
        return "anthropic"
    return "openai"


def _select_default_compare_models(n=3):
    """Pick top-ranked models, one per configured provider, for cross-provider comparison."""
    from .LLM import model_rankings, _get_model_name
    providers = set(_get_providers())

    best_entries = model_rankings.get("best", [])

    selected = []
    used_providers = set()

    for entry in best_entries:
        model = _get_model_name(entry)
        provider = _model_to_provider(model)
        if provider in providers and provider not in used_providers:
            selected.append(model)
            used_providers.add(provider)
            if len(selected) >= n:
                break

    # Fill remaining slots with next best models from any configured provider
    if len(selected) < n:
        for entry in best_entries:
            model = _get_model_name(entry)
            provider = _model_to_provider(model)
            if provider in providers and model not in selected:
                selected.append(model)
                if len(selected) >= n:
                    break

    return selected


def _select_compare_models():
    """Interactive multi-select model picker grouped by category. Returns list of model names."""
    import questionary
    from questionary import Choice, Separator
    from rich.console import Console
    console = Console()

    from .LLM import model_rankings, _get_model_name, _get_reasoning_effort

    defaults = set(_select_default_compare_models(3))

    categories = ["best", "fast", "cheap", "open"]
    choices = []
    seen = set()

    for cat in categories:
        entries = model_rankings.get(cat, [])
        if not entries:
            continue
        choices.append(Separator(f"── {cat} ──"))
        count = 0
        for e in entries:
            name = _get_model_name(e)
            if name in seen:
                continue
            seen.add(name)
            effort = _get_reasoning_effort(e)
            suffix = f"  (reasoning: {effort})" if effort and effort != "default" else ""
            label = f"{name}{suffix}"
            choices.append(Choice(label, value=name, checked=name in defaults))
            count += 1
            if count >= 6:
                break

    selected = questionary.checkbox(
        "Select models to compare (space to toggle, enter to confirm):",
        choices=choices,
        validate=lambda x: len(x) >= 2 or "Select at least 2 models",
    ).ask()

    if selected is None:
        return None

    console.print(f"\n  [bold]{len(selected)} model(s) selected[/bold]\n")
    return selected


def _to_anchor(label):
    """Convert model label to markdown anchor ID."""
    import re
    anchor = label.lower()
    anchor = re.sub(r'[/.]', '-', anchor)
    anchor = re.sub(r'[^a-z0-9\-]', '', anchor)
    anchor = re.sub(r'-+', '-', anchor).strip('-')
    return anchor


def _write_compare_md(result, filepath):
    """Write comparison results to a single markdown file with internal links."""
    import os
    from datetime import date

    prompt = result["prompt"]
    responses = result["responses"]
    models = result["models"]

    os.makedirs(os.path.dirname(os.path.abspath(filepath)) or ".", exist_ok=True)

    lines = []
    # YAML frontmatter
    lines.append("---")
    lines.append(f"prompt: {prompt}")
    lines.append("models:")
    for m in models:
        lines.append(f"  - {m}")
    lines.append(f"date: {date.today().isoformat()}")
    lines.append("---\n")

    # Header
    lines.append("# Model Comparison\n")
    lines.append(f"> **Prompt:** {prompt}\n")

    # Table of contents with links
    lines.append("| Model | Link |")
    lines.append("|-------|------|")
    for m in models:
        anchor = _to_anchor(m)
        lines.append(f"| {m} | [View response](#{anchor}) |")
    lines.append("")

    # Model response sections
    for m, response in zip(models, responses):
        anchor = _to_anchor(m)
        lines.append("---\n")
        lines.append(f"## {m}\n")
        lines.append(f"{response}\n")

    with open(filepath, "w") as f:
        f.write("\n".join(lines))

    return filepath


def _append_eval_to_md(filepath, eval_result, model_labels, eval_model_name):
    """Append pairwise evaluation results to an existing comparison markdown file."""
    with open(filepath, "r") as f:
        content = f.read()

    rankings = eval_result["rankings"]
    scores = eval_result["scores"]
    metrics = eval_result["raw"]["metrics_used"]

    # Build evaluation section
    eval_lines = []
    eval_lines.append("---\n")
    eval_lines.append("## Evaluation\n")
    eval_lines.append(f"**Eval model:** {eval_model_name}\n")
    eval_lines.append("**Metrics:**")
    for m in metrics:
        eval_lines.append(f"- {m}")
    eval_lines.append("")
    eval_lines.append("| Rank | Model | Score |")
    eval_lines.append("|------|-------|-------|")
    for rank, idx in enumerate(rankings, 1):
        label = model_labels[idx]
        anchor = _to_anchor(label)
        score = scores[idx]
        eval_lines.append(f"| {rank} | [{label}](#{anchor}) | {score:.2f} |")
    eval_lines.append("")

    eval_section = "\n".join(eval_lines)

    # Update the ToC table to include scores
    import re
    # Replace the simple table header and rows with scored versions
    old_header = "| Model | Link |"
    new_header = "| Rank | Model | Score | Link |"
    if old_header in content:
        content = content.replace(old_header, new_header)
        content = content.replace("|-------|------|", "|------|-------|-------|------|")
        for idx in range(len(model_labels)):
            label = model_labels[idx]
            anchor = _to_anchor(label)
            old_row = f"| {label} | [View response](#{anchor}) |"
            rank = rankings.index(idx) + 1
            score = scores[idx]
            new_row = f"| {rank} | {label} | {score:.2f} | [View response](#{anchor}) |"
            content = content.replace(old_row, new_row)

    # Append evaluation section
    content = content.rstrip() + "\n\n" + eval_section

    with open(filepath, "w") as f:
        f.write(content)


def _prompt_slug(prompt, max_len=40):
    """Generate a filesystem-safe slug from a prompt string."""
    import re
    slug = prompt.lower().strip()
    slug = re.sub(r'[^a-z0-9\s-]', '', slug)
    slug = re.sub(r'[\s]+', '-', slug)
    return slug[:max_len].rstrip('-')


def _run_compare(prompt, models, evaluate, eval_model, save_dir, no_view):
    """Execute model comparison pipeline."""
    import threading
    from rich.console import Console
    from .compare_models import compare_models, validate_model
    from .evaluate import pairwise_evaluate
    from .viewer import view_background

    console = Console()

    # Resolve models
    resolved = []
    for m in models:
        resolved.append(validate_model(m))
    models = resolved

    console.print(f"\n[bold]Comparing {len(models)} models:[/bold]")
    for i, m in enumerate(models, 1):
        console.print(f"  [{i}] {m}")
    console.print()

    # Run comparison (no evaluation yet)
    result = compare_models(prompt, models=models, evaluate=False, v=True)

    # Determine save path
    if not save_dir:
        slug = _prompt_slug(prompt)
        save_dir = f"compare_{slug}"

    import os
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, "comparison.md")

    # Write markdown
    _write_compare_md(result, filepath)
    console.print(f"\n[bold green]Saved to: {filepath}[/bold green]")

    # Open viewer in background
    if not no_view:
        console.print(f"[dim]Opening viewer...[/dim]")
        view_background(save_dir)

    # Run pairwise evaluation in background thread if requested
    if evaluate:
        eval_model_name = eval_model or "rank 1 (default)"
        console.print(f"\n[bold cyan]Running pairwise evaluation...[/bold cyan]")
        console.print(f"  [dim]Eval model: {eval_model_name}[/dim]")

        def run_eval():
            try:
                eval_result = pairwise_evaluate(
                    prompt=prompt,
                    results=result["responses"],
                    eval_model=eval_model,
                    v=True,
                )
                _append_eval_to_md(filepath, eval_result, result["models"], eval_model_name)
                console.print(f"\n[bold green]Evaluation complete. Results updated in {filepath}[/bold green]")
            except Exception as e:
                console.print(f"\n[red]Evaluation failed: {e}[/red]")

        eval_thread = threading.Thread(target=run_eval, daemon=True)
        eval_thread.start()
        eval_thread.join()  # Wait for eval to finish before process exits


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
        elif mode == "prompt":
            prompt = questionary.text(
                "Enter your prompt:",
                validate=lambda t: len(t.strip()) > 0 or "Enter a prompt",
            ).ask()
            if prompt is None:
                continue

            from .prompt_probe import list_transforms
            all_transforms = list_transforms(v=False)

            _TRANSFORM_HINTS = {
                "inverse": "Invert syntax/perspective, preserve meaning",
                "abstract_up": "One level more general",
                "abstract_down": "One level more specific",
                "abstract_up3": "Three levels more general",
                "abstract_down3": "Three levels more specific",
                "active_up": "More active, direct, and passionate",
                "passive_up": "More passive, detached, and timeless",
                "reflection": "Mirror/flip across a meaningful axis",
                "rotation": "Rotate perspective or framing",
                "shear": "Skew one dimension while fixing another",
                "scaling": "Amplify scope and intensity",
                "recursion": "Apply the prompt to itself",
                "dimension_up": "Elevate by one level of dimensionality",
                "dimension_down": "Collapse by one level of dimensionality",
                "improve_naive": "Straightforward prompt improvement",
                "user_profile": "Add a first-person persona/context",
                "double": "Duplicate the prompt (deterministic)",
                "translate_chinese": "Translate to Chinese",
                "translate_korean": "Translate to Korean",
                "translate_hindi": "Translate to Hindi",
                "translate_french": "Translate to French",
                "translate_arabic": "Translate to Arabic",
            }

            console.print(f"\n  [dim]({len(all_transforms)} transforms selected)[/dim]\n")
            selected = questionary.checkbox(
                "Select transforms to apply:",
                choices=[
                    questionary.Choice(
                        f"{name:<20} {_TRANSFORM_HINTS.get(name, '')}",
                        value=name,
                        checked=True,
                    )
                    for name in all_transforms
                ],
            ).ask()
            if selected is None:
                continue
            if not selected:
                console.print("[yellow]No transforms selected.[/yellow]\n")
                continue

            console.print(f"\n  [bold]{len(selected)}[/bold] of {len(all_transforms)} transforms selected\n")

            console.print("\n[bold]Select manipulation model[/bold] (for transforming prompts):")
            model = _select_model(None)
            if model is None:
                continue

            generate_outputs = questionary.confirm(
                "Generate outputs too?",
                default=False,
            ).ask()
            if generate_outputs is None:
                continue

            output_model = None
            if generate_outputs:
                console.print("\n[bold]Select output model[/bold] (for generating responses):")
                output_model = _select_model(None)
                if output_model is None:
                    continue

            default_dir = os.path.join("output", _prompt_to_slug(prompt))
            save_dir = questionary.text(
                "Output folder:",
                default=default_dir,
            ).ask()
            if save_dir is None:
                continue

            _run_prompt_manipulate(
                prompt.strip(), model,
                transforms=selected,
                output_model=output_model,
                save_dir=save_dir.strip(),
            )
        elif mode == "compare":
            prompt = questionary.text(
                "Enter your prompt:",
                validate=lambda t: len(t.strip()) > 0 or "Enter a prompt",
            ).ask()
            if prompt is None:
                continue
            models = _select_compare_models()
            if models is None:
                continue
            run_eval = questionary.confirm(
                "Run pairwise evaluation after responses?",
                default=False,
            ).ask()
            if run_eval is None:
                continue
            eval_model = None
            if run_eval:
                console.print("\n[dim]Select a model to judge the evaluation:[/dim]")
                eval_model = _select_model(None)
                if eval_model is None:
                    continue
            _run_compare(prompt.strip(), models, evaluate=run_eval, eval_model=eval_model, save_dir=None, no_view=False)
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

    # compare subcommand
    c = sub.add_parser("compare", help="Compare a prompt across multiple models")
    c.add_argument("prompt", help="Prompt to compare across models")
    c.add_argument("--models", "-m", nargs="+", default=None, help="Models to compare (names, shorthands, or rank numbers). Auto-selects if omitted.")
    c.add_argument("--evaluate", "-e", action="store_true", help="Run pairwise evaluation after generating responses")
    c.add_argument("--eval-model", default=None, help="Model to use for pairwise evaluation (default: rank 1)")
    c.add_argument("--save", "-s", default=None, help="Output directory (auto-named from prompt if omitted)")
    c.add_argument("--no-view", action="store_true", help="Don't auto-open viewer when done")

    # agent subcommand (dev/testing — not in interactive menu)
    a = sub.add_parser("agent", help="Run autonomous research agent on a topic")
    a.add_argument("prompt", help="Research topic or question")
    a.add_argument("--model", "-m", default=None, help="Model to use (name, shorthand, or rank number). Interactive picker if omitted.")
    a.add_argument("--no-view", action="store_true", help="Don't auto-open viewer when done")
    a.add_argument("--save", "-s", default=None, help="Output directory (auto-named from prompt if omitted)")

    # manipulate subcommand
    m = sub.add_parser("manipulate", help="Apply prompt transforms and optionally generate outputs")
    m.add_argument("prompt", help="The prompt to manipulate")
    m.add_argument("--model", "-m", default=None, help="Model for prompt transforms (name, shorthand, or rank). Interactive picker if omitted.")
    m.add_argument("--output-model", default=None, help="Model for generating outputs. Omit to skip output generation.")
    m.add_argument("--transforms", "-t", nargs="+", default=None, help="Transform names to apply (default: all). Use '--transforms list' to see available.")
    m.add_argument("--save", "-s", default=None, help="Output directory (auto-named from prompt if omitted)")

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
    elif args.command == "compare":
        _ensure_api_keys()
        models = args.models or _select_default_compare_models(3)
        _run_compare(args.prompt, models, args.evaluate, args.eval_model, args.save, args.no_view)
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
    elif args.command == "manipulate":
        # Handle --transforms list
        if args.transforms and args.transforms == ["list"]:
            from .prompt_probe import list_transforms
            list_transforms(v=True)
            raise SystemExit(0)

        _ensure_api_keys()
        model = _select_model(args.model)
        if model is None:
            raise SystemExit(0)

        output_model = None
        if args.output_model:
            output_model = _select_model(args.output_model)
            if output_model is None:
                raise SystemExit(0)

        _run_prompt_manipulate(
            args.prompt, model,
            transforms=args.transforms,
            output_model=output_model,
            save_dir=args.save,
        )
    else:
        # Ensure API keys before interactive mode
        _ensure_api_keys()
        _interactive_main()
