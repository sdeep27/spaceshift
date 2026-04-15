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


def _select_transforms(checked_default=True):
    """Interactive multi-select transform picker. Returns list of transform names or None."""
    import questionary
    from questionary import Separator as Sep
    from .prompt_probe import list_transforms

    all_transforms = list_transforms(v=False)
    transform_choices = [
        Sep(f"  ({len(all_transforms)} transforms available)"),
    ] + [
        questionary.Choice(
            f"{name:<20} {_TRANSFORM_HINTS.get(name, '')}",
            value=name,
            checked=checked_default,
        )
        for name in all_transforms
    ]
    return questionary.checkbox(
        "Select transforms to apply:",
        choices=transform_choices,
    ).ask()


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


def _select_model(model_arg, prompt_text=None):
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
        prompt_text or "Select a model (use arrow keys):",
        choices=choices,
        default=default_model,
    ).ask()

    if answer is None:
        return None

    console.print(f"\n  Using [bold]{answer}[/bold]\n")
    return answer


def _prompt_to_slug(prompt, max_len=40):
    """Convert a prompt to a filesystem-safe slug for directory naming."""
    import re
    slug = re.sub(r'[^\w\s-]', '', prompt.lower().strip())
    slug = re.sub(r'[\s_]+', '_', slug)
    return slug[:max_len].rstrip('_')


def _print_view_hint(console, path):
    """Print a copy-pasteable command for opening the saved path in the viewer.

    For a file, hints at viewing the parent directory (since `spaceshift view`
    serves a directory). For a directory, uses the path directly.
    """
    target = path
    if os.path.isfile(target) or target.endswith(".md"):
        target = os.path.dirname(target) or "."
    abs_path = os.path.abspath(target)
    console.print(f"[dim]View in browser:[/dim] [cyan]spaceshift view {abs_path}[/cyan]\n")


def _calculate_tree_count(n_list):
    """Total prompts for a direction given its n array. Returns (total, breakdown_str)."""
    if not n_list:
        return 0, ""
    parts = []
    running = 1
    total = 0
    for num in n_list:
        running *= num
        parts.append(str(running))
        total += running
    return total, " + ".join(parts)


def _build_md_tree(nodes, root_prompt, reverse=False):
    """Build indented dash lines from flat node list using parent references."""
    children_of = {}
    for node in nodes:
        parent = node.get("parent", root_prompt)
        children_of.setdefault(parent, []).append(node)

    lines = []

    def _recurse(parent_prompt, indent=0):
        for child in children_of.get(parent_prompt, []):
            prefix = "  " * indent + "- "
            lines.append(f"{prefix}{child['prompt']}")
            _recurse(child["prompt"], indent + 1)

    if reverse:
        top_level = children_of.get(root_prompt, [])
        all_groups = []
        for top_node in top_level:
            group_lines = []
            group_lines.append(f"- {top_node['prompt']}")
            def _collect(parent_prompt, indent=1):
                for child in children_of.get(parent_prompt, []):
                    group_lines.append(f"{'  ' * indent}- {child['prompt']}")
                    _collect(child["prompt"], indent + 1)
            _collect(top_node["prompt"])
            all_groups.append(group_lines)

        for group in reversed(all_groups):
            lines.extend(group)
    else:
        _recurse(root_prompt)

    return lines


def _write_tree_md(tree, path, model, sub_n, super_n, side_n, output_model=None):
    """Write a consolidated tree.md with hierarchical dash formatting."""
    from datetime import datetime

    if not path.endswith(".md"):
        path += ".md"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    prompt = tree["prompt"]

    total = 1
    for n_list in (sub_n, super_n, side_n):
        if n_list:
            count, _ = _calculate_tree_count(n_list)
            total += count

    meta_lines = ["---"]
    meta_lines.append(f"original_prompt: \"{prompt}\"")
    meta_lines.append(f"tree_model: {model}")
    if output_model:
        meta_lines.append(f"output_model: {output_model}")
    meta_lines.append(f"date: {datetime.now().strftime('%Y-%m-%d')}")
    meta_lines.append("directions:")
    if sub_n:
        meta_lines.append(f"  sub: {sub_n}")
    if super_n:
        meta_lines.append(f"  super: {super_n}")
    if side_n:
        meta_lines.append(f"  side: {side_n}")
    meta_lines.append(f"total_prompts: {total}")
    meta_lines.append("---")

    lines = meta_lines + [""]
    lines.append("# Prompt Tree\n")
    lines.append(f"> **{prompt}**\n")

    if "super" in tree and tree["super"]:
        lines.append("---\n")
        lines.append("## Super (above)\n")
        lines.append("Broader prompts that contain the root as a subtopic.\n")
        lines.extend(_build_md_tree(tree["super"], prompt, reverse=True))
        lines.append("")

    if "side" in tree and tree["side"]:
        lines.append("---\n")
        lines.append("## Side (alongside)\n")
        lines.append("Alternative prompts at the same level of abstraction.\n")
        lines.extend(_build_md_tree(tree["side"], prompt))
        lines.append("")

    if "sub" in tree and tree["sub"]:
        lines.append("---\n")
        lines.append("## Sub (below)\n")
        lines.append("Focused subprompts that decompose the root into specifics.\n")
        lines.extend(_build_md_tree(tree["sub"], prompt))
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


def _run_prompt_tree(prompt, model, sub_n=None, super_n=None, side_n=None,
                     output_model=None, save_dir=None, concurrency=5):
    """Execute the prompt tree pipeline."""
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

    from rich.console import Console
    from .prompt_space import prompt_tree, research_tree

    console = Console()

    if not save_dir:
        save_dir = os.path.join("output", _prompt_to_slug(prompt))

    console.print(f"\n[bold]Building prompt tree...[/bold]")
    console.print(f"  [dim]Tree model: {model}[/dim]")
    if output_model:
        console.print(f"  [dim]Output model: {output_model}[/dim]")
    console.print(f"  [dim]Output folder: {save_dir}/[/dim]")

    tree = prompt_tree(
        prompt, sub_n=sub_n, super_n=super_n, side_n=side_n,
        model=model, viz=True, v=True,
    )

    total = 1
    for direction in ("sub", "super", "side"):
        if direction in tree:
            total += len(tree[direction])
    console.print(f"\n[bold green]{total} prompts generated.[/bold green]")

    tree_path = os.path.join(save_dir, "tree.md")
    _write_tree_md(tree, tree_path, model, sub_n, super_n, side_n, output_model=output_model)
    console.print(f"  [dim]{tree_path}[/dim]")

    graph = tree.get("graph")
    if graph:
        graph_path = os.path.join(save_dir, "tree")
        graph.render(graph_path, cleanup=True)
        console.print(f"  [dim]{graph_path}.svg[/dim]")

    if output_model:
        console.print(f"\n[bold]Generating outputs for {total} prompts...[/bold]")
        result = research_tree(
            tree,
            output_model=output_model,
            search="auto",
            save=save_dir,
            v=True,
        )
        output_count = len(result.get("outputs", []))
        console.print(f"\n[bold green]{output_count} outputs saved.[/bold green]")
        saved = result.get("saved", [])
        if saved:
            console.print(f"  [dim]{os.path.dirname(saved[0])}/[/dim]")

    console.print()
    _print_view_hint(console, save_dir)


def _run_prompt_chain(prompts, model, save_dir=None):
    """Run a sequence of user prompts as a multi-turn chat; save the whole discussion."""
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

    from datetime import datetime
    from rich.console import Console
    from .LLM import LLM

    console = Console()

    if not save_dir:
        save_dir = os.path.join("output", _prompt_to_slug(prompts[0]))
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "chain.md")

    console.print(f"\n[bold]Running chain ({len(prompts)} turn(s))...[/bold]")
    console.print(f"  [dim]Model: {model}[/dim]")
    console.print(f"  [dim]Output: {path}[/dim]\n")

    llm = LLM(model=model, v=False)
    turns = []
    for i, p in enumerate(prompts, 1):
        console.print(f"[bold cyan]Turn {i}/{len(prompts)}[/bold cyan] [dim]{p[:80]}{'...' if len(p) > 80 else ''}[/dim]")
        llm.user(p).chat()
        reply = llm.last() or ""
        turns.append((p, reply))
        console.print(f"  [dim]{len(reply)} chars[/dim]")

    # Build markdown
    lines = ["---"]
    lines.append(f"model: {model}")
    lines.append(f"date: {datetime.now().strftime('%Y-%m-%d')}")
    lines.append(f"turns: {len(prompts)}")
    lines.append("---")
    lines.append("")
    lines.append("# Prompt Chain")
    lines.append("")
    for i, (user_msg, assistant_msg) in enumerate(turns, 1):
        lines.append(f"## Turn {i}")
        lines.append("")
        lines.append("### User")
        lines.append("")
        lines.append(user_msg)
        lines.append("")
        lines.append("### Assistant")
        lines.append("")
        lines.append(assistant_msg)
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))

    console.print(f"\n[bold green]Saved[/bold green] [dim]{path}[/dim]\n")
    _print_view_hint(console, path)


def _run_prompt_manipulate(prompt, model, transforms=None, output_model=None,
                           save_dir=None, concurrency=5):
    """Execute the prompt manipulation pipeline."""
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

    from concurrent.futures import ThreadPoolExecutor, as_completed
    from rich.console import Console
    from .prompt_probe import prompt_transform, list_transforms, _make_translator
    from .utils import _write_consolidated_md

    console = Console()

    # Resolve transforms
    if transforms is None:
        transforms = list_transforms(v=False)

    if not transforms:
        console.print("[yellow]No transforms selected.[/yellow]")
        return

    # Always include the untransformed prompt as a baseline (index 0)
    transforms = ["original"] + [t for t in transforms if t != "original"]

    # Resolve save path early so user sees it
    if not save_dir:
        save_dir = os.path.join("output", _prompt_to_slug(prompt))

    console.print(f"\n[bold]Running {len(transforms)} transforms...[/bold]")
    console.print(f"  [dim]Manipulation model: {model}[/dim]")
    if output_model:
        console.print(f"  [dim]Output model: {output_model}[/dim]")
    console.print(f"  [dim]Output folder: {save_dir}/[/dim]")

    # Phase 1: Run transforms concurrently. Translators are two-phase:
    # mutate returns a foreign-language prompt, postprocess translates the
    # response back to English later. Index 0 ("original") is pre-filled
    # with the untransformed prompt.
    transformed_prompts = [None] * len(transforms)
    postprocessors = [None] * len(transforms)
    transformed_prompts[0] = prompt
    failed = []
    prompts_path = os.path.join(save_dir, "manipulations.md")

    def _save_manipulations():
        return _write_consolidated_md(
            prompts_path, prompt, transforms, transformed_prompts,
            manipulation_model=model,
        )

    def _run_single(idx, name):
        if name.startswith("translate_"):
            language = name[len("translate_"):]
            mutate_fn, postprocess_fn = _make_translator(language, prompt_model=model)
            return idx, name, mutate_fn(prompt), postprocess_fn
        return idx, name, prompt_transform(prompt, name, model=model), None

    executor = ThreadPoolExecutor(max_workers=concurrency)
    interrupted = False
    try:
        futures = {
            executor.submit(_run_single, i, name): i
            for i, name in enumerate(transforms) if i != 0
        }
        console.print(f"  [1/{len(transforms)}] [green]original[/green]")
        done_count = 1
        for future in as_completed(futures):
            try:
                idx, name, result, postprocess_fn = future.result()
                transformed_prompts[idx] = result
                postprocessors[idx] = postprocess_fn
                done_count += 1
                console.print(f"  [{done_count}/{len(transforms)}] [green]{name}[/green]")
            except Exception as e:
                idx = futures[future]
                failed.append((transforms[idx], str(e)))
                done_count += 1
                console.print(f"  [{done_count}/{len(transforms)}] [red]{transforms[idx]} — failed: {e}[/red]")
            _save_manipulations()
        executor.shutdown(wait=True)
    except KeyboardInterrupt:
        interrupted = True
        executor.shutdown(wait=False, cancel_futures=True)
        _save_manipulations()
        console.print(f"\n[yellow]Interrupted — saved partial manipulations to {prompts_path}[/yellow]\n")
        _print_view_hint(console, prompts_path)
        return

    # Remove failed / incomplete transforms
    good_indices = [i for i in range(len(transforms)) if transformed_prompts[i] is not None]
    if len(good_indices) < len(transforms):
        transforms = [transforms[i] for i in good_indices]
        transformed_prompts = [transformed_prompts[i] for i in good_indices]
        postprocessors = [postprocessors[i] for i in good_indices]
        console.print(f"\n[yellow]{len(failed)} transform(s) failed, continuing with {len(transforms)}[/yellow]")

    if not transforms:
        console.print("[red]All transforms failed.[/red]")
        return

    saved = _save_manipulations()
    console.print(f"\n[bold green]{len(transforms)} transforms saved.[/bold green]")
    console.print(f"  [dim]{saved}[/dim]\n")
    if not output_model:
        _print_view_hint(console, saved)

    # Phase 2: Generate outputs and save consolidated file
    if output_model:
        console.print(f"\n[bold]Generating {len(transforms)} outputs with {output_model}...[/bold]")
        from .LLM import LLM

        def _generate_single(idx, name, transformed_prompt):
            llm = LLM(model=output_model, v=False)
            response = llm.user(transformed_prompt).result()
            return idx, name, response

        responses = [None] * len(transforms)
        outputs_path = os.path.join(save_dir, "outputs.md")

        def _save_outputs():
            return _write_consolidated_md(
                outputs_path, prompt, transforms, transformed_prompts,
                manipulation_model=model, output_model=output_model, responses=responses,
            )

        executor = ThreadPoolExecutor(max_workers=concurrency)
        try:
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
                _save_outputs()
            executor.shutdown(wait=True)
        except KeyboardInterrupt:
            executor.shutdown(wait=False, cancel_futures=True)
            _save_outputs()
            done = sum(1 for r in responses if r is not None)
            console.print(f"\n[yellow]Interrupted — saved {done}/{len(transforms)} outputs to {outputs_path}[/yellow]\n")
            _print_view_hint(console, outputs_path)
            return

        # Translate foreign-language responses back to English for translate_* transforms
        def _postprocess_single(idx, name, postprocess_fn, response):
            return idx, name, postprocess_fn(response)

        pending = [
            (i, transforms[i], postprocessors[i], responses[i])
            for i in range(len(transforms))
            if postprocessors[i] is not None and responses[i] is not None
        ]
        if pending:
            console.print(f"\n[bold]Translating {len(pending)} response(s) back to english...[/bold]")
            executor = ThreadPoolExecutor(max_workers=concurrency)
            try:
                futures = {
                    executor.submit(_postprocess_single, i, name, fn, resp): i
                    for i, name, fn, resp in pending
                }
                for future in as_completed(futures):
                    try:
                        idx, name, final = future.result()
                        responses[idx] = final
                        console.print(f"  [green]{name}[/green]")
                    except Exception as e:
                        console.print(f"  [red]{transforms[futures[future]]} — back-translate failed: {e}[/red]")
                    _save_outputs()
                executor.shutdown(wait=True)
            except KeyboardInterrupt:
                executor.shutdown(wait=False, cancel_futures=True)
                _save_outputs()
                console.print(f"\n[yellow]Interrupted during back-translation — saved partial outputs to {outputs_path}[/yellow]\n")
                _print_view_hint(console, outputs_path)
                return

        saved_outputs = _save_outputs()
        output_count = sum(1 for r in responses if r is not None)
        console.print(f"\n[bold green]{output_count} outputs saved.[/bold green]")
        console.print(f"  [dim]{saved_outputs}[/dim]\n")
        _print_view_hint(console, saved_outputs)


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


def _select_compare_models(checked_default=True):
    """Interactive multi-select model picker grouped by category. Returns list of model names."""
    import questionary
    from questionary import Choice, Separator
    from rich.console import Console
    console = Console()

    from .LLM import model_rankings, _get_model_name, _get_reasoning_effort

    defaults = set(_select_default_compare_models(3)) if checked_default else set()

    categories = ["optimal", "best", "fast", "cheap", "open"]
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

    console.print(f"\n  [bold]{len(selected)} model(s) selected:[/bold]")
    for m in selected:
        console.print(f"    • {m}")
    console.print()
    return selected


def _to_anchor(label):
    """Convert model label to a safe HTML anchor ID."""
    import re
    anchor = label.lower()
    anchor = re.sub(r'[^\w-]', '-', anchor)
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
        lines.append(f'<a id="{anchor}"></a>\n')
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



def _collect_eval_metrics():
    """Collect evaluation metrics from user. Returns None for auto-generate, or list of custom metric strings."""
    import questionary
    from rich.console import Console
    console = Console()

    choice = questionary.select(
        "Evaluation metrics:",
        choices=[
            questionary.Choice("Auto-generate metrics", value="auto"),
            questionary.Choice("Define custom metrics", value="custom"),
        ],
        default="auto",
    ).ask()
    if choice is None or choice == "auto":
        return None

    metrics = []
    console.print("\n  [dim]Add evaluation metrics one at a time. Press Enter with empty input when done.[/dim]\n")
    while True:
        label = f"  Metric {len(metrics) + 1}" if not metrics else f"  Metric {len(metrics) + 1} (Enter to finish)"
        metric = questionary.text(label).ask()
        if metric is None:
            # Cancelled
            if metrics:
                return metrics
            return None
        if not metric.strip():
            break
        metrics.append(metric.strip())
        console.print(f"    [green]Added:[/green] {metric.strip()}")

    if not metrics:
        console.print("  [dim]No custom metrics — will auto-generate.[/dim]")
        return None

    console.print(f"\n  [bold]{len(metrics)} custom metric(s) defined[/bold]\n")
    return metrics


def _run_compare(prompt, models, evaluate, eval_model, save_dir, metrics=None):
    """Execute model comparison pipeline."""
    import threading
    from rich.console import Console
    from .compare_models import compare_models, validate_model
    from .evaluate import pairwise_evaluate

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

    # Write markdown
    filepath = os.path.join(save_dir, "comparison.md")
    _write_compare_md(result, filepath)
    console.print(f"\n[bold green]Saved to: {filepath}[/bold green]")

    # Run pairwise evaluation in background thread if requested
    if evaluate:
        eval_model_name = eval_model or "rank 1 (default)"
        console.print(f"\n[bold cyan]Running pairwise evaluation...[/bold cyan]")
        console.print(f"  [dim]Eval model: {eval_model_name}[/dim]")
        if metrics:
            console.print(f"  [dim]Custom metrics: {len(metrics)}[/dim]")

        def run_eval():
            try:
                eval_result = pairwise_evaluate(
                    prompt=prompt,
                    results=result["responses"],
                    metrics=metrics,
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

    console.print()
    _print_view_hint(console, filepath)


def _write_grid_md(result, filepath):
    """Write grid search results to a single consolidated markdown file."""
    import os
    from datetime import date

    prompt = result["prompt"]
    grid = result["grid"]  # sorted by rank when evaluated
    models = result["models"]
    transforms = result["transforms"]
    evaluation = result.get("evaluation")

    os.makedirs(os.path.dirname(os.path.abspath(filepath)) or ".", exist_ok=True)

    lines = []
    # YAML frontmatter
    lines.append("---")
    lines.append(f"prompt: {prompt}")
    lines.append("models:")
    for m in models:
        lines.append(f"  - {m}")
    lines.append("transforms:")
    for t in transforms:
        lines.append(f"  - {t}")
    if result.get("top_transform"):
        lines.append(f"best_transform: {result['top_transform']}")
    if result.get("top_model"):
        lines.append(f"best_model: {result['top_model']}")
    lines.append(f"date: {date.today().isoformat()}")
    lines.append("---\n")

    # Header
    lines.append("# Grid Search Results\n")
    lines.append(f"> **Prompt:** {prompt}\n")

    # Summary
    if result.get("top_transform") and result.get("top_model"):
        best_score = grid[0].get("score", "N/A")
        score_str = f"{best_score:.2f}" if isinstance(best_score, (int, float)) else str(best_score)
        lines.append(f"**Best:** {result['top_transform']} x {result['top_model']} (score: {score_str})\n")

    # Rankings table
    lines.append("## Rankings\n")
    lines.append("| Rank | Transform | Model | Score | Link |")
    lines.append("|------|-----------|-------|-------|------|")
    for cell in grid:
        rank = cell.get("rank", "-")
        score = cell.get("score", "-")
        score_str = f"{score:.2f}" if isinstance(score, (int, float)) else str(score)
        anchor = _to_anchor(f"{cell['transform']}--{cell['model']}")
        lines.append(f"| {rank} | {cell['transform']} | {cell['model']} | {score_str} | [View](#{anchor}) |")
    lines.append("")

    # Evaluation metadata
    if evaluation:
        eval_metrics = evaluation.get("raw", {}).get("metrics_used", [])
        if eval_metrics:
            lines.append("## Evaluation\n")
            lines.append("**Metrics:**")
            for m in eval_metrics:
                lines.append(f"- {m}")
            lines.append("")

    # Individual cell sections (ordered by rank)
    for cell in grid:
        anchor = _to_anchor(f"{cell['transform']}--{cell['model']}")
        rank = cell.get("rank", "?")
        score = cell.get("score")
        score_str = f" ({score:.2f})" if isinstance(score, (int, float)) else ""

        lines.append("---\n")
        lines.append(f'<a id="{anchor}"></a>\n')
        lines.append(f"## #{rank}: {cell['transform']} x {cell['model']}{score_str}\n")

        # Show the transformed prompt if different from original
        if cell["transform"] != "original":
            lines.append("**Prompt used:**\n")
            for pline in cell["prompt"].splitlines():
                lines.append(f"> {pline}" if pline.strip() else ">")
            lines.append("")

        lines.append(cell["response"])
        lines.append("")

    with open(filepath, "w") as f:
        f.write("\n".join(lines))

    return filepath


def _run_grid_search(prompt, models, transforms, eval_model, save_dir):
    """Execute grid search pipeline."""
    from rich.console import Console
    from .grid_search import grid_search
    from .compare_models import validate_model

    console = Console()

    # Resolve models for display
    resolved = []
    for m in models:
        resolved.append(validate_model(m))
    models = resolved

    console.print(f"\n[bold]Running grid search...[/bold]")
    console.print(f"  [dim]{len(transforms)} transforms x {len(models)} models[/dim]")
    console.print()

    # Run grid search with evaluation
    result = grid_search(
        prompt,
        models=models,
        transforms=transforms,
        evaluate=True,
        save=os.path.join(save_dir, "grid"),
        concurrency=5,
        v=True,
        eval_model=eval_model,
    )

    # Write consolidated markdown
    filepath = os.path.join(save_dir, "grid_search.md")
    _write_grid_md(result, filepath)
    console.print(f"\n[bold green]Saved to: {filepath}[/bold green]")

    # Also show individual cell files
    if result.get("saved"):
        console.print(f"  [dim]{len(result['saved'])} individual cell files in {save_dir}/grid/[/dim]")

    # Print best combination summary
    if result.get("top_transform"):
        best_score = result["evaluation"]["scores"][result["rankings"][0]]
        console.print(f"\n[bold cyan]Best combination:[/bold cyan]")
        console.print(f"  Transform: [bold]{result['top_transform']}[/bold]")
        console.print(f"  Model: [bold]{result['top_model']}[/bold]")
        console.print(f"  Score: [bold]{best_score:.2f}[/bold]")

    console.print()
    _print_view_hint(console, filepath)


def _interactive_main():
    """Interactive mode — shown when user runs `spaceshift` with no args."""
    import questionary
    from questionary import Choice, Separator
    from rich.console import Console
    console = Console()

    console.print("\n[bold]spaceshift[/bold] [dim]— move through prompts in every direction[/dim]\n")

    # Show current providers
    providers = _get_providers()
    if providers:
        console.print(f"  [dim]{len(providers)} provider(s): {', '.join(providers)}[/dim]\n")

    while True:
        mode = questionary.select(
            "What would you like to do?",
            choices=[
                Choice("Prompt Manipulate", value="prompt"),
                Choice("Compare Models", value="compare"),
                Choice("Grid Search and Evaluation", value="grid"),
                Choice("Prompt Tree", value="tree"),
                Choice("Prompt Chain", value="chain"),
                Separator(),
                Choice("Manage API Keys", value="keys"),
            ],
        ).ask()

        if mode is None:
            break

        if mode == "prompt":
            step = 0
            prompt = selected = model = generate_outputs = output_model = save_dir = None
            while step >= 0:
                if step == 0:
                    prompt = questionary.text(
                        "Enter your prompt:",
                        validate=lambda t: len(t.strip()) > 0 or "Enter a prompt",
                    ).ask()
                    if prompt is None:
                        step = -1
                    else:
                        step = 1
                elif step == 1:
                    selected = _select_transforms(checked_default=True)
                    if selected is None:
                        step = 0
                    elif not selected:
                        console.print("[yellow]No transforms selected.[/yellow]\n")
                    else:
                        from .prompt_probe import list_transforms
                        all_transforms = list_transforms(v=False)
                        console.print(f"\n  [bold]{len(selected)}[/bold] of {len(all_transforms)} transforms selected\n")
                        step = 2
                elif step == 2:
                    console.print("\n[bold]Select manipulation model[/bold] (for transforming prompts):")
                    model = _select_model(None)
                    if model is None:
                        step = 1
                    else:
                        step = 3
                elif step == 3:
                    generate_outputs = questionary.confirm(
                        "Generate outputs too?",
                        default=False,
                    ).ask()
                    if generate_outputs is None:
                        step = 2
                    elif generate_outputs:
                        step = 4
                    else:
                        output_model = None
                        step = 5
                elif step == 4:
                    console.print("\n[bold]Select output model[/bold] (for generating responses):")
                    output_model = _select_model(None)
                    if output_model is None:
                        step = 3
                    else:
                        step = 5
                elif step == 5:
                    default_dir = os.path.join("output", _prompt_to_slug(prompt))
                    save_dir = questionary.text(
                        "Output folder:",
                        default=default_dir,
                    ).ask()
                    if save_dir is None:
                        step = 4 if generate_outputs else 3
                    else:
                        step = 6
                else:
                    break
            if step < 0:
                continue
            _run_prompt_manipulate(
                prompt.strip(), model,
                transforms=selected,
                output_model=output_model,
                save_dir=save_dir.strip(),
            )
        elif mode == "compare":
            step = 0
            prompt = models = run_eval = eval_model = eval_metrics = save_dir = None
            while step >= 0:
                if step == 0:
                    prompt = questionary.text(
                        "Enter your prompt:",
                        validate=lambda t: len(t.strip()) > 0 or "Enter a prompt",
                    ).ask()
                    if prompt is None:
                        step = -1
                    else:
                        step = 1
                elif step == 1:
                    models = _select_compare_models()
                    if models is None:
                        step = 0
                    else:
                        step = 2
                elif step == 2:
                    eval_choice = questionary.select(
                        "Evaluation:",
                        choices=[
                            "Pairwise Evaluate after Generation",
                            "No Evaluation",
                        ],
                    ).ask()
                    if eval_choice is None:
                        step = 1
                    else:
                        run_eval = eval_choice == "Pairwise Evaluate after Generation"
                        if run_eval:
                            step = 3
                        else:
                            eval_model = None
                            step = 4
                elif step == 3:
                    eval_model = _select_model(None, prompt_text="Select a model to judge the evaluation (use arrow keys):")
                    if eval_model is None:
                        step = 2
                    else:
                        eval_metrics = _collect_eval_metrics()
                        step = 4
                elif step == 4:
                    default_dir = os.path.join("output", _prompt_to_slug(prompt))
                    save_dir = questionary.text(
                        "Output folder:",
                        default=default_dir,
                    ).ask()
                    if save_dir is None:
                        step = 3 if run_eval else 2
                    else:
                        step = 5
                else:
                    break
            if step < 0:
                continue
            eval_metrics = eval_metrics if run_eval else None
            _run_compare(prompt.strip(), models, evaluate=run_eval, eval_model=eval_model, save_dir=save_dir.strip(), metrics=eval_metrics)
            break
        elif mode == "grid":
            console.print("\n[bold]Grid Search and Evaluation[/bold]\n")
            console.print("  Combines [bold]prompt transforms[/bold] x [bold]models[/bold] into a grid,")
            console.print("  generates all responses, then uses [bold]pairwise evaluation[/bold]")
            console.print("  to rank and find the best combination.\n")

            step = 0
            prompt = models = selected_transforms = eval_model = save_dir = None
            while step >= 0:
                if step == 0:
                    prompt = questionary.text(
                        "Enter your prompt:",
                        validate=lambda t: len(t.strip()) > 0 or "Enter a prompt",
                    ).ask()
                    if prompt is None:
                        step = -1
                    else:
                        step = 1
                elif step == 1:
                    models = _select_compare_models(checked_default=False)
                    if models is None:
                        step = 0
                    else:
                        step = 2
                elif step == 2:
                    selected_transforms = _select_transforms(checked_default=False)
                    if selected_transforms is None:
                        step = 1
                    elif not selected_transforms:
                        console.print("[yellow]No transforms selected.[/yellow]\n")
                    else:
                        # Grid summary
                        n_models = len(models)
                        n_transforms = len(selected_transforms)
                        n_cells = n_transforms * n_models + n_models
                        if n_cells <= 5:
                            n_pairs = n_cells * (n_cells - 1) // 2
                        else:
                            all_possible = n_cells * (n_cells - 1) // 2
                            n_pairs = min(all_possible, n_cells * 5)
                        n_eval_calls = n_pairs * 2

                        console.print(f"\n[bold]Grid summary:[/bold]")
                        console.print(f"  {n_transforms} transform{'s' if n_transforms != 1 else ''} + 1 original x {n_models} models = Evaluating [bold]{n_cells}[/bold] prompt/model combinations")
                        console.print(f"  Pairwise comparisons: [bold]{n_pairs}[/bold] ({n_eval_calls} LLM eval calls with position swap)\n")
                        step = 3
                elif step == 3:
                    eval_model = _select_model(None, prompt_text="Select a model to judge the evaluation (use arrow keys):")
                    if eval_model is None:
                        step = 2
                    else:
                        step = 4
                elif step == 4:
                    default_dir = os.path.join("output", _prompt_to_slug(prompt))
                    save_dir = questionary.text(
                        "Output folder:",
                        default=default_dir,
                    ).ask()
                    if save_dir is None:
                        step = 3
                    else:
                        step = 5
                else:
                    break
            if step < 0:
                continue
            _run_grid_search(
                prompt.strip(), models, selected_transforms,
                eval_model=eval_model, save_dir=save_dir.strip(),
            )
            break
        elif mode == "keys":
            _manage_api_keys(first_time=False)
            # Reload provider display
            providers = _get_providers()
            if providers:
                console.print(f"  [dim]{len(providers)} provider(s): {', '.join(providers)}[/dim]\n")
        elif mode == "tree":
            prompt = questionary.text(
                "Enter your prompt:",
                validate=lambda t: len(t.strip()) > 0 or "Enter a prompt",
            ).ask()
            if prompt is None:
                continue

            console.print("\n[bold]Configure tree directions[/bold]\n")

            # --- Sub ---
            console.print("  [bold cyan]Sub (Decomposition)[/bold cyan]")
            console.print("  [dim]Breaks your prompt into narrower, focused subprompts — goes DOWN into specifics.[/dim]\n")
            sub_depth = questionary.text(
                "  How many levels deep? (0 to skip):",
                default="1",
                validate=lambda t: t.strip().isdigit() or "Enter a number (0 to skip)",
            ).ask()
            if sub_depth is None:
                continue
            sub_n = None
            sub_depth = int(sub_depth.strip())
            if sub_depth > 0:
                sub_n = []
                for level in range(sub_depth):
                    default = "5" if level == 0 else "3"
                    count = questionary.text(
                        f"  Prompts at level {level + 1}?",
                        default=default,
                        validate=lambda t: (t.strip().isdigit() and int(t.strip()) > 0) or "Enter a positive number",
                    ).ask()
                    if count is None:
                        sub_n = None
                        break
                    sub_n.append(int(count.strip()))
                if sub_n is not None and len(sub_n) != sub_depth:
                    continue

            # --- Super ---
            console.print("\n  [bold red]Super (Abstraction)[/bold red]")
            console.print("  [dim]Generates broader parent prompts that contain yours — goes UP in abstraction.[/dim]\n")
            super_depth = questionary.text(
                "  How many levels deep? (0 to skip):",
                default="1",
                validate=lambda t: t.strip().isdigit() or "Enter a number (0 to skip)",
            ).ask()
            if super_depth is None:
                continue
            super_n = None
            super_depth = int(super_depth.strip())
            if super_depth > 0:
                super_n = []
                for level in range(super_depth):
                    default = "3"
                    count = questionary.text(
                        f"  Prompts at level {level + 1}?",
                        default=default,
                        validate=lambda t: (t.strip().isdigit() and int(t.strip()) > 0) or "Enter a positive number",
                    ).ask()
                    if count is None:
                        super_n = None
                        break
                    super_n.append(int(count.strip()))
                if super_n is not None and len(super_n) != super_depth:
                    continue

            # --- Side ---
            console.print("\n  [bold green]Side (Lateral)[/bold green]")
            console.print("  [dim]Creates alternative prompts at the same level — goes SIDEWAYS to explore parallels.[/dim]\n")
            side_depth = questionary.text(
                "  How many levels deep? (0 to skip):",
                default="1",
                validate=lambda t: t.strip().isdigit() or "Enter a number (0 to skip)",
            ).ask()
            if side_depth is None:
                continue
            side_n = None
            side_depth = int(side_depth.strip())
            if side_depth > 0:
                side_n = []
                for level in range(side_depth):
                    default = "4"
                    count = questionary.text(
                        f"  Prompts at level {level + 1}?",
                        default=default,
                        validate=lambda t: (t.strip().isdigit() and int(t.strip()) > 0) or "Enter a positive number",
                    ).ask()
                    if count is None:
                        side_n = None
                        break
                    side_n.append(int(count.strip()))
                if side_n is not None and len(side_n) != side_depth:
                    continue

            if sub_n is None and super_n is None and side_n is None:
                console.print("[yellow]At least one direction must have depth > 0.[/yellow]\n")
                continue

            console.print("")
            total = 1
            for label, n_list in [("Sub", sub_n), ("Super", super_n), ("Side", side_n)]:
                if n_list:
                    count, breakdown = _calculate_tree_count(n_list)
                    if " + " in breakdown:
                        console.print(f"  [dim]{label}: {breakdown} = {count}[/dim]")
                    else:
                        console.print(f"  [dim]{label}: {count}[/dim]")
                    total += count
            console.print(f"\n  [bold]{total} total prompts[/bold] will be generated\n")

            console.print("[bold]Select tree-building model[/bold] (for generating prompts):")
            model = _select_model(None)
            if model is None:
                continue

            output_choice = questionary.select(
                "Outputs:",
                choices=[
                    "No outputs (tree only)",
                    "Generate LLM outputs for every prompt in the tree",
                ],
            ).ask()
            if output_choice is None:
                continue
            generate_outputs = output_choice.startswith("Generate")

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

            _run_prompt_tree(
                prompt.strip(), model,
                sub_n=sub_n, super_n=super_n, side_n=side_n,
                output_model=output_model,
                save_dir=save_dir.strip(),
            )
            break
        elif mode == "chain":
            prompt = questionary.text(
                "Enter your initial prompt:",
                validate=lambda t: len(t.strip()) > 0 or "Enter a prompt",
            ).ask()
            if prompt is None:
                continue

            prompts = [prompt.strip()]
            console.print("\n[dim]Add followups one at a time. Leave blank and press Enter to finish.[/dim]\n")
            while True:
                followup = questionary.text(
                    f"  Followup {len(prompts)} (blank to finish):",
                ).ask()
                if followup is None:
                    prompts = None
                    break
                if not followup.strip():
                    break
                prompts.append(followup.strip())

            if prompts is None:
                continue

            console.print(f"\n  [bold]{len(prompts)} turn(s)[/bold] queued\n")

            console.print("[bold]Select model[/bold]:")
            model = _select_model(None)
            if model is None:
                continue

            default_dir = os.path.join("output", _prompt_to_slug(prompts[0]))
            save_dir = questionary.text(
                "Output folder:",
                default=default_dir,
            ).ask()
            if save_dir is None:
                continue

            _run_prompt_chain(prompts, model, save_dir=save_dir.strip())
            break
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

    # compare subcommand
    c = sub.add_parser("compare", help="Compare a prompt across multiple models")
    c.add_argument("prompt", help="Prompt to compare across models")
    c.add_argument("--models", "-m", nargs="+", default=None, help="Models to compare (names, shorthands, or rank numbers). Auto-selects if omitted.")
    c.add_argument("--evaluate", "-e", action="store_true", help="Run pairwise evaluation after generating responses")
    c.add_argument("--eval-model", default=None, help="Model to use for pairwise evaluation (default: rank 1)")
    c.add_argument("--metrics", nargs="+", default=None, help="Custom evaluation metrics (default: auto-generate)")
    c.add_argument("--save", "-s", default=None, help="Output directory (auto-named from prompt if omitted)")

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
    elif args.command == "compare":
        _ensure_api_keys()
        models = args.models or _select_default_compare_models(3)
        save_dir = args.save or os.path.join("output", _prompt_to_slug(args.prompt))
        _run_compare(args.prompt, models, args.evaluate, args.eval_model, save_dir, metrics=args.metrics)
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
