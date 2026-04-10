from concurrent.futures import ThreadPoolExecutor, as_completed

from .LLM import LLM
from .compare_models import validate_model, parse_model_string, _model_label, _DEFAULT_MODELS, _is_shorthand, _has_search, _top_search_models
from .prompt_probe import _resolve_transforms
from .evaluate import pairwise_evaluate


_DEFAULT_N_MODELS = 4
_DEFAULT_N_TRANSFORMS = 4


def grid_search(
    prompt,
    models=None,
    transforms=None,
    n_transforms=None,
    metrics=None,
    evaluate=True,
    prompt_model=None,
    search=False,
    search_prompts=False,
    save=None,
    concurrency=5,
    v=True,
    **eval_kwargs,
):
    """
    Search across models × prompt transforms and rank all responses flat.

    Combines compare_models (vary the model) with prompt_probe (vary the prompt)
    into a single N×M grid, then uses pairwise evaluation to find the best cell.

    Args:
        prompt: The original prompt.
        models: List of model identifiers (shorthands, litellm names, or 'name(effort)').
            Defaults to top 4 ranked models.
        transforms: List of transform names/callables, or None to auto-select.
        n_transforms: How many random transforms when transforms is None. Defaults to 4.
        metrics: Evaluation metrics for pairwise_evaluate. None = auto-generate.
        evaluate: If False, skip evaluation and return the grid only.
        prompt_model: Model override for LLM-based prompt transforms.
        search (bool): Enable web search for response generation (output step).
            When models are all shorthands, auto-selects search-capable models.
            When explicit models are given, validates search support.
        search_prompts (bool): Enable web search for LLM-based prompt transforms.
        save: Path to save responses as .md files. None = don't save.
        concurrency: Max parallel threads. Defaults to 5.
        v: Verbose output.
        **eval_kwargs: Additional kwargs passed to pairwise_evaluate.

    Returns:
        dict with keys: top_output, top_model, top_transform, top_prompt,
        grid (list of cell dicts), rankings, scores, evaluation
    """
    n_transforms = n_transforms or _DEFAULT_N_TRANSFORMS
    models = models or _DEFAULT_MODELS[:_DEFAULT_N_MODELS]

    all_shorthands = all(_is_shorthand(parse_model_string(m)[0]) for m in models)
    if search and all_shorthands:
        models = _top_search_models(len(models))
        if v:
            print(f"Auto-selected search models: {models}")

    # Resolve models
    parsed = [parse_model_string(m) for m in models]
    resolved_models = []
    efforts = []
    for model_id, effort in parsed:
        resolved_models.append(validate_model(model_id))
        efforts.append(effort)

    if search and not all_shorthands:
        no_search = [name for name in resolved_models if not _has_search(name)]
        if no_search:
            suggestions = _top_search_models()
            raise ValueError(
                "These models do not support search:\n"
                + "".join(f"  - {name}\n" for name in no_search)
                + "Models with search support:\n"
                + "".join(f"  - {name}\n" for name in suggestions)
            )

    model_labels = [_model_label(resolved_models[i], efforts[i]) for i in range(len(models))]

    # Resolve transforms (includes "original" prepended after)
    resolved_transforms = _resolve_transforms(transforms, n_transforms, prompt_model, search=search_prompts, v=v)
    transform_names = [name for name, _ in resolved_transforms]

    if v:
        n_total = len(transform_names) * len(model_labels) + len(model_labels)
        print(f"Grid: {len(transform_names)} transform{'s' if len(transform_names) != 1 else ''} + 1 original × {len(model_labels)} models = {n_total} prompt/model combinations")
        print(f"  Transforms: {transform_names}")
        print(f"  Models: {model_labels}")

    # Phase 1: Run transforms concurrently
    mutated_prompts = [None] * len(resolved_transforms)
    postprocessors = [None] * len(resolved_transforms)

    def _run_transform(idx, name, transform):
        if isinstance(transform, tuple):
            mutate_fn, postprocess_fn = transform
            return idx, mutate_fn(prompt), postprocess_fn
        return idx, transform(prompt), None

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(_run_transform, i, name, t): i
            for i, (name, t) in enumerate(resolved_transforms)
        }
        for future in as_completed(futures):
            idx, result, postprocess_fn = future.result()
            mutated_prompts[idx] = result
            postprocessors[idx] = postprocess_fn
            if v:
                print(f"  [{transform_names[idx]}] {result[:80]}{'...' if len(result) > 80 else ''}")

    # Build full prompt list: original + transforms
    all_transform_names = ["original"] + transform_names
    all_prompts = [prompt] + mutated_prompts
    all_postprocessors = [None] + postprocessors

    # Phase 2: Build grid and generate responses
    # Each cell = (transform_idx, model_idx)
    cells = []
    for t_idx in range(len(all_prompts)):
        for m_idx in range(len(resolved_models)):
            cells.append((t_idx, m_idx))

    if v:
        print(f"\nGenerating {len(cells)} responses...")

    responses = [None] * len(cells)

    def _run_cell(cell_idx):
        t_idx, m_idx = cells[cell_idx]
        kwargs = {"model": resolved_models[m_idx], "v": False, "sub_closest_model": False}
        if efforts[m_idx]:
            kwargs["reasoning"] = True
            kwargs["reasoning_effort"] = efforts[m_idx]
        if search:
            kwargs["search"] = True
        llm = LLM(**kwargs)
        return cell_idx, llm.user(all_prompts[t_idx]).result()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_run_cell, i): i for i in range(len(cells))}
        done = 0
        for future in as_completed(futures):
            cell_idx, response = future.result()
            responses[cell_idx] = response
            done += 1
            if v:
                t_idx, m_idx = cells[cell_idx]
                print(f"  [{done}/{len(cells)}] {all_transform_names[t_idx]} × {model_labels[m_idx]}")

    # Phase 3: Postprocess (translations)
    for cell_idx, (t_idx, m_idx) in enumerate(cells):
        postprocess_fn = all_postprocessors[t_idx]
        if postprocess_fn is not None:
            if v:
                print(f"  Translating {all_transform_names[t_idx]} × {model_labels[m_idx]} back...")
            responses[cell_idx] = postprocess_fn(responses[cell_idx])

    # Build grid structure
    grid = []
    for cell_idx, (t_idx, m_idx) in enumerate(cells):
        grid.append({
            "transform": all_transform_names[t_idx],
            "model": model_labels[m_idx],
            "prompt": all_prompts[t_idx],
            "response": responses[cell_idx],
        })

    if not evaluate:
        result = {
            "prompt": prompt,
            "grid": grid,
            "transforms": all_transform_names,
            "models": model_labels,
            "responses": responses,
        }
        if save:
            _save_grid(result, save)
        return result

    # Phase 4: Evaluate all responses flat
    if v:
        print(f"\nEvaluating {len(responses)} responses...")

    eval_kwargs.setdefault('v', v)
    evaluation = pairwise_evaluate(prompt=prompt, results=responses, metrics=metrics, **eval_kwargs)

    rankings = evaluation['rankings']
    best_idx = rankings[0]
    best_t_idx, best_m_idx = cells[best_idx]

    # Attach scores to grid
    for cell_idx, cell in enumerate(grid):
        cell["score"] = evaluation['scores'][cell_idx]
        cell["rank"] = rankings.index(cell_idx) + 1

    result = {
        "prompt": prompt,
        "top_output": responses[best_idx],
        "top_model": model_labels[best_m_idx],
        "top_transform": all_transform_names[best_t_idx],
        "top_prompt": all_prompts[best_t_idx],
        "grid": sorted(grid, key=lambda c: c["rank"]),
        "transforms": all_transform_names,
        "models": model_labels,
        "responses": responses,
        "scores": evaluation['scores'],
        "rankings": rankings,
        "evaluation": evaluation,
    }

    if save:
        _save_grid(result, save)

    return result


def _save_grid(result, save):
    from .utils import to_md, _write_md
    import os

    base = save.removesuffix(".md")
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)

    grid = result['grid']
    scores = result.get('scores')

    paths = []
    for i, cell in enumerate(grid):
        label = f"{cell['transform']}__{cell['model']}"
        if 'rank' in cell:
            label = f"{cell['rank']}_{label}"

        meta = {
            "prompt": cell["prompt"],
            "transform": cell["transform"],
            "model": cell["model"],
        }
        if 'score' in cell:
            meta["score"] = cell["score"]
        if 'rank' in cell:
            meta["rank"] = cell["rank"]

        p = f"{base}_{label}.md"
        _write_md(p, cell["response"], meta)
        paths.append(p)

    result["saved"] = paths
