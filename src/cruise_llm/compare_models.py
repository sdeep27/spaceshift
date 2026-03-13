import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import litellm
import rapidfuzz

from .LLM import LLM, model_rankings, _get_model_name, _WEB_SEARCH_OVERRIDES
from .evaluate import pairwise_evaluate

_available_models = None

def _get_available_models():
    global _available_models
    if _available_models is None:
        _available_models = LLM(v=False).get_models()
    return _available_models


def parse_model_string(model_str):
    """Parse 'model_name (effort)' or 'model_name(effort)' into (model, effort).
    Returns (model_str, None) if no parentheses."""
    if not isinstance(model_str, str):
        return model_str, None
    m = re.match(r'^(.+?)\s*\((\w+)\)\s*$', model_str)
    if m:
        return m.group(1).strip(), m.group(2).lower()
    return model_str, None


def _is_shorthand(model_id):
    """Check if model_id is a category shorthand (int, 'best1', 'open', etc.)."""
    if isinstance(model_id, (int, float)):
        return True
    if not isinstance(model_id, str):
        return False
    s = model_id.lower()
    categories = {'best', 'cheap', 'fast', 'open', 'optimal', 'codex', 'reasoning', 'search'}
    if s.isdigit():
        return True
    if s in categories:
        return True
    m = re.match(r'^([a-z]+)\d+$', s)
    return m is not None and m.group(1) in categories


def validate_model(model_id):
    """Resolve model_id to a litellm name. Shorthands (1, best1, open) resolve normally.
    Exact litellm names pass through. Anything else raises with fuzzy suggestions."""
    available = _get_available_models()
    if isinstance(model_id, str) and model_id in available:
        return model_id

    if _is_shorthand(model_id):
        try:
            llm = LLM(model=model_id, v=False)
            return llm.model
        except Exception:
            pass

    clean = model_id.strip() if isinstance(model_id, str) else str(model_id)
    matches = rapidfuzz.process.extract(clean, available, scorer=rapidfuzz.fuzz.WRatio, limit=5)
    suggestions = "\n".join(f"    - {name}" for name, score, _ in matches)
    raise ValueError(f"Model '{clean}' not recognized. Did you mean:\n{suggestions}")


def _model_label(resolved_name, effort):
    """Filesystem-safe label for a model, e.g. 'claude-opus-4-6' or 'gpt-5_xhigh'."""
    base = resolved_name.replace("/", "-")
    if effort:
        return f"{base}_{effort}"
    return base


def _has_search(model):
    return model in _WEB_SEARCH_OVERRIDES or litellm.supports_web_search(model=model) == True


def _top_search_models(n=5):
    """Return top n search-capable models, zipped best/optimal."""
    best_entries = model_rankings.get('best', [])
    optimal_entries = model_rankings.get('optimal', [])
    zipped = []
    for i in range(max(len(best_entries), len(optimal_entries))):
        if i < len(best_entries):
            zipped.append(_get_model_name(best_entries[i]))
        if i < len(optimal_entries):
            zipped.append(_get_model_name(optimal_entries[i]))
    seen = set()
    result = []
    for name in zipped:
        if name not in seen and _has_search(name):
            seen.add(name)
            result.append(name)
            if len(result) >= n:
                return result
    return result


_DEFAULT_MODELS = [1, 2, 3, 4, 5]


def compare_models(prompt, models=None, metrics=None, evaluate=True, search=False, concurrency=5, v=True, **eval_kwargs):
    """
    Compare a prompt across different models and rank the results.

    Runs a prompt through multiple models in parallel, then uses pairwise evaluation
    to rank which model produced the best response.

    Args:
        prompt (str): The prompt to test across models.
        models: List of model identifiers. Supports cruise-llm shorthands (1, 'best1', 'fast2'),
            litellm model names ('claude-opus-4-6'), and inline reasoning effort via
            parentheses ('gpt-5.4(xhigh)', 'claude-opus-4-6 (low)'). Defaults to [1,2,3,4,5].
        metrics: Evaluation metrics passed to pairwise_evaluate. None = auto-generate.
        evaluate (bool): If False, skip evaluation and return responses only.
        search (bool): Enable web search. When models are all shorthands, auto-selects
            search-capable models. When explicit models are given, validates search support.
        concurrency (int): Max parallel model calls. Defaults to 5.
        v (bool): Verbose output. Defaults to True.
        **eval_kwargs: Additional kwargs passed to pairwise_evaluate.

    Returns:
        dict with keys: rankings, responses, models, configs, evaluation
        (without evaluate: responses, models)
    """
    models = models or _DEFAULT_MODELS
    all_shorthands = all(_is_shorthand(parse_model_string(m)[0]) for m in models)

    if search and all_shorthands:
        models = _top_search_models(len(models))
        if v:
            print(f"Auto-selected search models: {models}")

    if v:
        print(f"Comparing models: {models}")

    parsed = [parse_model_string(m) for m in models]

    resolved = []
    efforts = []
    for (model_id, effort), original in zip(parsed, models):
        resolved.append(validate_model(model_id))
        efforts.append(effort)

    if search and not all_shorthands:
        no_search = [name for name in resolved if not _has_search(name)]
        if no_search:
            suggestions = _top_search_models()
            raise ValueError(
                "These models do not support search:\n"
                + "".join(f"  - {name}\n" for name in no_search)
                + "Models with search support:\n"
                + "".join(f"  - {name}\n" for name in suggestions)
            )

    if v:
        for i, (name, effort) in enumerate(zip(resolved, efforts)):
            tag = f" (reasoning={effort})" if effort else ""
            search_tag = " [search]" if search else ""
            print(f"  [{i+1}] {name}{tag}{search_tag}")

    results = [None] * len(models)

    def run_model(idx):
        kwargs = {"model": resolved[idx], "v": False, "sub_closest_model": False}
        if efforts[idx]:
            kwargs["reasoning"] = True
            kwargs["reasoning_effort"] = efforts[idx]
        if search:
            kwargs["search"] = True
        llm = LLM(**kwargs)
        return idx, llm.user(prompt).res()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(run_model, i): i for i in range(len(models))}
        for future in as_completed(futures):
            idx, response = future.result()
            results[idx] = response
            if v:
                print(f"  [{idx+1}/{len(models)}] {resolved[idx]} done")

    model_labels = [_model_label(resolved[i], efforts[i]) for i in range(len(models))]

    if not evaluate:
        return {
            "responses": results,
            "models": model_labels,
        }

    if v:
        print(f"\nEvaluating {len(results)} responses...")

    eval_kwargs.setdefault('v', v)
    evaluation = pairwise_evaluate(prompt=prompt, results=results, metrics=metrics, **eval_kwargs)

    configs = {
        model_labels[i]: {
            "model_shorthand": models[i],
            "model_resolved": resolved[i],
            "reasoning_effort": efforts[i],
            "rank": evaluation['rankings'].index(i) + 1,
            "score": evaluation['scores'][i]
        }
        for i in range(len(models))
    }

    return {
        "rankings": [model_labels[i] for i in evaluation['rankings']],
        "responses": results,
        "models": model_labels,
        "configs": configs,
        "evaluation": evaluation,
    }
