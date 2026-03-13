import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from .LLM import LLM
from .evaluate import pairwise_evaluate

_TRANSFORMS_DIR = os.path.join(os.path.dirname(__file__), "prompt_transforms")

_LLM_TRANSFORMS = {
    "inverse": "inverse.json",
    "abstract_up": "abstract_up.json",
    "abstract_down": "abstract_down.json",
    "abstract_up3": "abstract_up3.json",
    "abstract_down3": "abstract_down3.json",
    "active_up": "active_up.json",
    "passive_up": "passive_up.json",
    "reflection": "reflection.json",
    "rotation": "rotation.json",
    "shear": "shear.json",
    "scaling": "scaling.json",
    "recursion": "recursion.json",
    "dimension_up": "dimension_up.json",
    "dimension_down": "dimension_down.json",
    "improve_naive": "improve_naive.json",
}

_DETERMINISTIC_TRANSFORMS = {
    "double": lambda prompt: prompt + "\n\n" + prompt,
}

_BUILTIN_TRANSLATORS = ["translate_chinese", "translate_korean", "translate_hindi", "translate_french", "translate_arabic"]

_ALL_BUILTINS = list(_LLM_TRANSFORMS.keys()) + list(_DETERMINISTIC_TRANSFORMS.keys()) + _BUILTIN_TRANSLATORS


def _load_transform(name, prompt_model=None, v=False):
    """Load a saved LLM transform by name, returning a callable (prompt) -> str."""
    path = os.path.join(_TRANSFORMS_DIR, _LLM_TRANSFORMS[name])
    llm = LLM.load_llm(path)
    llm.v = v
    def transform(prompt):
        kwargs = {"prompt": prompt}
        if prompt_model is not None:
            kwargs["model"] = prompt_model
        return llm.run(**kwargs)["output_prompt"]
    return transform


def _make_translator(language, prompt_model=None, v=False):
    """Build a two-phase (mutate, postprocess) translator transform."""
    path = os.path.join(_TRANSFORMS_DIR, "translator.json")

    def mutate(prompt):
        llm = LLM.load_llm(path)
        llm.v = v
        kwargs = {"text": prompt, "language": language}
        if prompt_model is not None:
            kwargs["model"] = prompt_model
        return llm.run(**kwargs)["output"]

    def postprocess(response):
        llm = LLM.load_llm(path)
        llm.v = v
        kwargs = {"text": response, "language": "english"}
        if prompt_model is not None:
            kwargs["model"] = prompt_model
        return llm.run(**kwargs)["output"]

    return (mutate, postprocess)


def _resolve_transforms(transforms, n, prompt_model, v=False):
    """Resolve a list of transform specs into (name, callable_or_tuple) pairs."""
    if transforms is None:
        names = random.sample(_ALL_BUILTINS, min(n, len(_ALL_BUILTINS)))
        transforms = names

    resolved = []
    for t in transforms:
        if callable(t):
            resolved.append((repr(t), t))
        elif isinstance(t, str):
            if t in _LLM_TRANSFORMS:
                resolved.append((t, _load_transform(t, prompt_model, v=v)))
            elif t in _DETERMINISTIC_TRANSFORMS:
                resolved.append((t, _DETERMINISTIC_TRANSFORMS[t]))
            elif t.startswith("translate_"):
                language = t[len("translate_"):]
                resolved.append((t, _make_translator(language, prompt_model, v=v)))
            else:
                raise ValueError(f"Unknown transform: '{t}'. Available: {_ALL_BUILTINS}")
        else:
            raise ValueError(f"Transform must be a string or callable, got {type(t)}")

    return resolved


def language_transform(prompt, language, output_language='english', translator_model='gemini/gemini-3-flash-preview', output_model=1, v=False):
    """
    Translate prompt to another language, generate a response, translate back.

    Args:
        prompt: The original prompt.
        language: Language to translate the prompt into.
        output_language: Language to translate the response back into.
        translator_model: Model for translation steps.
        output_model: Model for generating the response (passed to LLM).
        v: Verbose output - passed to LLM runs.

    Returns:
        dict with keys: translated_prompt, translated_response, output_response
    """
    translator = LLM.load_llm(os.path.join(_TRANSFORMS_DIR, "translator.json"))
    translator.v = v

    translated_prompt = translator.run(text=prompt, language=language, model=translator_model)["output"]
    translated_response = LLM(model=output_model, v=v).user(translated_prompt).result()
    output_response = translator.run(text=translated_response, language=output_language, model=translator_model)["output"]

    return {
        "translated_prompt": translated_prompt,
        "translated_response": translated_response,
        "output_response": output_response,
    }


def prompt_search(prompt, transforms=None, n=6, output_model=1, prompt_model=None, metrics=None, evaluate=True, concurrency=5, v=True):
    """
    Search for the best prompt variant by transforming, generating responses, and evaluating.

    Applies prompt transforms (LLM-based, deterministic, or translation), generates responses
    for each variant, then uses pairwise evaluation to rank them.

    Args:
        prompt: The original prompt to transform.
        transforms: List of transform names (str) or callables, or None to auto-select n random built-ins.
        n: How many random built-in transforms to pick when transforms is None.
        output_model: Model for generating final responses (passed to LLM(model=...)).
        prompt_model: Model override for LLM-based prompt transforms. None = use transform's saved config.
        metrics: Passed to pairwise_evaluate. None = auto-generate.
        evaluate: If False, skip pairwise evaluation and return responses only.
        concurrency: Max parallel threads for transform + generation phases.
        v: Verbose output.

    Returns:
        dict with keys: top_output, top_prompt, top_transform, transforms, prompts, responses,
        rankings_output, rankings_transforms, scores, evaluation
    """
    resolved = _resolve_transforms(transforms, n, prompt_model, v=v)

    if v:
        names = [name for name, _ in resolved]
        print(f"Transforms: {names}")

    # Phase 1: Run all mutate functions concurrently
    mutated_prompts = [None] * len(resolved)
    postprocessors = [None] * len(resolved)

    def _run_transform(idx, name, transform):
        if isinstance(transform, tuple):
            mutate_fn, postprocess_fn = transform
            return idx, name, mutate_fn(prompt), postprocess_fn
        else:
            return idx, name, transform(prompt), None

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(_run_transform, i, name, t): i
            for i, (name, t) in enumerate(resolved)
        }
        for future in as_completed(futures):
            idx, name, result, postprocess_fn = future.result()
            mutated_prompts[idx] = result
            postprocessors[idx] = postprocess_fn
            if v:
                print(f"  [{name}] {result[:80]}{'...' if len(result) > 80 else ''}")

    # Prepend original prompt (index 0 = baseline)
    all_transforms = ["original"] + [name for name, _ in resolved]
    all_prompts = [prompt] + mutated_prompts
    all_postprocessors = [None] + postprocessors

    if v:
        print(f"\nGenerating {len(all_prompts)} responses with model={output_model}...")

    # Phase 2: Generate responses
    llm = LLM(model=output_model, v=False)
    all_responses = llm.result_batch(all_prompts, concurrency=concurrency)

    # Phase 3: Apply postprocessors (translations only)
    for i, (postprocess_fn, response) in enumerate(zip(all_postprocessors, all_responses)):
        if postprocess_fn is not None:
            if v:
                print(f"  Translating response {i} back to english...")
            all_responses[i] = postprocess_fn(response)

    if not evaluate:
        return {
            "transforms": all_transforms,
            "prompts": all_prompts,
            "responses": all_responses,
        }

    # Phase 4: Evaluate
    if v:
        print(f"\nEvaluating {len(all_responses)} responses...")

    evaluation = pairwise_evaluate(
        prompt=prompt,
        results=all_responses,
        metrics=metrics,
        v=v,
    )

    rankings = evaluation["rankings"]
    best_idx = rankings[0]

    return {
        "top_output": all_responses[best_idx],
        "top_prompt": all_prompts[best_idx],
        "top_transform": all_transforms[best_idx],
        "transforms": all_transforms,
        "prompts": all_prompts,
        "responses": all_responses,
        "rankings_output": rankings,
        "rankings_transforms": [all_transforms[i] for i in rankings],
        "scores": evaluation["scores"],
        "evaluation": evaluation,
    }
