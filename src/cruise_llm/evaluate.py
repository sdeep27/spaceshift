"""
Evaluation module for comparing and scoring LLM outputs.

Provides pairwise comparison with position bias mitigation,
auto-generated metrics, and Bradley-Terry ranking for large sets.
"""

import random
import re
import math
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor, as_completed


def pairwise_evaluate(
    prompt=None,
    prompts=None,
    results=None,
    additional_information='',
    metrics=None,
    weights=None,
    position_swap=True,
    penalize_verbosity=False,
    per_metric=False,
    batch_size=4,
    v=True
):
    """
    Evaluate and rank a set of LLM outputs using pairwise comparison.

    Args:
        prompt: Single prompt used for all responses (e.g. original prompt before stretching).
        prompts: List of prompts (optional). If provided with results, evaluates prompt-result pairs.
        results: List of results to evaluate. Required if prompts is empty.
        additional_information: Domain context injected into evaluation prompt.
        metrics: List of evaluation questions (e.g. ["How clear?", "How thorough?"]). Empty = auto-generate.
        weights: Optional dict of metric weights. Keys must match metric strings.
        position_swap: If True, run (A,B) and (B,A) to mitigate position bias.
        penalize_verbosity: If True, add "reward conciseness" to eval prompt.
        per_metric: If True, make one LLM call per metric (legacy). Default False batches all metrics per comparison.
        batch_size: Number of pair comparisons to run concurrently. Default 4. Set to 1 for sequential.
        v: Verbose output - print progress.

    Returns:
        dict with:
            - rankings: list of indices sorted best to worst
            - scores: dict of {index: normalized_score (0-1)}
            - raw: method-specific data (comparison_matrix, win_counts, metrics_used)
    """
    from .LLM import LLM

    if prompt is not None and prompts is not None:
        raise ValueError("Cannot specify both 'prompt' (singular) and 'prompts' (plural). "
                         "Use 'prompt' for a single prompt shared across all responses, "
                         "or 'prompts' for per-response prompts.")

    prompts = prompts or []
    results = results or []
    metrics = metrics if metrics is not None else []

    # Validation
    if not prompt and not prompts and not results:
        raise ValueError("At least one of prompt, prompts, or results must be non-empty")

    if prompts and results and len(prompts) != len(results):
        raise ValueError(f"Length mismatch: {len(prompts)} prompts vs {len(results)} results")

    # Determine what we're evaluating
    items = results if results else prompts
    n_items = len(items)

    if n_items == 0:
        raise ValueError("No items to evaluate")

    # Single item edge case
    if n_items == 1:
        return {
            "rankings": [0],
            "scores": {0: 1.0},
            "raw": {
                "comparison_matrix": [[None]],
                "win_counts": {0: 1.0},
                "metrics_used": metrics or [],
                "auto_generated_metrics": False
            }
        }

    # Auto-generate metrics if not provided
    auto_generated = False
    if not metrics:
        if v:
            print("Auto-generating evaluation metrics...")
        gen_prompts = [prompt] * len(items) if prompt else prompts
        metrics = _generate_metrics(items, additional_information, gen_prompts, results, mode="pairwise")
        auto_generated = True
        if v:
            print(f"Generated metrics: {metrics}")

    # Validate weights if provided
    if weights:
        if set(weights.keys()) != set(metrics):
            raise ValueError(f"Weight keys must match metric keys. Got {set(weights.keys())} vs {set(metrics)}")

    # Determine comparison pairs
    if n_items <= 5:
        pairs = list(combinations(range(n_items), 2))
    else:
        # Bradley-Terry: sample N*5 pairs
        pairs = _sample_pairs_bradley_terry(n_items)

    if v:
        print(f"Evaluating {n_items} items with {len(pairs)} comparisons across {len(metrics)} metrics")

    # Initialize results structure for all metrics
    all_metric_results = {}
    for metric_question in metrics:
        all_metric_results[metric_question] = {
            "comparison_matrix": [[None] * n_items for _ in range(n_items)],
            "win_counts": {i: 0.0 for i in range(n_items)},
        }

    # Run pairwise comparisons concurrently
    def _resolve_prompts(i, j):
        if prompt:
            return prompt, prompt
        if prompts:
            return prompts[i], prompts[j]
        return None, None

    def _run_pair(i, j):
        pa, pb = _resolve_prompts(i, j)
        if per_metric:
            winners = {}
            for metric_question in metrics:
                winners[metric_question] = _compare_pair(
                    item_a=items[i], item_b=items[j],
                    prompt_a=pa, prompt_b=pb,
                    metric_question=metric_question,
                    additional_information=additional_information,
                    position_swap=position_swap,
                    penalize_verbosity=penalize_verbosity
                )
            return winners
        return _compare_pair_multi_metric(
            item_a=items[i], item_b=items[j],
            prompt_a=pa, prompt_b=pb,
            metrics=metrics,
            additional_information=additional_information,
            position_swap=position_swap,
            penalize_verbosity=penalize_verbosity
        )

    if v:
        print(f"Running comparisons with batch_size={batch_size}")

    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        future_to_pair = {
            executor.submit(_run_pair, i, j): (i, j)
            for i, j in pairs
        }
        completed = 0
        for future in as_completed(future_to_pair):
            i, j = future_to_pair[future]
            metric_winners = future.result()
            completed += 1

            if v:
                print(f"\n[{completed}/{len(pairs)}] {i} vs {j}")

            for metric_question, winner in metric_winners.items():
                if v:
                    print(f"  {metric_question[:60]}... {winner}")

                comparison_matrix = all_metric_results[metric_question]["comparison_matrix"]
                win_counts = all_metric_results[metric_question]["win_counts"]

                if winner == "A":
                    comparison_matrix[i][j] = "A"
                    comparison_matrix[j][i] = "B"
                    win_counts[i] += 1.0
                elif winner == "B":
                    comparison_matrix[i][j] = "B"
                    comparison_matrix[j][i] = "A"
                    win_counts[j] += 1.0
                else:  # tie
                    comparison_matrix[i][j] = "tie"
                    comparison_matrix[j][i] = "tie"
                    win_counts[i] += 0.5
                    win_counts[j] += 0.5

    # Aggregate scores across metrics
    final_scores = _aggregate_metric_scores(all_metric_results, weights, n_items, len(pairs))

    # Compute rankings
    rankings = sorted(range(n_items), key=lambda i: final_scores[i], reverse=True)

    # Use first metric's comparison matrix for raw output
    first_metric = metrics[0]

    return {
        "rankings": rankings,
        "scores": final_scores,
        "raw": {
            "comparison_matrix": all_metric_results[first_metric]["comparison_matrix"],
            "win_counts": all_metric_results[first_metric]["win_counts"],
            "metrics_used": metrics,
            "auto_generated_metrics": auto_generated,
            "all_metric_results": all_metric_results
        }
    }


def evaluate_single(
    response,
    prompt=None,
    additional_information='',
    metrics=None,
    penalize_verbosity=False,
    per_metric=False,
    v=True
):
    """
    Evaluate a single response using absolute scoring.

    Used by LLM.evaluate_last(). Returns scores for each metric.

    Args:
        response: The response text to evaluate.
        prompt: Optional prompt that generated the response.
        additional_information: Domain context for evaluator.
        metrics: Dict of {"evaluation question": "scale"}. Empty = auto-generate.
        penalize_verbosity: Add conciseness instruction.
        per_metric: If True, make one LLM call per metric (legacy). Default False batches all metrics.
        v: Verbose output.

    Returns:
        dict with:
            - score: normalized 0-1 average
            - metric_scores: raw scores per metric
            - scales: the scale used for each metric
    """
    from .LLM import LLM

    metrics = metrics if metrics is not None else {}

    # Auto-generate if needed
    if not metrics:
        if v:
            print("Auto-generating evaluation metrics...")
        metrics = _generate_metrics([response], additional_information, [prompt] if prompt else [], [response])
        if v:
            print(f"Generated metrics: {list(metrics.keys())}")

    if per_metric:
        metric_scores = {}
        for metric_question, scale in metrics.items():
            if v:
                print(f"Scoring: {metric_question}... ", end="", flush=True)
            score = _score_single(
                response=response, prompt=prompt,
                metric_question=metric_question, scale=scale,
                additional_information=additional_information,
                penalize_verbosity=penalize_verbosity
            )
            if v:
                print(score)
            metric_scores[metric_question] = score
    else:
        if v:
            print(f"Scoring all {len(metrics)} metrics in one call...")
        metric_scores = _score_single_multi_metric(
            response=response, prompt=prompt, metrics=metrics,
            additional_information=additional_information,
            penalize_verbosity=penalize_verbosity
        )
        if v:
            for question, score in metric_scores.items():
                print(f"  {question[:60]}... {score}")

    scales = {q: s for q, s in metrics.items()}

    # Normalize and average
    normalized_scores = []
    for question, raw_score in metric_scores.items():
        scale = scales[question]
        min_val, max_val = _parse_scale(scale)
        if max_val > min_val:
            normalized = (raw_score - min_val) / (max_val - min_val)
        else:
            normalized = raw_score
        normalized_scores.append(normalized)

    avg_score = sum(normalized_scores) / len(normalized_scores) if normalized_scores else 0.0

    return {
        "score": avg_score,
        "metric_scores": metric_scores,
        "scales": scales
    }


def _generate_metrics(items, additional_information, prompts, results, mode="absolute"):
    """Auto-generate 3 evaluation metrics based on content.

    mode="pairwise" returns list[str], mode="absolute" returns dict.
    """
    from .LLM import LLM

    # Sample items for context
    sample = items[:3] if len(items) > 3 else items
    sample_text = "\n\n---\n\n".join(str(s)[:500] for s in sample)

    context_type = "responses"
    if prompts and not results:
        context_type = "prompts"
    elif prompts and results:
        context_type = "prompt-response pairs"

    if mode == "pairwise":
        result = LLM(model=1, v=False).sys(
            "You suggest evaluation metrics for comparing LLM outputs. "
            'Return exactly 3 metrics as JSON: {"metrics": ["How ...?", ...]}'
        ).user(
            f"Suggest 3 evaluation metrics for comparing these {context_type}:\n\n"
            f"{sample_text}\n\n"
            f"Additional context: {additional_information or 'General evaluation'}"
        ).res_json()

        return result.get("metrics", [
            "How clear and well-structured is this?",
            "How interesting and well-written is this?",
            "How complete and thorough is this?"
        ])

    result = LLM(model=1, v=False).sys(
        "You suggest evaluation metrics for comparing LLM outputs. "
        'Return exactly 3 metrics as JSON: {"metrics": {"How [question]?": "1-10", ...}}'
    ).user(
        f"Suggest 3 evaluation metrics for comparing these {context_type}:\n\n"
        f"{sample_text}\n\n"
        f"Additional context: {additional_information or 'General evaluation'}"
    ).res_json()

    return result.get("metrics", {
        "How clear and well-structured is this?": "1-10",
        "How interesting and well-written is this?": "1-10",
        "How complete and thorough is this?": "1-10"
    })


def _compare_pair(item_a, item_b, prompt_a, prompt_b, metric_question,
                  additional_information, position_swap, penalize_verbosity):
    """Compare two items on a metric. Returns 'A', 'B', or 'tie'."""
    from .LLM import LLM

    def do_comparison(first, second, first_prompt, second_prompt):
        prompt_context = ""
        if first_prompt and second_prompt:
            if first_prompt == second_prompt:
                prompt_context = f"\n## Original Prompt\n{first_prompt}\n"
            else:
                prompt_context = f"\n## Prompt for Response A\n{first_prompt}\n\n## Prompt for Response B\n{second_prompt}\n"

        verbosity_note = ""
        if penalize_verbosity:
            verbosity_note = "\nNote: Reward conciseness. Longer responses are not necessarily better."

        context_section = ""
        if additional_information:
            context_section = f"\n## Context\n{additional_information}\n"

        eval_prompt = f"""You are an expert evaluator. Compare the following two responses.
{context_section}
## Response A
{first}

## Response B
{second}
{prompt_context}
Evaluate based on: {metric_question}
{verbosity_note}
Which response is better? Reply with JSON only:
{{"winner": "A" or "B" or "tie", "reasoning": "brief explanation"}}"""

        result = LLM(model=1, v=False).user(eval_prompt).res_json()
        return result.get("winner", "tie").upper()

    # First comparison: A, B order
    result1 = do_comparison(item_a, item_b, prompt_a, prompt_b)

    if not position_swap:
        if result1 == "A":
            return "A"
        elif result1 == "B":
            return "B"
        else:
            return "tie"

    # Second comparison: B, A order (swapped)
    result2 = do_comparison(item_b, item_a, prompt_b, prompt_a)

    # Translate result2 back (if B won when shown first, that means original A won)
    if result2 == "A":
        result2_translated = "B"
    elif result2 == "B":
        result2_translated = "A"
    else:
        result2_translated = "tie"

    # Check consistency
    if result1 == result2_translated:
        return result1 if result1 in ["A", "B"] else "tie"
    else:
        # Inconsistent = tie
        return "tie"


def _compare_pair_multi_metric(item_a, item_b, prompt_a, prompt_b, metrics,
                               additional_information, position_swap, penalize_verbosity):
    """Compare two items on all metrics in a single LLM call. Returns {metric_question: winner}."""
    from .LLM import LLM

    def build_prompt(first, second, first_prompt, second_prompt):
        prompt_context = ""
        if first_prompt and second_prompt:
            if first_prompt == second_prompt:
                prompt_context = f"\n## Original Prompt\n{first_prompt}\n"
            else:
                prompt_context = f"\n## Prompt for Response A\n{first_prompt}\n\n## Prompt for Response B\n{second_prompt}\n"

        verbosity_note = ""
        if penalize_verbosity:
            verbosity_note = "\nNote: Reward conciseness. Longer responses are not necessarily better."

        context_section = ""
        if additional_information:
            context_section = f"\n## Context\n{additional_information}\n"

        metric_lines = "\n".join(
            f"metric_{idx}: {q}"
            for idx, q in enumerate(metrics, 1)
        )

        example_keys = ", ".join(
            f'"metric_{idx}": {{"winner": "A" or "B" or "tie", "reasoning": "brief"}}'
            for idx in range(1, len(metrics) + 1)
        )

        return f"""You are an expert evaluator. Compare the following two responses.
{context_section}
## Response A
{first}

## Response B
{second}
{prompt_context}
Evaluate on each of the following metrics:
{metric_lines}
{verbosity_note}
For EACH metric, determine which response is better. You MUST use the exact keys metric_1, metric_2, etc. Reply with JSON only:
{{{example_keys}}}"""

    def do_comparison(first, second, first_prompt, second_prompt):
        eval_prompt = build_prompt(first, second, first_prompt, second_prompt)
        return LLM(model=1, v=False).user(eval_prompt).res_json()

    def extract_winners(result, warn_label=""):
        winners = {}
        for idx, question in enumerate(metrics, 1):
            key = f"metric_{idx}"
            if key not in result:
                print(f"Warning: {key} missing from {warn_label}LLM response, defaulting to tie")
                winners[question] = "tie"
                continue
            entry = result[key]
            winner = entry.get("winner", "tie").upper() if isinstance(entry, dict) else "tie"
            winners[question] = winner if winner in ("A", "B") else "tie"
        return winners

    result1 = do_comparison(item_a, item_b, prompt_a, prompt_b)
    winners1 = extract_winners(result1)

    if not position_swap:
        return winners1

    result2 = do_comparison(item_b, item_a, prompt_b, prompt_a)
    winners2 = extract_winners(result2, "swapped ")

    # Reconcile per metric
    final = {}
    for question in metrics:
        r1 = winners1[question]
        r2 = winners2[question]
        # Translate r2 back (positions were swapped)
        r2_translated = {"A": "B", "B": "A"}.get(r2, "tie")
        if r1 == r2_translated:
            final[question] = r1 if r1 in ("A", "B") else "tie"
        else:
            final[question] = "tie"
    return final


def _score_single(response, prompt, metric_question, scale, additional_information, penalize_verbosity):
    """Score a single response on a metric. Returns raw score."""
    from .LLM import LLM

    prompt_context = ""
    if prompt:
        prompt_context = f"\n## Original Prompt\n{prompt}\n"

    verbosity_note = ""
    if penalize_verbosity:
        verbosity_note = "\nNote: Reward conciseness. Longer responses are not necessarily better."

    context_section = ""
    if additional_information:
        context_section = f"\n## Context\n{additional_information}\n"

    min_val, max_val = _parse_scale(scale)

    eval_prompt = f"""You are an expert evaluator. Score the following response.
{context_section}
## Response
{response}
{prompt_context}
Evaluate based on: {metric_question}
Scale: {scale}
{verbosity_note}
Reply with JSON only:
{{"score": <number between {min_val} and {max_val}>, "reasoning": "brief explanation"}}"""

    result = LLM(model=1, v=False).user(eval_prompt).res_json()

    score = result.get("score", (min_val + max_val) / 2)

    # Clamp to scale
    return max(min_val, min(max_val, float(score)))


def _score_single_multi_metric(response, prompt, metrics, additional_information, penalize_verbosity):
    """Score a single response on all metrics in a single LLM call. Returns {metric_question: score}."""
    from .LLM import LLM

    metric_keys = list(metrics.keys())

    prompt_context = ""
    if prompt:
        prompt_context = f"\n## Original Prompt\n{prompt}\n"

    verbosity_note = ""
    if penalize_verbosity:
        verbosity_note = "\nNote: Reward conciseness. Longer responses are not necessarily better."

    context_section = ""
    if additional_information:
        context_section = f"\n## Context\n{additional_information}\n"

    metric_lines = "\n".join(
        f"metric_{idx}: {q} (Scale: {metrics[q]})"
        for idx, q in enumerate(metric_keys, 1)
    )

    example_keys = ", ".join(
        f'"metric_{idx}": {{"score": <number>, "reasoning": "brief"}}'
        for idx in range(1, len(metric_keys) + 1)
    )

    eval_prompt = f"""You are an expert evaluator. Score the following response.
{context_section}
## Response
{response}
{prompt_context}
Evaluate on each of the following metrics:
{metric_lines}
{verbosity_note}
For EACH metric, provide a score. You MUST use the exact keys metric_1, metric_2, etc. Reply with JSON only:
{{{example_keys}}}"""

    result = LLM(model=1, v=False).user(eval_prompt).res_json()

    scores = {}
    for idx, question in enumerate(metric_keys, 1):
        key = f"metric_{idx}"
        scale = metrics[question]
        min_val, max_val = _parse_scale(scale)
        midpoint = (min_val + max_val) / 2

        if key not in result:
            print(f"Warning: {key} missing from LLM response, defaulting to midpoint ({midpoint})")
            scores[question] = midpoint
        elif isinstance(result[key], dict):
            raw = result[key].get("score", midpoint)
            scores[question] = max(min_val, min(max_val, float(raw)))
        else:
            scores[question] = midpoint

    return scores


def _parse_scale(scale_str):
    """Parse scale string like '1-10' or '0-100' to (min, max)."""
    match = re.match(r'(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)', str(scale_str))
    if match:
        return float(match.group(1)), float(match.group(2))
    # Default
    return 0.0, 1.0


def _sample_pairs_bradley_terry(n_items):
    """Sample pairs for Bradley-Terry estimation. Returns ~N*5 pairs."""
    all_pairs = list(combinations(range(n_items), 2))
    n_pairs = min(len(all_pairs), n_items * 5)
    return random.sample(all_pairs, n_pairs)


def _aggregate_metric_scores(all_metric_results, weights, n_items, n_pairs):
    """Aggregate scores across metrics into final normalized scores."""
    if not weights:
        # Equal weights
        weights = {k: 1.0 / len(all_metric_results) for k in all_metric_results}

    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}

    final_scores = {i: 0.0 for i in range(n_items)}

    for metric, data in all_metric_results.items():
        win_counts = data["win_counts"]

        # Normalize win counts to 0-1
        max_possible = n_pairs  # Maximum wins possible
        if max_possible > 0:
            normalized = {i: count / max_possible for i, count in win_counts.items()}
        else:
            normalized = {i: 0.5 for i in range(n_items)}

        # Apply weight
        metric_weight = weights.get(metric, 1.0 / len(all_metric_results))
        for i in range(n_items):
            final_scores[i] += normalized[i] * metric_weight

    return final_scores
