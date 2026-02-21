"""
Evaluation module for comparing and scoring LLM outputs.

Provides pairwise comparison with position bias mitigation,
auto-generated metrics, and Bradley-Terry ranking for large sets.
"""

import random
import re
import math
from itertools import combinations


def evaluate(
    prompts=None,
    results=None,
    additional_information='',
    method='pairwise',
    metrics=None,
    weights=None,
    position_swap=True,
    penalize_verbosity=False,
    v=True
):
    """
    Evaluate and rank a set of LLM outputs using pairwise comparison.

    Args:
        prompts: List of prompts (optional). If provided with results, evaluates prompt-result pairs.
        results: List of results to evaluate. Required if prompts is empty.
        additional_information: Domain context injected into evaluation prompt.
        method: Evaluation method. Only 'pairwise' supported in v1.
        metrics: Dict of {"evaluation question": "scale"}. Empty = auto-generate 3 metrics.
        weights: Optional dict of metric weights. Keys must match metrics.
        position_swap: If True, run (A,B) and (B,A) to mitigate position bias.
        penalize_verbosity: If True, add "reward conciseness" to eval prompt.
        v: Verbose output - print progress.

    Returns:
        dict with:
            - rankings: list of indices sorted best to worst
            - scores: dict of {index: normalized_score (0-1)}
            - raw: method-specific data (comparison_matrix, win_counts, metrics_used)
    """
    from .LLM import LLM

    prompts = prompts or []
    results = results or []
    metrics = metrics if metrics is not None else {}

    # Validation
    if not prompts and not results:
        raise ValueError("At least one of prompts or results must be non-empty")

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
                "metrics_used": metrics or {},
                "auto_generated_metrics": False
            }
        }

    # Auto-generate metrics if not provided
    auto_generated = False
    if not metrics:
        if v:
            print("Auto-generating evaluation metrics...")
        metrics = _generate_metrics(items, additional_information, prompts, results)
        auto_generated = True
        if v:
            print(f"Generated metrics: {list(metrics.keys())}")

    # Validate weights if provided
    if weights:
        if set(weights.keys()) != set(metrics.keys()):
            raise ValueError(f"Weight keys must match metric keys. Got {set(weights.keys())} vs {set(metrics.keys())}")

    # Determine comparison pairs
    if n_items <= 5:
        pairs = list(combinations(range(n_items), 2))
    else:
        # Bradley-Terry: sample N*5 pairs
        pairs = _sample_pairs_bradley_terry(n_items)

    if v:
        print(f"Evaluating {n_items} items with {len(pairs)} comparisons across {len(metrics)} metrics")

    # Run pairwise comparisons for each metric
    all_metric_results = {}

    for metric_question, scale in metrics.items():
        if v:
            print(f"\nMetric: {metric_question}")

        comparison_matrix = [[None] * n_items for _ in range(n_items)]
        win_counts = {i: 0.0 for i in range(n_items)}

        for idx, (i, j) in enumerate(pairs):
            if v:
                print(f"  Comparing {i} vs {j}... ", end="", flush=True)

            winner = _compare_pair(
                item_a=items[i],
                item_b=items[j],
                prompt_a=prompts[i] if prompts else None,
                prompt_b=prompts[j] if prompts else None,
                metric_question=metric_question,
                scale=scale,
                additional_information=additional_information,
                position_swap=position_swap,
                penalize_verbosity=penalize_verbosity
            )

            if v:
                print(winner)

            # Record result
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

        all_metric_results[metric_question] = {
            "comparison_matrix": comparison_matrix,
            "win_counts": win_counts,
            "scale": scale
        }

    # Aggregate scores across metrics
    final_scores = _aggregate_metric_scores(all_metric_results, weights, n_items, len(pairs))

    # Compute rankings
    rankings = sorted(range(n_items), key=lambda i: final_scores[i], reverse=True)

    # Use first metric's comparison matrix for raw output
    first_metric = list(metrics.keys())[0]

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

    metric_scores = {}
    scales = {}

    for metric_question, scale in metrics.items():
        if v:
            print(f"Scoring: {metric_question}... ", end="", flush=True)

        score = _score_single(
            response=response,
            prompt=prompt,
            metric_question=metric_question,
            scale=scale,
            additional_information=additional_information,
            penalize_verbosity=penalize_verbosity
        )

        if v:
            print(score)

        metric_scores[metric_question] = score
        scales[metric_question] = scale

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


def _generate_metrics(items, additional_information, prompts, results):
    """Auto-generate 3 evaluation metrics based on content."""
    from .LLM import LLM

    # Sample items for context
    sample = items[:3] if len(items) > 3 else items
    sample_text = "\n\n---\n\n".join(str(s)[:500] for s in sample)

    context_type = "responses"
    if prompts and not results:
        context_type = "prompts"
    elif prompts and results:
        context_type = "prompt-response pairs"

    result = LLM(model=1, v=False).sys(
        "You suggest evaluation metrics for comparing LLM outputs. "
        "Return exactly 3 metrics as JSON: {\"metrics\": {\"How [question]?\": \"1-10\", ...}}"
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


def _compare_pair(item_a, item_b, prompt_a, prompt_b, metric_question, scale,
                  additional_information, position_swap, penalize_verbosity):
    """Compare two items on a metric. Returns 'A', 'B', or 'tie'."""
    from .LLM import LLM

    def do_comparison(first, second, first_prompt, second_prompt):
        prompt_context = ""
        if first_prompt and second_prompt:
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
Scale: {scale}
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
