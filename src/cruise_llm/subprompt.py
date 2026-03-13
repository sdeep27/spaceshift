from .LLM import LLM


def subprompt(prompt, n=[5], model=1, search=False, concurrency=5, v=True):
    """
    Recursively decompose a prompt into smaller subprompts across multiple levels.

    Each level splits every prompt from the previous level into n[level] subprompts,
    processed in parallel via run_batch. Returns the final level as a flat list of
    dicts with lineage metadata.

    Args:
        prompt (str): The original prompt to decompose.
        n (int | list[int]): Number of subprompts per split at each level.
            int treated as [int]. e.g. [4,4,4] = 3 levels, 4 splits each.
        model: Model identifier passed to LLM(). Defaults to 1 (best1).
        search (bool): Enable web search for grounded decomposition.
        concurrency (int): Max parallel API calls per level. Defaults to 5.

    Returns:
        list[dict]: Flat list of final-level subprompts, each with:
            - "prompt": the subprompt text
            - "depth": the level index that produced it
            - "parent": the parent prompt text it was split from
    """
    if isinstance(n, int):
        n = [n]

    current = [{"prompt": prompt, "depth": -1, "parent": None}]

    for depth, num_splits in enumerate(n):
        system_prompt = (
            f"You are a prompt decomposition expert. Given a prompt, break it down into "
            f"exactly {num_splits} smaller, focused subprompts that collectively cover the "
            f"full scope of the original prompt.\n\n"
            f"Each subprompt should be self-contained (you do not need to know the prompt "
            f"it derived from to answer it properly), actionable, and when all subprompts "
            f"are addressed together, they should fully satisfy the original prompt.\n\n"
            f'Return JSON with key:\n- "subprompts": an array of exactly {num_splits} strings, '
            f"each being a subprompt"
        )

        user_msg = "Prompt = {prompt}"
        if search:
            user_msg += "</end_prompt> \n Options = Search for up-to-date, current information to generate the prompts"

        llm = LLM(model=model, search=search, v=False).sys(system_prompt).user(user_msg)

        if v:
            print(f"[Level {depth}] Splitting {len(current)} prompt(s):")
            for i, entry in enumerate(current):
                print(f"  [{depth}:{i}] {entry['prompt'][:90]}{'...' if len(entry['prompt']) > 90 else ''}")

        inputs = [{"prompt": entry["prompt"]} for entry in current]
        results = llm.run_batch(inputs, concurrency=concurrency)

        next_level = []
        for parent_entry, result in zip(current, results):
            for sp in result["subprompts"]:
                next_level.append({"prompt": sp, "depth": depth, "parent": parent_entry["prompt"]})

        current = next_level

    if v:
        print(f"[Result] {len(current)} final prompt(s):")
        for i, entry in enumerate(current):
            print(f"  [R:{i}] {entry['prompt'][:90]}{'...' if len(entry['prompt']) > 90 else ''}")

    return current
