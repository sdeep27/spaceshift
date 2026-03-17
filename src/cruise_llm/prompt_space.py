from concurrent.futures import ThreadPoolExecutor, as_completed
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
            f"exactly {num_splits} smaller subprompts that together decompose the "
            f"scope of the original prompt.\n\n"
            f"Each subprompt is on its own, with no awareness of or reference to the original prompt."
            f"Each subprompt should match the syntax, tone, and approximate length of the original prompt.\n\n"
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


_SUPERPROMPT_SYSTEM = (
    "You are a super-prompter. Given a prompt, you infer plausible parent prompts "
    "\u2014 prompts that, if someone ran prompt decomposition on them, could yield the "
    "given prompt as one of the subprompts.\n\n"
    "A parent/super prompt:\n"
    "- Stands entirely on its own. It has no awareness of or reference to the child prompt. "
    "It never hints at the child prompt\u2019s existence.\n"
    "- Is a natural, self-contained prompt that someone would genuinely write.\n"
    "- Matches the syntax, tone, and approximate length of the given prompt. If the input is "
    "terse, the super prompt is terse. If the input uses casual language, the super prompt "
    "uses casual language.\n"
    "- Does NOT inflate scope by injecting words like \"comprehensive,\" \"extensive,\" "
    "\"complete,\" \"curriculum,\" \"series,\" \"end-to-end,\" \"holistic,\" or similar "
    "breadth-expanding language unless such words already appear in the input prompt.\n"
    "- Does NOT add qualifiers, adjectives, or extra clauses to artificially broaden or "
    "lengthen the prompt beyond what the input prompt\u2019s style warrants.\n"
    "- Goes \"one level up\" in abstraction naturally \u2014 the way a broader task naturally "
    "contains the given task as an implicit subtask, not by explicitly enumerating subtasks "
    "or bolting on extra requirements.\n\n"
    'Return JSON with key "superprompts" containing an array of n super prompts.'
)

_SIBLING_SYSTEM = (
    "You are an expert prompt engineer specializing in lateral thinking and \"sibling\" prompts. "
    "Given an original prompt and a number 'n', your task is to generate 'n' sibling prompts.\n\n"
    "A sibling prompt:\n"
    "- Exists at the exact same level of abstraction, detail, and complexity as the original prompt.\n"
    "- Explores a different perspective, viewpoint, alternative approach, or a closely related "
    "parallel subject matter.\n"
    "- Is a completely standalone, natural prompt that does not explicitly reference or acknowledge "
    "the existence of the original prompt.\n"
    "- Matches the tone, syntax, and approximate length of the original prompt.\n"
    "- Does NOT narrow the scope into smaller tasks (subprompting) and does NOT broaden the scope "
    "into an overarching task (superprompting).\n\n"
    'Return JSON with a single key:\n- "sibling_prompts": an array of exactly n strings, '
    "each being a generated sibling prompt."
)


def superprompt(prompt, n=[3], model=1, concurrency=5, v=True):
    """
    Iteratively generate parent/super prompts — prompts one level up in abstraction.

    Each level takes every prompt from the previous level and generates n[level]
    superprompts for it, climbing the abstraction ladder.

    Args:
        prompt (str): The original prompt to find parents for.
        n (int | list[int]): Number of superprompts per prompt at each level.
            int treated as [int]. e.g. [3, 2] = level 0 generates 3, level 1
            generates 2 per each of those 3 (6 total).
        model: Model identifier passed to LLM(). Defaults to 1.
        concurrency (int): Max parallel API calls per level. Defaults to 5.

    Returns:
        list[dict]: Flat list of final-level superprompts, each with:
            - "prompt": the superprompt text
            - "depth": the level index that produced it
            - "parent": the prompt it was generated from
    """
    if isinstance(n, int):
        n = [n]

    current = [{"prompt": prompt, "depth": -1, "parent": None}]

    for depth, num_splits in enumerate(n):
        llm = LLM(model=model, v=False).sys(_SUPERPROMPT_SYSTEM).user("{prompt}\n\nn={n}")

        if v:
            print(f"[Level {depth}] Generating {num_splits} superprompt(s) for {len(current)} prompt(s):")
            for i, entry in enumerate(current):
                print(f"  [{depth}:{i}] {entry['prompt'][:90]}{'...' if len(entry['prompt']) > 90 else ''}")

        inputs = [{"prompt": entry["prompt"], "n": num_splits} for entry in current]
        results = llm.run_batch(inputs, concurrency=concurrency)

        next_level = []
        for parent_entry, result in zip(current, results):
            for sp in result["superprompts"]:
                next_level.append({"prompt": sp, "depth": depth, "parent": parent_entry["prompt"]})

        current = next_level

    if v:
        print(f"[Result] {len(current)} final superprompt(s):")
        for i, entry in enumerate(current):
            print(f"  [R:{i}] {entry['prompt'][:90]}{'...' if len(entry['prompt']) > 90 else ''}")

    return current


def sideprompt(prompt, n=[4], model=1, concurrency=5, v=True):
    """
    Iteratively generate side prompts — prompts at the same abstraction level
    exploring different perspectives.

    Each level takes every prompt from the previous level and generates n[level]
    side prompts for it, expanding laterally.

    Args:
        prompt (str): The original prompt to find lateral alternatives for.
        n (int | list[int]): Number of side prompts per prompt at each level.
            int treated as [int]. e.g. [4, 3] = level 0 generates 4 side prompts,
            level 1 generates 3 per each of those 4 (12 total).
        model: Model identifier passed to LLM(). Defaults to 1.
        concurrency (int): Max parallel API calls per level. Defaults to 5.

    Returns:
        list[dict]: Flat list of final-level side prompts, each with:
            - "prompt": the side prompt text
            - "depth": the level index that produced it
            - "parent": the prompt it was generated from
    """
    if isinstance(n, int):
        n = [n]

    current = [{"prompt": prompt, "depth": -1, "parent": None}]

    for depth, num_splits in enumerate(n):
        llm = LLM(model=model, v=False).sys(_SIBLING_SYSTEM).user(
            "Original Prompt: {prompt}\n\nGenerate exactly {n} sibling prompts."
        )

        if v:
            print(f"[Level {depth}] Generating {num_splits} sideprompt(s) for {len(current)} prompt(s):")
            for i, entry in enumerate(current):
                print(f"  [{depth}:{i}] {entry['prompt'][:90]}{'...' if len(entry['prompt']) > 90 else ''}")

        inputs = [{"prompt": entry["prompt"], "n": num_splits} for entry in current]
        results = llm.run_batch(inputs, concurrency=concurrency)

        next_level = []
        for parent_entry, result in zip(current, results):
            for sp in result["sibling_prompts"]:
                next_level.append({"prompt": sp, "depth": depth, "parent": parent_entry["prompt"]})

        current = next_level

    if v:
        print(f"[Result] {len(current)} final sideprompt(s):")
        for i, entry in enumerate(current):
            print(f"  [R:{i}] {entry['prompt'][:90]}{'...' if len(entry['prompt']) > 90 else ''}")

    return current


# --- prompt_tree internals ---

def _run_expansion(prompt, n, direction, model, concurrency, v):
    """Run one direction of expansion, returning (final_level, all_nodes, all_edges)."""
    if isinstance(n, int):
        n = [n]

    current = [{"id": "root", "prompt": prompt}]
    all_nodes = []
    all_edges = []
    counter = 0

    for depth, num_splits in enumerate(n):
        if direction == "sub":
            sys_prompt = (
                f"You are a prompt decomposition expert. Given a prompt, break it down into "
                f"exactly {num_splits} smaller, focused subprompts that collectively cover the "
                f"full scope of the original prompt.\n\n"
                f"Each subprompt should be self-contained, actionable, and when all subprompts "
                f"are addressed together, they should fully satisfy the original prompt.\n\n"
                f'Return JSON with key "subprompts": an array of exactly {num_splits} strings.'
            )
            llm = LLM(model=model, v=False).sys(sys_prompt).user("Prompt = {prompt}")
            inputs = [{"prompt": e["prompt"]} for e in current]
            output_key = "subprompts"
        elif direction == "super":
            llm = LLM(model=model, v=False).sys(_SUPERPROMPT_SYSTEM).user("{prompt}\n\nn={n}")
            inputs = [{"prompt": e["prompt"], "n": num_splits} for e in current]
            output_key = "superprompts"
        elif direction == "side":
            llm = LLM(model=model, v=False).sys(_SIBLING_SYSTEM).user(
                "Original Prompt: {prompt}\n\nGenerate exactly {n} sibling prompts."
            )
            inputs = [{"prompt": e["prompt"], "n": num_splits} for e in current]
            output_key = "sibling_prompts"

        if v:
            print(f"  [{direction} L{depth}] Expanding {len(current)} prompt(s) x {num_splits}:")
            for i, e in enumerate(current):
                p = e["prompt"]
                print(f"    [{depth}:{i}] {p[:90]}{'...' if len(p) > 90 else ''}")

        results = llm.run_batch(inputs, concurrency=concurrency)

        next_level = []
        for parent_entry, result in zip(current, results):
            for sp in result[output_key]:
                node_id = f"{direction}_{counter}"
                counter += 1
                node = {"id": node_id, "prompt": sp, "depth": depth, "direction": direction}
                next_level.append(node)
                all_nodes.append(node)
                all_edges.append((parent_entry["id"], node_id))

        current = next_level

        if v:
            for i, nd in enumerate(current):
                print(f"    -> {nd['prompt'][:90]}{'...' if len(nd['prompt']) > 90 else ''}")

    final = [{"prompt": nd["prompt"], "depth": nd["depth"], "parent": nd.get("parent", prompt)} for nd in current]
    return final, all_nodes, all_edges


def _wrap_label(text, width=40, max_lines=3):
    """Wrap text into multi-line label for graphviz nodes."""
    text = text.replace('\n', ' ').strip()
    words = text.split()
    lines, line = [], ""
    for word in words:
        if line and len(line) + 1 + len(word) > width:
            lines.append(line)
            line = word
        else:
            line = f"{line} {word}" if line else word
    if line:
        lines.append(line)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][:width - 3] + "..."
    return "\n".join(lines)


_DIRECTION_COLORS = {"sub": "#3498DB", "super": "#E74C3C", "side": "#27AE60"}


def _build_graph(root_prompt, all_nodes, all_edges):
    """Build a graphviz Digraph from collected nodes and edges.

    Layout: super levels stack above root, side and sub levels stack below.
    Deeper levels within each direction go further from root.
    """
    try:
        import graphviz
    except ImportError:
        raise ImportError("graphviz required for visualization: pip install graphviz")

    dot = graphviz.Digraph(comment='Prompt Tree', format='svg')
    dot.attr(rankdir='TB', fontname='Helvetica', bgcolor='#FAFAFA', newrank='true')
    dot.attr('node', fontname='Helvetica', fontsize='10', shape='box', style='rounded,filled')
    dot.attr('edge', arrowsize='0.7')

    dot.node("root", _wrap_label(root_prompt), fillcolor="#F39C12", fontcolor="white", penwidth="2")

    node_dir = {}
    node_map = {}
    levels = {}
    for node in all_nodes:
        color = _DIRECTION_COLORS.get(node["direction"], "#95A5A6")
        dot.node(node["id"], _wrap_label(node["prompt"]), fillcolor=color, fontcolor="white")
        node_dir[node["id"]] = node["direction"]
        node_map[node["id"]] = node
        key = (node["direction"], node["depth"])
        levels.setdefault(key, []).append(node["id"])

    # Constrain each (direction, depth) group to same rank
    for (direction, depth), node_ids in levels.items():
        with dot.subgraph() as s:
            s.attr(rank='same')
            for nid in node_ids:
                s.node(nid)

    # Force side depth-0 nodes onto same rank as root
    if ("side", 0) in levels:
        with dot.subgraph() as s:
            s.attr(rank='same')
            s.node("root")
            for nid in levels[("side", 0)]:
                s.node(nid)

    # Visible edges
    for src, dst in all_edges:
        color = _DIRECTION_COLORS.get(node_dir.get(dst), "#999999")
        if node_dir.get(dst) == "super":
            dot.edge(dst, src, dir='back', color=color)
        elif node_dir.get(dst) == "side":
            dot.edge(src, dst, color=color, constraint='false')
        else:
            dot.edge(src, dst, color=color)

    return dot


def prompt_tree(prompt, sub=None, super=None, side=None, model=1,
                sub_model=None, super_model=None, side_model=None,
                concurrency=5, v=True, viz=True):
    """
    Fan out a prompt in all three directions and visualize the tree.

    Runs subprompt, superprompt, and sideprompt expansions in parallel from
    the root prompt. Each direction accepts its own n array and model override.

    Args:
        prompt: The root prompt.
        sub: n array for subprompts (e.g. [4] or [4, 3]). None to skip.
        super: n array for superprompts. None to skip.
        side: n array for sideprompts. None to skip.
        model: Default model for all directions.
        sub_model: Override model for sub direction.
        super_model: Override model for super direction.
        side_model: Override model for side direction.
        concurrency: Max parallel API calls per expansion level.
        v: Verbose output.
        viz: Generate graphviz visualization.

    Returns:
        dict with keys: prompt, sub, super, side (present only if requested),
        graph (graphviz.Digraph, if viz=True)
    """
    directions = []
    if sub is not None:
        directions.append(("sub", sub, sub_model or model))
    if super is not None:
        directions.append(("super", super, super_model or model))
    if side is not None:
        directions.append(("side", side, side_model or model))

    if not directions:
        raise ValueError("At least one direction required (sub, super, or side)")

    if v:
        print(f"prompt_tree: {len(directions)} direction(s) from root")

    all_nodes = []
    all_edges = []
    result = {"prompt": prompt}

    with ThreadPoolExecutor(max_workers=len(directions)) as executor:
        futures = {
            executor.submit(_run_expansion, prompt, n, direction, m, concurrency, v): direction
            for direction, n, m in directions
        }
        for future in as_completed(futures):
            direction = futures[future]
            final, nodes, edges = future.result()
            result[direction] = final
            all_nodes.extend(nodes)
            all_edges.extend(edges)
            if v:
                print(f"  [{direction}] {len(final)} final prompt(s)")

    if viz:
        result["graph"] = _build_graph(prompt, all_nodes, all_edges)

    return result
