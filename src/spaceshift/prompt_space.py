from concurrent.futures import ThreadPoolExecutor, as_completed
from .LLM import LLM


_SUBPROMPT_SYSTEM_FN = lambda n: (
    f"You are a prompt decomposition expert. Given a prompt, break it down into "
    f"exactly {n} smaller subprompts that together decompose the "
    f"scope of the original prompt.\n\n"
    f"Each subprompt is on its own, with no awareness of or reference to the original prompt."
    f"Each subprompt should match the syntax, tone, and approximate length of the original prompt.\n\n"
    f'Return JSON with key:\n- "subprompts": an array of exactly {n} strings, '
    f"each being a subprompt"
)

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


def _expand_loop(prompt, n, model, search, concurrency, v,
                 system_prompt_fn, user_template, output_key, label):
    """Shared expansion loop for subprompt/superprompt/sideprompt.
    Expects search to already be resolved (True/False), not "auto".
    """
    if isinstance(n, int):
        n = [n]
    current = [{"prompt": prompt, "depth": -1, "parent": None}]
    for depth, num_splits in enumerate(n):
        sys_prompt = system_prompt_fn(num_splits)
        llm = LLM(model=model, search=search, v=False).sys(sys_prompt).user(user_template)
        if v:
            print(f"[Level {depth}] {label} {len(current)} prompt(s):")
            for i, entry in enumerate(current):
                print(f"  [{depth}:{i}] {entry['prompt'][:90]}{'...' if len(entry['prompt']) > 90 else ''}")
        inputs = [{"prompt": entry["prompt"], "n": num_splits} for entry in current]
        results = llm.run_batch(inputs, concurrency=concurrency)
        next_level = []
        for parent_entry, result in zip(current, results):
            for sp in result[output_key]:
                next_level.append({"prompt": sp, "depth": depth, "parent": parent_entry["prompt"]})
        current = next_level
    if v:
        print(f"[Result] {len(current)} final prompt(s):")
        for i, entry in enumerate(current):
            print(f"  [R:{i}] {entry['prompt'][:90]}{'...' if len(entry['prompt']) > 90 else ''}")
    return current


def subprompt(prompt, n=[5], model=1, search="auto", concurrency=5, v=True):
    """
    Recursively decompose a prompt into smaller subprompts across multiple levels.

    Args:
        prompt (str): The original prompt to decompose.
        n (int | list[int]): Number of subprompts per split at each level.
        model: Model identifier passed to LLM(). Defaults to 1 (best1).
        search: Enable web search. "auto" enables if model supports it.
        concurrency (int): Max parallel API calls per level. Defaults to 5.

    Returns:
        list[dict]: Each with "prompt", "depth", "parent".
    """
    search = _resolve_search_auto(search, model, v=v)
    user_msg = "Prompt = {prompt}"
    if search:
        user_msg += "</end_prompt> \n Options = Search for up-to-date, current information to generate the prompts"
    return _expand_loop(prompt, n, model, search, concurrency, v,
                        _SUBPROMPT_SYSTEM_FN, user_msg, "subprompts", "Splitting")


def superprompt(prompt, n=[3], model=1, search="auto", concurrency=5, v=True):
    """
    Generate parent/super prompts — prompts one level up in abstraction.

    Args:
        prompt (str): The original prompt to find parents for.
        n (int | list[int]): Number of superprompts per prompt at each level.
        model: Model identifier passed to LLM(). Defaults to 1.
        search: Enable web search. "auto" enables if model supports it.
        concurrency (int): Max parallel API calls per level. Defaults to 5.

    Returns:
        list[dict]: Each with "prompt", "depth", "parent".
    """
    search = _resolve_search_auto(search, model, v=v)
    return _expand_loop(prompt, n, model, search, concurrency, v,
                        lambda _: _SUPERPROMPT_SYSTEM, "{prompt}\n\nn={n}",
                        "superprompts", "Generating superprompt(s) for")


def sideprompt(prompt, n=[4], model=1, search="auto", concurrency=5, v=True):
    """
    Generate side prompts — prompts at the same abstraction level exploring different perspectives.

    Args:
        prompt (str): The original prompt to find lateral alternatives for.
        n (int | list[int]): Number of side prompts per prompt at each level.
        model: Model identifier passed to LLM(). Defaults to 1.
        search: Enable web search. "auto" enables if model supports it.
        concurrency (int): Max parallel API calls per level. Defaults to 5.

    Returns:
        list[dict]: Each with "prompt", "depth", "parent".
    """
    search = _resolve_search_auto(search, model, v=v)
    return _expand_loop(prompt, n, model, search, concurrency, v,
                        lambda _: _SIBLING_SYSTEM,
                        "Original Prompt: {prompt}\n\nGenerate exactly {n} sibling prompts.",
                        "sibling_prompts", "Generating sideprompt(s) for")


# --- prompt_tree internals ---

def _run_expansion(prompt, n, direction, model, search, concurrency, v):
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
            llm = LLM(model=model, search=search, v=False).sys(sys_prompt).user("Prompt = {prompt}")
            inputs = [{"prompt": e["prompt"]} for e in current]
            output_key = "subprompts"
        elif direction == "super":
            llm = LLM(model=model, search=search, v=False).sys(_SUPERPROMPT_SYSTEM).user("{prompt}\n\nn={n}")
            inputs = [{"prompt": e["prompt"], "n": num_splits} for e in current]
            output_key = "superprompts"
        elif direction == "side":
            llm = LLM(model=model, search=search, v=False).sys(_SIBLING_SYSTEM).user(
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

    # Build parent lookup from edges
    parent_map = {dst: src for src, dst in all_edges}
    all_prompts = []
    for nd in all_nodes:
        parent_id = parent_map.get(nd["id"], "root")
        parent_prompt = prompt if parent_id == "root" else next(
            n2["prompt"] for n2 in all_nodes if n2["id"] == parent_id
        )
        all_prompts.append({
            "id": nd["id"],
            "prompt": nd["prompt"],
            "depth": nd["depth"],
            "direction": nd["direction"],
            "parent": parent_prompt,
        })

    return all_prompts, all_nodes, all_edges


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


def prompt_tree(prompt, sub_n=None, super_n=None, side_n=None, model=1,
                sub_model=None, super_model=None, side_model=None,
                search="auto", concurrency=5, v=True, viz=True):
    """
    Fan out a prompt in all three directions and visualize the tree.

    Runs subprompt, superprompt, and sideprompt expansions in parallel from
    the root prompt. Each direction accepts its own n array and model override.

    Args:
        prompt: The root prompt.
        sub_n: n array for subprompts (e.g. [4] or [4, 3]). None to skip.
        super_n: n array for superprompts. None to skip.
        side_n: n array for sideprompts. None to skip.
        model: Default model for all directions.
        sub_model: Override model for sub direction.
        super_model: Override model for super direction.
        side_model: Override model for side direction.
        search (bool): Enable web search for grounded prompt generation.
        concurrency: Max parallel API calls per expansion level.
        v: Verbose output.
        viz: Generate graphviz visualization.

    Returns:
        dict with keys: prompt, sub, super, side (present only if requested,
        each containing ALL nodes across all depths), graph (graphviz.Digraph, if viz=True)
    """
    search = _resolve_search_auto(search, model, v=v)

    directions = []
    if sub_n is not None:
        directions.append(("sub", sub_n, sub_model or model))
    if super_n is not None:
        directions.append(("super", super_n, super_model or model))
    if side_n is not None:
        directions.append(("side", side_n, side_model or model))

    if not directions:
        raise ValueError("At least one direction required (sub, super, or side)")

    if v:
        print(f"prompt_tree: {len(directions)} direction(s) from root")

    all_nodes = []
    all_edges = []
    result = {"prompt": prompt}

    with ThreadPoolExecutor(max_workers=len(directions)) as executor:
        futures = {
            executor.submit(_run_expansion, prompt, n, direction, m, search, concurrency, v): direction
            for direction, n, m in directions
        }
        for future in as_completed(futures):
            direction = futures[future]
            dir_prompts, nodes, edges = future.result()
            result[direction] = dir_prompts
            all_nodes.extend(nodes)
            all_edges.extend(edges)
            if v:
                print(f"  [{direction}] {len(dir_prompts)} prompt(s) across {max(p['depth'] for p in dir_prompts) + 1} level(s)")

    if viz:
        result["graph"] = _build_graph(prompt, all_nodes, all_edges)
        result["all_nodes"] = all_nodes
        result["all_edges"] = all_edges

    return result


def _prompt_to_dirname(prompt, max_len=60):
    """Sanitize a prompt string into a filesystem-safe directory name."""
    import re
    slug = prompt.strip()[:max_len].lower()
    slug = re.sub(r'[^\w\s-]', '', slug)
    slug = re.sub(r'[\s_]+', '_', slug).strip('_')
    return slug or "research"


def _title_to_filename(title, max_len=80):
    """Convert a title to a filesystem-safe filename, preserving readability."""
    import re
    name = title.strip()[:max_len]
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    name = name.strip('. ')
    return name or "untitled"


def _unique_path(directory, filename):
    """Return a unique file path, adding counter suffix if needed."""
    import os
    path = os.path.join(directory, filename)
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(filename)
    i = 2
    while os.path.exists(path):
        path = os.path.join(directory, f"{base} {i}{ext}")
        i += 1
    return path


_RESEARCH_SYSTEM = (
    "You are a research assistant. Return a JSON object with exactly two keys:\n"
    '- "title": a concise 2-6 word descriptive title for your response\n'
    '- "content": your full response in markdown'
)


def _extract_citations(search_annotations):
    """Extract unique citation URLs from search annotations."""
    urls = []
    seen = set()
    for ann_list in search_annotations:
        for ann in ann_list:
            url = getattr(ann, 'url', None) or (ann.get('url') if isinstance(ann, dict) else None)
            if url and url not in seen:
                urls.append(url)
                seen.add(url)
    return urls


def _resolve_search_auto(search, model, v=True):
    """Resolve search='auto' for a given model. Returns True/False."""
    if search != "auto":
        return search
    probe = LLM(model=model, v=False)
    if LLM._has_search(probe.model):
        return True
    if v:
        print(f"Search not available for {probe.model}, proceeding without search")
    return False


_DEFAULT_SUB_N = [5, 3]
_DEFAULT_SUPER_N = [5]
_DEFAULT_SIDE_N = [3]


def research_tree(
    prompt,
    sub_n=_DEFAULT_SUB_N, super_n=_DEFAULT_SUPER_N, side_n=_DEFAULT_SIDE_N,
    prompt_model=1,
    sub_model=None, super_model=None, side_model=None,
    output_model=None,
    sub_output_model=None, super_output_model=None, side_output_model=None,
    search="auto", search_tree="auto",
    save=True,
    concurrency=5,
    v=True,
):
    """
    Deep research: build a prompt tree and generate responses for every node.

    Expands a prompt in sub/super/side directions, then generates LLM responses
    for the root prompt and every node in the tree. Optionally saves all outputs
    as markdown files with YAML frontmatter and the tree visualization.

    Args:
        prompt: A prompt string, or an existing prompt_tree result dict.
        sub_n: n array for subprompts. Defaults to [5, 3]. None to skip.
        super_n: n array for superprompts. Defaults to [5]. None to skip.
        side_n: n array for sideprompts. Defaults to [3]. None to skip.
        prompt_model: Default model for tree building. Defaults to 1.
        sub_model: Override tree-building model for sub direction.
        super_model: Override tree-building model for super direction.
        side_model: Override tree-building model for side direction.
        output_model: Default model for generating responses. Defaults to prompt_model.
        sub_output_model: Override output model for sub nodes.
        super_output_model: Override output model for super nodes.
        side_output_model: Override output model for side nodes.
        search (bool): Enable web search for response generation.
        search_tree (bool): Enable web search for tree building.
        save: Directory path to save outputs, or True to auto-name from prompt. False/None = don't save.
        concurrency: Max parallel API calls. Defaults to 5.
        v: Verbose output.

    Returns:
        dict with keys: prompt, tree, outputs (list of dicts with prompt/response/direction/depth),
        root_output (response for the original prompt)
    """
    # Resolve search auto
    output_model = output_model or prompt_model
    search = _resolve_search_auto(search, output_model, v=v)
    search_tree = _resolve_search_auto(search_tree, prompt_model, v=v)

    # Build or reuse tree
    if isinstance(prompt, dict) and "prompt" in prompt:
        tree = prompt
        prompt = tree["prompt"]
    else:
        if sub_n is None and super_n is None and side_n is None:
            raise ValueError("At least one direction required (sub_n, super_n, or side_n)")
        tree = prompt_tree(
            prompt, sub_n=sub_n, super_n=super_n, side_n=side_n,
            model=prompt_model, sub_model=sub_model, super_model=super_model, side_model=side_model,
            search=search_tree, concurrency=concurrency, v=v, viz=True,
        )

    # Resolve save path
    if save is True:
        save = _prompt_to_dirname(prompt)

    # Collect all prompts to generate responses for
    # Each entry: (prompt_text, direction, depth, node_id, output_model_for_this_node)
    jobs = [{"prompt": prompt, "direction": "root", "depth": -1, "id": "root",
             "output_model": output_model}]

    for direction in ("sub", "super", "side"):
        if direction not in tree:
            continue
        dir_output_model = {
            "sub": sub_output_model,
            "super": super_output_model,
            "side": side_output_model,
        }[direction] or output_model

        # Count per depth for indexing
        depth_counters = {}
        for node in tree[direction]:
            d = node["depth"]
            idx = depth_counters.get(d, 0)
            depth_counters[d] = idx + 1
            jobs.append({
                "prompt": node["prompt"],
                "direction": direction,
                "depth": d,
                "id": f"{direction}_L{d}_{idx}",
                "parent": node.get("parent"),
                "output_model": dir_output_model,
            })

    if v:
        print(f"\nGenerating {len(jobs)} responses...")

    # Generate responses concurrently
    responses = [None] * len(jobs)
    titles = [None] * len(jobs)
    citations_list = [None] * len(jobs)

    def _run_job(idx):
        job = jobs[idx]
        llm = LLM(model=job["output_model"], search=search, v=False)
        llm.sys(_RESEARCH_SYSTEM)
        result = llm.user(job["prompt"]).result_json()
        title = result.get("title", "")
        content = result.get("content") or llm.last()
        citations = _extract_citations(llm.search_annotations)
        return idx, title, content, citations

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_run_job, i): i for i in range(len(jobs))}
        done = 0
        for future in as_completed(futures):
            idx, title, response, citations = future.result()
            responses[idx] = response
            titles[idx] = title
            citations_list[idx] = citations
            done += 1
            if v:
                job = jobs[idx]
                print(f"  [{done}/{len(jobs)}] {job['prompt'][:80]}{'...' if len(job['prompt']) > 80 else ''}")

    # Build outputs
    outputs = []
    for job, response, title, citations in zip(jobs, responses, titles, citations_list):
        entry = {
            "id": job["id"],
            "prompt": job["prompt"],
            "response": response,
            "title": title,
            "direction": job["direction"],
            "depth": job["depth"],
        }
        if "parent" in job:
            entry["parent"] = job["parent"]
        if citations:
            entry["citations"] = citations
        outputs.append(entry)

    result = {
        "prompt": prompt,
        "tree": tree,
        "root_output": responses[0],
        "outputs": outputs,
    }

    if save:
        result["saved"] = _save_research(result, save)

    return result


_FOLLOWUP_SYSTEM = (
    "You are an expert prompt engineer and conceptual explorer. Your task is to analyze "
    "an original prompt and its corresponding LLM output, then generate a comprehensive "
    "set of follow-up prompts across three distinct dimensions. \n\n"
    "You must return a JSON object with the following keys:\n"
    '- "subprompts": An array of exactly 5 strings. These are decompositions, diving '
    "deeper into specific aspects of the original prompt or output.\n"
    '- "superprompts": An array of exactly 5 strings. These climb upward, expanding the '
    "context, seeking broader understanding or meta-perspectives.\n"
    '- "sideprompts": An array of exactly 5 strings. These explore different perspectives, '
    "alternative scenarios, or parallel paths at the same conceptual level as the original.\n\n"
    "Important Constraints:\n"
    '- Ensure none of your generated prompts overlap with or repeat the items listed in the "used_prompts" input.\n'
    "- Output ONLY valid JSON."
)

_FOLLOWUP_USER = (
    "Original Prompt:\n{prompt}\n\n"
    "LLM Output:\n{llm_output}\n\n"
    "Used Prompts (Do not repeat these):\n{used_prompts?}"
)


def research_expand(
    prompt,
    depth=2,
    model=1,
    followup_model=None,
    search="auto",
    concurrency=5,
    save=True,
    viz=True,
    v=True,
):
    """
    Recursive research expansion: generate an output, expand into followup prompts
    across sub/super/side directions, generate outputs for those, and recurse.

    Saves results incrementally as markdown files with YAML frontmatter.

    Args:
        prompt: Initial research prompt.
        depth: Recursion depth. 1 = root + one level of followups. 2 = two levels. Defaults to 2.
        model: Model for generating research outputs. Single value for all depths,
            or a list where model[i] is used for depth i. Defaults to 1.
        followup_model: Model for generating followup prompts. None uses a built-in
            prompt engineer. Can be a model name/int, or an LLM instance with
            template vars {prompt}, {llm_output}, {used_prompts?}.
        search: Enable web search for output generation.
        concurrency: Max parallel API calls. Defaults to 5.
        save: True to auto-name directory from prompt, string for explicit path, False to skip.
        viz: Generate tree visualization (requires graphviz).
        v: Verbose output.

    Returns:
        dict with keys: prompt, root_output, outputs (list of node dicts),
        graph (graphviz Digraph if viz), saved (file paths if save).
    """
    import os
    from .utils import _write_md

    # Resolve search auto
    search = _resolve_search_auto(search, model if not isinstance(model, list) else model[0], v=v)

    # Resolve save path
    if save is True:
        save = _prompt_to_dirname(prompt)

    if save:
        os.makedirs(save, exist_ok=True)

    # Resolve model list
    model_list = model if isinstance(model, list) else [model]

    # Build followup LLM
    if isinstance(followup_model, LLM):
        fup_llm = followup_model
    else:
        fup_llm = LLM(model=followup_model or 1, reasoning=True, v=False)
        fup_llm.sys(_FOLLOWUP_SYSTEM).user(_FOLLOWUP_USER)

    all_edges = []
    used_prompts = set()
    outputs = []
    saved_paths = []
    counter = 0

    def _model_for_depth(d):
        return model_list[min(max(0, d), len(model_list) - 1)]

    def _save_node(node):
        if not save:
            return
        meta = {
            "id": node["id"],
            "title": node.get("title", ""),
            "prompt": node["prompt"],
            "direction": node["direction"],
            "depth": node["depth"],
        }
        if "parent" in node:
            meta["parent"] = node["parent"]
        if "model" in node:
            meta["model"] = node["model"]
        if node.get("citations"):
            meta["citations"] = node["citations"]
        title = node.get("title", "")
        if title:
            filename = f"{_title_to_filename(title)}.md"
        else:
            filename = f"{node['id']}.md"
        path = _unique_path(save, filename)
        _write_md(path, node["response"], meta)
        saved_paths.append(path)

    root_model = _model_for_depth(0)
    if v:
        print(f"Generating root output with model={root_model}...")
    root_llm = LLM(model=root_model, search=search, v=False)
    root_llm.sys(_RESEARCH_SYSTEM)
    root_result = root_llm.user(prompt).result_json()
    root_title = root_result.get("title", "")
    root_response = root_result.get("content") or root_llm.last()
    root_citations = _extract_citations(root_llm.search_annotations)
    used_prompts.add(prompt)

    root_node = {
        "id": "root", "prompt": prompt, "response": root_response,
        "title": root_title,
        "direction": "root", "depth": -1, "model": str(root_model),
    }
    if root_citations:
        root_node["citations"] = root_citations
    outputs.append(root_node)
    _save_node(root_node)

    if v:
        print(f"  root: {prompt[:70]}{'...' if len(prompt) > 70 else ''}")

    pending = [root_node]

    for d in range(depth):
        if not pending:
            break

        output_model = _model_for_depth(d)

        if v:
            print(f"\n[Depth {d}] Expanding {len(pending)} node(s), generating followups...")

        followup_results = [None] * len(pending)
        used_str = "\n".join(used_prompts) if used_prompts else ""

        def _gen_followup(idx, node):
            clone = fup_llm._clone_for_batch()
            result = clone.run(prompt=node["prompt"], llm_output=node["response"], used_prompts=used_str)
            return idx, result

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(_gen_followup, i, node): i for i, node in enumerate(pending)}
            for future in as_completed(futures):
                idx, result = future.result()
                followup_results[idx] = result

        new_children = []
        for parent_node, followups in zip(pending, followup_results):
            if not followups:
                continue
            for direction in ("sub", "super", "side"):
                prompts_list = followups.get(f"{direction}prompts", [])
                for fp in prompts_list:
                    if fp in used_prompts:
                        continue
                    node_id = f"{direction}_{counter}"
                    counter += 1
                    used_prompts.add(fp)
                    child = {
                        "id": node_id, "prompt": fp, "response": None,
                        "direction": direction, "depth": d,
                        "parent": parent_node["prompt"], "parent_id": parent_node["id"],
                        "model": str(output_model),
                    }
                    new_children.append(child)
                    all_edges.append((parent_node["id"], node_id))

        if v:
            print(f"[Depth {d}] Generated {len(new_children)} followup prompts, generating outputs with model={output_model}...")

        output_llm = LLM(model=output_model, search=search, v=False)
        output_llm.sys(_RESEARCH_SYSTEM)

        def _gen_output(idx, child):
            llm = output_llm._clone_for_batch()
            result = llm.user(child["prompt"]).result_json()
            title = result.get("title", "")
            content = result.get("content") or llm.last()
            citations = _extract_citations(llm.search_annotations)
            return idx, title, content, citations

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(_gen_output, i, child): i for i, child in enumerate(new_children)}
            done = 0
            for future in as_completed(futures):
                idx, title, response, citations = future.result()
                new_children[idx]["response"] = response
                new_children[idx]["title"] = title
                if citations:
                    new_children[idx]["citations"] = citations
                _save_node(new_children[idx])
                done += 1
                if v:
                    child = new_children[idx]
                    print(f"  [{done}/{len(new_children)}] {child['prompt'][:80]}{'...' if len(child['prompt']) > 80 else ''}")

        outputs.extend(new_children)
        pending = new_children

    graph = None
    if viz and all_edges:
        all_nodes = [
            {"id": n["id"], "prompt": n["prompt"], "depth": n["depth"], "direction": n["direction"]}
            for n in outputs if n["id"] != "root"
        ]
        graph = _build_graph(prompt, all_nodes, all_edges)
        if save:
            graph_path = os.path.join(save, "tree")
            graph.render(graph_path, cleanup=True)
            saved_paths.append(f"{graph_path}.svg")

    result = {
        "prompt": prompt,
        "root_output": root_response,
        "outputs": outputs,
    }
    if graph:
        result["graph"] = graph
    if save:
        result["saved"] = saved_paths

    if v:
        print(f"\nDone! {len(outputs)} total nodes across {depth} depth levels.")
        if save:
            print(f"Saved to: {save}/")

    return result


def _save_research(result, save_dir):
    from .utils import _write_md
    import os

    os.makedirs(save_dir, exist_ok=True)
    saved = []

    for entry in result["outputs"]:
        meta = {
            "id": entry["id"],
            "title": entry.get("title", ""),
            "prompt": entry["prompt"],
            "direction": entry["direction"],
            "depth": entry["depth"],
        }
        if "parent" in entry:
            meta["parent"] = entry["parent"]
        if entry.get("citations"):
            meta["citations"] = entry["citations"]

        title = entry.get("title", "")
        if title:
            filename = f"{_title_to_filename(title)}.md"
        else:
            filename = f"{entry['id']}.md"
        path = _unique_path(save_dir, filename)
        _write_md(path, entry["response"], meta)
        saved.append(path)

    # Save tree visualization
    tree = result.get("tree", {})
    graph = tree.get("graph")
    if graph:
        graph_path = os.path.join(save_dir, "tree")
        graph.render(graph_path, cleanup=True)
        saved.append(f"{graph_path}.svg")

    return saved
