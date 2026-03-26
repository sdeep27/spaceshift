import os


def _write_md(path, content, meta=None):
    if not path.endswith(".md"):
        path += ".md"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        if meta:
            f.write("---\n")
            for key, val in meta.items():
                if isinstance(val, list):
                    f.write(f"{key}:\n")
                    for item in val:
                        f.write(f"  - {item}\n")
                elif "\n" in str(val):
                    f.write(f"{key}: |\n")
                    for line in str(val).splitlines():
                        f.write(f"  {line}\n")
                else:
                    f.write(f"{key}: {val}\n")
            f.write("---\n\n")
        f.write(str(content))
    return path


def _dict_to_md(result, path):
    """Handle dict results from compare_models, prompt_probe, or language_transform."""
    # language_transform shape: has output_response but no responses list
    if 'output_response' in result and 'responses' not in result:
        if path is None:
            return result['output_response']
        meta = {}
        for key in ['original_prompt', 'translated_prompt', 'translated_response', 'language']:
            if key in result:
                meta[key] = result[key]
        return _write_md(path, result['output_response'], meta or None)

    # prompt_probe / compare_models shape: has responses list
    responses = result['responses']

    if 'models' in result:
        names = result['models']
    elif 'transforms' in result:
        names = result['transforms']
    else:
        names = [str(i + 1) for i in range(len(responses))]

    scores = result.get('scores')
    if scores:
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        rank_map = {i: rank + 1 for rank, i in enumerate(ranked)}

    labels = []
    for i, name in enumerate(names):
        if scores:
            labels.append(f"{rank_map[i]}_{name}")
        else:
            labels.append(name)

    if path is None:
        return responses

    base = path.removesuffix(".md")
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)

    prompt = result.get('prompt')
    prompts = result.get('prompts')
    transforms = result.get('transforms')
    models_list = result.get('models')

    paths = []
    for i, (response, label) in enumerate(zip(responses, labels)):
        meta = {}
        if prompts:
            meta['prompt'] = prompts[i]
        elif prompt:
            meta['prompt'] = prompt
        if transforms:
            meta['transform'] = transforms[i]
        if models_list:
            meta['model'] = models_list[i]

        p = f"{base}_{label}.md"
        _write_md(p, response, meta or None)
        paths.append(p)
    return paths


def to_md(text, path=None, labels=None):
    """Write text (or list of texts) to Markdown file(s). If no path given, returns text unchanged.

    Accepts raw text, a list of texts, or a dict result from compare_models/prompt_probe/language_transform.
    For dicts, auto-extracts responses, builds labels with rank prefix when evaluated,
    and writes YAML frontmatter with prompt metadata.
    """
    if isinstance(text, dict) and ('responses' in text or 'output_response' in text):
        return _dict_to_md(text, path)

    if not isinstance(text, list):
        if path is None:
            return text
        return _write_md(path, text)

    if path is None:
        return text

    base = path.removesuffix(".md")
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
    paths = []
    for i, item in enumerate(text):
        suffix = labels[i] if labels else str(i + 1)
        p = f"{base}_{suffix}.md"
        _write_md(p, item)
        paths.append(p)
    return paths
