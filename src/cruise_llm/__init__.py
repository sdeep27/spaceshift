from .LLM import LLM
from .evaluate import pairwise_evaluate
from .subprompt import subprompt
from .prompt_probe import prompt_probe, language_transform
from .compare_models import compare_models


def to_md(text, path=None, labels=None):
    """Write text (or list of texts) to Markdown file(s). If no path given, returns text unchanged.

    Accepts raw text, a list of texts, or a dict result from compare_models/prompt_probe.
    For dicts, auto-extracts responses and builds labels with rank prefix when evaluated.

    For a list, writes numbered files (path_1.md, path_2.md, ...) or labeled files
    (path_original.md, path_abstract_up.md, ...) if labels are provided.
    """
    if isinstance(text, dict) and 'responses' in text:
        return _dict_to_md(text, path)

    if not isinstance(text, list):
        if path is None:
            return text
        import os
        if not path.endswith(".md"):
            path += ".md"
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(str(text))
        return path

    if path is None:
        return text

    import os
    base = path.removesuffix(".md")
    os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
    paths = []
    for i, item in enumerate(text):
        suffix = labels[i] if labels else str(i + 1)
        p = f"{base}_{suffix}.md"
        with open(p, "w") as f:
            f.write(str(item))
        paths.append(p)
    return paths


def _dict_to_md(result, path):
    """Handle dict results from compare_models or prompt_probe."""
    import os

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
    paths = []
    for i, (response, label) in enumerate(zip(responses, labels)):
        p = f"{base}_{label}.md"
        with open(p, "w") as f:
            f.write(str(response))
        paths.append(p)
    return paths


__all__ = ["LLM", "pairwise_evaluate", "subprompt", "prompt_probe", "language_transform", "compare_models", "to_md"]
