from .LLM import LLM
from .evaluate import pairwise_evaluate
from .subprompt import subprompt
from .prompt_probe import prompt_probe, language_transform
from .compare_models import compare_models


def to_md(text, path=None, labels=None):
    """Write text (or list of texts) to Markdown file(s). If no path given, returns text unchanged.

    For a list, writes numbered files (path_1.md, path_2.md, ...) or labeled files
    (path_original.md, path_abstract_up.md, ...) if labels are provided.
    """
    if not isinstance(text, list):
        if path is None:
            return text
        if not path.endswith(".md"):
            path += ".md"
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


__all__ = ["LLM", "pairwise_evaluate", "subprompt", "prompt_probe", "language_transform", "compare_models", "to_md"]
