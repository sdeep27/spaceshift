"""Autonomous research agent — alternative to the prompt-tree pipeline.

Dev-only for now. Test via: spaceshift agent "topic" --model <model>
"""

import os
import re

from .LLM import LLM
from .tools import ResearchTools


def _prompt_to_dirname(prompt, max_len=60):
    slug = prompt.strip()[:max_len].lower()
    slug = re.sub(r'[^\w\s-]', '', slug)
    slug = re.sub(r'[\s_]+', '_', slug).strip('_')
    return slug or "research"


def research_agent(prompt, model=1, save=True, v=True):
    """Run the autonomous research agent.

    Args:
        prompt: Research topic or question.
        model: Model specifier (name, rank, or shorthand).
        save: True to auto-name output dir, string for explicit path, False to skip.
        v: Verbose output.

    Returns:
        dict with output_dir and outputs.
    """
    # Resolve save directory
    if save is True:
        save = _prompt_to_dirname(prompt)
    if save:
        os.makedirs(save, exist_ok=True)
        output_dir = os.path.realpath(save)
    else:
        output_dir = None

    # Set up agent
    agent = LLM(model=model, v=v)

    if output_dir:
        rt = ResearchTools(output_dir)
        agent.tools(fns=rt.all_tools(), tool_choice="auto", max_turns=30)

    agent.sys(
        "You are an autonomous research agent. Given a topic, research it thoroughly "
        "and produce comprehensive markdown outputs. Use the available tools to save "
        "your findings as structured markdown files with YAML frontmatter."
    )

    result = agent.user(prompt).chat()

    return {
        "output_dir": output_dir,
        "outputs": [],
    }
