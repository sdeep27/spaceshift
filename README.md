# spaceshift

An open research toolkit powered by LLMs. Branch into questions through tree structures, navigate the full space of perspectives, and grid-search evaluate across prompts and models to find what works best.

**[Full documentation at spcshft.com](https://spcshft.com)**

```python
from spaceshift import research_tree

# Decompose a topic and generate responses for every angle
result = research_tree("What are the second-order effects of AI on labor markets?")
# Saves markdown files + tree visualization for every sub/super/side angle
```

---

### Research Tree

Decompose a question, explore it from every direction, and generate responses for every node. Saves structured markdown with YAML frontmatter and a graphviz tree visualization.

```python
from spaceshift import research_tree

result = research_tree(
    "How do general-purpose technologies reshape economic structures?",
    sub_n=[5, 3],      # 5 subtopics, each split into 3 more
    super_n=[5],        # 5 broader framings
    side_n=[3],         # 3 lateral perspectives
)

print(result["root_output"])       # Response to the original question
print(len(result["outputs"]))     # Responses for every node in the tree
# All saved to markdown files with metadata + tree.svg visualization
```

---

### Research Expand

Recursive research expansion. Generates an output, expands into followup prompts across sub/super/side directions, generates outputs for those, and recurses to arbitrary depth.

```python
from spaceshift import research_expand

result = research_expand(
    "What are the second-order effects of AI on labor markets?",
    depth=2,            # root + 2 levels of followup expansion
    model=[1, 2],       # model 1 for depth 0, model 2 for depth 1+
)

print(result["root_output"])       # Response to the original question
print(len(result["outputs"]))     # All nodes across the expansion tree
# Each node saved as a titled markdown file with citations
```

---

### Directional Prompt Exploration

Most prompt decomposition only goes downward — breaking big into small. spaceshift moves in three directions:

```python
from spaceshift import subprompt, superprompt, sideprompt, prompt_tree

# Down — decompose into focused subtopics
subs = subprompt("What are the effects of AI on labor markets?", n=[5])

# Up — discover the bigger question this is a piece of
supers = superprompt("What are the effects of AI on labor markets?", n=[3])
# e.g. "How do general-purpose technologies reshape economic structures?"

# Sideways — explore sibling questions at the same abstraction level
sides = sideprompt("What are the effects of AI on labor markets?", n=[4])
# e.g. "What are the effects of AI on education systems?"

# All three at once, with tree visualization
tree = prompt_tree(
    "What are the effects of AI on labor markets?",
    sub_n=[5], super_n=[3], side_n=[4],
    viz=True,
)
tree["graph"].render("exploration_tree", cleanup=True)  # saves SVG
```

---

### Language Transform

Route a question through another language to sample different reasoning. Not a translation utility — an exploration axis.

```python
from spaceshift import language_transform

# Think about the question in Korean, get the answer back in English
result = language_transform(
    "What is the role of honor in modern society?",
    language="korean",
    save="honor_korean.md",
)

print(result["translated_prompt"])     # The question in Korean
print(result["translated_response"])   # Response generated in Korean
print(result["output_response"])       # That response translated back to English
```

Built-in languages: Chinese, Korean, Hindi, French, Arabic — or pass any language string.

---

### Compare Models

Run the same question across multiple models, auto-evaluate and rank the responses.

```python
from spaceshift import compare_models

result = compare_models(
    "Explain why the sky is blue",
    models=[1, 2, 3, 4, 5],               # top 5 ranked models
    save="sky_comparison",                  # saves ranked markdown files
)

print(result["top_model"])                 # which model won
print(result["rankings"])                  # full ranking
print(result["scores"])                    # normalized scores
```

Supports shorthand selection (`1`, `"best"`, `"fast3"`), explicit model names (`"claude-opus-4-6"`), and inline reasoning effort (`"gpt-5.4(xhigh)"`).

---

### Grid Search

Search across models and prompt transforms simultaneously. Find the best combination.

```python
from spaceshift import grid_search

# 4 transforms x 4 models = 16 cells + 4 original = 20 total, all evaluated
result = grid_search(
    "Explain quantum entanglement",
    models=[1, 2, 3, 4],
    n_transforms=4,                        # auto-select 4 random transforms
    save="quantum_grid",
)

print(result["top_output"])                # best response across entire grid
print(result["top_model"])                 # which model won
print(result["top_transform"])             # which transform won
print(result["grid"][:3])                  # top 3 cells with scores
```

---

### Prompt Probe

Sample the output space by applying transforms to your question, generating responses for each variant, and evaluating to find the best one.

```python
from spaceshift import prompt_probe

result = prompt_probe(
    "Explain why the sky is blue",
    n=6,                                   # try 6 random transforms
    save="sky_probe",
)

print(result["top_output"])                # best response found
print(result["top_transform"])             # which transform produced it
print(result["rankings_transforms"])       # full ranking of transforms
```

---

### Pairwise Evaluate

Rank any set of LLM outputs. Uses position-swap bias mitigation, auto-generated metrics, and Bradley-Terry sampling for large sets.

```python
from spaceshift import LLM, pairwise_evaluate

# Generate responses however you want, then evaluate
responses = [LLM(model=m).user("Explain dark matter").res() for m in [1, 2, 3]]

result = pairwise_evaluate(
    results=responses,
    metrics=["How accessible is this for a general audience?"],
)

print(result["rankings"])                  # [2, 0, 1] = third response was best
print(result["scores"])                    # normalized scores per response
```

Score a single response:

```python
llm = LLM().user("Write a haiku about recursion").chat()
score = llm.evaluate_last(
    metrics={"How well does this follow 5-7-5 structure?": "1-10"}
)
print(score["score"])                      # 0.0 - 1.0
```

---

### Prompt Transforms

20+ built-in transforms that reframe questions in different ways. Used internally by prompt_probe and grid_search, but available directly.

```python
from spaceshift import prompt_transform, list_transforms

list_transforms()                          # see all available transforms

# Apply a single transform
result = prompt_transform("Explain gravity", "abstract_up")
# e.g. "Explain fundamental forces in physics"

result = prompt_transform("Explain gravity", "inverse")
# e.g. "What would a universe without gravity look like?"
```

Available transforms include abstraction shifts, perspective changes, dimensional manipulation, language translations, and more.

---

### Everything Saves to Markdown

Every tool supports `save=` to write structured markdown with YAML frontmatter. Outputs are ranked, labeled, and organized for comparison.

```python
# Compare models — saves ranked files: 1_claude-opus-4-6.md, 2_gpt-5.2.md, ...
compare_models("Explain dark matter", save="dark_matter")

# Research tree — saves per-node files + tree.svg visualization
research_tree("Effects of AI on education", save="ai_education")

# Grid search — saves per-cell files with transform + model in filename
grid_search("Explain gravity", save="gravity_grid")
```

Browse any output directory in the browser with the built-in viewer:

```bash
spaceshift view gravity_grid
```

```python
from spaceshift import view
view("gravity_grid")
```

Two-panel layout: sidebar with smart-sorted file list, content area with rendered markdown and frontmatter metadata cards. No dependencies — runs on Python's stdlib server with client-side markdown rendering.

---

### The LLM Interface

Everything above is built on a clean, composable LLM class. One class, all providers via [litellm](https://github.com/BerriAI/litellm).

```python
from spaceshift import LLM

# Basic usage
LLM().user("Explain quantum computing").chat(stream=True)

# LLM as function — define with {placeholders}, returns parsed JSON
sentiment = LLM().sys("Classify sentiment. Return JSON with key 'sentiment'.").user("{text}")
sentiment.run("I love this product!")    # {"sentiment": "positive"}
sentiment.run(text="This is terrible")   # {"sentiment": "negative"}

# Batch processing
results = sentiment.run_batch(
    [{"text": "Love it!"}, {"text": "Terrible"}, {"text": "It's okay"}],
    concurrency=10,
)

# Model selection
LLM(model="gpt-5.2")
LLM(model="best")       # top intelligence rankings
LLM(model="fast")       # optimized for speed
LLM(model="cheap")
LLM(model="open")       # open-source models
LLM(model=1)            # top overall (default)

# Multi-turn conversation
chat = (
    LLM(model="fast")
    .sys("You are a bitcoin analyst")
    .user("What is proof of work?").chat()
    .user("Steel man the case for bitcoin mining").chat()
)

# Tool calling
def search_docs(query: str):
    """Search internal documentation."""
    return f"Found: '{query}' appears in onboarding.md"

LLM().tools(fns=[search_docs]).user("Find the onboarding guide").chat()

# Images and audio
LLM().user("What's in this image?", image="photo.jpg").chat()
LLM().user("Summarize this", audio="meeting.mp3").chat()

# Cost tracking
llm = LLM(model="best").user("Explain quantum computing").chat()
print(f"Cost: ${llm.total_cost():.6f}")

# Save, load, export
llm.save_llm("agents/researcher.json")
loaded = LLM.load_llm("agents/researcher.json")
llm.to_md("conversation.md")
```

Prompt queues, context compaction, web search, reasoning, vision auto-switching, and more — see the source for full capabilities.

---

### Install

```bash
pip install spaceshift
```

Your access to models is based on your API keys from the various providers — keys are available for free from most providers. Create a local `.env` file with at least one API key:

```env
OPENAI_API_KEY=sk-proj-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=AIza...
TOGETHERAI_API_KEY=...
XAI_API_KEY=xai-...
```
