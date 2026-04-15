# Changelog

**spaceshift** is an open prompt exploration toolkit powered by LLMs. Manipulate prompts through transforms, navigate the full space of perspectives, and evaluate across prompts and models to find what works best.

---

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-04-15

Major pivot: spaceshift is now an **interactive CLI prompt exploration toolkit**. The deep-research CLI surface is gone; the library has been rebuilt around five guided modes that branch, navigate, and evaluate prompts.

### Added
- **Interactive CLI** (`spaceshift`): launches a guided menu with five modes
  - **Prompt Manipulate**: apply any of 22 built-in transforms (abstraction, inversion, reflection, rotation, dimension shifts, translations…) and optionally generate outputs for each variant. Includes an automatic `original` baseline.
  - **Prompt Tree**: explore a prompt in three directions (sub/super/side), configure depth per direction, render a tree visualization
  - **Prompt Chain**: build a multi-turn conversation by adding followups one at a time
  - **Compare Models**: run one prompt across multiple models with optional pairwise evaluation (auto-generated or custom metrics)
  - **Grid Search**: sweep models × transforms and rank every combination
- **Built-in markdown viewer** (`spaceshift view <dir>`): two-panel browser UI with sidebar file list, YAML frontmatter cards, KaTeX math rendering, and anchor-link navigation
- **Copy-pasteable `spaceshift view` hint** printed after every save so results open in one step from any cwd
- **Global API key management**: first-run setup wizard writes to `~/.spaceshift/config.json`; editable from the main menu
- **Step-back navigation**: every multi-step interactive flow supports going back to the previous question
- **Per-response incremental save + Ctrl-C handling** in Prompt Manipulate — interrupting a run keeps completed outputs and marks pending ones
- **Custom evaluation metrics** in pairwise eval (fall back to auto-generated metrics)

### Changed
- **BREAKING**: deep-research CLI (`cruise-research`, `research`, `research_expand`) and its docs have been removed. The underlying `research_tree` helper still exists but is unsupported — the CLI is the primary interface.
- **`research_tree` remains available** as an advanced unsupported API for programmatic use
- Grid search UX: no default model pre-selection, cleaner eval prompt, optimal-category summary
- Transform-selection menu surfaces descriptions and a running count of selected transforms
- Viewer: heading IDs added to rendered markdown and in-page hash links scroll the content pane (previously they did nothing)
- Translate transforms run response through a back-translation phase so final output is in English

### Fixed
- `_enforce_json` no longer crashes when context contains brace-delimited tokens
- Compare-models anchor links, eval toggle UX, and output-folder prompt
- Translate-transform outputs now correctly return English (back-translation runs after the main output)

---

## [1.1.0] - 2026-03-26

### Added
- **`research_expand()`**: Recursive research expansion — generate an output, expand into followup prompts across sub/super/side directions, generate outputs for those, and recurse to arbitrary depth
  - Per-depth model selection: `model=[1, 2]` uses model 1 for depth 0, model 2 for depth 1+
  - Custom followup generation via `followup_model=` — pass a model name or a pre-configured `LLM` instance with template variables
  - Incremental file saving with human-readable titles as filenames
  - Citation extraction from web search results
  - Deduplication of followup prompts across the full expansion tree
- **Titled research outputs**: `research_tree` and `research_expand` now generate a title for each output, used as the saved filename instead of node IDs
- **Citation tracking**: Web search citations are extracted and saved in YAML frontmatter as a list of URLs

### Changed
- **`search="auto"` is the new default**: All prompt space functions (`subprompt`, `superprompt`, `sideprompt`, `prompt_tree`, `research_tree`, `research_expand`) now auto-detect whether the model supports web search and enable it if available — no need to pass `search=True` explicitly
- **Search Responses API uses `tool_choice="auto"`**: The model now decides when to search rather than being forced to, producing more natural outputs
- **YAML frontmatter handles lists**: Markdown output now correctly serializes list values (e.g., citations) as proper YAML arrays
- `LLM._has_search()` is now a static method

---

## [0.9.0] - 2026-02-15

### Changed
- **`run()` is the unified JSON interface**: `run()`, `run_json()`, `result_json()`, `res_json()`, and `rjson()` all route to a single method — always returns parsed JSON dicts. `run()` works with or without template variables.
- **`batch_run()` renamed to `run_batch()`**: Consistent naming under `run`. `batch_run` and `batch_run_json` kept as aliases.
- **`run_batch()` default concurrency lowered**: From 5 to 3 for safer defaults with rate-limited APIs.
- **`tool_choice` accepts callables**: Pass a function reference to force the model to call that specific tool
  - `LLM().tools(fns=[get_weather], tool_choice=get_weather)`
  - Also supports `"required"`, `"none"`, and `"auto"` (default)
- Internal: extracted `_parse_json()` helper — JSON parsing, markdown stripping, and LLM repair in one place
- Updated model rankings (2026-02-06)

---

## [0.8.0] - 2026-02-09

### Added
- **`batch_run()`**: Run the same LLM template across many inputs concurrently
  - `classifier.batch_run([{"text": "great"}, {"text": "awful"}], concurrency=10)`
  - Uses `ThreadPoolExecutor` with isolated LLM instances per call
  - Results returned in input order
  - `return_errors=True` returns `{"error": str, "input": dict}` instead of raising
  - Cost aggregation printed when `v=True`
- **`batch_run_json()`**: Same as `batch_run()` but returns parsed JSON dicts
  - Supports `enforce=True` (default) for LLM-based JSON repair on parse failure
- **`cost_report()`**: Full cost summary with per-model breakdown
  - Returns `total_cost`, `num_calls`, `avg_cost`, `total_input_tokens`, `total_output_tokens`, `by_model`
- **`assistant` alias**: Fixed `asssistant` typo — `assistant` now works (triple-s kept for backward compat)
- **`max_retries` parameter**: `LLM(max_retries=5)` — configurable instead of hardcoded `2`
- **Return type hints** on all public methods (`chat() -> LLM`, `run() -> str`, `last() -> str | None`, etc.)

### Changed
- README reorganized: `.run()` is now the headline API pattern, batch processing section added, `.chat()` documented as the multi-turn option

---

## [0.7.0] - 2026-02-06

### Added
- **`.compact()`**: Summarize older messages to manage long conversations
  - Keeps last 10 messages, summarizes the rest into a structured summary appended to system prompt
  - Iterative: subsequent compactions merge into existing summary rather than regenerating
  - Optional `model=` override for the summarization LLM
- **`auto_compact`**: Automatic compaction when conversations get long
  - `LLM(auto_compact=30)` (default) — compacts when messages reach 30
  - `LLM(auto_compact=0)` to disable

### Changed
- Updated model rankings (2026-02-06)

---

## [0.6.0] - 2026-02-06

### Added
- **Audio support**: `.user(audio="file.mp3")` sends audio natively to models
  - Supports WAV, MP3, FLAC, OGG, M4A, AAC, OPUS, WebM formats
  - `prompt` is now optional in `.user()` — audio can be the entire input
  - Auto-switches to audio-capable model when current model doesn't support it
  - Multiple audio files: `.user("Compare", audio=["a.wav", "b.wav"])`
  - Combined with images: `.user("Describe", image="photo.jpg", audio="clip.wav")`
  - URL audio downloaded and base64-encoded automatically
- **`.transcribe()`**: Standalone transcription utility via Whisper
  - `LLM().transcribe("recording.wav")` — tries whisper-1, then groq/whisper variants
  - Supports single files or lists, local paths or URLs
- **`models_with_audio_input()`**: Discover audio-capable models
- **`evaluate()`**: Pairwise LLM output comparison and ranking
  - `evaluate(results, prompts, metrics)` for ranking multiple outputs
  - `LLM.evaluate_last()` for scoring single responses with absolute metrics
  - Auto-generated metrics when none provided
  - Position swap for bias mitigation (default on)
  - Bradley-Terry sampling for >5 items
- **`require_audio()` generator tool**: `generate()` can now flag audio capability

### Changed
- `get_models_for_category()` accepts `audio=True` filter
- Model auto-switch now normalizes reasoning_effort for cross-provider compatibility

---

## [0.5.0] - 2026-02-02

### Added
- **`generate()` method**: Create configured LLM instances from natural language descriptions
  - `LLM().generate("A DCF analyst that takes a stock ticker")` returns a ready-to-use LLM
  - Uses tool-calling internally to configure system prompt, inputs, reasoning, search, etc.
  - Generator LLM configuration (model, reasoning) influences quality of generated instance
- **Optional template variables**: `{var?}` syntax for optional inputs that default to empty string
  - `llm.user("Analyze {ticker} {context?}")` - context becomes "" if not provided
  - `get_template_vars(split=True)` returns `{'required': set(), 'optional': set()}`
- **Positional argument for `run()`**: When exactly one required variable, pass it directly
  - `dcf.run("TSLA")` instead of `dcf.run(ticker="TSLA")`
- **Simple numeric model selection**: Numbers 1-N zip optimal and best rankings
  - `LLM(model=1)` = top optimal, `LLM(model=2)` = top best, `LLM(model=3)` = second optimal, etc.

### Changed
- Model rank suffixes are now 1-indexed: `best1` (not `best0`) selects the top model
- Default model selection changed from `optimal` to `optimal1` (deterministic top optimal)

---

## [0.4.0] - 2026-01-27

### Added
- **LLM as function pattern**: New `run()` and `run_json()` methods with template interpolation
  - Define prompts with `{placeholders}`, then call `llm.run(var="value")`
  - `get_template_vars()` returns all placeholder names in the LLM
- **JSON enforcement**: Auto-fix malformed JSON using a fast LLM
  - `result_json()`, `run_json()`, and `last_json()` now have `enforce=True` by default
  - Falls back to LLM-based repair when `json.loads()` fails

### Changed
- Improved `_strip_markdown_json()` to handle edge cases with markdown code fences

---

## [0.3.0] - 2026-01-21

### Added
- **New model categories**: `optimal` (balanced best+fast) and `codex` (code-focused models)
- **Deterministic model selection**: Use `best0`, `fast1`, `cheap2` etc. to select exact rank
- **Auto reasoning effort**: Rankings now include `reasoning_effort` metadata, auto-applied when selecting category models
- **Vision support**: Auto-switches to vision-capable model when images are attached
- **`result_json()`**: Run in JSON mode, return parsed dict, reset history
- **`models_with_vision()`**: List models with vision support

### Changed
- Default model selection now uses `optimal` category instead of best/fast intersection
- `res_json()` now strips markdown code fences before parsing
- Improved open source model detection (llama, deepseek, qwen, mistral, kimi, etc.)

---

## [0.2.2] - 2026-01-21

### Added
- **Image support**: Attach images to prompts via `.user(prompt, image="path")` or with multiple images as a list
  - Supports local files (automatically converted to base64)
  - Supports URLs (passed directly to vision-capable models)
- **Cost tracking**: Track token usage and costs across completions
  - `last_cost()` - cost of most recent completion
  - `total_cost()` - sum of all completion costs in session
  - `all_costs()` - full array of cost objects with token breakdowns
  - Uses litellm's model_cost database (98% coverage)
  - Warns when search is enabled (search costs vary by provider and aren't captured)

---

## [0.2.1] - 2026-01-15

### Fixed
- Claude web search now works correctly (server-handled tool calls no longer trigger client execution)
- Groq search uses correct `browser_search` tool format
- Grok/Groq reasoning parameter compatibility

### Changed
- README caveat addition

---

## [0.2.0] - 2026-01-02

### Breaking Changes
- Renamed `.add_followup()` method to `.queue()` for clearer intent

### Added
- Comprehensive docstrings across all public methods and the `LLM` class
- Public method `get_models_for_category(category_str)` to retrieve ranked models for a given category
- Bigger test suite to tests/test_llm.py

### Changed
- Removed hard-coded `max_tokens` default, now respects model/provider defaults unless explicitly set

### Fixed
- Removed broken or unavailable models from the static rankings list

---

## [0.1.3] - 2025-12-19

### Added
- **Model categories**: Pass `"best"`, `"cheap"`, `"fast"`, or `"open"` as the `model` parameter to auto-select from top-ranked models in that category
  ```python
  LLM(model="fast")   # Selects a fast model
  LLM(model="cheap")  # Selects a budget-friendly model
  LLM(model="best")   # Selects a top-performing model
  LLM(model="open")   # Selects an open-source model
  ```
- Static model rankings bundled with the package (`rankings/static_rankings_2025-12-19.json`)
- Smarter default model selection: when no model is specified, selects from models that rank high in both "best" and "fast" categories

### Changed
- Moved ranking generation scripts to `scripts/` folder (not distributed with package)

## [0.1.2] - 2024

### Added
- Initial public release
- Core `LLM` class with chat, result, and JSON modes
- Tool/function calling support
- Web search integration
- Reasoning model support
- Multi-provider support via litellm (OpenAI, Anthropic, Gemini, etc.)
- Fuzzy model name matching with `rapidfuzz`
- Save/load LLM state to JSON
- Export conversations to Markdown

