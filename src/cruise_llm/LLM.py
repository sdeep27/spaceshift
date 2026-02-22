from __future__ import annotations
import litellm
from litellm import completion, responses
import json
import copy
import base64
import mimetypes
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from function_schema import get_function_schema
import logging
import rapidfuzz
import random
from pathlib import Path

_RANKINGS_PATH = Path(__file__).parent / "rankings" / "static_rankings_2026-02-06.json"
with open(_RANKINGS_PATH, "r") as f:
    _raw_rankings = json.load(f)

def _parse_rankings(raw):
    """Convert new dict format to model list, preserving reasoning_effort metadata."""
    parsed = {}
    for category, items in raw.items():
        if isinstance(items, list) and items and isinstance(items[0], dict):
            parsed[category] = items
        else:
            parsed[category] = [{"model": m} for m in items]
    return parsed

model_rankings = _parse_rankings(_raw_rankings)

def _get_model_name(entry):
    """Extract model name from ranking entry (dict or string)."""
    return entry["model"] if isinstance(entry, dict) else entry

def _get_reasoning_effort(entry):
    """Extract reasoning_effort from ranking entry if present."""
    if isinstance(entry, dict):
        return entry.get("reasoning_effort")
    return None

load_dotenv()
litellm.drop_params = True

_AUDIO_FORMAT_MAP = {
    '.wav': 'wav', '.mp3': 'mp3', '.m4a': 'mp3',
    '.ogg': 'ogg', '.flac': 'flac', '.aac': 'aac',
    '.opus': 'opus', '.webm': 'webm',
}

# litellm hasn't caught up with audio support flags for these yet
_AUDIO_INPUT_OVERRIDES = {
    "gemini/gemini-3-flash-preview",
    "gemini/gemini-2.5-flash",
    "gemini/gemini-2.5-flash-lite",
}

class LLM:
    """
    A chainable, stateful wrapper around LiteLLM for building composable agents and workflows.

    This class manages a highly flexible conversation history with interchangeable models. 
    The API is meant for method chaining with a concise syntax, supporting quick prototyping. 
    It supports dynamic model aliasing (e.g., "best", "fast") with fuzzy model matching, 
    saving and loading preset workflows and prompt queues, 
    and easy tool integration without manual schema definition.
    """
    def __init__(self, model=None, temperature=None, stream=False, v=True, debug=False, max_tokens=None, search=False, reasoning=False, search_context_size="medium", reasoning_effort="medium",sub_closest_model=True, auto_compact=30, max_retries=2):
        """
        Initialize the LLM client.

        Args (tends to match OpenAI/litellm completion API spec):
            model (str, optional): The model name (e.g., "gpt-4o", "claude-3-5-sonnet") or a category alias
                ("best", "fast", "cheap", "open", "optimal", "codex"). Supports deterministic selection
                with 1-indexed suffix (e.g., "best1", "optimal2") or simple numbers (1, 2, 3...) which
                zip best and optimal rankings (1=top best, 2=top optimal, 3=second best, etc.).
                Defaults to "best1" (top best model).
            temperature (float, optional): Sampling temperature.
            stream (bool): If True, streams output to stdout.
            v (bool): Verbosity flag. If True, prints prompts, tool calls, and responses to stdout.
            debug (bool): If True, enables verbose LiteLLM logging.
            max_tokens (int, optional): Max tokens for generation.
            search (bool): If True, enables web search capabilities for all messages. Can instead be enabled on a per message basis, using .tools method
            reasoning (bool): If True, enables reasoning capabilities for all messages. Can instead be enabled on a per message basis, using .tools method
            search_context_size (str): Context size for search ("short", "medium", "long"). Defaults to "medium".
            reasoning_effort (str): Effort level for reasoning models ("low", "medium", "high").
            sub_closest_model (bool):  Defaults to True. Attempts to fuzzy match the model name if the exact name isn't found (e.g., "gpt4" -> "gpt-4").

        Raises:
            ValueError: If no API keys are found in the environment.
        """
        self.chat_msgs = []
        self.logs = []
        self.response_metadatas = []
        self.costs = []
        self.prompt_queue = []
        self.prompt_queue_remaining = 0
        self.last_chunk_metadata = None
        self.available_models = self.get_models()
        if not self.available_models:
            raise ValueError("Make sure you have at least one API key configured. Create a .env file in your project and add a variable line: (e.g., OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY)")
        self.sub_closest_model = sub_closest_model
        self.model = self._check_model(model)
        self.temperature = temperature
        self.stream = stream
        self.v = v #verbosity
        self.available_tools = None
        self.schemas = None
        self.tool_choice = "auto"
        self.parallel_tool_calls = True
        self.fn_map = None
        self.search_enabled = search
        self.search_context_size = search_context_size
        if not getattr(self, 'reasoning_enabled', None):
            self.reasoning_enabled = reasoning
        if not getattr(self, 'reasoning_effort', None):
            self.reasoning_effort = reasoning_effort
        if self.search_enabled:
            if not self._has_search(self.model):
                self._update_model_to_search()
        if self.reasoning_enabled:
            if not self._has_reasoning(self.model):
                self._update_model_to_reasoning()
        self.auto_compact = auto_compact
        self.max_retries = max_retries
        self.reasoning_contents = []
        self.search_annotations = []
        self.max_tokens = max_tokens
        if debug == True:
            self.turn_on_debug()
        else:
            self.turn_off_debug()
    
    def _resolve_args(self, **kwargs):
        """Merges chat, chat_json, res, res_json kwargs with instance init defaults."""
        instance_defaults   = {
            "model": kwargs.get("model", self.model),
            "temperature": kwargs.get("temperature", self.temperature),
            "stream": kwargs.get("stream", self.stream),
            "v": kwargs.get("v", self.v),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }
        return {**instance_defaults, **kwargs}

    def _logger_fn(self, model_call_dict):
        self.logs.append(model_call_dict)

    def _process_image(self, image_source):
        if image_source.startswith(('http://', 'https://')):
            return {"type": "image_url", "image_url": {"url": image_source}}
        mime_type, _ = mimetypes.guess_type(image_source)
        mime_type = mime_type or "image/png"
        with open(image_source, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}}

    def _detect_audio_format(self, source):
        """Detect audio format from file path or URL extension."""
        from urllib.parse import urlparse
        path = urlparse(source).path if source.startswith(('http://', 'https://')) else source
        ext = Path(path).suffix.lower()
        if ext in _AUDIO_FORMAT_MAP:
            return _AUDIO_FORMAT_MAP[ext]
        if source.startswith(('http://', 'https://')):
            try:
                import urllib.request
                req = urllib.request.Request(source, method='HEAD')
                with urllib.request.urlopen(req, timeout=10) as resp:
                    ct = resp.headers.get('Content-Type', '')
                    # e.g. "audio/mpeg" -> "mp3", "audio/wav" -> "wav"
                    if '/' in ct:
                        sub = ct.split('/')[1].split(';')[0].strip()
                        if sub == 'mpeg':
                            return 'mp3'
                        return sub
            except Exception:
                pass
        return 'wav'

    def _process_audio(self, audio_source):
        """Process an audio file path or URL into a litellm input_audio content block."""
        import urllib.request
        fmt = self._detect_audio_format(audio_source)
        if audio_source.startswith(('http://', 'https://')):
            req = urllib.request.Request(audio_source)
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
        else:
            with open(audio_source, "rb") as f:
                data = f.read()
        b64 = base64.b64encode(data).decode("utf-8")
        return {"type": "input_audio", "input_audio": {"data": b64, "format": fmt}}

    def _strip_markdown_json(self, text):
        """Strip markdown code block wrapper from JSON if present."""
        if not isinstance(text, str):
            return text
        text = text.strip()
        if text.startswith('```'):
            lines = text.split('\n')
            # Remove first line (```json or ```)
            if lines[-1].strip() == '```':
                lines = lines[1:-1]
            else:
                lines = lines[1:]
            text = '\n'.join(lines).strip()
        return text

    def _enforce_json(self, text):
        """Use LLM to fix malformed JSON. Returns parsed dict or {} on failure."""
        enforcer = LLM(model="fast", stream=False, v=False) \
            .sys('You are a JSON enforcer. The user will provide text that should be valid JSON but may have issues. Return ONLY valid JSON that can be parsed by json.loads. Fix any syntax errors, missing brackets, or malformed structures. Output nothing except the corrected JSON.') \
            .user('{text}')
        return enforcer.run(text=text, enforce=False)

    def _parse_json(self, text, enforce=True):
        """Strip markdown fences, parse JSON, optionally enforce with LLM repair."""
        text = self._strip_markdown_json(text)
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            if enforce:
                return self._enforce_json(text)
            print(f"!! Error parsing JSON: {text}")
            return {}

    def _track_cost(self, response, model):
        usage = getattr(response, 'usage', None)
        if not usage:
            return
        input_tokens = getattr(usage, 'prompt_tokens', 0) or 0
        output_tokens = getattr(usage, 'completion_tokens', 0) or 0
        try:
            total_cost = litellm.completion_cost(completion_response=response, model=model)
        except:
            total_cost = 0
        self.costs.append({
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_cost": total_cost,
        })

    def tools(self, fns = [], tool_choice="auto", parallel_tool_calls=True, search=False, search_context_size=None, reasoning=False, reasoning_effort=None) -> LLM:
        """
        Register tools (functions) or enable capabilities like Web Search or Reasoning.
        Automatically generates JSON schemas from the provided Python functions.

        Args:
            fns (list): A list of Python callable functions. Type hints and docstrings
                are recommended on the functions for accurate schema generation.
            tool_choice: Controls how the model selects tools. Accepts:
                - "auto" (default): model decides whether to call a tool
                - "required": model must call at least one tool
                - "none": model must not call any tools
                - A callable (function reference): forces the model to call that specific function.
                  e.g. tool_choice=get_weather
            parallel_tool_calls (bool): Allow the model to call multiple tools in parallel.
                Defaults to True.
            search (bool): Enable web search. If the current model doesn't support it,
                attempts to switch to a supported model.
            search_context_size (str, optional): "short", "medium", "long".
            reasoning (bool): Enable reasoning. If current model doesn't support it,
                attempts to switch to a supported model.
            reasoning_effort (str, optional): "low", "medium", "high".

        Returns:
            self: For chaining.
        """
        schemas = [get_function_schema(fn) for fn in fns]
        self.schemas = schemas
        self.fn_map = {schema['name']: fn for schema, fn in zip(schemas, fns)}
        tools = []
        for fn in fns:
            tool = {
                "type": "function",
                "function": get_function_schema(fn)
            }
            tool["function"]["strict"] = True
            tool["function"]["parameters"]["additionalProperties"] = False
            tools.append(tool)
        self.available_tools = tools
        if callable(tool_choice):
            tool_choice = {"type": "function", "function": {"name": tool_choice.__name__}}
        self.tool_choice = tool_choice
        self.parallel_tool_calls = parallel_tool_calls
        if search:
            if not self._has_search(self.model):
                self._update_model_to_search()
            self.search_enabled = True
            self.temperature = None # openAI search model does not accept temperature
            if search_context_size:
                self.search_context_size = search_context_size
        else:
            self.search_enabled = False
            if search_context_size:
                self.search_context_size = None
        if reasoning:
            if not self._has_reasoning(self.model):
                self._update_model_to_reasoning()
            self.reasoning_enabled = True
            self.temperature = None # Anthropic doesnt want temperature when theres reasoning 
            if reasoning_effort:
                self.reasoning_effort = reasoning_effort
        else:
            self.reasoning_enabled = False
            if reasoning_effort:
                self.reasoning_effort = None
        return self

    def chat(self, **kwargs) -> LLM:
        """
        Run the LLM prediction based on current history, appending the response to the internal log.
        Generally follows a .user update, e.g. LLM().user("hi").chat()
        This is a stateful call; it updates `self.chat_msgs`.

        Args:
            **kwargs: Overrides for run-specific settings (temperature, model, etc.).

        Returns:
            self: For chaining (e.g., `.chat().user("Next question")`).
        """
        self._run_prediction(**kwargs)
        return self

    c = ch = chat

    def chat_json(self, **kwargs) -> LLM:
        """
        Same as `chat()`, but enforces JSON mode on the model response.

        Returns:
            self: For chaining.
        """
        self._run_prediction(jsn_mode=True, **kwargs)
        return self
    
    cjson = c_json = ch_json = chat_json

    def result(self, **kwargs) -> str:
        """
        Run the prediction, return the response text, and reset the chat history
        (preserving the System prompt).

        Useful for single-turn tasks where you don't want history functionality 
        cluttering the context window.

        Args:
            **kwargs: Overrides for run-specific settings.

        Returns:
            str: The assistant's response content.
        """
        self._run_prediction(**kwargs)
        last_res = self.last()
        self._reset_msgs()
        return last_res
    
    r = res = result

    def _interpolate_templates(self, **kwargs):
        """
        Interpolate {placeholder} template variables in all chat messages.
        Handles both required {var} and optional {var?} syntax.
        Uses regex substitution so literal JSON braces are never touched.

        Args:
            **kwargs: Template variables to interpolate (e.g., ticker="TSLA")

        Returns:
            self: For chaining.
        """
        import re

        def interpolate_text(text):
            def replace_optional(match):
                var_name = match.group(1)
                return str(kwargs.get(var_name, ''))
            text = re.sub(r'\{(\w+)\?\}', replace_optional, text)
            def replace_required(match):
                var_name = match.group(1)
                if var_name in kwargs:
                    return str(kwargs[var_name])
                return match.group(0)
            return re.sub(r'\{(\w+)\}', replace_required, text)

        for msg in self.chat_msgs:
            content = msg.get('content', '')
            if isinstance(content, str):
                msg['content'] = interpolate_text(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        item['text'] = interpolate_text(item['text'])
        self.prompt_queue = [interpolate_text(p) for p in self.prompt_queue]
        return self

    def get_template_vars(self, split=False):
        """
        Return placeholder variable names found in all prompts.

        Args:
            split: If True, returns dict with 'required' and 'optional' sets.
                   If False (default), returns flat set of all var names (without ? suffix).

        Returns:
            set or dict: Variable names. Optional vars are marked with ? suffix in templates.
                e.g., {text} is required, {context?} is optional
        """
        import re
        required_pattern = re.compile(r'\{(\w+)\}')  # {var}
        optional_pattern = re.compile(r'\{(\w+)\?\}')  # {var?}

        required = set()
        optional = set()

        def extract_from_text(text):
            # Find optional first (so we don't double-count)
            opt = set(optional_pattern.findall(text))
            optional.update(opt)
            # Find all {word} patterns, exclude those that are optional
            all_vars = set(required_pattern.findall(text))
            required.update(all_vars - opt)

        for msg in self.chat_msgs:
            content = msg.get('content', '')
            if isinstance(content, str):
                extract_from_text(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        extract_from_text(item.get('text', ''))

        for prompt in self.prompt_queue:
            extract_from_text(prompt)

        if split:
            return {'required': required, 'optional': optional}
        return required | optional

    def run(self, __input__=None, enforce=True, **kwargs) -> dict:
        """
        Execute the LLM with template variable interpolation, returning parsed JSON.

        Interpolates {key} placeholders in all chat_msgs with provided kwargs,
        runs the prediction in JSON mode, and returns the parsed response.

        Supports optional variables with {var?} syntax - these become empty string if not provided.

        This enables "LLM as function" usage (dict in -> dict out):
            dcf = LLM().sys("Analyze {ticker}. Return JSON with key 'valuation'.").user("{ticker}")
            result = dcf.run("TSLA")  # returns {"valuation": ...}

        Args:
            __input__: Single positional arg mapped to the sole required variable (if exactly one).
            enforce (bool): If True (default), uses an LLM to fix malformed JSON on parse failure.
            **kwargs: Template variables to interpolate (e.g., ticker="TSLA").
                      Any unknown kwargs are passed to the prediction (model, temperature, etc.)

        Returns:
            dict: The parsed JSON response. Returns empty dict on parsing error.
        """
        template_vars = self.get_template_vars(split=True)

        if __input__ is not None:
            if len(template_vars['required']) != 1:
                raise ValueError(f"Positional argument only works with exactly 1 required variable, found: {template_vars['required']}")
            var_name = next(iter(template_vars['required']))
            kwargs[var_name] = __input__

        all_vars = template_vars['required'] | template_vars['optional']
        interp_kwargs = {k: v for k, v in kwargs.items() if k in all_vars}
        pred_kwargs = {k: v for k, v in kwargs.items() if k not in all_vars}

        missing = template_vars['required'] - set(interp_kwargs.keys())
        if missing:
            raise ValueError(f"Missing required template variables: {missing}")

        saved_msgs = copy.deepcopy(self.chat_msgs)
        self._interpolate_templates(**interp_kwargs)
        self._run_prediction(jsn_mode=True, **pred_kwargs)
        result = self._parse_json(self.last(), enforce)
        self.chat_msgs = saved_msgs
        return result

    run_json = run
    result_json = rjson = res_json = run

    def _clone_for_batch(self) -> LLM:
        """Create an isolated LLM copy for batch execution (shares no mutable state)."""
        clone = LLM.__new__(LLM)
        clone.chat_msgs = copy.deepcopy(self.chat_msgs)
        clone.prompt_queue = list(self.prompt_queue)
        clone.prompt_queue_remaining = self.prompt_queue_remaining
        clone.logs = []
        clone.response_metadatas = []
        clone.costs = []
        clone.last_chunk_metadata = None
        clone.reasoning_contents = []
        clone.search_annotations = []
        clone.available_models = self.available_models
        clone.model = self.model
        clone.temperature = self.temperature
        clone.stream = False
        clone.v = False
        clone.max_tokens = self.max_tokens
        clone.max_retries = self.max_retries
        clone.sub_closest_model = self.sub_closest_model
        clone.available_tools = self.available_tools
        clone.schemas = self.schemas
        clone.tool_choice = self.tool_choice
        clone.parallel_tool_calls = self.parallel_tool_calls
        clone.fn_map = self.fn_map
        clone.search_enabled = self.search_enabled
        clone.search_context_size = self.search_context_size
        clone.reasoning_enabled = self.reasoning_enabled
        clone.reasoning_effort = self.reasoning_effort
        clone.auto_compact = self.auto_compact
        return clone

    def run_batch(self, inputs, concurrency=3, return_errors=False, enforce=True) -> list[dict]:
        """
        Run the LLM template across multiple inputs concurrently, returning parsed JSON.

        Each input gets an isolated LLM instance. Results are returned in input order.

        Args:
            inputs (list[dict]): List of template variable dicts, e.g. [{"text": "hello"}, {"text": "bye"}]
            concurrency (int): Max parallel threads. Defaults to 3.
            return_errors (bool): If True, failed calls return {"error": str, "input": dict} instead of raising.
            enforce (bool): If True (default), uses an LLM to fix malformed JSON on parse failure.

        Returns:
            list[dict]: Parsed JSON results in input order.

        Example:
            classifier = LLM().sys("Classify sentiment. Return JSON with key 'sentiment'.").user("{text}")
            results = classifier.run_batch([{"text": "great"}, {"text": "awful"}], concurrency=10)
            # results: [{"sentiment": "positive"}, {"sentiment": "negative"}]
        """
        results = [None] * len(inputs)

        def _exec(index, input_kwargs):
            clone = self._clone_for_batch()
            try:
                return index, clone.run(enforce=enforce, **input_kwargs), clone.costs
            except Exception as e:
                if return_errors:
                    return index, {"error": str(e), "input": input_kwargs}, []
                raise

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = {pool.submit(_exec, i, inp): i for i, inp in enumerate(inputs)}
            for future in as_completed(futures):
                idx, result, costs = future.result()
                results[idx] = result
                self.costs.extend(costs)

        if self.v:
            total = sum(c["total_cost"] for c in self.costs)
            print(f"run_batch: {len(inputs)} calls, total cost: ${total:.6f}")

        return results

    batch_run = run_batch
    batch_run_json = run_batch

    def last_json(self, enforce=True) -> dict:
        """Parse the last assistant response as JSON, stripping markdown fences if present."""
        return self._parse_json(self.last(), enforce)

    def _check_model(self, inputted_model):
        if not self.sub_closest_model:
            return inputted_model
        avail_models = self.get_models()
        if inputted_model in avail_models:
            return inputted_model

        category_result = self._handle_model_category(inputted_model)
        if category_result:
            return category_result

        def closest_match(inputted_model, choices):
            return rapidfuzz.process.extractOne(inputted_model, choices, scorer=rapidfuzz.fuzz.WRatio)[0]
        print(f"{inputted_model} not a valid model name.")
        new_model = closest_match(inputted_model, avail_models)
        print(f"Substituting {new_model}")
        return new_model
    
    def _handle_model_category(self, category_str):
        """Handle category aliases like 'best', 'fast', 'optimal', 'codex'.

        Supports:
        - Simple numbers (1, 2, 3...): zipped ranking where odd=optimal, even=best
          1=top optimal, 2=top best, 3=second optimal, 4=second best, etc.
        - Category with 1-indexed suffix (best1, optimal2): select exact rank in category
        - Category name alone (best, optimal): random selection from top N

        Returns None if not a valid category.
        """
        import re
        valid_categories = ['best', 'cheap', 'fast', 'open', 'optimal', 'codex', 'reasoning', 'search']

        if category_str is None:
            category_str = "best1"

        category_str = str(category_str)

        # Simple number: zipped best/optimal ranking (1=best[0], 2=optimal[0], 3=best[1], etc.)
        if category_str.isdigit():
            rank = int(category_str)
            if rank < 1:
                raise ValueError(f"Model rank must be >= 1, got {rank}")
            optimal_entries = model_rankings.get('optimal', [])
            best_entries = model_rankings.get('best', [])
            zipped = []
            max_len = max(len(optimal_entries), len(best_entries))
            for i in range(max_len):
                if i < len(best_entries):
                    zipped.append(('best', best_entries[i]))
                if i < len(optimal_entries):
                    zipped.append(('optimal', optimal_entries[i]))
            if rank > len(zipped):
                raise ValueError(f"Rank {rank} not available (max: {len(zipped)})")
            category, entry = zipped[rank - 1]
            model_name = _get_model_name(entry)
            effort = _get_reasoning_effort(entry)
            if effort and not getattr(self, 'reasoning_effort', None):
                self.reasoning_effort = effort
                self.reasoning_enabled = True
            return model_name

        # Category with 1-indexed suffix (e.g., best1, optimal2)
        match = re.match(r'^([a-z]+)(\d+)$', category_str.lower())
        if match:
            base_category = match.group(1)
            rank = int(match.group(2))
            if base_category in valid_categories:
                entries = model_rankings.get(base_category, [])
                if rank < 1:
                    raise ValueError(f"Rank must be >= 1, got {rank}")
                rank_index = rank - 1
                if rank_index < len(entries):
                    entry = entries[rank_index]
                    model_name = _get_model_name(entry)
                    effort = _get_reasoning_effort(entry)
                    if effort and not getattr(self, 'reasoning_effort', None):
                        self.reasoning_effort = effort
                        self.reasoning_enabled = True
                    return model_name
                else:
                    raise ValueError(f"Rank {rank} not available for {base_category} (max: {len(entries)})")
            return None

        # Random selection from category
        if category_str.lower() in valid_categories:
            candidates = self.get_models_for_category(category_str.lower())
            if not candidates:
                raise ValueError(f"No models available for category: {category_str}")
            selected = random.choice(candidates)
            for entry in model_rankings.get(category_str.lower(), []):
                if _get_model_name(entry) == selected:
                    effort = _get_reasoning_effort(entry)
                    if effort and not getattr(self, 'reasoning_effort', None):
                        self.reasoning_effort = effort
                        self.reasoning_enabled = True
                    break
            return selected

        return None

    def get_models_for_category(self, category_str, search=False, reasoning=False, vision=False, audio=False) -> list[str]:
        """
        Retrieve the list of models associated with a specific alias category.
        Optionally filter by capability (search, reasoning, vision, audio).

        Categories are defined in the static rankings file.

        Args:
            category_str (str): One of "best", "fast", "cheap", "open", "optimal", "codex".
            search (bool): If True, only return models that support web search.
            reasoning (bool): If True, only return models that support reasoning.
            vision (bool): If True, only return models that support vision.
            audio (bool): If True, only return models that support audio input.

        Returns:
            list: A list of model names (strings) sorted by rank for that category.
        """
        entries = model_rankings.get(category_str, [])

        # If filtering by capability, get all models first, filter, then apply limit
        if search or reasoning or vision or audio:
            models = [_get_model_name(entry) for entry in entries]
            if search:
                models = [m for m in models if self._has_search(m)]
            if reasoning:
                models = [m for m in models if self._has_reasoning(m)]
            if vision:
                models = [m for m in models if self._has_vision(m)]
            if audio:
                models = [m for m in models if self._has_audio_input(m)]
            return models[:10]
        
        # Default behavior: apply category-specific limits
        total = len(entries)
        category_limits = {
            "best": min(10, total),
            "cheap": min(10, total),
            "fast": min(10, total),
            "open": min(5, total),  
            "optimal": min(10, total),
            "codex": min(5, total), 
        }
        top_n = category_limits.get(category_str, min(10, total))

        return [_get_model_name(entry) for entry in entries[:top_n]]

    get_models_category = get_models_for_category

    def _run_xai_search(self, args, jsn_mode=False):
        """Handle xAI search using the Responses API (Agent Tools API).
        xAI deprecated Live Search Dec 2025 - web_search tool only works via Responses API.
        """
        # Convert chat messages to Responses API input format
        input_messages = []
        for msg in self.chat_msgs:
            role = msg.get('role')
            content = msg.get('content', '')
            if role == 'system':
                input_messages.append({"role": "developer", "content": content})
            elif role in ('user', 'assistant'):
                input_messages.append({"role": role, "content": content})

        resp_args = {
            "model": args['model'],
            "input": input_messages,
            "tools": [{"type": "web_search"}],
        }
        if args['temperature'] is not None:
            resp_args["temperature"] = args['temperature']
        if args['max_tokens'] is not None:
            resp_args["max_output_tokens"] = args['max_tokens']

        if args['v']:
            print(f"Requesting {args['model']} (Responses API with web_search)")

        resp = responses(**resp_args)
        self.response_metadatas.append(resp)

        # Extract text from the Responses API output
        output_text = ""
        for output_item in resp.output:
            if hasattr(output_item, 'content'):
                for content_item in output_item.content:
                    if hasattr(content_item, 'text'):
                        output_text += content_item.text

        self.asst(output_text, merge=False)
        self._track_cost_responses(resp, args['model'])
        if args['v']:
            actual_model = resp.model or args['model']
            print(f"ASSISTANT ({actual_model}):")
            print(f"{output_text}\n")

        # Handle prompt queue
        if self.prompt_queue and self.prompt_queue_remaining > 0:
            prompt_queue_index = len(self.prompt_queue) - self.prompt_queue_remaining
            self.user(self.prompt_queue[prompt_queue_index])
            self.prompt_queue_remaining -= 1
            self._run_xai_search(args, jsn_mode)

    def _track_cost_responses(self, resp, model):
        """Track costs from Responses API format."""
        usage = getattr(resp, 'usage', None)
        if not usage:
            return
        input_tokens = getattr(usage, 'input_tokens', 0) or 0
        output_tokens = getattr(usage, 'output_tokens', 0) or 0
        total_cost = getattr(usage, 'cost_in_usd_ticks', 0)
        if total_cost:
            total_cost = total_cost / 1e9  # Convert from ticks to USD
        self.costs.append({
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_cost": total_cost,
        })

    def _run_prediction(self, jsn_mode=False, **kwargs):
        if self.auto_compact and len(self.chat_msgs) >= self.auto_compact:
            self.compact()
        args = self._resolve_args(**kwargs)
        args['model'] = self._check_model(args['model'])
        model_lower = args['model'].lower()

        # xAI with search requires Responses API (Agent Tools API)
        if self.search_enabled and model_lower.startswith('xai/'):
            return self._run_xai_search(args, jsn_mode)

        chat_args = {
            "model": args['model'],
            "temperature": args['temperature'],
            "messages": self.chat_msgs,
            "logger_fn": self._logger_fn,
            "max_tokens": args['max_tokens'],
            "num_retries": self.max_retries,
            "stream": args['stream'],
        }
        if self.available_tools:
            chat_args["tools"] = self.available_tools
            chat_args["tool_choice"] = self.tool_choice
            chat_args["parallel_tool_calls"] = self.parallel_tool_calls

        if self.reasoning_enabled:
            if 'grok' in model_lower and 'reasoning' in model_lower:
                pass
            elif model_lower.startswith('groq/'):
                pass
            elif self.reasoning_effort == 'default':
                pass
            else:
                chat_args['reasoning_effort'] = self.reasoning_effort
        else:
            if "gemini-3" in self.model:
                chat_args['reasoning_effort'] = "minimal"
        if self.search_enabled:
            if model_lower.startswith('groq/'):
                chat_args["tools"] = [{"type": "browser_search"}]
                chat_args["tool_choice"] = "required"
            else:
                chat_args["web_search_options"] = {
                    "search_context_size": self.search_context_size
                }

        if jsn_mode:
            chat_args["response_format"] = {"type": "json_object"}
        if args['v']: print(f"Requesting {args['model']}")
        comp = completion(**chat_args)
        ## saving metadata
        self.response_metadatas.append(comp)

        if args['stream']:
            printed_header = False
            for chunk in comp:
                if not printed_header and args['v']:
                    actual_model = chunk.model or args['model']
                    print(f"ASSISTANT ({actual_model}):")
                    printed_header = True
                chunk_content = chunk.choices[0].delta.content or ""
                self.asst(chunk_content, merge=True)
                if args['v']: print(chunk_content, end="")
            self.last_chunk_metadata = chunk
            self._track_cost(chunk, args['model'])
            if args['v']: print("\n")
        else:
            self._save_reasoning_trace(self.response_metadatas[-1])
            self._save_search_trace(self.response_metadatas[-1])
            asst_msg = comp.choices[0].message
            if args['v']:
                actual_model = comp.model or args['model']
                print(f"ASSISTANT ({actual_model}):")
            # Tool loop
            while self._requests_tool(asst_msg):
                self._execute_tools(asst_msg)
                comp = completion(**chat_args)
                self.response_metadatas.append(comp)
                self._track_cost(comp, args['model'])
                asst_msg = comp.choices[0].message
            # Final text response
            self.asst(asst_msg.content, merge=False)
            self._track_cost(comp, args['model'])
            if args['v']: print(f"{asst_msg.content}\n")
        
        if self.prompt_queue and self.prompt_queue_remaining > 0:
            prompt_queue_index = len(self.prompt_queue) - self.prompt_queue_remaining
            self.user(self.prompt_queue[prompt_queue_index])
            self.prompt_queue_remaining -= 1
            self._run_prediction(jsn_mode, **kwargs)

    def _execute_tools(self, asst_msg):
        """Execute all tool calls, append results to history"""
        self.chat_msgs.append(asst_msg.to_dict())

        for tool_call in asst_msg.tool_calls:
            name = tool_call.function.name
            if name == 'web_search':
                if self.v: print(f'Skipping server-handled tool: {name}')
                continue
            arguments_str = tool_call.function.arguments
            args = json.loads(arguments_str) if arguments_str and arguments_str.strip() not in ("", "null", "{}") else None
            if self.v: print(f'Found Tool {name} {args}.')
            result = self.fn_map[name](**args) if args else self.fn_map[name]()
            if self.v: print('Executed Tool. Result:', result,"\n")

            self.chat_msgs.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result),
            })
   
    def _requests_tool(self, asst_msg):
        if asst_msg.tool_calls:
            if self.v: print('Received Tool Call Request.\n')
            return True
        else:
            return False
    def _requests_tool_streaming(self):
        "Currently tool calls are not supported for streaming responses. To be added."
        pass
    
    def user(self, prompt=None, image=None, audio=None) -> LLM:
        """
        Add a User message to the history.

        Args:
            prompt (str, optional): The message content. Can be omitted when audio is the entire input.
            image (str or list, optional): Path(s) to image file(s) or URL(s) to attach.
                If provided, will check if current model supports vision and switch if needed.
            audio (str or list, optional): Path(s) to audio file(s) or URL(s) to attach.
                If provided, will check if current model supports audio input and switch if needed.

        Returns:
            self: For chaining.
            Generally is followed by an inference call -> .chat, .result etc. or a preset .asst message
        """
        has_media = image is not None or audio is not None

        if not has_media:
            content = prompt
        else:
            if image is not None and not self._has_vision(self.model):
                self._update_model_to_vision()
            if audio is not None and not self._has_audio_input(self.model):
                self._update_model_to_audio_input()

            content = []
            if prompt is not None:
                content.append({"type": "text", "text": prompt})
            if image is not None:
                images = [image] if isinstance(image, str) else image
                for img in images:
                    content.append(self._process_image(img))
            if audio is not None:
                audios = [audio] if isinstance(audio, str) else audio
                for aud in audios:
                    content.append(self._process_audio(aud))

        user_msg_obj = {"role": "user", "content": content}
        self.chat_msgs.append(user_msg_obj)
        if self.v:
            print(f"USER:")
            if prompt:
                print(f"{prompt}\n")
            if image:
                img_count = 1 if isinstance(image, str) else len(image)
                print(f"[{img_count} image(s) attached]\n")
            if audio:
                aud_count = 1 if isinstance(audio, str) else len(audio)
                print(f"[{aud_count} audio file(s) attached]\n")
        return self
    
    u = usr = user

    def transcribe(self, audio, model=None):
        """
        Standalone transcription utility using litellm.transcription().

        Args:
            audio (str or list): Path(s) or URL(s) to audio file(s).
            model (str, optional): Transcription model. Defaults to trying whisper-1,
                then groq/whisper-large-v3-turbo.

        Returns:
            str or list: Transcribed text. String for single file, list for multiple.
        """
        models_to_try = [model] if model else ["whisper-1", "groq/whisper-large-v3-turbo", "groq/whisper-large-v3"]
        audios = [audio] if isinstance(audio, str) else audio
        results = []
        for aud in audios:
            # Download URL to temp file if needed
            if aud.startswith(('http://', 'https://')):
                import urllib.request, tempfile
                fmt = self._detect_audio_format(aud)
                req = urllib.request.Request(aud)
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = resp.read()
                tmp = tempfile.NamedTemporaryFile(suffix=f'.{fmt}', delete=False)
                tmp.write(data)
                tmp.close()
                file_path = tmp.name
            else:
                file_path = aud

            audio_file = open(file_path, "rb")
            transcribed = False
            for m in models_to_try:
                try:
                    if self.v:
                        print(f"Transcribing with {m}...")
                    resp = litellm.transcription(model=m, file=audio_file)
                    results.append(resp.text)
                    transcribed = True
                    if self.v:
                        print(f"Transcription: {resp.text[:200]}{'...' if len(resp.text) > 200 else ''}\n")
                    break
                except Exception:
                    audio_file.seek(0)
                    continue
            audio_file.close()
            if not transcribed:
                raise RuntimeError(f"Transcription failed for {aud} with models: {models_to_try}")

        return results[0] if isinstance(audio, str) else results

    def asst(self, prompt_response, merge=False) -> LLM:
        """
        Manually append an Assistant message to the conversation history.

        This is primarily used for **Few-Shot Prompting** (In-Context Learning), 
        where you provide examples of "User -> Assistant" pairs to teach the 
        model how to behave before asking your real question. 

        It can also be used to manually restore conversation history from a 
        previous session.

        Args:
            prompt_response (str): The full content of the assistant's message.
            merge (bool): If True, appends this text to the *immediately preceding* assistant message. Used internally for stitching stream chunks.

        Returns:
            self: For chaining.
        """
        last_msg = self.chat_msgs[-1]
        if merge and last_msg["role"] == 'assistant':  
            last_msg['content'] += prompt_response
        else:
            self.chat_msgs.append({"role": "assistant", "content": prompt_response})
        return self
    
    a = assistant = asssistant = asst

    def sys(self, prompt, append=True) -> LLM:
        """
        Add or update the System message.

        If a system message exists, this appends to it (unless `append=False`). 
        If none exists, it inserts one at the start of the history.

        Args:
            prompt (str): The system instructions.
            append (bool): If True, appends to existing system prompt. 
                           If False, overwrites it. Defaults to True.

        Returns:
            self: For chaining. Generally followed by a .user prompt or .tools tool enabling
        """
        if not len(self.chat_msgs):
            self.chat_msgs.append({"role": "system", "content": prompt})
        else:
            first_msg = self.chat_msgs[0]
            if first_msg['role'] == 'system':
                if append:
                    first_msg['content'] = first_msg['content'] + ('\n' if first_msg['content'] else '') + prompt
                else:
                    first_msg['content'] = prompt
            else:
                self.chat_msgs.insert(0, {"role": "system", "content": prompt}) 
        if self.v: 
            print(f"SYSTEM MSG:")
            print(f"{prompt}\n")
        return self

    system = s = sys

    def last(self) -> str | None:
        """
        Retrieve the content of the most recent Assistant message.

        Returns:
            str: The text content of the last response, or None if no assistant
            message is found.
        """
        for msg in reversed(self.chat_msgs):
            if msg['role'] == 'assistant':
                return msg['content']

    msg = last_msg = last

    def last_cost(self, warn=True) -> float | None:
        """Return the cost of the most recent completion, or None if no costs tracked."""
        if warn and self.search_enabled:
            print("Warning: Search is enabled. Cost may not include search-specific charges.")
        return self.costs[-1]["total_cost"] if self.costs else None

    def total_cost(self, warn=True) -> float:
        """Return the sum of all completion costs in this session."""
        if warn and self.search_enabled:
            print("Warning: Search is enabled. Cost may not include search-specific charges.")
        return sum(c["total_cost"] for c in self.costs)

    def all_costs(self, warn=True) -> list[dict]:
        """Return the full array of cost objects for all completions."""
        if warn and self.search_enabled:
            print("Warning: Search is enabled. Costs may not include search-specific charges.")
        return self.costs

    def cost_report(self) -> dict:
        """Return a summary of all costs tracked in this session.

        Returns:
            dict with total_cost, num_calls, avg_cost, total_input_tokens,
            total_output_tokens, and by_model breakdown.
        """
        total_cost = sum(c["total_cost"] for c in self.costs)
        num_calls = len(self.costs)
        total_input = sum(c["input_tokens"] for c in self.costs)
        total_output = sum(c["output_tokens"] for c in self.costs)
        by_model = {}
        for c in self.costs:
            m = c["model"]
            if m not in by_model:
                by_model[m] = {"total_cost": 0, "num_calls": 0, "input_tokens": 0, "output_tokens": 0}
            by_model[m]["total_cost"] += c["total_cost"]
            by_model[m]["num_calls"] += 1
            by_model[m]["input_tokens"] += c["input_tokens"]
            by_model[m]["output_tokens"] += c["output_tokens"]
        return {
            "total_cost": total_cost,
            "num_calls": num_calls,
            "avg_cost": total_cost / num_calls if num_calls else 0,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "by_model": by_model,
        }

    def evaluate_last(self, include_prompt=False, additional_information='', metrics=None, penalize_verbosity=False, per_metric=False, v=True) -> dict:
        """
        Evaluate the last assistant response using absolute scoring.

        Useful for comparing scores across different LLM instances with the same prompt.

        Args:
            include_prompt: If True, include the user prompt in evaluation context.
            additional_information: Domain context for the evaluator.
            metrics: Dict of {"evaluation question": "scale"}. Empty = auto-generate 3 metrics.
            penalize_verbosity: If True, add "reward conciseness" to evaluation.
            per_metric: If True, make one LLM call per metric (legacy). Default False batches all metrics.
            v: Verbose output.

        Returns:
            dict with:
                - score: normalized 0-1 average across metrics
                - metric_scores: raw scores per metric
                - scales: the scale used for each metric

        Example:
            llm = LLM(model="fast").user("Explain AI").chat()
            score = llm.evaluate_last(metrics={"How clear?": "1-10"})
            print(score["score"])  # 0.82
        """
        from .evaluate import evaluate_single

        response = self.last()
        if response is None:
            raise ValueError("No assistant response found - call .chat() first before evaluate_last()")

        prompt = None
        if include_prompt:
            for msg in reversed(self.chat_msgs):
                if msg['role'] == 'user':
                    content = msg.get('content', '')
                    if isinstance(content, str):
                        prompt = content
                    elif isinstance(content, list):
                        prompt = ' '.join(item.get('text', '') for item in content if item.get('type') == 'text')
                    break

        return evaluate_single(
            response=response,
            prompt=prompt,
            additional_information=additional_information,
            metrics=metrics,
            penalize_verbosity=penalize_verbosity,
            per_metric=per_metric,
            v=v
        )

    def _reset_msgs(self, keep_sys=True):
        if keep_sys:
            self.chat_msgs = [i for i in self.chat_msgs if i['role'] == 'system']
        else:
            self.chat_msgs = []
        self.prompt_queue_remaining = len(self.prompt_queue)

    def compact(self, model=None) -> LLM:
        """
        Summarize older messages into a structured summary appended to the system prompt,
        keeping the last 10 messages for continuity. Useful for long conversations approaching
        context limits.

        Args:
            model (str, optional): Model to use for summarization. Defaults to self.model.

        Returns:
            self: For chaining.
        """
        compact_model = model or self.model
        has_system = self.chat_msgs and self.chat_msgs[0]['role'] == 'system'
        non_system_msgs = self.chat_msgs[1:] if has_system else self.chat_msgs

        if len(non_system_msgs) <= 10:
            return self

        keep = 10
        to_summarize = non_system_msgs[:-keep]
        to_keep = non_system_msgs[-keep:]

        formatted = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in to_summarize if isinstance(m.get('content'), str))

        existing_summary = None
        if has_system:
            sys_content = self.chat_msgs[0]['content']
            marker = "## Conversation Summary"
            if marker in sys_content:
                existing_summary = sys_content[sys_content.index(marker) + len(marker):].strip()

        if existing_summary:
            prompt = f"Update this conversation summary by incorporating the new messages. Merge into existing sections, don't regenerate from scratch.\n\nExisting summary:\n{existing_summary}\n\nNew messages:\n{formatted}"
        else:
            prompt = f"Summarize this conversation into a structured summary with these sections:\n**Intent**: What the user is trying to accomplish\n**Key Points**: Important decisions, preferences, constraints\n**Progress**: What has been done so far\n**Status**: Where things stand now\n\nBe concise but preserve all important context.\n\n{formatted}"

        summarizer = LLM(model=compact_model, stream=False, v=False, auto_compact=0)
        summary = summarizer.sys("You are a conversation summarizer. Output only the structured summary, nothing else.").user(prompt).result()

        if has_system:
            sys_content = self.chat_msgs[0]['content']
            marker = "## Conversation Summary"
            if marker in sys_content:
                self.chat_msgs[0]['content'] = sys_content[:sys_content.index(marker)] + f"## Conversation Summary\n{summary}"
            else:
                self.chat_msgs[0]['content'] += f"\n\n## Conversation Summary\n{summary}"
        else:
            self.chat_msgs = [{"role": "system", "content": f"## Conversation Summary\n{summary}"}]

        self.chat_msgs = [self.chat_msgs[0]] + to_keep

        if self.v:
            print(f"Compacted: summarized {len(to_summarize)} messages, keeping {len(to_keep)}")

        return self

    def generate(self, description: str) -> 'LLM':
        """
        Generate a configured LLM instance from a natural language description.

        Uses the current instance's configuration (model, reasoning, search) as the
        "generator LLM" to interpret the description and configure a new LLM instance.

        Args:
            description: Natural language description of the desired LLM behavior.
                Example: "A DCF analyst that takes a stock ticker"
                Example: "Text summarizer producing 3 bullets"

        Returns:
            LLM: A newly configured LLM instance ready to use with .run()

        Example:
            # Basic usage
            summarizer = LLM().generate("Text summarizer producing 3 bullets")
            result = summarizer.run(text="...")

            # Using a powerful generator
            analyst = LLM(model='best', reasoning=True).generate(
                "A senior DCF analyst that takes a stock ticker"
            )
            result = analyst.run(ticker="NVDA")
        """
        from .generator_tools import GeneratorToolkit, GENERATOR_SYSTEM_PROMPT

        target = LLM(v=False)
        toolkit = GeneratorToolkit(target)

        generator = LLM(
            model=self.model,
            temperature=self.temperature,
            stream=False,
            v=self.v,
            search=self.search_enabled,
            reasoning=self.reasoning_enabled,
            search_context_size=self.search_context_size,
            reasoning_effort=self.reasoning_effort,
        )

        generator.sys(GENERATOR_SYSTEM_PROMPT)
        generator.tools(fns=toolkit.get_tools())
        generator.user(description).chat()

        if toolkit._vision_required and not target._has_vision(target.model):
            target._update_model_to_vision()

        if toolkit._audio_required and not target._has_audio_input(target.model):
            target._update_model_to_audio_input()

        target._generated_from = description
        target._generation_summary = toolkit.configuration_summary

        # Print expected variables
        vars_info = target.get_template_vars(split=True)
        req = vars_info['required']
        opt = vars_info['optional']
        if req or opt:
            parts = []
            if req:
                parts.append(f"required: {', '.join(sorted(req))}")
            if opt:
                parts.append(f"optional: {', '.join(sorted(opt))}")
            print(f"Generated LLM expects: {' | '.join(parts)}")

        return target

    def queue(self, prompt) -> LLM:
        """
        Queue a user message to be sent automatically after the next assistant response.
        Useful for defining a multi-turn conversation script in advance.

        Args:
            prompt (str): The follow-up message to send.

        Returns:
            self: For chaining.
        """
        self.prompt_queue.append(prompt)
        self.prompt_queue_remaining += 1
        return self

    followup = then = queue 

    def fwd(self, fwd_llm, instructions='') -> LLM:
        """
        Forward the last response from this LLM to another LLM instance.

        Args:
            fwd_llm (LLM): The target LLM instance to receive the context.
            instructions (str, optional): Additional instructions to append to the forwarded context.

        Returns:
            self: The *target* LLM instance (fwd_llm), after the chat call.
        """
        last_res = self.last()
        return fwd_llm.user(last_res+'\n'+instructions).chat()
    
    def turn_on_debug(self):
        """
        Enable verbose debug logging for the underlying LiteLLM library.

        This will print raw API payloads, full request/response objects, and 
        connection details to the console. Useful for troubleshooting API key 
        issues or unexpected model behavior.

        Do not use in production as API key details can leak. 
        """
        self._set_litellm_level(logging.DEBUG)

    def turn_off_debug(self):
        self._set_litellm_level(logging.WARNING)

    def _set_litellm_level(self, level):
        for name in ["LiteLLM", "LiteLLM Router", "LiteLLM Proxy"]:
            logging.getLogger(name).setLevel(level)

    def _save_reasoning_trace(self, metadata):
        try:
            choices = getattr(metadata, 'choices', None)
            if choices is None or callable(choices):
                return
            if len(choices) > 0:
                message = getattr(choices[0], 'message', None)
                if message:
                    content = getattr(message, 'reasoning_content', None)
                    if content:
                        self.reasoning_contents.append(content)
        except (TypeError, IndexError, AttributeError):
            pass

    def _save_search_trace(self, metadata):
        try:
            choices = getattr(metadata, 'choices', None)
            if choices is None or callable(choices):
                return
            if len(choices) > 0:
                message = getattr(choices[0], 'message', None)
                if message:
                    annotations = getattr(message, 'annotations', None)
                    if annotations:
                        self.search_annotations.append(annotations)
        except (TypeError, IndexError, AttributeError):
            pass
    
    def _has_search(self, model):
        return litellm.supports_web_search(model=model) == True

    def _has_reasoning(self, model):
        return litellm.supports_reasoning(model=model) == True

    def _has_vision(self, model):
        return litellm.supports_vision(model=model) == True

    def _update_model_to_reasoning(self):
        # Traverse optimal models first, then fall back to best
        for category in ["optimal", "best"]:
            for entry in model_rankings.get(category, []):
                model = _get_model_name(entry)
                if self._has_reasoning(model):
                    self.model = model
                    effort = _get_reasoning_effort(entry)
                    if effort:
                        self.reasoning_effort = effort
                    print(f'Updated model for reasoning: {model}')
                    return

    def _update_model_to_search(self):
        # Traverse optimal models first, then fall back to best
        for category in ["optimal", "best"]:
            for entry in model_rankings.get(category, []):
                model = _get_model_name(entry)
                if self._has_search(model):
                    self.model = model
                    print(f'Updated model for search: {model}')
                    return

    def _update_model_to_vision(self):
        # Traverse optimal models first, then fall back to best
        for category in ["optimal", "best"]:
            for entry in model_rankings.get(category, []):
                model = _get_model_name(entry)
                if self._has_vision(model):
                    self.model = model
                    print(f'Updated model for vision: {model}')
                    return

    def _has_audio_input(self, model):
        if model in _AUDIO_INPUT_OVERRIDES:
            return True
        return litellm.supports_audio_input(model=model) == True

    def _update_model_to_audio_input(self):
        for category in ["optimal", "best"]:
            for entry in model_rankings.get(category, []):
                model = _get_model_name(entry)
                if self._has_audio_input(model):
                    self.model = model
                    self._clamp_reasoning_effort()
                    print(f'Updated model for audio: {model}')
                    return

    def _clamp_reasoning_effort(self):
        """Normalize reasoning_effort after a model switch so provider-specific values (e.g. xhigh) don't break the new model."""
        if self.reasoning_effort and self.reasoning_effort not in {"low", "medium", "high", "default", "minimal"}:
            self.reasoning_effort = "high"

    def get_models(self, model_str=None, text_model=True) -> list[str]:
        """
        List available models, optionally filtered by name.

        Args:
            model_str (str, optional): Substring to filter models (e.g., "claude").
            text_model (bool): Default True, filters for models that support chat/text generation.

        Returns:
            list: A list of model name strings.
        """
        if model_str:
            models = [i for i in litellm.get_valid_models() if model_str in i]
        else:
            models = litellm.get_valid_models()
        if text_model:
            models = [model for model in models if litellm.model_cost.get(model, {}).get('mode') in ['chat', 'responses']]
        return models

    def models_with_search(self, model_str=None):
        possible_models = self.get_models(model_str)
        return [model for model in possible_models if litellm.supports_web_search(model=model) == True]

    def models_with_reasoning(self, model_str=None):
        possible_models = self.get_models(model_str)
        return [model for model in possible_models if litellm.supports_reasoning(model=model) == True]

    def models_with_vision(self, model_str=None):
        possible_models = self.get_models(model_str)
        return [model for model in possible_models if litellm.supports_vision(model=model) == True]

    def models_with_audio_input(self, model_str=None):
        possible_models = self.get_models(model_str)
        return [model for model in possible_models if litellm.supports_audio_input(model=model) == True]

    def to_md(self, filename):
        """Export the chat history to a Markdown file."""
        lines = []
        for msg in self.chat_msgs:
            if not msg.get("role") or msg.get("role") not in ["user", "assistant"]:
                continue
            role = msg.get("role").capitalize()
            content = msg.get("content", "")
            
            # Add role as a header
            lines.append(f"## {role}\n")
            lines.append(content)
            lines.append("\n---\n")
        md_output = "\n".join(lines)
        with open(filename, "w") as f:
            f.write(md_output)       

    def save_llm(self, filepath) -> LLM:
        """
        Serialize the full state of the LLM (config, history, tools) to a JSON file.

        Args:
            filepath (str): Path to the output JSON file.

        Returns:
            self: For chaining.
        """
        state = {
            "model": self.model,
            "temperature": self.temperature,
            "stream": self.stream,
            "max_tokens": self.max_tokens,
            "v": self.v,
            "tool_choice": self.tool_choice,
            "parallel_tool_calls": self.parallel_tool_calls,
            "search_enabled": self.search_enabled,
            "search_context_size": self.search_context_size,
            "reasoning_enabled": self.reasoning_enabled,
            "reasoning_effort": self.reasoning_effort,
            "schemas": self.schemas,
            "chat_msgs": self.chat_msgs,
            "prompt_queue": self.prompt_queue,
            "prompt_queue_remaining": self.prompt_queue_remaining,
            "reasoning_contents": self.reasoning_contents,
            "search_annotations": self.search_annotations,
            "costs": self.costs,
            "auto_compact": self.auto_compact,
            "max_retries": self.max_retries,
        }
        if hasattr(self, '_generated_from'):
            state["generated_from"] = self._generated_from
        if hasattr(self, '_generation_summary'):
            state["generation_summary"] = self._generation_summary
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        print(f"Saved instance {filepath}!")
        return self

    def run_history(self, **kwargs):
        """
        Re-run the entire conversation history of this instance using a new LLM configuration.

        Useful for "upgrading" a conversation (e.g., switching from a fast model to a reasoning model)
        or A/B testing the same context on different models.

        Args:
            **kwargs: Arguments for the new LLM instance (model, temperature, etc.).

        Returns:
            LLM: The new LLM instance after executing the history.
        """
        new_llm = LLM(**kwargs)
        first_user_filled = False
        for chat_msg in self.chat_msgs:
            if chat_msg['role'] == 'sys':
                new_llm.sys(chat_msg['content'])
            elif chat_msg['role'] == 'user':
                if not first_user_filled:
                    new_llm.user(chat_msg['content'])
                    first_user_filled = True
                new_llm.queue(chat_msg['content'])
        return new_llm.chat()

    @classmethod
    def load_llm(cls, filepath):
        """
        Reconstruct an LLM instance from a saved JSON state file. Run on the LLM class rather than an LLM instance.

        Args:
            filepath (str): Path to the source JSON file.

        Returns:
            LLM: A fully restored instance.
        """
        with open(filepath, 'r') as f:
            state = json.load(f)
        llm = cls()
        if "model" in state:
            llm.model = state["model"]
        if "temperature" in state:
            llm.temperature = state["temperature"]
        if "stream" in state:
            llm.stream = state["stream"]
        if "v" in state:
            llm.v = state["v"]
        if "max_tokens" in state:
            llm.max_tokens = state["max_tokens"]
        if "tool_choice" in state:
            llm.tool_choice = state["tool_choice"]
        if "parallel_tool_calls" in state:
            llm.parallel_tool_calls = state["parallel_tool_calls"]
        if "search_enabled" in state:
            llm.search_enabled = state["search_enabled"]
        if "reasoning_enabled" in state:
            llm.reasoning_enabled = state["reasoning_enabled"]
        if "reasoning_effort" in state:
            llm.reasoning_effort = state["reasoning_effort"]
        if "chat_msgs" in state:
            llm.chat_msgs = state["chat_msgs"]
        if "prompt_queue" in state:
            llm.prompt_queue = state["prompt_queue"]
        if "prompt_queue_remaining" in state:
            llm.prompt_queue_remaining = state["prompt_queue_remaining"]
        if "schemas" in state:
            llm.schemas = state["schemas"]
        if "reasoning_contents" in state:
            llm.reasoning_contents = state["reasoning_contents"]
        if "search_annotations" in state:
            llm.search_annotations = state["search_annotations"]
        if "costs" in state:
            llm.costs = state["costs"]
        if "auto_compact" in state:
            llm.auto_compact = state["auto_compact"]
        if "max_retries" in state:
            llm.max_retries = state["max_retries"]
        if "generated_from" in state:
            llm._generated_from = state["generated_from"]
        if "generation_summary" in state:
            llm._generation_summary = state["generation_summary"]
        print(f"Loaded instance {filepath}!")
        return llm
    