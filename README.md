# spaceshift

An interactive CLI prompt exploration toolkit powered by LLMs. Manipulate prompts through transforms, navigate the full space of perspectives, and evaluate across prompts and models to find what works best.

**[Full documentation at spcshft.com](https://spcshft.com)**

```bash
# Launch interactive mode
spaceshift

# Select from guided menus:
# → Prompt Manipulate — explore prompt transformations
# → Compare Models — rank model responses
# → Grid Search — search across models × transforms
# → Prompt Tree — visualize exploration paths
```

All functionality is also available as a Python library for programmatic use.

---

### Install

```bash
pip install spaceshift
```

On first run, spaceshift will guide you through setting up your API keys:

```bash
$ spaceshift

No API keys found. Let's set up your providers.

  OpenAI (press Enter to skip): sk-proj-...
  ✓ OpenAI key saved

  Anthropic (press Enter to skip): sk-ant-...
  ✓ Anthropic key saved

  Google Gemini (press Enter to skip): [Enter]
  Together.AI (press Enter to skip): [Enter]
  xAI (press Enter to skip): [Enter]

✓ Configuration saved to ~/.spaceshift/config.json
```

Keys are stored securely in `~/.spaceshift/config.json` and available globally. You can update or add keys anytime via the "Manage API Keys" option in the main menu.

**For Python library usage:** You can still use a `.env` file in your project directory - it will be loaded automatically when you import spaceshift.

---

### Interactive CLI

Launch the interactive mode:

```bash
spaceshift
```

The CLI guides you through:
- **Prompt Manipulate** — Transform prompts and explore variations
- **Compare Models** — Run the same prompt across models and rank responses
- **Grid Search** — Search across models × transforms simultaneously
- **Prompt Tree** — Visualize the exploration space
- **Manage API Keys** — Add, update, or view your configured API providers

Pick a model from categorized rankings (optimal, best, fast, cheap, open), enter your prompt, and explore. Results are saved as structured markdown with YAML frontmatter, and the built-in viewer opens automatically.

---

### Viewing Results

Browse any output directory in the browser with the built-in viewer:

```bash
spaceshift view output_directory
```

Two-panel layout: sidebar with smart-sorted file list, content area with rendered markdown and frontmatter metadata cards. No dependencies — runs on Python's stdlib server with client-side markdown rendering.

---

## Advanced Usage

While spaceshift is designed as a CLI tool, advanced users can import and use the underlying modules programmatically. **This is unsupported** — the CLI is the primary interface, and internal APIs may change without notice.

For those who want to explore anyway, the main modules are in `spaceshift/` including `LLM`, `prompt_tree`, `compare_models`, `grid_search`, etc. See the source code for details.
