"""
Build static rankings from Artificial Analysis API data.

Replaces the notebook-based generation with a deterministic, reviewable pipeline:
1. Fetch AA API data
2. Rank by category metrics (intelligence, speed, price)
3. Map AA slugs to litellm using static map + rule-based auto-discovery
4. Validate each model exists in litellm
5. Apply reasoning_effort from pinned config
6. Compute optimal (50/50 blend), open (pattern match), codex (separate)
7. Auto-persist newly discovered mappings
8. Diff against current active rankings
9. Save with confirmation

Usage:
    ARTIFICIAL_ANALYSIS_KEY=xxx python scripts/build_rankings.py
"""

import json
import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from rapidfuzz import fuzz
import os

load_dotenv()

SCRIPT_DIR = Path(__file__).parent
RANKINGS_DIR = Path(__file__).parent.parent / "src" / "cruise_llm" / "rankings"
AA_MAP_PATH = SCRIPT_DIR / "aa_to_litellm_map.json"
REASONING_CONFIG_PATH = SCRIPT_DIR / "reasoning_config.json"


def load_static_map():
    with open(AA_MAP_PATH) as f:
        return json.load(f)


def load_reasoning_config():
    with open(REASONING_CONFIG_PATH) as f:
        return json.load(f)


def get_current_rankings_path():
    """Find the most recent rankings file."""
    files = sorted(RANKINGS_DIR.glob("static_rankings_*.json"))
    if not files:
        return None
    return files[-1]


def load_current_rankings():
    path = get_current_rankings_path()
    if path is None:
        return {}
    with open(path) as f:
        return json.load(f)


# --- AA API ---

def fetch_aa_data():
    api_key = os.getenv("ARTIFICIAL_ANALYSIS_KEY")
    if not api_key:
        print("ERROR: Set ARTIFICIAL_ANALYSIS_KEY environment variable")
        sys.exit(1)

    url = "https://artificialanalysis.ai/api/v2/data/llms/models"
    resp = requests.get(url, headers={"x-api-key": api_key})
    resp.raise_for_status()
    data = resp.json()["data"]
    df = pd.json_normalize(data)
    print(f"Fetched {len(df)} models from AA API")
    return df


# --- Reasoning effort extraction ---

def extract_reasoning_effort(name, slug):
    """Extract reasoning effort from model name like 'GPT-5.2 (xhigh)'."""
    name_lower = name.lower()
    slug_lower = slug.lower()

    match = re.search(r'\((xhigh|high|medium|low|minimal)\)', name_lower)
    if match:
        return match.group(1)

    if any(x in slug_lower for x in ['-thinking', '-reasoning']):
        return 'default'

    return None


# --- Category ranking ---

def rank_best(df):
    """Sort by intelligence index descending, skip nulls."""
    col = 'evaluations.artificial_analysis_intelligence_index'
    df_valid = df.dropna(subset=[col])
    return df_valid.sort_values(col, ascending=False)


def rank_fast(df):
    """Score = tokens/s / ttft, descending."""
    speed_cols = ['median_output_tokens_per_second', 'median_time_to_first_token_seconds']
    df_speed = df.dropna(subset=speed_cols).copy()
    df_speed = df_speed[~(df_speed[speed_cols] == 0).any(axis=1)]
    df_speed['fast_score'] = (
        df_speed['median_output_tokens_per_second'] /
        df_speed['median_time_to_first_token_seconds']
    )
    return df_speed.sort_values('fast_score', ascending=False)


def rank_cheap(df, best_slugs_top100):
    """From top 100 best, sort by blended price ascending."""
    price_col = 'pricing.price_1m_blended_3_to_1'
    df_price = df[df['slug'].isin(best_slugs_top100)].copy()
    df_price = df_price[df_price[price_col] > 0].dropna(subset=[price_col])
    return df_price.sort_values(price_col)


# --- Auto-discovery ---

def generate_candidates(aa_slug, available_set):
    """Generate ordered litellm model name candidates for an AA slug.

    Tries predictable transformations and validates each against available models.
    Returns list of candidates that exist in available_set.
    """
    candidates = []
    slug = aa_slug.lower()

    # 1. Identity
    if slug in available_set:
        candidates.append(slug)

    # 2. Provider prefix by family
    if slug.startswith('grok'):
        prefixed = f"xai/{slug}"
        if prefixed in available_set:
            candidates.append(prefixed)
    if slug.startswith('gemini'):
        prefixed = f"gemini/{slug}"
        if prefixed in available_set:
            candidates.append(prefixed)
        # Also try dash-to-dot version: gemini-2-5-pro -> gemini/gemini-2.5-pro
        dotted = re.sub(r'(\d+)-(\d+)', r'\1.\2', slug)
        if dotted != slug:
            prefixed_dotted = f"gemini/{dotted}"
            if prefixed_dotted in available_set:
                candidates.append(prefixed_dotted)
    if slug.startswith('gpt-oss'):
        prefixed = f"together_ai/openai/{slug}"
        if prefixed in available_set:
            candidates.append(prefixed)

    # 3. Dash-to-dot version normalization: gpt-5-2 -> gpt-5.2
    dotted = re.sub(r'(\d+)-(\d+)', r'\1.\2', slug)
    if dotted != slug and dotted in available_set:
        candidates.append(dotted)

    # 4. Substring search: normalize slug, check if it appears inside any available model
    # catches: llama-4-maverick -> together_ai/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8
    normalized = slug.replace('-instruct', '').replace('-chat', '')
    for model in sorted(available_set):
        if normalized in model and model not in candidates:
            candidates.append(model)

    # 5. Fuzzy match (>= 90 threshold) as last resort
    if not candidates:
        best_score, best_match = 0, None
        for model in available_set:
            # Compare against the last segment (after last /) for provider-prefixed models
            model_tail = model.rsplit('/', 1)[-1]
            score = max(fuzz.ratio(slug, model), fuzz.ratio(slug, model_tail))
            if score > best_score:
                best_score, best_match = score, model
        if best_score >= 90 and best_match:
            candidates.append(best_match)

    return candidates


def resolve_litellm_name(aa_slug, static_map, available_set):
    """Resolve an AA slug to a litellm model name.

    Returns (litellm_name, source) where source is 'static', 'auto', or None.
    """
    if aa_slug in static_map:
        return static_map[aa_slug], 'static'

    candidates = generate_candidates(aa_slug, available_set)
    if candidates:
        return candidates[0], 'auto'

    return None, None


# --- Slug mapping ---

def map_to_litellm(aa_entries, aa_to_litellm, available_set, allow_codex=False):
    """Convert AA ranking entries to litellm format using static map + auto-discovery.

    Returns (mapped_entries, unmapped_slugs, auto_discovered dict).
    """
    result = []
    seen = set()
    unmapped = []
    auto_discovered = {}

    for item in aa_entries:
        slug = item['slug']
        litellm_model, source = resolve_litellm_name(slug, aa_to_litellm, available_set)

        if litellm_model is None:
            unmapped.append(slug)
            continue

        if source == 'auto':
            auto_discovered[slug] = litellm_model

        if not allow_codex and 'codex' in litellm_model.lower():
            continue

        if litellm_model in seen:
            continue
        seen.add(litellm_model)

        entry = {'model': litellm_model}
        if item.get('reasoning_effort'):
            entry['reasoning_effort'] = item['reasoning_effort']
        result.append(entry)

    return result, unmapped, auto_discovered


# --- Validation ---

def validate_models(rankings, available_models):
    """Check each model exists in litellm. Returns (valid_rankings, invalid_models)."""
    available_set = set(available_models)
    valid = {}
    invalid = []

    for category, entries in rankings.items():
        valid_entries = []
        for entry in entries:
            if entry['model'] in available_set:
                valid_entries.append(entry)
            else:
                invalid.append((category, entry['model']))
        valid[category] = valid_entries

    return valid, invalid


# --- Reasoning effort application ---

def apply_reasoning_config(rankings, reasoning_config):
    """Apply pinned reasoning_effort from config, falling back to AA-extracted values."""
    for category, entries in rankings.items():
        for entry in entries:
            model = entry['model']
            if model in reasoning_config:
                entry['reasoning_effort'] = reasoning_config[model]
            # else: keep whatever AA extraction gave us (or nothing)


# --- Optimal computation ---

SPEED_PENALTIES = {
    'together_ai/': 0.25,
}

def compute_optimal(best_list, fast_list, fast_weight=0.5, best_weight=0.5):
    """50/50 weighted rank blend of best + fast positions, with provider speed penalties."""
    best_models = [e['model'] for e in best_list]
    fast_models = [e['model'] for e in fast_list]
    all_models = set(best_models) | set(fast_models)

    best_rank = {m: i for i, m in enumerate(best_models)}
    fast_rank = {m: i for i, m in enumerate(fast_models)}
    n_fast = len(fast_models)

    scores = []
    for m in all_models:
        if 'codex' in m.lower():
            continue
        b = best_rank.get(m, len(best_models))
        f = fast_rank.get(m, n_fast)
        penalty = sum(v for prefix, v in SPEED_PENALTIES.items() if m.startswith(prefix))
        if penalty:
            f = f + penalty * n_fast
        scores.append((m, fast_weight * f + best_weight * b))

    scores.sort(key=lambda x: x[1])
    return [m for m, _ in scores]


# --- Open source detection ---

OPEN_SOURCE_PATTERNS = [
    'together_ai/',
    'llama', 'deepseek', 'qwen', 'mistral', 'kimi', 'gemma',
    'phi-', 'falcon', 'yi-', 'vicuna', 'wizardlm', 'openchat',
    'solar', 'internlm', 'codellama', 'starcoder', 'codestral',
    'devstral', 'mixtral', 'mathstral', 'ministral', 'gpt-oss',
    'nemotron', 'granite', 'olmo', 'command-r',
    'glm', 'cogito', 'apriel', 'lfm',
]

CLOSED_PATTERNS = [
    'gpt-4', 'gpt-5', 'gpt-3.5',
    'claude', 'gemini', 'o1', 'o3', 'o4', 'grok',
]


def is_open_source(model_name):
    name_lower = model_name.lower()
    for pattern in OPEN_SOURCE_PATTERNS:
        if pattern in name_lower:
            is_closed = any(cp in name_lower for cp in CLOSED_PATTERNS if cp != 'gpt-oss')
            if not is_closed or 'gpt-oss' in name_lower:
                return True
    return False


def build_open_ranking(all_rankings):
    """Build open source ranking from all category models, sorted by best rank."""
    all_models = set()
    model_to_effort = {}
    for cat in ['best', 'fast', 'cheap']:
        for entry in all_rankings[cat]:
            all_models.add(entry['model'])
            if entry.get('reasoning_effort'):
                model_to_effort[entry['model']] = entry['reasoning_effort']

    open_models = [m for m in all_models if is_open_source(m) and 'codex' not in m.lower()]

    best_order = [e['model'] for e in all_rankings['best']]

    def get_best_rank(m):
        try:
            return best_order.index(m)
        except ValueError:
            return 999

    open_models_sorted = sorted(open_models, key=get_best_rank)

    result = []
    for m in open_models_sorted:
        entry = {'model': m}
        if model_to_effort.get(m):
            entry['reasoning_effort'] = model_to_effort[m]
        result.append(entry)
    return result


# --- Diff ---

def diff_rankings(old, new):
    """Print human-readable diff between old and new rankings."""
    all_cats = sorted(set(list(old.keys()) + list(new.keys())))

    changes_found = False
    for cat in all_cats:
        old_models = [e['model'] for e in old.get(cat, [])]
        new_models = [e['model'] for e in new.get(cat, [])]

        old_set = set(old_models)
        new_set = set(new_models)
        added = new_set - old_set
        removed = old_set - new_set

        # Check for effort changes on models in both
        old_efforts = {e['model']: e.get('reasoning_effort') for e in old.get(cat, [])}
        new_efforts = {e['model']: e.get('reasoning_effort') for e in new.get(cat, [])}
        effort_changes = []
        for m in old_set & new_set:
            if old_efforts.get(m) != new_efforts.get(m):
                effort_changes.append((m, old_efforts.get(m), new_efforts.get(m)))

        # Check for rank changes (position shifts)
        rank_changes = []
        for m in old_set & new_set:
            old_pos = old_models.index(m)
            new_pos = new_models.index(m)
            if old_pos != new_pos:
                rank_changes.append((m, old_pos + 1, new_pos + 1))

        if not added and not removed and not effort_changes and not rank_changes:
            print(f"\n  {cat}: no changes ({len(new_models)} models)")
            continue

        changes_found = True
        print(f"\n  {cat}: {len(old_models)} -> {len(new_models)} models")

        if added:
            for m in sorted(added):
                pos = new_models.index(m) + 1
                print(f"    + {m} (rank {pos})")
        if removed:
            for m in sorted(removed):
                print(f"    - {m}")
        if effort_changes:
            for m, old_e, new_e in effort_changes:
                print(f"    ~ {m}: reasoning_effort {old_e} -> {new_e}")
        if rank_changes:
            # Only show significant rank changes (moved 3+ positions)
            big_moves = [(m, o, n) for m, o, n in rank_changes if abs(o - n) >= 3]
            if big_moves:
                print(f"    Rank shifts (3+ positions):")
                for m, old_pos, new_pos in sorted(big_moves, key=lambda x: x[2]):
                    direction = "up" if new_pos < old_pos else "down"
                    print(f"      {m}: #{old_pos} -> #{new_pos} ({direction} {abs(old_pos - new_pos)})")

    if not changes_found:
        print("\n  No changes detected - rankings are identical.")

    return changes_found


# --- Main ---

def main():
    print("=" * 60)
    print("Build Rankings Pipeline")
    print("=" * 60)

    # Load config files
    aa_to_litellm = load_static_map()
    reasoning_config = load_reasoning_config()
    print(f"Loaded static map: {len(aa_to_litellm)} AA slug mappings")
    print(f"Loaded reasoning config: {len(reasoning_config)} model entries")

    # Fetch AA data
    print("\nFetching AA API data...")
    df = fetch_aa_data()

    # Extract reasoning effort from AA model names
    df['reasoning_effort'] = df.apply(
        lambda row: extract_reasoning_effort(row['name'], row['slug']), axis=1
    )

    # Build slug -> effort lookup
    slug_to_effort = {}
    for _, row in df.iterrows():
        if row.get('reasoning_effort'):
            slug_to_effort[row['slug']] = row['reasoning_effort']

    # --- Category rankings ---
    print("\nRanking models by category...")

    # Best
    df_best = rank_best(df)
    best_entries = []
    for _, row in df_best.iterrows():
        entry = {'slug': row['slug']}
        if row.get('reasoning_effort'):
            entry['reasoning_effort'] = row['reasoning_effort']
        if 'codex' not in row['slug'].lower():
            best_entries.append(entry)

    # Fast
    df_fast = rank_fast(df)
    fast_entries = []
    for _, row in df_fast.iterrows():
        if 'codex' in row['slug'].lower():
            continue
        entry = {'slug': row['slug']}
        if row.get('reasoning_effort'):
            entry['reasoning_effort'] = row['reasoning_effort']
        fast_entries.append(entry)

    # Cheap
    top_100_best_slugs = list(df_best['slug'].head(100))
    df_cheap = rank_cheap(df, top_100_best_slugs)
    cheap_entries = []
    for _, row in df_cheap.iterrows():
        if 'codex' in row['slug'].lower():
            continue
        entry = {'slug': row['slug']}
        if row.get('reasoning_effort'):
            entry['reasoning_effort'] = row['reasoning_effort']
        cheap_entries.append(entry)

    # Codex
    codex_entries = []
    for _, row in df_best.iterrows():
        if 'codex' in row['slug'].lower():
            entry = {'slug': row['slug']}
            if row.get('reasoning_effort'):
                entry['reasoning_effort'] = row['reasoning_effort']
            codex_entries.append(entry)

    print(f"  best: {len(best_entries)} AA models")
    print(f"  fast: {len(fast_entries)} AA models")
    print(f"  cheap: {len(cheap_entries)} AA models")
    print(f"  codex: {len(codex_entries)} AA models")

    # --- Load litellm available models (needed for auto-discovery + validation) ---
    print("\nLoading litellm available models...")
    from cruise_llm import LLM
    available_models = LLM(v=False).get_models()
    available_set = set(available_models)
    print(f"  litellm has {len(available_models)} available models")

    # Augment with Together.AI models (litellm's static list is outdated)
    together_key = os.getenv("TOGETHERAI_API_KEY")
    if together_key:
        try:
            resp = requests.get("https://api.together.xyz/v1/models",
                              headers={"Authorization": f"Bearer {together_key}"})
            resp.raise_for_status()
            together_models = resp.json()
            excluded_types = {"image", "embedding", "moderation", "code", "audio"}
            if isinstance(together_models, list):
                chat_ids = [m["id"] for m in together_models if m.get("type") not in excluded_types]
            else:
                chat_ids = [m["id"] for m in together_models.get("data", []) if m.get("type") not in excluded_types]
            prefixed = [f"together_ai/{mid}" for mid in chat_ids]
            available_set.update(prefixed)
            print(f"  + {len(prefixed)} Together.AI chat models added")
        except Exception as e:
            print(f"  WARNING: Failed to fetch Together.AI models: {e}")

    # --- Map to litellm ---
    print("\nMapping AA slugs to litellm model names...")

    all_unmapped = set()
    all_auto_discovered = {}
    rankings = {}

    rankings['best'], unmapped, auto = map_to_litellm(best_entries, aa_to_litellm, available_set)
    all_unmapped.update(unmapped)
    all_auto_discovered.update(auto)

    rankings['fast'], unmapped, auto = map_to_litellm(fast_entries, aa_to_litellm, available_set)
    all_unmapped.update(unmapped)
    all_auto_discovered.update(auto)

    rankings['cheap'], unmapped, auto = map_to_litellm(cheap_entries, aa_to_litellm, available_set)
    all_unmapped.update(unmapped)
    all_auto_discovered.update(auto)

    rankings['codex'], unmapped, auto = map_to_litellm(codex_entries, aa_to_litellm, available_set, allow_codex=True)
    all_unmapped.update(unmapped)
    all_auto_discovered.update(auto)

    print(f"  best: {len(rankings['best'])} mapped models")
    print(f"  fast: {len(rankings['fast'])} mapped models")
    print(f"  cheap: {len(rankings['cheap'])} mapped models")
    print(f"  codex: {len(rankings['codex'])} mapped models")

    if all_auto_discovered:
        print(f"\n  Auto-discovered {len(all_auto_discovered)} new mappings:")
        for slug, model in sorted(all_auto_discovered.items()):
            print(f"    {slug} -> {model}")

    if all_unmapped:
        top_slugs = set()
        for entries_list in [best_entries[:50], fast_entries[:50], cheap_entries[:50], codex_entries]:
            for e in entries_list:
                top_slugs.add(e['slug'])
        notable_unmapped = sorted(all_unmapped & top_slugs)
        if notable_unmapped:
            print(f"\n  UNMAPPED (top 50 in any category) - needs manual mapping:")
            for slug in notable_unmapped:
                print(f"    {slug}")
        trivial_unmapped = len(all_unmapped) - len(notable_unmapped)
        if trivial_unmapped:
            print(f"  ({trivial_unmapped} more unmapped slugs outside top 50)")

    # --- Validate against litellm ---
    print("\nValidating mapped models against litellm...")
    rankings, invalid = validate_models(rankings, available_set)
    if invalid:
        print(f"\n  INVALID models (not in litellm, removed from rankings):")
        for cat, model in invalid:
            print(f"    [{cat}] {model}")
    else:
        print("  All models validated OK")

    # --- Apply reasoning config ---
    print("\nApplying reasoning_effort config...")
    apply_reasoning_config(rankings, reasoning_config)

    # --- Compute optimal (exclude non-default reasoning from blend) ---
    print("\nComputing optimal ranking (50/50 blend)...")
    non_default_efforts = {'high', 'xhigh', 'medium', 'low', 'minimal'}
    best_for_optimal = [e for e in rankings['best'] if e.get('reasoning_effort', 'default') not in non_default_efforts]
    fast_for_optimal = [e for e in rankings['fast'] if e.get('reasoning_effort', 'default') not in non_default_efforts]
    excluded = len(rankings['best']) - len(best_for_optimal)
    if excluded:
        print(f"  Excluded {excluded} non-default reasoning models from optimal blend")
    optimal_models = compute_optimal(best_for_optimal, fast_for_optimal)

    # Build effort lookup from all categories
    model_to_effort = {}
    for cat in ['best', 'fast', 'cheap']:
        for entry in rankings[cat]:
            if entry.get('reasoning_effort'):
                model_to_effort[entry['model']] = entry['reasoning_effort']
    # Override with reasoning config
    model_to_effort.update(reasoning_config)

    rankings['optimal'] = []
    for m in optimal_models:
        entry = {'model': m}
        if model_to_effort.get(m):
            entry['reasoning_effort'] = model_to_effort[m]
        rankings['optimal'].append(entry)
    print(f"  optimal: {len(rankings['optimal'])} models")

    # --- Build open ---
    print("Building open source ranking...")
    rankings['open'] = build_open_ranking(rankings)
    print(f"  open: {len(rankings['open'])} models")

    # --- Summary ---
    print("\n" + "=" * 60)
    print("Generated Rankings Summary")
    print("=" * 60)
    for cat in ['best', 'fast', 'cheap', 'optimal', 'open', 'codex']:
        entries = rankings.get(cat, [])
        print(f"\n  {cat} ({len(entries)} models):")
        for i, e in enumerate(entries[:5]):
            effort = f" ({e['reasoning_effort']})" if e.get('reasoning_effort') else ""
            print(f"    {i+1}. {e['model']}{effort}")
        if len(entries) > 5:
            print(f"    ... and {len(entries) - 5} more")

    # --- Auto-persist new mappings ---
    if all_auto_discovered:
        aa_to_litellm.update(all_auto_discovered)
        with open(AA_MAP_PATH, 'w') as f:
            json.dump(aa_to_litellm, f, indent=2)
            f.write('\n')
        print(f"\nAuto-saved {len(all_auto_discovered)} new mappings to {AA_MAP_PATH.name}")

    # --- Diff ---
    print("\n" + "=" * 60)
    print("Diff against current rankings")
    print("=" * 60)
    current = load_current_rankings()
    current_path = get_current_rankings_path()
    if current:
        print(f"Comparing with: {current_path.name}")
        changes = diff_rankings(current, rankings)
    else:
        print("No current rankings found - this will be the first file.")
        changes = True

    # --- Save ---
    print("\n" + "=" * 60)
    today = date.today().isoformat()
    output_path = RANKINGS_DIR / f"static_rankings_{today}.json"

    if output_path.exists():
        print(f"WARNING: {output_path.name} already exists!")

    confirm = input(f"\nSave rankings to {output_path.name}? [y/N] ").strip().lower()
    if confirm != 'y':
        print("Aborted. Rankings not saved.")
        # Still save to a staging file for review
        staging_path = SCRIPT_DIR / f"staged_rankings_{today}.json"
        with open(staging_path, 'w') as f:
            json.dump(rankings, f, indent=2)
        print(f"Staged rankings saved to {staging_path} for review.")
        return

    # Reorder categories for consistent output
    ordered = {}
    for cat in ['best', 'fast', 'cheap', 'codex', 'open', 'optimal']:
        if cat in rankings:
            ordered[cat] = rankings[cat]

    with open(output_path, 'w') as f:
        json.dump(ordered, f, indent=2)
    print(f"\nSaved to {output_path}")
    print(f"To activate: update _RANKINGS_PATH in src/cruise_llm/LLM.py line 13")


if __name__ == '__main__':
    main()
