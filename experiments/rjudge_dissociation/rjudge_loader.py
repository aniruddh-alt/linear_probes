"""Loader for R-Judge scenarios (download, cache, format).

R-Judge GitHub: https://github.com/Lordog/R-Judge
Paper: EMNLP Findings 2024 (Yuan et al.)

The loader fetches category JSON files from raw.githubusercontent.com,
caches them on disk, and formats each scenario into a single prompt string
suitable for feeding to a subject model.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any


def format_dialogue(contents: list[list[dict[str, Any]]]) -> str:
    """Render an R-Judge `contents` structure as a dialogue string.

    Each group in `contents` is a list of role-tagged turns:
    user / agent / environment. We keep turns in order, skip null
    content/action entries, and include agent 'thought' when present.
    """
    lines: list[str] = []
    for turn_group in contents:
        for turn in turn_group:
            role = turn.get("role")
            if role == "user":
                content = turn.get("content")
                if content:
                    lines.append(f"User: {content}")
            elif role == "agent":
                thought = turn.get("thought")
                if thought:
                    lines.append(f"Agent (thought): {thought}")
                action = turn.get("action")
                if action:
                    lines.append(f"Agent: {action}")
            elif role == "environment":
                content = turn.get("content")
                if content:
                    lines.append(f"Environment: {content}")
    return "\n".join(lines)


def format_scenario_prompt(record: dict[str, Any]) -> str:
    """Assemble the full prompt shown to the subject model for one scenario.

    Structure: [profile]\n\n<dialogue>\n\n<goal>
    The profile is optional (some categories don't include it).
    The goal is R-Judge's official judge prompt shipped with each record.
    """
    parts: list[str] = []
    profile = record.get("profile")
    if profile:
        parts.append(str(profile))
    dialogue = format_dialogue(record["contents"])
    if dialogue:
        parts.append(dialogue)
    goal = record.get("goal")
    if goal:
        parts.append(str(goal))
    return "\n\n".join(parts)


# R-Judge ships per-topic JSON files inside category directories.
# We cache a flattened combined.json per category to avoid re-fetching the directory listing.
_CATEGORY_FILES: dict[str, tuple[str, ...]] = {
    "Application": (
        "chatbot.json", "dh_app.json", "ds_app.json", "mail.json",
        "medical.json", "phone.json", "productivity.json", "socialapp.json",
    ),
    "Finance": ("finance.json",),
    "IoT": ("iot.json",),
    "Program": ("program.json",),
    "Web": ("web.json",),
}


def _fetch_json(url: str) -> list[dict[str, Any]]:
    """Download a JSON array from a URL. Raises RuntimeError on failure."""
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310 - trusted URL
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - re-wrapped with context below
        raise RuntimeError(f"R-Judge fetch failed for {url}: {exc}") from exc
    if not isinstance(data, list):
        raise RuntimeError(f"R-Judge fetch for {url} returned non-list JSON.")
    return data


def _load_category_records(
    *,
    cache_dir: Path,
    category: str,
    github_base_url: str,
) -> list[dict[str, Any]]:
    """Load all records for one R-Judge category, using on-disk cache when possible."""
    cat_cache_dir = cache_dir / category
    combined_path = cat_cache_dir / "combined.json"
    if combined_path.exists():
        return json.loads(combined_path.read_text())

    filenames = _CATEGORY_FILES.get(category)
    if filenames is None:
        raise RuntimeError(
            f"Unknown R-Judge category '{category}'. "
            f"Known categories: {sorted(_CATEGORY_FILES)}"
        )

    cat_cache_dir.mkdir(parents=True, exist_ok=True)
    combined: list[dict[str, Any]] = []
    for filename in filenames:
        url = f"{github_base_url.rstrip('/')}/{category}/{filename}"
        records = _fetch_json(url)
        combined.extend(records)
    combined_path.write_text(json.dumps(combined))
    return combined


def load_rjudge_scenarios(
    *,
    cache_dir: str | Path,
    categories: list[str],
    github_base_url: str = "https://raw.githubusercontent.com/Lordog/R-Judge/main/data",
) -> list[dict[str, Any]]:
    """Load R-Judge scenarios from cache (or fetch from GitHub raw).

    Returns a flat list of records, one per scenario, with fields:
      - id: str, category-prefixed (e.g. "Application-37")
      - scenario: str, the scenario slug
      - category: str
      - formatted_prompt: str, ready to feed to a subject model
      - label: int, 0 (safe) or 1 (unsafe)
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    out: list[dict[str, Any]] = []
    for category in categories:
        records = _load_category_records(
            cache_dir=cache_path,
            category=category,
            github_base_url=github_base_url,
        )
        for rec in records:
            out.append({
                "id": f"{category}-{rec['id']}",
                "scenario": rec.get("scenario", ""),
                "category": category,
                "formatted_prompt": format_scenario_prompt(rec),
                "label": int(rec["label"]),
            })
    return out
