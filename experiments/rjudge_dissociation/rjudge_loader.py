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
    """Assemble the full prompt shown to the subject model for one scenario."""
    raise NotImplementedError  # implemented in a later step


def load_rjudge_scenarios(
    *,
    cache_dir: str | Path,
    categories: list[str],
    github_base_url: str = "https://raw.githubusercontent.com/Lordog/R-Judge/main/data",
) -> list[dict[str, Any]]:
    """Load R-Judge scenarios from GitHub (or cache)."""
    raise NotImplementedError  # implemented in a later step
