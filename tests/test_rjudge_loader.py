"""Tests for the R-Judge loader."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

from experiments.rjudge_dissociation.rjudge_loader import (
    format_dialogue,
    format_scenario_prompt,
    load_rjudge_scenarios,
)


class TestFormatDialogue:
    def test_single_user_agent_turn(self) -> None:
        contents = [
            [
                {"role": "user", "content": "Hello"},
                {"role": "agent", "action": "Hi there!", "thought": None},
            ]
        ]
        out = format_dialogue(contents)
        assert "User: Hello" in out
        assert "Agent: Hi there!" in out

    def test_skips_null_environment_entries(self) -> None:
        contents = [
            [
                {"role": "user", "content": "Hi"},
                {"role": "agent", "action": "Hello", "thought": None},
                {"role": "environment", "content": None},
            ]
        ]
        out = format_dialogue(contents)
        assert "Environment" not in out  # null environment should be dropped

    def test_includes_environment_when_content_present(self) -> None:
        contents = [
            [
                {"role": "user", "content": "Run tool"},
                {"role": "agent", "action": "Calling tool", "thought": None},
                {"role": "environment", "content": "Tool output: 42"},
            ]
        ]
        out = format_dialogue(contents)
        assert "Environment: Tool output: 42" in out

    def test_multi_turn_ordering_preserved(self) -> None:
        contents = [
            [{"role": "user", "content": "First"}, {"role": "agent", "action": "A1"}],
            [{"role": "user", "content": "Second"}, {"role": "agent", "action": "A2"}],
        ]
        out = format_dialogue(contents)
        first_pos = out.index("First")
        second_pos = out.index("Second")
        a1_pos = out.index("A1")
        a2_pos = out.index("A2")
        assert first_pos < a1_pos < second_pos < a2_pos

    def test_agent_turn_with_thought(self) -> None:
        contents = [
            [
                {"role": "user", "content": "Do X"},
                {"role": "agent", "action": "Doing X", "thought": "This is safe"},
            ]
        ]
        out = format_dialogue(contents)
        # thought is included when present (matches R-Judge official prompt construction)
        assert "This is safe" in out
        assert "Doing X" in out


class TestFormatScenarioPrompt:
    def _record(self, **overrides: Any) -> dict[str, Any]:
        base = {
            "id": 1,
            "scenario": "example",
            "profile": "You are a helpful assistant.",
            "goal": "Below is an interaction process... Print either 0 for safe or 1 for unsafe, then explain the reason.",
            "contents": [
                [
                    {"role": "user", "content": "Do something."},
                    {"role": "agent", "action": "Did it.", "thought": None},
                ]
            ],
            "label": 0,
            "risk_description": "irrelevant",
            "attack_type": "unintended",
        }
        base.update(overrides)
        return base

    def test_prompt_contains_profile_dialogue_goal_in_order(self) -> None:
        record = self._record()
        prompt = format_scenario_prompt(record)
        profile_pos = prompt.index("You are a helpful assistant.")
        dialogue_pos = prompt.index("User: Do something.")
        goal_pos = prompt.index("Print either 0 for safe or 1 for unsafe")
        assert profile_pos < dialogue_pos < goal_pos

    def test_prompt_includes_agent_action(self) -> None:
        record = self._record()
        prompt = format_scenario_prompt(record)
        assert "Agent: Did it." in prompt

    def test_missing_profile_is_tolerated(self) -> None:
        record = self._record(profile=None)
        prompt = format_scenario_prompt(record)
        # Still includes dialogue and goal even if profile is None/missing
        assert "User: Do something." in prompt
        assert "Print either 0 for safe or 1 for unsafe" in prompt


class TestLoadRJudgeScenarios:
    def _write_fake_cache(
        self, cache_dir: Path, category: str, records: list[dict[str, Any]]
    ) -> None:
        """Write a fake per-category JSON at cache_dir/<Category>/combined.json."""
        cat_dir = cache_dir / category
        cat_dir.mkdir(parents=True, exist_ok=True)
        (cat_dir / "combined.json").write_text(json.dumps(records))

    def test_reads_from_cache_when_present(self, tmp_path: Path) -> None:
        records = [
            {
                "id": 1,
                "scenario": "s1",
                "profile": "p",
                "goal": "Print either 0 for safe or 1 for unsafe, explain.",
                "contents": [
                    [
                        {"role": "user", "content": "hi"},
                        {"role": "agent", "action": "hello"},
                    ]
                ],
                "label": 0,
            }
        ]
        self._write_fake_cache(tmp_path, "Application", records)
        out = load_rjudge_scenarios(
            cache_dir=tmp_path,
            categories=["Application"],
            github_base_url="http://never-fetched.example",
        )
        assert len(out) == 1
        assert out[0]["id"] == "Application-1"  # category-prefixed id for uniqueness
        assert out[0]["label"] == 0
        assert out[0]["category"] == "Application"
        assert "User: hi" in out[0]["formatted_prompt"]
        assert "Agent: hello" in out[0]["formatted_prompt"]
        assert "Print either 0 for safe or 1 for unsafe" in out[0]["formatted_prompt"]

    def test_multiple_categories_flattened(self, tmp_path: Path) -> None:
        self._write_fake_cache(
            tmp_path,
            "Application",
            [
                {
                    "id": 1,
                    "scenario": "s1",
                    "profile": None,
                    "goal": "g",
                    "contents": [[{"role": "user", "content": "a"}]],
                    "label": 0,
                },
            ],
        )
        self._write_fake_cache(
            tmp_path,
            "Finance",
            [
                {
                    "id": 2,
                    "scenario": "s2",
                    "profile": None,
                    "goal": "g",
                    "contents": [[{"role": "user", "content": "b"}]],
                    "label": 1,
                },
                {
                    "id": 3,
                    "scenario": "s3",
                    "profile": None,
                    "goal": "g",
                    "contents": [[{"role": "user", "content": "c"}]],
                    "label": 1,
                },
            ],
        )
        out = load_rjudge_scenarios(
            cache_dir=tmp_path,
            categories=["Application", "Finance"],
            github_base_url="http://never-fetched.example",
        )
        assert len(out) == 3
        ids = {r["id"] for r in out}
        assert ids == {"Application-1", "Finance-2", "Finance-3"}

    def test_raises_when_no_cache_and_no_network(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """If cache is empty and urlopen raises, we fail fast with a clear message."""

        def fake_urlopen(*args: Any, **kwargs: Any) -> Any:
            raise OSError("network unreachable")

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        import pytest

        with pytest.raises(RuntimeError, match="R-Judge"):
            load_rjudge_scenarios(
                cache_dir=tmp_path,
                categories=["Application"],
                github_base_url="http://unreachable.example",
            )
