"""Tests for the R-Judge loader."""

from __future__ import annotations

import json

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
