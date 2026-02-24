"""Tests for dot-notation config override utility."""

from __future__ import annotations

from core.configs.overrides import apply_dot_overrides


class TestConfigOverrides:
    def test_apply_dot_overrides_nested_values(self):
        payload = {
            "probe": {"learning_rate": 1e-2},
            "split": {"auto_group_by_id_when_none": True},
        }
        out = apply_dot_overrides(
            payload,
            {"probe.learning_rate": "0.001", "split.auto_group_by_id_when_none": "false"},
        )
        assert out["probe"]["learning_rate"] == 0.001
        assert out["split"]["auto_group_by_id_when_none"] is False

    def test_apply_dot_overrides_top_level(self):
        payload = {"run_name": "old", "seed": 0}
        out = apply_dot_overrides(payload, {"run_name": "new", "seed": "42"})
        assert out["run_name"] == "new"
        assert out["seed"] == 42

    def test_apply_dot_overrides_empty_overrides(self):
        payload = {"a": 1}
        out = apply_dot_overrides(payload, {})
        assert out == {"a": 1}

    def test_apply_dot_overrides_deeply_nested(self):
        payload = {"model": {"config": {"hidden_size": 768}}}
        out = apply_dot_overrides(payload, {"model.config.hidden_size": "1024"})
        assert out["model"]["config"]["hidden_size"] == 1024

    def test_apply_dot_overrides_string_values_preserved(self):
        payload = {"model": {"model_name": "old-model"}}
        out = apply_dot_overrides(payload, {"model.model_name": "new-model"})
        assert out["model"]["model_name"] == "new-model"
