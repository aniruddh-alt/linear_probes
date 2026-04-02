"""Tests for GenerateConfig."""
from __future__ import annotations

from core.configs.generate_config import GenerateConfig


class TestGenerateConfig:
    def test_defaults(self):
        cfg = GenerateConfig()
        assert cfg.run_name == ""
        assert cfg.seed == 0
        assert cfg.action == "generate"
        assert cfg.model.model_name == ""
        assert cfg.generation.max_new_tokens == 256
        assert cfg.io.output_dir == "artifacts"

    def test_from_dict(self):
        cfg = GenerateConfig.from_dict({
            "run_name": "test_gen",
            "model": {"model_name": "Qwen/Qwen2.5-1.5B-Instruct"},
            "generation": {"max_new_tokens": 128},
            "io": {"output_dir": "artifacts/refusal/"},
        })
        assert cfg.run_name == "test_gen"
        assert cfg.model.model_name == "Qwen/Qwen2.5-1.5B-Instruct"
        assert cfg.generation.max_new_tokens == 128
        assert cfg.io.output_dir == "artifacts/refusal/"

    def test_steering_defaults(self):
        cfg = GenerateConfig()
        assert cfg.steering.enabled is False
        assert cfg.steering.mode == "project_subtract"

    def test_steering_from_dict(self):
        cfg = GenerateConfig.from_dict({
            "model": {"model_name": "test"},
            "steering": {
                "enabled": True,
                "vector_path": "v.pt",
                "layers": [14, 15],
                "strength": 20.0,
                "mode": "additive",
            },
        })
        assert cfg.steering.enabled is True
        assert cfg.steering.layers == [14, 15]
        assert cfg.steering.strength == 20.0

    def test_steering_yaml_roundtrip(self, tmp_path):
        cfg = GenerateConfig.from_dict({
            "steering": {
                "enabled": True,
                "vector_path": "w.pt",
                "layers": [10],
                "mode": "project_subtract",
            },
        })
        path = tmp_path / "gen_steer.yaml"
        cfg.to_yaml(path)
        loaded = GenerateConfig.from_yaml(path)
        assert loaded.steering.enabled is True
        assert loaded.steering.layers == [10]

    def test_yaml_roundtrip(self, tmp_path):
        cfg = GenerateConfig(run_name="gen_test")
        cfg.model.model_name = "test-model"
        path = tmp_path / "gen.yaml"
        cfg.to_yaml(path)
        loaded = GenerateConfig.from_yaml(path)
        assert loaded.run_name == "gen_test"
