"""Tests for GenerationParams config."""

from __future__ import annotations

from core.configs.params.generation_params import GenerationParams


class TestGenerationParams:
    def test_defaults(self):
        params = GenerationParams()
        assert params.max_new_tokens == 256
        assert params.temperature == 1.0
        assert params.top_p == 1.0
        assert params.do_sample is False
        assert params.batch_size == 8

    def test_round_trip_yaml(self, tmp_path):
        params = GenerationParams(max_new_tokens=128, temperature=0.7)
        path = tmp_path / "gen.yaml"
        params.to_yaml(path)
        loaded = GenerationParams.from_yaml(path)
        assert loaded.max_new_tokens == 128
        assert loaded.temperature == 0.7

    def test_run_config_includes_generation(self):
        from core.configs.run_config import RunConfig
        cfg = RunConfig()
        assert hasattr(cfg, "generation")
        assert cfg.generation.max_new_tokens == 256

    def test_run_config_yaml_with_generation(self, tmp_path):
        from core.configs.run_config import RunConfig
        yaml_text = """
run_name: gen-test
action: generate
model:
  model_name: test-model
generation:
  max_new_tokens: 64
  temperature: 0.5
"""
        path = tmp_path / "run.yaml"
        path.write_text(yaml_text, encoding="utf-8")
        cfg = RunConfig.from_yaml(path)
        assert cfg.generation.max_new_tokens == 64
        assert cfg.generation.temperature == 0.5
