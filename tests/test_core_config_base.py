"""Tests for core config base and params modules."""

from __future__ import annotations

from core.configs.params.model_params import ModelParams
from core.configs.params.extraction_params import ExtractionParams
from core.configs.params.probe_params import ProbeParams
from core.configs.params.split_params import SplitParams
from core.configs.params.output_params import OutputParams


class TestBaseConfig:
    def test_base_config_round_trip_yaml(self, tmp_path):
        cfg = ModelParams(model_name="Qwen/Qwen2.5-1.5B-Instruct")
        path = tmp_path / "model.yaml"
        cfg.to_yaml(path)
        loaded = ModelParams.from_yaml(path)
        assert loaded == cfg

    def test_base_config_round_trip_with_non_defaults(self, tmp_path):
        cfg = ModelParams(
            model_name="meta-llama/Llama-3.1-8B",
            device="cuda:0",
            dtype="bfloat16",
            load_in_8bit=True,
        )
        path = tmp_path / "model2.yaml"
        cfg.to_yaml(path)
        loaded = ModelParams.from_yaml(path)
        assert loaded == cfg
        assert loaded.device == "cuda:0"

    def test_extraction_params_defaults(self):
        ep = ExtractionParams()
        assert ep.batch_size == 8
        assert ep.token_index == -1
        assert ep.to_cpu is True

    def test_probe_params_defaults(self):
        pp = ProbeParams()
        assert pp.epochs == 10
        assert pp.learning_rate == 1e-3

    def test_split_params_defaults(self):
        sp = SplitParams()
        assert sp.train_fraction == 0.7
        assert sp.val_fraction == 0.15
        assert sp.test_fraction == 0.15

    def test_output_params_defaults(self):
        op = OutputParams()
        assert op.save_plots is True

    def test_split_params_round_trip_yaml(self, tmp_path):
        cfg = SplitParams(train_fraction=0.8, val_fraction=0.1, test_fraction=0.1)
        path = tmp_path / "split.yaml"
        cfg.to_yaml(path)
        loaded = SplitParams.from_yaml(path)
        assert loaded == cfg

    def test_probe_params_round_trip_yaml(self, tmp_path):
        cfg = ProbeParams(epochs=20, learning_rate=0.01, weight_decay=0.001)
        path = tmp_path / "probe.yaml"
        cfg.to_yaml(path)
        loaded = ProbeParams.from_yaml(path)
        assert loaded == cfg
