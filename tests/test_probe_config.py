"""Tests for ProbeConfig."""

from __future__ import annotations

from core.configs.probe_config import ProbeConfig


class TestProbeConfig:
    def test_defaults(self):
        cfg = ProbeConfig()
        assert cfg.run_name == ""
        assert cfg.action == "probe_sweep"
        assert cfg.probe.learning_rate == 1e-3
        assert cfg.split.train_fraction == 0.7
        assert cfg.sweep.selection_metric == "auroc"
        assert cfg.io.input_path == ""
        assert cfg.io.output_dir == "artifacts"
        assert cfg.output.save_plots is True

    def test_from_dict(self):
        cfg = ProbeConfig.from_dict(
            {
                "run_name": "refusal_probe",
                "probe": {"learning_rate": 0.01, "epochs": 20},
                "io": {
                    "input_path": "artifacts/activations/",
                    "output_dir": "artifacts/",
                },
            }
        )
        assert cfg.run_name == "refusal_probe"
        assert cfg.probe.learning_rate == 0.01
        assert cfg.io.input_path == "artifacts/activations/"

    def test_yaml_roundtrip(self, tmp_path):
        cfg = ProbeConfig(run_name="probe_test")
        path = tmp_path / "probe.yaml"
        cfg.to_yaml(path)
        loaded = ProbeConfig.from_yaml(path)
        assert loaded.run_name == "probe_test"
        assert loaded.action == "probe_sweep"

    def test_probe_params_has_probe_type_field(self):
        from core.configs import ProbeParams

        params = ProbeParams()
        assert params.probe_type == "linear"
        assert params.probe_kwargs == {}

    def test_probe_params_accepts_custom_probe_type(self):
        from core.configs import ProbeParams

        params = ProbeParams(probe_type="attention", probe_kwargs={"foo": 1})
        assert params.probe_type == "attention"
        assert params.probe_kwargs == {"foo": 1}
