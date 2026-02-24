"""Tests for backward-compatible config imports."""

from __future__ import annotations


class TestConfigCompatImports:
    def test_legacy_config_imports_still_work(self):
        from configs import LayerProbeSweepConfig, ProbeConfig, ModelConfig, ActivationConfig  # noqa: F401

    def test_new_config_imports_via_configs_package(self):
        from configs import RunConfig, ModelParams, ProbeParams, SplitParams  # noqa: F401

    def test_new_config_imports_via_core(self):
        from core.configs import RunConfig, BaseConfig, ModelParams  # noqa: F401

    def test_legacy_probe_config_has_expected_defaults(self):
        from configs import ProbeConfig
        pc = ProbeConfig()
        assert pc.epochs == 10
        assert pc.learning_rate == 1e-3

    def test_new_probe_params_has_expected_defaults(self):
        from configs import ProbeParams
        pp = ProbeParams()
        assert pp.epochs == 10
        assert pp.learning_rate == 1e-3
