"""Integration tests for the wired probe_sweep and diff_means runner actions."""

from __future__ import annotations

import torch

from sonde.activation.storage import save_extraction
from sonde.core.configs.diff_means_config import DiffMeansConfig
from sonde.core.configs.probe_config import ProbeConfig
from sonde.probes import ProbeArtifact
from sonde.runners.experiment_runner import dispatch_action


def _write_synthetic_extraction(tmp_path) -> str:
    """A small, separable two-layer extraction; returns the manifest path."""
    torch.manual_seed(0)
    n, d = 30, 12
    labels = [1] * 15 + [0] * 15
    sample_ids = [f"s-{i}" for i in range(n)]

    def layer(sep: float) -> torch.Tensor:
        feats = torch.randn(n, d) * 0.5
        feats[:, 0] += torch.tensor([sep if y else -sep for y in labels])
        return feats

    result = {
        "model": {"name": "synthetic", "num_layers": 2, "hidden_size": d},
        "requested": ["layers_output:0", "layers_output:1"],
        "activations": {
            "layers_output:0": layer(0.5),
            "layers_output:1": layer(2.0),
        },
        "sample_ids": sample_ids,
        "labels": labels,
    }
    storage = save_extraction(result, tmp_path / "ext", overwrite=True)
    return storage["manifest_path"]


class TestProbeSweepAction:
    def test_probe_sweep_produces_loadable_artifact(self, tmp_path):
        manifest = _write_synthetic_extraction(tmp_path)
        out = tmp_path / "out"
        cfg = ProbeConfig.from_dict(
            {
                "run_name": "rt",
                "seed": 0,
                "action": "probe_sweep",
                "probe": {"epochs": 40, "learning_rate": 0.05},
                "sweep": {
                    "activation_targets": ["layers_output:0", "layers_output:1"],
                    "enforce_control_sanity": True,
                },
                "io": {"input_path": manifest, "output_dir": str(out)},
                "output": {"output_dir": str(out)},
            }
        )
        result = dispatch_action(cfg)
        assert result.summary["status"] == "completed"
        assert result.summary["best_layer"].startswith("layers_output:")
        art = ProbeArtifact.load(result.artifacts["probe"])
        assert art.direction.numel() == 12
        assert art.layer in (0, 1)

    def test_probe_sweep_is_deterministic(self, tmp_path):
        manifest = _write_synthetic_extraction(tmp_path)
        cfgs = [
            ProbeConfig.from_dict(
                {
                    "run_name": f"rt{i}",
                    "seed": 0,
                    "action": "probe_sweep",
                    "probe": {"epochs": 40, "learning_rate": 0.05},
                    "sweep": {
                        "activation_targets": ["layers_output:0", "layers_output:1"]
                    },
                    "io": {
                        "input_path": manifest,
                        "output_dir": str(tmp_path / f"o{i}"),
                    },
                    "output": {"output_dir": str(tmp_path / f"o{i}")},
                }
            )
            for i in range(2)
        ]
        r0 = dispatch_action(cfgs[0])
        r1 = dispatch_action(cfgs[1])
        assert r0.summary["best_layer"] == r1.summary["best_layer"]


class TestDiffMeansAction:
    def test_diff_means_produces_direction_artifact(self, tmp_path):
        manifest = _write_synthetic_extraction(tmp_path)
        out = tmp_path / "dm"
        cfg = DiffMeansConfig.from_dict(
            {
                "run_name": "dm",
                "seed": 0,
                "action": "diff_means",
                "sweep": {
                    "activation_targets": ["layers_output:0", "layers_output:1"],
                    "enforce_control_sanity": True,
                },
                "io": {"input_path": manifest, "output_dir": str(out)},
                "output": {"output_dir": str(out)},
            }
        )
        result = dispatch_action(cfg)
        assert result.summary["status"] == "completed"
        art = ProbeArtifact.load(result.artifacts["direction"])
        assert art.direction.numel() == 12
        assert art.metadata["method"] == "diff_means"
