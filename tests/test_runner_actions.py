"""Integration tests for the wired probe_sweep and diff_means runner actions."""

from __future__ import annotations

import pytest
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
        # Layer 1 is the more separable layer (sep 2.0 vs 0.5) and must be picked.
        assert result.summary["best_layer"] == "layers_output:1"
        assert result.summary["test_auroc"] >= 0.9
        art = ProbeArtifact.load(result.artifacts["probe"])
        assert art.direction.numel() == 12
        assert art.layer == 1

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
        # Separation lives in feature dim 0, so the direction must point there.
        assert int(art.direction.abs().argmax()) == 0

    def test_diff_means_rejects_sequence_mode_extraction(self, tmp_path):
        # diff_means needs pooled (N, D) features; a sequence-mode extraction
        # (list of per-token tensors) must fail loudly, not IndexError.
        seqs = [torch.randn(4, 8) for _ in range(12)]
        result = {
            "model": {},
            "requested": ["layers_output:0"],
            "activations": {"layers_output:0": seqs},
            "sample_ids": [f"s{i}" for i in range(12)],
            "labels": [1] * 6 + [0] * 6,
        }
        storage = save_extraction(result, tmp_path / "seq", overwrite=True)
        cfg = DiffMeansConfig.from_dict(
            {
                "action": "diff_means",
                "seed": 0,
                "sweep": {
                    "activation_targets": ["layers_output:0"],
                    "enforce_control_sanity": False,
                },
                "io": {
                    "input_path": storage["manifest_path"],
                    "output_dir": str(tmp_path / "o"),
                },
                "output": {"output_dir": str(tmp_path / "o")},
            }
        )
        with pytest.raises(ValueError, match="requires pooled"):
            dispatch_action(cfg)


class TestControlSanityGuard:
    def test_probe_sweep_control_sanity_raises_when_real_not_above_controls(self):
        from sonde.probes.sweep import LayerProbeSweepRunner

        runner = LayerProbeSweepRunner()
        controls = {
            "real": {"auroc_mean": 0.55},
            "shuffled_labels": {"auroc_mean": 0.60},
            "random_features": {"auroc_mean": 0.50},
        }
        with pytest.raises(ValueError, match="Control sanity check failed"):
            runner._enforce_control_sanity(controls)

    def test_probe_sweep_control_sanity_passes_when_real_above(self):
        from sonde.probes.sweep import LayerProbeSweepRunner

        runner = LayerProbeSweepRunner()
        controls = {
            "real": {"auroc_mean": 0.95},
            "shuffled_labels": {"auroc_mean": 0.50},
            "random_features": {"auroc_mean": 0.52},
        }
        runner._enforce_control_sanity(controls)  # no raise

    def test_diff_means_control_sanity_raises(self):
        from sonde.directions.sweep import DiffMeansSweepRunner

        runner = DiffMeansSweepRunner()
        controls = {
            "real": {"auroc_mean": 0.50},
            "shuffled_labels": {"auroc_mean": 0.60},
        }
        with pytest.raises(ValueError, match="Control sanity check failed"):
            runner._enforce_control_sanity(controls)


class TestExtractAction:
    def test_extract_action_writes_loadable_manifest(self, tmp_path, monkeypatch):
        import sonde.activation as activation_pkg
        from sonde.activation.storage import load_extraction_manifest, save_extraction
        from sonde.core.configs.extract_config import ExtractConfig

        # Stub the extractor so no model is needed: it writes a real artifact.
        class _StubExtractor:
            def __init__(self, *, model, extraction):
                self.extraction = extraction

            def extract(self, bundle):
                import torch

                result = {
                    "model": {"name": "stub"},
                    "requested": ["layers_output:0"],
                    "activations": {"layers_output:0": torch.randn(len(bundle.ids), 4)},
                    "sample_ids": list(bundle.ids),
                    "labels": list(bundle.labels),
                }
                result["storage"] = save_extraction(
                    result, self.extraction.save_path, overwrite=True
                )
                return result

        monkeypatch.setattr(activation_pkg, "ActivationExtractor", _StubExtractor)

        inp = tmp_path / "data.jsonl"
        inp.write_text(
            '{"id": "a", "text": "hello", "label": 1}\n'
            '{"id": "b", "text": "world", "label": 0}\n',
            encoding="utf-8",
        )
        cfg = ExtractConfig.from_dict(
            {
                "run_name": "ext",
                "action": "extract",
                "model": {"model_name": "stub"},
                "extraction": {"activations": ["layers_output:0"]},
                "io": {"input_path": str(inp), "output_dir": str(tmp_path / "out")},
            }
        )
        result = dispatch_action(cfg)
        manifest_path = result.artifacts["extraction_path"]
        assert manifest_path.endswith("_manifest.json")
        loaded = load_extraction_manifest(manifest_path)
        assert loaded["sample_ids"] == ["a", "b"]


class TestMapLabel:
    def test_strict_label_map_raises_on_unmapped(self):
        from sonde.runners.experiment_runner import _map_label

        label_map = {"entailment": 1, "contradiction": 0}
        assert _map_label("entailment", label_map) == 1
        with pytest.raises(KeyError, match="not in dataset"):
            _map_label("neutral", label_map)

    def test_empty_label_map_coerces_int(self):
        from sonde.runners.experiment_runner import _map_label

        assert _map_label("1", {}) == 1
        assert _map_label(0, {}) == 0
