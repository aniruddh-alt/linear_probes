"""Smoke test: full rjudge_dissociation pipeline with mocked model modules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from experiments.rjudge_dissociation import run_pipeline


class _FakeTrainer:
    def __init__(self, hidden_dim: int) -> None:
        self.device = torch.device("cpu")
        self.model = torch.nn.Linear(hidden_dim, 1)

    def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        # Return deterministic non-trivial scores derived from features.
        with torch.no_grad():
            return torch.sigmoid(features.float().mean(dim=-1)).cpu()


def _write_rjudge_cache(cache_dir: Path) -> None:
    """Write a fake R-Judge cache with 12 scenarios (6 risky, 6 safe)."""
    records = []
    for i in range(6):
        records.append({
            "id": 100 + i,
            "scenario": f"safe_{i}",
            "profile": "You are helpful.",
            "goal": "Print 0 for safe or 1 for unsafe, then explain.",
            "contents": [[
                {"role": "user", "content": f"benign query {i}"},
                {"role": "agent", "action": f"benign reply {i}"},
            ]],
            "label": 0,
        })
    for i in range(6):
        records.append({
            "id": 200 + i,
            "scenario": f"risky_{i}",
            "profile": "You are helpful.",
            "goal": "Print 0 for safe or 1 for unsafe, then explain.",
            "contents": [[
                {"role": "user", "content": f"risky query {i}"},
                {"role": "agent", "action": f"risky reply {i}"},
            ]],
            "label": 1,
        })
    (cache_dir / "Application").mkdir(parents=True, exist_ok=True)
    (cache_dir / "Application" / "combined.json").write_text(json.dumps(records))


def test_pipeline_end_to_end_with_mocks(tmp_path: Path, monkeypatch) -> None:
    output_dir = tmp_path / "data"
    cache_dir = tmp_path / "data" / "rjudge_raw"
    _write_rjudge_cache(cache_dir)

    config = {
        "run_name": "smoke",
        "seed": 42,
        "model": {"model_name": "FAKE-MODEL", "dtype": "float32"},
        "judgment": {"max_new_tokens": 8, "batch_size": 2},
        "extraction": {
            "batch_size": 2,
            "token_index": -1,
            "activations": ["layers_output:0", "layers_output:1"],
            "save_path": str(output_dir / "activations"),
        },
        "probe": {
            "probe_type": "linear",
            "epochs": 2,
            "learning_rate": 0.01,
            "weight_decay": 0.1,
            "bootstrap_samples": 2,
            "early_stopping_patience": 2,
            "seed": 0,
            "threshold": 0.5,
        },
        "split": {"train_fraction": 0.5, "val_fraction": 0.25, "test_fraction": 0.25},
        "sweep": {"batch_size": 4, "selection_metric": "auroc"},
        "io": {"output_dir": str(output_dir)},
        "rjudge": {
            "cache_dir": str(cache_dir),
            "categories": ["Application"],
            "github_base_url": "http://unused.example",
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(json.dumps(config))  # YAML is JSON-compatible

    # Mock ResponseGenerator: make the model "get half right and half wrong"
    # to produce a non-empty FN cell.
    class _FakeResponseGenerator:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def generate(self, bundle: Any) -> Any:
            from generation.types import GenerationResult
            ids = list(bundle.ids)
            # For risky ids (200+), half will be correctly flagged ('1'), half missed ('0') → FN.
            # For safe ids (100+), most correctly flagged ('0'), one flagged as '1' → FP.
            responses: list[str] = []
            for sid in ids:
                numeric = int(sid.split("-")[-1])
                if numeric >= 200:  # risky
                    responses.append("1 reason" if numeric % 2 == 0 else "0 reason")
                else:  # safe
                    responses.append("1 reason" if numeric == 105 else "0 reason")
            return GenerationResult(
                prompts=[f"p-{sid}" for sid in ids],
                responses=responses,
                sample_ids=list(ids),
                labels=[None] * len(ids),
            )

    # Mock ActivationExtractor: returns a fake ExtractionResult with random features.
    class _FakeExtractor:
        def __init__(self, **kwargs: Any) -> None:
            self.save_path = kwargs["extraction"].save_path
            self.activations = list(kwargs["extraction"].activations)

        def extract(self, bundle: Any) -> dict[str, Any]:
            n = len(bundle.ids)
            hidden_dim = 8
            torch.manual_seed(0)
            acts = {
                key: torch.randn(n, hidden_dim) for key in self.activations
            }
            Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)
            return {
                "model": {"name": "fake", "num_layers": 2, "hidden_size": hidden_dim,
                          "num_heads": 1, "vocab_size": 10},
                "requested": list(self.activations),
                "activations": acts,
                "sample_ids": list(bundle.ids),
                "labels": list(bundle.labels),
                "storage": {"mode": "in_memory"},
            }

    monkeypatch.setattr(run_pipeline, "ResponseGenerator", _FakeResponseGenerator)
    monkeypatch.setattr(run_pipeline, "ActivationExtractor", _FakeExtractor)

    # Lower the SweepRunner's control sanity to tolerate the tiny fake data.
    from probes.sweep import LayerProbeSweepRunner

    original_init = LayerProbeSweepRunner.__init__

    def _patched_init(self: LayerProbeSweepRunner, probe: Any = None, sweep: Any = None) -> None:
        original_init(self, probe=probe, sweep=sweep)
        self.sweep.enforce_control_sanity = False

    monkeypatch.setattr(LayerProbeSweepRunner, "__init__", _patched_init)

    run_pipeline.main(config_path=config_path)

    assert (output_dir / "judgments.jsonl").exists()
    assert (output_dir / "cells.json").exists()
    assert (output_dir / "best_probe.pt").exists()
    assert (output_dir / "results.json").exists()

    results = json.loads((output_dir / "results.json").read_text())
    assert results["run_name"] == "smoke"
    assert "dissociation" in results
    assert "auroc_fn_vs_tn" in results["dissociation"]
    assert results["cells"]["FN"] > 0, "expected at least one FN in the fake data"
