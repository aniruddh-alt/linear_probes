"""Run the R-Judge dissociation experiment end-to-end.

Loads `experiments/rjudge_dissociation/config.yaml`, runs:
  1. R-Judge scenario loading + prompt formatting
  2. Behavioral judgment via ResponseGenerator (greedy, short)
  3. 4-cell classification (TP/FP/FN/TN)
  4. Activation extraction at last prompt token across multiple layers
  5. Probe sweep trained on TP ∪ TN only
  6. Dissociation evaluation on FN cell
  7. Report + save results.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from activation import ActivationExtractor
from core.configs import (
    ExtractionParams,
    GenerationParams,
    ModelParams,
    ProbeParams,
    SweepParams,
)
from dataset import ProbingSampleBuilder
from dataset.probing_dataset import ProbingDataset
from dataset.splitting import stratified_train_val_test_split
from experiments.rjudge_dissociation.cells import classify_cells
from experiments.rjudge_dissociation.judgment import parse_first_digit
from experiments.rjudge_dissociation.metrics import (
    auroc_between_cells,
    classification_rate_at_threshold,
)
from experiments.rjudge_dissociation.rjudge_loader import load_rjudge_scenarios
from generation import ResponseGenerator
from probes import LayerProbeSweepRunner


def _load_config(config_path: Path) -> dict[str, Any]:
    raw = OmegaConf.load(config_path)
    return OmegaConf.to_container(raw, resolve=True)  # type: ignore[return-value]


def _run_judgment(
    *,
    scenarios: list[dict[str, Any]],
    cfg: dict[str, Any],
    output_dir: Path,
) -> dict[str, int]:
    """Generate 1-token judgments and parse 0/1. Writes judgments.jsonl."""
    judgment_cfg = cfg["judgment"]
    model_cfg = cfg["model"]

    rows = [
        {"id": s["id"], "text": s["formatted_prompt"], "label": s["label"]}
        for s in scenarios
    ]
    bundle = ProbingSampleBuilder.from_iterable(rows).to_samples(text_key="text")

    generator = ResponseGenerator(
        model=ModelParams(model_name=model_cfg["model_name"], dtype=model_cfg.get("dtype")),
        generation=GenerationParams(
            max_new_tokens=judgment_cfg["max_new_tokens"],
            batch_size=judgment_cfg.get("batch_size", 2),
            do_sample=False,  # greedy
        ),
    )
    result = generator.generate(bundle)

    predictions: dict[str, int] = {}
    jsonl_path = output_dir / "judgments.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for sid, prompt, response in zip(
            result.sample_ids, result.prompts, result.responses, strict=True
        ):
            pred = parse_first_digit(response)
            predictions[sid] = pred if pred is not None else -1
            f.write(json.dumps({
                "id": sid,
                "raw_output": response,
                "predicted_label": predictions[sid],
            }) + "\n")
    return predictions


def _run_extraction(
    *,
    scenarios: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Extract activations at last prompt token for each scenario."""
    extraction_cfg = cfg["extraction"]
    model_cfg = cfg["model"]

    rows = [
        {"id": s["id"], "text": s["formatted_prompt"], "label": s["label"]}
        for s in scenarios
    ]
    bundle = ProbingSampleBuilder.from_iterable(rows).to_samples(text_key="text")

    extractor = ActivationExtractor(
        model=ModelParams(model_name=model_cfg["model_name"], dtype=model_cfg.get("dtype")),
        extraction=ExtractionParams(
            save_path=extraction_cfg["save_path"],
            activations=list(extraction_cfg["activations"]),
            batch_size=extraction_cfg["batch_size"],
            token_index=extraction_cfg["token_index"],
        ),
    )
    return extractor.extract(bundle)


def _build_split_on_subset(
    *,
    scenarios: list[dict[str, Any]],
    cells: dict[str, str],
    split_cfg: dict[str, Any],
    seed: int,
) -> tuple[list[int], list[int], list[int], list[int], list[int]]:
    """Build train/val/test indices over the TP ∪ TN subset.

    Returns:
        train_rel, val_rel, test_rel: indices RELATIVE to tp_tn_indices
            (0..len(tp_tn_indices)-1). These are what the sweep runner consumes
            after the extraction is subset to TP ∪ TN rows.
        fn_indices: ABSOLUTE indices into the full scenarios list, for the FN cell.
        tp_tn_indices: ABSOLUTE indices into the full scenarios list, used to
            subset the extraction before probe training.
    """
    tp_tn_indices = [
        i for i, s in enumerate(scenarios)
        if cells.get(s["id"]) in ("TP", "TN")
    ]
    fn_indices = [
        i for i, s in enumerate(scenarios)
        if cells.get(s["id"]) == "FN"
    ]
    labels_subset = [scenarios[i]["label"] for i in tp_tn_indices]

    train_rel, val_rel, test_rel = stratified_train_val_test_split(
        labels=labels_subset,
        train_fraction=split_cfg["train_fraction"],
        val_fraction=split_cfg["val_fraction"],
        test_fraction=split_cfg["test_fraction"],
        seed=seed,
    )
    return train_rel, val_rel, test_rel, fn_indices, tp_tn_indices


def _subset_extraction(
    extraction: dict[str, Any],
    indices: list[int],
) -> dict[str, Any]:
    """Return a new extraction dict with only the rows at the given absolute indices.

    Used to filter the full extraction to TP ∪ TN rows before probe training,
    so the sweep runner's coverage check and control-sanity metrics see clean data.
    """
    activations = extraction["activations"]
    subset_acts: dict[str, Any] = {}
    for key, value in activations.items():
        if isinstance(value, list):
            subset_acts[key] = [value[i] for i in indices]
        else:
            subset_acts[key] = value[indices]
    sample_ids = extraction.get("sample_ids", [])
    subset_sample_ids = [sample_ids[i] for i in indices] if sample_ids else []
    labels = extraction.get("labels", [])
    subset_labels = [labels[i] for i in indices] if labels else []
    return {
        **extraction,
        "activations": subset_acts,
        "sample_ids": subset_sample_ids,
        "labels": subset_labels,
    }


def _evaluate_dissociation(
    *,
    extraction: dict[str, Any],
    best_key: str,
    best_probe: Any,
    scenarios: list[dict[str, Any]],
    cells: dict[str, str],
    threshold: float,
) -> dict[str, Any]:
    """Score every scenario with the best probe and compute dissociation metrics."""
    dataset = ProbingDataset.from_extraction_result(
        extraction,
        activation_key=best_key,
        labels=[s["label"] for s in scenarios],
    )
    features = dataset.features  # (N, D)
    scores = best_probe.trainer.predict_proba(features.to(best_probe.trainer.device))

    def mask_for(cell_name: str) -> torch.Tensor:
        return torch.tensor(
            [cells.get(s["id"]) == cell_name for s in scenarios],
            dtype=torch.bool,
        )

    tp_mask = mask_for("TP")
    fp_mask = mask_for("FP")
    fn_mask = mask_for("FN")
    tn_mask = mask_for("TN")

    fn_scores = scores[fn_mask]

    results: dict[str, Any] = {
        "best_layer": best_key,
        "threshold": threshold,
        "cell_counts": {
            cell: int(mask.sum().item())
            for cell, mask in (("TP", tp_mask), ("FP", fp_mask),
                               ("FN", fn_mask), ("TN", tn_mask))
        },
        "auroc_fn_vs_tn": auroc_between_cells(scores=scores, mask_a=fn_mask, mask_b=tn_mask),
        "auroc_fn_vs_tp": auroc_between_cells(scores=scores, mask_a=fn_mask, mask_b=tp_mask),
        "auroc_fp_vs_tp": auroc_between_cells(scores=scores, mask_a=fp_mask, mask_b=tp_mask),
        "fn_classification_rate": classification_rate_at_threshold(
            scores=fn_scores, threshold=threshold
        ),
    }

    # Per-category FN classification rate for inspection.
    categories = sorted({s["category"] for s in scenarios})
    per_cat: dict[str, dict[str, float]] = {}
    for cat in categories:
        cat_fn_mask = torch.tensor(
            [cells.get(s["id"]) == "FN" and s["category"] == cat for s in scenarios],
            dtype=torch.bool,
        )
        cat_fn_scores = scores[cat_fn_mask]
        per_cat[cat] = {
            "fn_count": int(cat_fn_mask.sum().item()),
            "fn_classification_rate": classification_rate_at_threshold(
                scores=cat_fn_scores, threshold=threshold
            ),
        }
    results["per_category"] = per_cat
    return results


def main(config_path: Path | None = None) -> None:
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    cfg = _load_config(config_path)

    output_dir = Path(cfg["io"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[rjudge] Loading R-Judge scenarios from {cfg['rjudge']['cache_dir']}")
    scenarios = load_rjudge_scenarios(
        cache_dir=cfg["rjudge"]["cache_dir"],
        categories=list(cfg["rjudge"]["categories"]),
        github_base_url=cfg["rjudge"]["github_base_url"],
    )
    labels_by_id = {s["id"]: s["label"] for s in scenarios}
    print(f"[rjudge] Loaded {len(scenarios)} scenarios "
          f"({sum(labels_by_id.values())} risky, "
          f"{len(scenarios) - sum(labels_by_id.values())} safe)")

    print("[rjudge] Running behavioral judgment pass...")
    predictions = _run_judgment(scenarios=scenarios, cfg=cfg, output_dir=output_dir)

    print("[rjudge] Classifying TP/FP/FN/TN cells...")
    cells_result = classify_cells(labels=labels_by_id, predictions=predictions)
    print(f"[rjudge] Cells: {cells_result['counts']}")
    (output_dir / "cells.json").write_text(json.dumps(cells_result, indent=2))

    if cells_result["counts"]["FN"] < 20:
        print(
            f"[rjudge] WARNING: FN count is {cells_result['counts']['FN']} (<20). "
            "Dissociation test is underpowered. Continuing anyway."
        )

    print("[rjudge] Extracting activations (last prompt token, multi-layer)...")
    extraction = _run_extraction(scenarios=scenarios, cfg=cfg)

    print("[rjudge] Building train/val/test split within TP ∪ TN...")
    train_rel, val_rel, test_rel, fn_idx, tp_tn_idx = _build_split_on_subset(
        scenarios=scenarios,
        cells=cells_result["per_id"],
        split_cfg=cfg["split"],
        seed=cfg["seed"],
    )
    print(f"[rjudge] Split sizes: train={len(train_rel)}, val={len(val_rel)}, "
          f"test={len(test_rel)}, TP∪TN={len(tp_tn_idx)}, FN(held-out)={len(fn_idx)}")

    # Filter extraction to TP ∪ TN rows so the sweep runner sees clean data
    # (full coverage, uncontaminated test metrics, honest control-sanity check).
    subset_extraction = _subset_extraction(extraction, tp_tn_idx)
    subset_labels = [scenarios[i]["label"] for i in tp_tn_idx]

    probe_cfg = cfg["probe"]
    sweep_cfg = cfg["sweep"]
    runner = LayerProbeSweepRunner(
        probe=ProbeParams(
            probe_type=probe_cfg["probe_type"],
            epochs=probe_cfg["epochs"],
            learning_rate=probe_cfg["learning_rate"],
            weight_decay=probe_cfg["weight_decay"],
            bootstrap_samples=probe_cfg["bootstrap_samples"],
            early_stopping_patience=probe_cfg["early_stopping_patience"],
            seed=probe_cfg.get("seed"),
        ),
        sweep=SweepParams(
            activation_targets=list(cfg["extraction"]["activations"]),
            batch_size=sweep_cfg["batch_size"],
            selection_metric=sweep_cfg["selection_metric"],
        ),
    )
    sweep_result = runner.run(
        subset_extraction,
        train_indices=train_rel,
        val_indices=val_rel,
        test_indices=test_rel,
        labels=subset_labels,
    )

    best_probe = sweep_result.probes[sweep_result.best_key]
    print(f"[rjudge] Best layer: {sweep_result.best_key} "
          f"(val {sweep_cfg['selection_metric']}={sweep_result.best_score:.4f})")
    print(f"[rjudge] Test metrics: {sweep_result.test_metrics}")

    torch.save(
        best_probe.trainer.model.state_dict(),
        output_dir / "best_probe.pt",
    )

    print("[rjudge] Evaluating dissociation on FN cell...")
    dissoc = _evaluate_dissociation(
        extraction=extraction,
        best_key=sweep_result.best_key,
        best_probe=best_probe,
        scenarios=scenarios,
        cells=cells_result["per_id"],
        threshold=probe_cfg.get("threshold", 0.5),
    )

    results = {
        "run_name": cfg["run_name"],
        "model": cfg["model"]["model_name"],
        "cells": cells_result["counts"],
        "sweep": {
            "best_layer": sweep_result.best_key,
            "best_val_score": sweep_result.best_score,
            "test_metrics": {
                k: (v[0] if isinstance(v, tuple) else v)
                for k, v in sweep_result.test_metrics.items()
            },
            "controls": sweep_result.controls,
        },
        "dissociation": dissoc,
    }
    (output_dir / "results.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"[rjudge] Done. Results at {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
