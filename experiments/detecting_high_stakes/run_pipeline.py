"""High-stakes detection pipeline: attention probes on Llama 3.1 8B activations.

Full-sequence extraction with OOD evaluation on paper's external datasets.
"""

from __future__ import annotations

from pathlib import Path

import torch
from datasets import load_dataset

from activation import ActivationExtractor
from core.configs import ExtractionParams, ModelParams, ProbeParams, SweepParams
from dataset import ProbingSampleBuilder
from probes import LayerProbeSweepRunner
from probes.architectures import build_probe
from probes.linear import BinaryProbeTrainer

MODEL_NAME = "meta-llama/Llama-3.1-8B"
DATA_DIR = Path("experiments/models_under_pressure/data")
ARTIFACT_PATH = DATA_DIR / "activations"
MIDDLE_LAYERS = [10, 13, 15, 17, 20]
BATCH_SIZE = 1
TRAIN_SAMPLES = 1000  # paper shows 32 samples already helps; 1K is plenty
SEED = 42

OOD_DATASETS = [
    ("anthropic_hh_balanced", "test"),
    ("mt_balanced", "test"),
    ("toolace_balanced", "test"),
    ("mental_health_balanced", "test"),
    ("aya_redteaming_balanced", "test"),
]


def load_hf_split(
    config: str, split: str, max_samples: int | None = None
) -> tuple[list[dict], list[int]]:
    ds = load_dataset("Arrrlex/models-under-pressure", config, split=split)
    rows, labels = [], []
    for row in ds:  # type: ignore[union-attr]
        label = 1 if row["labels"] == "high-stakes" else 0  # type: ignore[index]
        rows.append({"id": row["ids"], "text": row["inputs"], "label": label})  # type: ignore[index]
        labels.append(label)
    if max_samples and len(rows) > max_samples:
        # Stratified subsample: keep class balance
        import random

        rng = random.Random(SEED)
        pos = [(r, lab) for r, lab in zip(rows, labels) if lab == 1]
        neg = [(r, lab) for r, lab in zip(rows, labels) if lab == 0]
        n_each = max_samples // 2
        rng.shuffle(pos)
        rng.shuffle(neg)
        sampled = pos[:n_each] + neg[:n_each]
        rng.shuffle(sampled)
        rows = [s[0] for s in sampled]
        labels = [s[1] for s in sampled]
    n_pos = sum(labels)
    print(
        f"  [{config}/{split}] {len(rows)} samples: {n_pos} high-stakes, {len(labels) - n_pos} low-stakes"
    )
    return rows, labels


def evaluate_ood(
    extractor: ActivationExtractor,
    best_key: str,
    probe_params: ProbeParams,
    best_probe_state: dict,
    input_dim: int,
) -> None:
    print("\n" + "=" * 60)
    print("OOD EVALUATION")
    print("=" * 60)
    print(f"{'Dataset':<32} {'AUROC':>8} {'Acc':>8} {'F1':>8}")
    print("-" * 60)

    for config, split in OOD_DATASETS:
        try:
            rows, labels = load_hf_split(config, split)
        except Exception as e:
            print(f"  {config:<30} SKIPPED: {e}")
            continue

        bundle = ProbingSampleBuilder.from_iterable(rows).to_samples(text_key="text")
        extraction = extractor.extract(bundle)

        from dataset.probing_dataset import ProbingDataset

        dataset = ProbingDataset.from_extraction_result(
            extraction,
            activation_key=best_key,
            labels=labels,
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=128,
            shuffle=False,
            collate_fn=_get_collate_fn(dataset),
        )

        model = build_probe(
            probe_params.probe_type, input_dim, **probe_params.probe_kwargs
        )
        model.load_state_dict(best_probe_state)
        evaluator = BinaryProbeTrainer(model=model, config=probe_params)
        metrics = evaluator.evaluate(loader)

        auroc = metrics.get("auroc", 0)
        acc = metrics.get("accuracy", 0)
        f1 = metrics.get("f1", 0)
        if isinstance(auroc, tuple):
            auroc = auroc[0]
        if isinstance(acc, tuple):
            acc = acc[0]
        if isinstance(f1, tuple):
            f1 = f1[0]
        print(f"  {config:<30} {auroc:>8.4f} {acc:>8.4f} {f1:>8.4f}")


def _get_collate_fn(dataset):
    if dataset.sequence_mode:
        from dataset.collate import sequence_collate_fn

        return sequence_collate_fn
    return None


def main() -> None:
    print("Loading training data...")
    rows, labels = load_hf_split("training", "train", max_samples=TRAIN_SAMPLES)

    bundle = ProbingSampleBuilder.from_iterable(rows).to_samples(text_key="text")
    train_idx, val_idx, test_idx = bundle.train_val_test_split(
        train_fraction=0.7,
        val_fraction=0.15,
        test_fraction=0.15,
        seed=SEED,
        group_ids=bundle.ids,
    )

    targets = [f"layers_output:{i}" for i in MIDDLE_LAYERS]
    extractor = ActivationExtractor(
        model=ModelParams(model_name=MODEL_NAME, dtype="bfloat16"),
        extraction=ExtractionParams(
            save_path=str(ARTIFACT_PATH),
            activations=targets,
            batch_size=BATCH_SIZE,
            token_index=None,  # full sequence
        ),
    )
    print(
        f"Extracting activations (full sequence, {TRAIN_SAMPLES} samples, {len(MIDDLE_LAYERS)} layers)..."
    )
    extraction = extractor.extract(bundle)

    probe_params = ProbeParams(
        epochs=20,
        learning_rate=1e-3,
        weight_decay=0.1,
        seed=SEED,
        bootstrap_samples=200,
        early_stopping_patience=5,
        probe_type="attention",
        probe_kwargs={"n_heads": 4},
    )
    runner = LayerProbeSweepRunner(
        probe=probe_params,
        sweep=SweepParams(activation_targets=targets, batch_size=128),
    )
    result = runner.run(
        extraction,
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        labels=labels,
        group_ids=bundle.ids,
    )

    print(f"\n{'=' * 60}")
    print("IN-DISTRIBUTION RESULTS")
    print(f"{'=' * 60}")
    print(
        f"Samples: {len(labels)} ({result.split_sizes[0]}/{result.split_sizes[1]}/{result.split_sizes[2]})"
    )
    print(f"Best layer: {result.best_key} (val_auroc={result.best_score:.4f})")
    print(f"\nTest metrics: {result.test_metrics}")
    print(f"Controls: {result.controls}")

    print(f"\n{'Layer':<20} {'val_auroc':>10} {'val_acc':>10}")
    print("-" * 42)
    for key, p in result.probes.items():
        auroc = p.val_metrics.get("auroc", 0)
        acc = p.val_metrics.get("accuracy", 0)
        if isinstance(auroc, tuple):
            auroc = auroc[0]
        if isinstance(acc, tuple):
            acc = acc[0]
        print(f"  {key:<18} {auroc:>10.4f} {acc:>10.4f}")

    # Save best probe
    best_probe = result.probes[result.best_key]
    probe_path = DATA_DIR / "best_probe.pt"
    probe_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_probe.trainer.model.state_dict(), probe_path)
    print(f"\nSaved best probe to {probe_path} (layer {result.best_key})")

    # OOD evaluation — reuse same model, extract only best layer
    ood_extractor = ActivationExtractor(
        model=ModelParams(model_name=MODEL_NAME, dtype="bfloat16"),
        extraction=ExtractionParams(
            save_path=str(DATA_DIR / "ood_activations"),
            activations=[result.best_key],
            batch_size=BATCH_SIZE,
            token_index=None,
        ),
    )
    input_dim = int(getattr(best_probe.trainer.model, "W_q").shape[0])
    evaluate_ood(
        ood_extractor,
        best_key=result.best_key,
        probe_params=probe_params,
        best_probe_state=best_probe.trainer.model.state_dict(),
        input_dim=input_dim,
    )


if __name__ == "__main__":
    main()
