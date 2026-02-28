"""Steps 4-5: Extract activations from labeled data and train refusal probes."""

from __future__ import annotations

import json
from pathlib import Path

from activation import ActivationExtractor
from core.configs import ExtractionParams, ModelParams, ProbeParams, SweepParams
from dataset import ProbingSampleBuilder
from probes import LayerProbeSweepRunner, ProbeAnalyzer

DATA_DIR = Path("examples/refusal_probing/data")
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
ARTIFACT_PATH = DATA_DIR / "activations"
NUM_LAYERS = 28
BATCH_SIZE = 8
SEED = 42


def load_labeled_data() -> tuple[list[dict], list[int]]:
    """Load labeled JSONL and convert refusal labels to binary."""
    # Also load responses for prompt+response concatenation
    responses = {}
    with (DATA_DIR / "responses.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            responses[r["sample_id"]] = r["response"]

    rows = []
    with (DATA_DIR / "labeled.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            label_str = row["refusal_label"]
            label = 1 if "refusal" in label_str and "non-refusal" not in label_str else 0
            sid = row["sample_id"]
            prompt = row["original_prompt"]
            response = responses.get(sid, row.get("original_response", ""))
            # Probe on prompt+response so activations capture refusal behavior
            rows.append({
                "id": sid,
                "text": f"{prompt}\n{response}",
                "label": label,
            })
    labels = [r["label"] for r in rows]
    print(f"Loaded {len(rows)} samples: {sum(labels)} refusal, {len(labels) - sum(labels)} non-refusal")
    return rows, labels


def main() -> None:
    # 1. Load labeled data
    rows, labels = load_labeled_data()

    builder = ProbingSampleBuilder.from_iterable(rows)
    bundle = builder.to_samples(text_key="text")
    train_idx, val_idx, test_idx = bundle.train_val_test_split(
        train_fraction=0.7, val_fraction=0.15, test_fraction=0.15,
        seed=SEED, group_ids=bundle.ids,
    )

    # 2. Extract activations
    targets = [f"layers_output:{i}" for i in range(NUM_LAYERS)]
    extractor = ActivationExtractor(
        model=ModelParams(model_name=MODEL_NAME, dtype="bfloat16"),
        extraction=ExtractionParams(
            save_path=str(ARTIFACT_PATH),
            activations=targets,
            batch_size=BATCH_SIZE,
            token_index=-1,
        ),
    )
    extraction = extractor.extract(bundle)

    # 3. Train probes
    runner = LayerProbeSweepRunner(
        probe=ProbeParams(
            epochs=20,
            learning_rate=1e-3,
            weight_decay=0.1,
            seed=SEED,
            bootstrap_samples=200,
            early_stopping_patience=5,
        ),
        sweep=SweepParams(activation_targets=targets, batch_size=32),
    )
    result = runner.run(
        extraction,
        train_indices=train_idx, val_indices=val_idx, test_indices=test_idx,
        labels=labels, group_ids=bundle.ids,
    )

    # 4. Report
    train_n, val_n, test_n = result.split_sizes
    print(f"\nSweep complete: {len(labels)} samples ({train_n}/{val_n}/{test_n})")
    print(f"Best layer: {result.best_key} (val_auroc={result.best_score:.4f})")
    print(f"\nTest metrics (best layer): {result.test_metrics}")
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

    # 5. Visualize
    analyzer = ProbeAnalyzer(list(result.probes.values()))
    analyzer.auroc_analysis()
    analyzer.cosine_similarity_analysis()


if __name__ == "__main__":
    main()
