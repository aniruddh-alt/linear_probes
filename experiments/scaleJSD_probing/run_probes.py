"""Linear probes for frequency category vs semantic category on ScaleJSD data.

Tests the normative argument from "From Tokens to Semantics":
  - Frequency probes should peak early (layers encode frequency first)
  - Semantic probes should peak late (layers encode semantics later)

Uses the sonde toolkit with binary classification:
  - Frequency: high=1, low=0 (natural binary)
  - Semantic: one-vs-rest per domain (5 binary sweeps)

Usage:
    # Single model, all probes
    python experiments/scaleJSD_probing/run_probes.py \
        --model EleutherAI/pythia-70m-deduped \
        --revision step143000 \
        --dataset-dir /path/to/scaleJSD/dataset/legacy/filtered \
        --output-dir artifacts/scaleJSD/pythia-70m/step143000

    # Specific probe only
    python experiments/scaleJSD_probing/run_probes.py \
        --model EleutherAI/pythia-70m-deduped \
        --probe frequency \
        --output-dir artifacts/scaleJSD/pythia-70m/step143000

    # Skip extraction (reuse cached activations)
    python experiments/scaleJSD_probing/run_probes.py \
        --model EleutherAI/pythia-70m-deduped \
        --skip-extraction \
        --output-dir artifacts/scaleJSD/pythia-70m/step143000
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch

from activation import ActivationExtractor
from core.configs import ExtractionParams, ModelParams, ProbeParams, SweepParams
from dataset import ProbingSampleBuilder
from probes import LayerProbeSweepRunner

# ── Model config ─────────────────────────────────────────────────────────────

MODEL_LAYERS = {
    "pythia-70m": 6,
    "pythia-160m": 12,
    "pythia-410m": 24,
    "pythia-1b": 16,
    "pythia-1.4b": 24,
    "pythia-2.8b": 32,
    "pythia-6.9b": 32,
    "pythia-12b": 36,
    "olmo-1b": 16,
    "olmo-7b": 32,
    "llama-3": 32,
    "llama-2": 32,
}

DATASET_NAMES = ["emotion", "medical", "legal", "scientific", "verb"]
DATASET_PATTERNS = [
    "{name}_ngrams_dedup_filtered.jsonl",
    "{name}_ngrams_filtered_pairs.jsonl",
    "{name}_filtered_pairs.jsonl",
]

SEED = 42


def infer_num_layers(model_id: str) -> int:
    for key, n in MODEL_LAYERS.items():
        if key in model_id.lower():
            return n
    return 12


# ── Data loading ─────────────────────────────────────────────────────────────


def find_dataset(dataset_dir: Path, name: str) -> Path | None:
    for pat in DATASET_PATTERNS:
        p = dataset_dir / pat.format(name=name)
        if p.exists():
            return p
    return None


def load_all_pairs(dataset_dir: Path) -> list[dict[str, Any]]:
    """Load all synonym pairs from all domains, tagging each with its domain."""
    all_pairs = []
    for name in DATASET_NAMES:
        path = find_dataset(dataset_dir, name)
        if path is None:
            print(f"  SKIP: {name} not found in {dataset_dir}")
            continue
        with open(path) as f:
            for line in f:
                row = json.loads(line.strip())
                row["domain"] = name
                all_pairs.append(row)
        print(
            f"  Loaded {name}: {sum(1 for p in all_pairs if p['domain'] == name)} pairs"
        )
    print(
        f"  Total: {len(all_pairs)} pairs across {len(set(p['domain'] for p in all_pairs))} domains"
    )
    return all_pairs


def build_frequency_samples(
    pairs: list[dict],
) -> tuple[list[dict], list[int], list[str]]:
    """Build samples for frequency probe: high=1, low=0.

    Text is truncated at [TERM] so token_index=-1 captures the target token.
    """
    records = []
    labels = []
    group_ids = []

    for i, pair in enumerate(pairs):
        pair_id = pair.get("synonym_pair_seed", f"pair_{i}")
        template = pair.get("sentence_template", "The word [TERM] means")

        # Find the [TERM] position and take text up to end of [TERM]
        term_pos = template.find("[TERM]")
        prefix = template[:term_pos] if term_pos >= 0 else template + " "

        for ngram, label in [
            (pair["high_freq_ngram"], 1),
            (pair["low_freq_ngram"], 0),
        ]:
            text = prefix + ngram
            records.append(
                {
                    "id": f"{pair_id}_{['low', 'high'][label]}",
                    "text": text,
                    "label": label,
                }
            )
            labels.append(label)
            group_ids.append(pair_id)

    return records, labels, group_ids


def build_semantic_samples(
    pairs: list[dict],
    target_domain: str,
) -> tuple[list[dict], list[int], list[str]]:
    """Build samples for semantic probe: target_domain=1, others=0.

    Uses both high and low freq variants to ensure the probe learns
    semantic category, not frequency.
    """
    records = []
    labels = []
    group_ids = []

    for i, pair in enumerate(pairs):
        pair_id = pair.get("synonym_pair_seed", f"pair_{i}")
        template = pair.get("sentence_template", "The word [TERM] means")
        label = 1 if pair["domain"] == target_domain else 0

        term_pos = template.find("[TERM]")
        prefix = template[:term_pos] if term_pos >= 0 else template + " "

        for ngram in [pair["high_freq_ngram"], pair["low_freq_ngram"]]:
            text = prefix + ngram
            records.append(
                {
                    "id": f"{pair_id}_{ngram}",
                    "text": text,
                    "label": label,
                }
            )
            labels.append(label)
            group_ids.append(pair_id)

    return records, labels, group_ids


# ── Extraction & probing ────────────────────────────────────────────────────


def extract_activations(
    records: list[dict],
    model_name: str,
    revision: str,
    num_layers: int,
    save_path: Path,
    batch_size: int = 4,
    dtype: str = "float16",
):
    """Extract activations for all layers, caching to disk."""
    builder = ProbingSampleBuilder.from_iterable(records)
    bundle = builder.to_samples(text_key="text")
    targets = [f"layers_output:{i}" for i in range(num_layers)]

    extractor = ActivationExtractor(
        model=ModelParams(
            model_name=model_name,
            dtype=dtype,
            revision=revision if revision != "main" else None,
        ),
        extraction=ExtractionParams(
            save_path=str(save_path),
            activations=targets,
            batch_size=batch_size,
            token_index=-1,
            to_cpu=True,
        ),
    )
    return extractor.extract(bundle)


def run_probe_sweep(
    extraction,
    labels: list[int],
    group_ids: list[str],
    num_layers: int,
    output_dir: Path,
    probe_name: str,
):
    """Train probes across all layers and save results."""
    from dataset.splitting import stratified_train_val_test_split

    train_idx, val_idx, test_idx = stratified_train_val_test_split(
        labels=labels,
        train_fraction=0.7,
        val_fraction=0.15,
        test_fraction=0.15,
        seed=SEED,
        group_ids=group_ids,
    )

    targets = [f"layers_output:{i}" for i in range(num_layers)]
    runner = LayerProbeSweepRunner(
        probe=ProbeParams(
            epochs=20,
            learning_rate=1e-3,
            weight_decay=0.01,
            seed=SEED,
            bootstrap_samples=200,
            early_stopping_patience=5,
        ),
        sweep=SweepParams(
            activation_targets=targets,
            batch_size=128,
        ),
    )

    result = runner.run(
        extraction,
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        labels=labels,
        group_ids=group_ids,
        manifest_path=str(output_dir / f"{probe_name}_manifest.json"),
    )

    # Save per-layer metrics CSV
    rows = []
    for key, p in result.probes.items():
        layer_idx = int(key.split(":")[-1])
        auroc = p.val_metrics.get("auroc", 0)
        acc = p.val_metrics.get("accuracy", 0)
        if isinstance(auroc, tuple):
            auroc = auroc[0]
        if isinstance(acc, tuple):
            acc = acc[0]
        rows.append(
            {
                "probe": probe_name,
                "layer": layer_idx,
                "val_auroc": auroc,
                "val_accuracy": acc,
            }
        )

    csv_path = output_dir / f"{probe_name}_layers.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["probe", "layer", "val_auroc", "val_accuracy"]
        )
        writer.writeheader()
        writer.writerows(rows)

    # Print summary
    print(f"\n{'=' * 50}")
    print(f"  {probe_name}: best={result.best_key} (auroc={result.best_score:.4f})")
    print(f"  Test: {result.test_metrics}")
    print(f"{'=' * 50}")
    for key, p in result.probes.items():
        auroc = p.val_metrics.get("auroc", 0)
        if isinstance(auroc, tuple):
            auroc = auroc[0]
        print(f"    {key:<20} auroc={auroc:.4f}")

    return result


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(
        description="ScaleJSD linear probes for frequency & semantic category"
    )
    p.add_argument("--model", default="EleutherAI/pythia-70m-deduped")
    p.add_argument("--revision", default="step143000")
    p.add_argument(
        "--dataset-dir", required=True, help="Path to ScaleJSD filtered JSONL dir"
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--probe",
        choices=["frequency", "semantic", "all"],
        default="all",
        help="Which probes to run",
    )
    p.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Reuse cached activations from output-dir",
    )
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--device", default=None, help="Override device (auto-detected)")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(args.dataset_dir)
    num_layers = infer_num_layers(args.model)

    print(f"Model: {args.model} @ {args.revision}")
    print(f"Layers: {num_layers}")
    print(f"Output: {output_dir}")

    # Load all pairs
    print("\nLoading datasets...")
    pairs = load_all_pairs(dataset_dir)
    if not pairs:
        raise RuntimeError(f"No datasets found in {dataset_dir}")

    # Build frequency samples (used for extraction too — covers all pairs)
    freq_records, freq_labels, freq_group_ids = build_frequency_samples(pairs)
    print(
        f"\nFrequency samples: {len(freq_records)} ({sum(freq_labels)} high, {len(freq_labels) - sum(freq_labels)} low)"
    )

    # Extract activations (shared across all probes since same text)
    act_path = output_dir / "activations"
    if args.skip_extraction and (act_path / "extraction.pt").exists():
        print(f"\nLoading cached activations from {act_path}")
        extraction = torch.load(act_path / "extraction.pt", weights_only=False)
    else:
        print("\nExtracting activations...")
        extraction = extract_activations(
            records=freq_records,
            model_name=args.model,
            revision=args.revision,
            num_layers=num_layers,
            save_path=act_path,
            batch_size=args.batch_size,
            dtype=args.dtype,
        )
        # Cache for reuse
        act_path.mkdir(parents=True, exist_ok=True)
        torch.save(extraction, act_path / "extraction.pt")
        print(f"Saved extraction to {act_path / 'extraction.pt'}")

    # ── Frequency probe ──────────────────────────────────────────────────
    if args.probe in ("frequency", "all"):
        print(f"\n{'#' * 50}")
        print("  FREQUENCY PROBE (high=1, low=0)")
        print(f"{'#' * 50}")
        run_probe_sweep(
            extraction=extraction,
            labels=freq_labels,
            group_ids=freq_group_ids,
            num_layers=num_layers,
            output_dir=output_dir,
            probe_name="frequency",
        )

    # ── Semantic probes (one-vs-rest) ────────────────────────────────────
    if args.probe in ("semantic", "all"):
        domains = sorted(set(p["domain"] for p in pairs))
        for domain in domains:
            _, sem_labels, sem_group_ids = build_semantic_samples(pairs, domain)
            n_pos = sum(sem_labels)
            n_neg = len(sem_labels) - n_pos
            print(f"\n{'#' * 50}")
            print(f"  SEMANTIC PROBE: {domain} vs rest ({n_pos} pos, {n_neg} neg)")
            print(f"{'#' * 50}")

            if n_pos < 10:
                print(f"  SKIP: too few positive samples ({n_pos})")
                continue

            run_probe_sweep(
                extraction=extraction,
                labels=sem_labels,
                group_ids=sem_group_ids,
                num_layers=num_layers,
                output_dir=output_dir,
                probe_name=f"semantic_{domain}",
            )

    # ── Combined summary CSV ─────────────────────────────────────────────
    all_csvs = sorted(output_dir.glob("*_layers.csv"))
    if all_csvs:
        combined = []
        for csv_path in all_csvs:
            with open(csv_path) as f:
                reader = csv.DictReader(f)
                combined.extend(reader)

        summary_path = output_dir / "all_probes_summary.csv"
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["probe", "layer", "val_auroc", "val_accuracy"]
            )
            writer.writeheader()
            writer.writerows(combined)
        print(f"\nCombined summary: {summary_path} ({len(combined)} rows)")

    print("\nDone.")


if __name__ == "__main__":
    main()
