"""End-to-end refusal probe: extract activations from Qwen and train linear probes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from datasets import load_dataset
from transformers import AutoTokenizer

from activation import ActivationExtractor
from core.configs import ExtractionParams, ModelParams, ProbeParams, SweepParams
from dataset import ProbingSampleBuilder
from probes import LayerProbeSweepRunner, ProbeAnalyzer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
ARTIFACT_PATH = Path("artifacts/qwen_refusal_activations")
NUM_PER_CLASS = 128
NUM_LAYERS = 8
BATCH_SIZE = 4
SEED = 0
FORCED_PREFIX = "Sure, here is"


# ── Dataset helpers ──────────────────────────────────────────────────────────


def _load_harmful(limit: int) -> list[str]:
    ds = load_dataset("walledai/AdvBench", split="train")
    out = [cast(dict[str, Any], row)["goal"] for row in ds if cast(dict[str, Any], row).get("goal", "").strip()][:limit]
    if len(out) < limit:
        raise ValueError(f"AdvBench yielded only {len(out)}/{limit} rows.")
    return out


def _load_harmless(limit: int) -> list[str]:
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    out = [cast(dict[str, Any], row)["instruction"] for row in ds if cast(dict[str, Any], row).get("instruction", "").strip()][:limit]
    if len(out) < limit:
        raise ValueError(f"Alpaca yielded only {len(out)}/{limit} rows.")
    return out


def _format_prompt(instruction: str) -> str:
    return f"User: {instruction}\nAssistant: {FORCED_PREFIX}"


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    # 1. Build dataset
    harmful = _load_harmful(NUM_PER_CLASS)
    harmless = _load_harmless(NUM_PER_CLASS)

    records = [
        {"id": f"advbench-{i}", "text": _format_prompt(t), "label": 1}
        for i, t in enumerate(harmful)
    ] + [
        {"id": f"alpaca-{i}", "text": _format_prompt(t), "label": 0}
        for i, t in enumerate(harmless)
    ]

    bundle = ProbingSampleBuilder.from_iterable(records).to_samples(text_key="text")
    labels = [int(l) for l in bundle.labels if l is not None]
    train_idx, val_idx, test_idx = bundle.train_val_test_split(
        train_fraction=0.7, val_fraction=0.15, test_fraction=0.15,
        seed=SEED, group_ids=bundle.ids,
    )

    # 2. Resolve token position for forced-prefix boundary
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    prefix_tokens = tokenizer.encode(FORCED_PREFIX, add_special_tokens=False)
    token_index = -len(prefix_tokens)

    # 3. Extract activations
    targets = [f"layers_output:{i}" for i in range(NUM_LAYERS)]
    extractor = ActivationExtractor(
        model=ModelParams(model_name=MODEL_NAME),
        extraction=ExtractionParams(
            save_path=str(ARTIFACT_PATH),
            activations=targets,
            batch_size=BATCH_SIZE,
            token_index=token_index,
        ),
    )
    extraction = extractor.extract(bundle)

    # 4. Train probes across layers
    runner = LayerProbeSweepRunner(
        probe=ProbeParams(epochs=10, learning_rate=1e-3, seed=SEED, bootstrap_samples=200),
        sweep=SweepParams(activation_targets=targets, batch_size=BATCH_SIZE),
    )
    result = runner.run(
        extraction,
        train_indices=train_idx, val_indices=val_idx, test_indices=test_idx,
        labels=labels, group_ids=bundle.ids,
    )

    # 5. Report
    train_n, val_n, test_n = result.split_sizes
    print(f"\nSweep complete: {len(labels)} samples ({train_n}/{val_n}/{test_n})")
    print(f"Best layer: {result.best_key} (val_auroc={result.best_score:.4f})")
    for key, p in result.probes.items():
        auroc = p.val_metrics["auroc"]
        print(f"  {key}: auroc={auroc:.4f}")

    # 6. Visualize
    analyzer = ProbeAnalyzer(list(result.probes.values()))
    analyzer.auroc_analysis()
    analyzer.cosine_similarity_analysis()


if __name__ == "__main__":
    main()
