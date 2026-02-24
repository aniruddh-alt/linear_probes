from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from activation import ActivationExtractor
from configs import ActivationConfig, LayerProbeSweepConfig, ModelConfig, ProbeConfig
from dataset import ProbingSampleBuilder
from probes import LayerProbeSweepRunner, ProbeAnalyzer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
ARTIFACT_PATH = Path("artifacts/qwen_refusal_activations")
NUM_PER_CLASS = 128
NUM_LAYERS = 8
BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 1e-2
TRAIN_FRACTION = 0.7
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15
SEED = 0
FORCED_ASSISTANT_PREFIX = "Sure, here is"
INCLUDE_ALPACA_INPUT_CONTEXT = False
ALLOW_SOURCE_LABEL_CONFOUND = True
STRICT_DATASET_VALIDATION = False


def _metric_value(metrics: dict[str, float | tuple[float, float]], name: str) -> float:
    value = metrics.get(name)
    if value is None:
        raise KeyError(
            f"Missing expected metric '{name}'. Available: {sorted(metrics.keys())}"
        )
    if isinstance(value, tuple):
        raise TypeError(f"Expected scalar metric '{name}', got interval value {value}.")
    return float(value)


def _first_present(record: dict[str, object], keys: list[str]) -> str | None:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _load_advbench_harmful(limit: int) -> list[str]:
    dataset = load_dataset("walledai/AdvBench", split="train")
    harmful: list[str] = []
    for row in dataset:
        text = _first_present(dict(row), ["goal", "instruction", "prompt", "text"])
        if text is None:
            continue
        harmful.append(text)
        if len(harmful) >= limit:
            break
    if len(harmful) < limit:
        raise ValueError(
            f"AdvBench only yielded {len(harmful)} usable rows (< {limit})."
        )
    return harmful


def _load_alpaca_harmless(limit: int) -> list[str]:
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    harmless: list[str] = []
    for row in dataset:
        record = dict(row)
        instruction = _first_present(record, ["instruction", "text", "prompt"])
        if instruction is None:
            continue
        extra = _first_present(record, ["input"])
        full_instruction = (
            f"{instruction}\n{extra}"
            if INCLUDE_ALPACA_INPUT_CONTEXT and extra is not None
            else instruction
        )
        harmless.append(full_instruction)
        if len(harmless) >= limit:
            break
    if len(harmless) < limit:
        raise ValueError(
            f"Alpaca only yielded {len(harmless)} usable rows (< {limit})."
        )
    return harmless


def _assistant_turn_prompt(instruction: str, forced_prefix: str) -> str:
    return f"User: {instruction}\nAssistant: {forced_prefix}"


def _source_from_id(sample_id: str) -> str:
    return sample_id.split("-", maxsplit=1)[0]


def _validate_dataset_confounding(records: list[dict[str, object]]) -> list[str]:
    source_to_labels: dict[str, Counter[int]] = defaultdict(Counter)
    marker_counts: dict[int, int] = defaultdict(int)
    class_counts: Counter[int] = Counter()
    warnings: list[str] = []

    for record in records:
        sample_id = str(record["id"])
        label = int(record["label"])
        text = str(record["text"])
        source_to_labels[_source_from_id(sample_id)][label] += 1
        class_counts[label] += 1
        if "additional context:" in text.lower():
            marker_counts[label] += 1

    deterministic_sources = [
        source
        for source, counts in source_to_labels.items()
        if len(counts.keys()) == 1 and sum(counts.values()) >= 10
    ]
    if deterministic_sources and not ALLOW_SOURCE_LABEL_CONFOUND:
        details = ", ".join(
            f"{source} -> label {next(iter(source_to_labels[source].keys()))}"
            for source in sorted(deterministic_sources)
        )
        message = (
            "Source-label confound detected: some sources map to a single label "
            f"({details}). This will usually produce inflated AUROC. "
            "Set ALLOW_SOURCE_LABEL_CONFOUND=True only for debugging."
        )
        if STRICT_DATASET_VALIDATION:
            raise ValueError(message)
        warnings.append(message)

    if class_counts:
        for label in sorted(class_counts):
            marker_rate = marker_counts[label] / class_counts[label]
            if marker_rate > 0.4:
                message = (
                    "Detected formatting artifact strongly associated with one class "
                    f"(label={label}, 'Additional context' rate={marker_rate:.2%})."
                )
                if STRICT_DATASET_VALIDATION:
                    raise ValueError(message)
                warnings.append(message)
    return warnings


def main() -> None:
    harmful = _load_advbench_harmful(NUM_PER_CLASS)
    harmless = _load_alpaca_harmless(NUM_PER_CLASS)
    records: list[dict[str, object]] = []
    for idx, text in enumerate(harmful):
        records.append(
            {
                "id": f"advbench-{idx}",
                "text": _assistant_turn_prompt(text, FORCED_ASSISTANT_PREFIX),
                "label": 1,
            }
        )
    for idx, text in enumerate(harmless):
        records.append(
            {
                "id": f"alpaca-{idx}",
                "text": _assistant_turn_prompt(text, FORCED_ASSISTANT_PREFIX),
                "label": 0,
            }
        )
    warnings = _validate_dataset_confounding(records)
    for warning in warnings:
        print(f"WARNING: {warning}")

    bundle = ProbingSampleBuilder.from_iterable(records).to_samples(text_key="text")
    labels = [int(label) for label in bundle.labels if label is not None]
    train_indices, val_indices, test_indices = bundle.train_val_test_split(
        train_fraction=TRAIN_FRACTION,
        val_fraction=VAL_FRACTION,
        test_fraction=TEST_FRACTION,
        seed=SEED,
        group_ids=bundle.ids,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    prefix_tokens = tokenizer.encode(FORCED_ASSISTANT_PREFIX, add_special_tokens=False)
    if not prefix_tokens:
        raise ValueError("Forced assistant prefix must tokenize to at least one token.")
    response_token_index = -len(prefix_tokens)

    model_config = ModelConfig(model_name=MODEL_NAME)
    activation_targets = [f"layers_output:{i}" for i in range(NUM_LAYERS)]
    activation_config = ActivationConfig(
        model_config=model_config,
        save_path=ARTIFACT_PATH,
        activations=activation_targets,
        batch_size=BATCH_SIZE,
        token_index=response_token_index,
        to_cpu=True,
    )
    extractor = ActivationExtractor(activation_config)
    extraction = extractor.extract(bundle)

    sweep_config = LayerProbeSweepConfig(
        activation_targets=[f"layers_output:{i}" for i in range(NUM_LAYERS)],
        batch_size=BATCH_SIZE,
        train_fraction=TRAIN_FRACTION,
        val_fraction=VAL_FRACTION,
        test_fraction=TEST_FRACTION,
        split_seed=SEED,
        selection_metric="auroc",
        enforce_control_sanity=False,
        probe=ProbeConfig(
            early_stopping_min_delta=0.0005,
            weight_decay=0.1,
            early_stopping_patience=5,
            epochs=min(EPOCHS, 10),
            learning_rate=min(LEARNING_RATE, 1e-3),
            seed=SEED,
            bootstrap_samples=200,
        ),
    )
    sweep_result = LayerProbeSweepRunner(sweep_config).run(
        extraction,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        labels=labels,
        group_ids=bundle.ids,
    )

    probes = sweep_result.probes
    best_probe = probes[sweep_result.best_key]
    train_size, val_size, test_size = sweep_result.split_sizes
    print(
        f"\nLayerwise probe sweep complete ({len(labels)} samples, "
        f"{train_size} train / {val_size} val / {test_size} test)"
    )
    print(
        f"Activation token index (assistant first token): {response_token_index} "
        f"for prefix '{FORCED_ASSISTANT_PREFIX}'"
    )
    print(
        f"Best layer: {sweep_result.best_key} "
        f"(val_{sweep_result.best_metric}={sweep_result.best_score:.4f}, "
        f"test_auroc={_metric_value(sweep_result.test_metrics, 'auroc'):.4f})"
    )
    print(f"Best direction shape: {tuple(best_probe.direction.shape)}")
    for key, probe_result in probes.items():
        print(
            f"{key}: val_acc={_metric_value(probe_result.val_metrics, 'accuracy'):.4f} "
            f"val_f1={_metric_value(probe_result.val_metrics, 'f1'):.4f} "
            f"val_auroc={_metric_value(probe_result.val_metrics, 'auroc'):.4f}"
        )

    analyzer = ProbeAnalyzer(list(probes.values()))
    best_probe = analyzer.auroc_analysis()
    heatmap = analyzer.cosine_similarity_analysis()


if __name__ == "__main__":
    main()
