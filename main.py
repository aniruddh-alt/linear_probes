from activation import ActivationExtractor
from configs import (
    ActivationConfig,
    LayerProbeSweepConfig,
    ModelConfig,
    ProbeConfig,
)
from dataset import ProbingSampleBuilder
from probes import LayerProbeSweepRunner

DATASET_PATH = "medical_ngrams_dedup_filtered.jsonl"
NUM_LAYERS = 6


def _metric_value(metrics: dict[str, float | tuple[float, float]], name: str) -> float:
    value = metrics[name]
    if isinstance(value, tuple):
        raise TypeError(f"Expected scalar metric '{name}', got interval value {value}.")
    return float(value)


def main() -> None:
    records = [
        {"id": "sample1", "text": "This is a sample text.", "label": 0},
        {"id": "sample2", "text": "This is another sample text.", "label": 1},
        {"id": "sample3", "text": "This is yet another sample text.", "label": 0},
        {"id": "sample4", "text": "This is a different sample text.", "label": 1},
    ]
    print(f"Loaded {len(records)} samples ({len(records) // 2} synonym pairs)")

    bundle = ProbingSampleBuilder.from_iterable(records).to_samples(text_key="text")

    model_config = ModelConfig(model_name="EleutherAI/pythia-70m")

    activation_config = ActivationConfig(
        model_config=model_config,
        save_path="artifacts/activations",
        activations=[f"layers_output:{i}" for i in range(NUM_LAYERS)],
        batch_size=8,
        token_index=-1,
        to_cpu=True,
    )
    extractor = ActivationExtractor(activation_config)
    result = extractor.extract(bundle)

    sweep_config = LayerProbeSweepConfig(
        activation_targets=list(range(NUM_LAYERS)),
        batch_size=8,
        train_fraction=0.8,
        selection_metric="auroc",
        probe=ProbeConfig(epochs=30, learning_rate=0.01),
    )
    sweep = LayerProbeSweepRunner(sweep_config)
    probes = sweep.run(result)
    if not probes:
        raise ValueError("Probe sweep returned no trained probes.")
    chooser = max if sweep_config.maximize_metric else min
    best_key = chooser(
        probes,
        key=lambda key: _metric_value(
            probes[key].metrics, sweep_config.selection_metric
        ),
    )
    best_probe = probes[best_key]
    total = len(records)
    train_size = min(max(int(total * sweep_config.train_fraction), 1), total - 1)
    val_size = total - train_size
    print(
        f"\nLayerwise probe sweep complete ({len(records)} samples, "
        f"{train_size} train / {val_size} val)"
    )
    print(
        f"Best layer: {best_key} "
        f"({sweep_config.selection_metric}="
        f"{_metric_value(best_probe.metrics, sweep_config.selection_metric):.4f})"
    )
    print(f"Best direction shape: {tuple(best_probe.direction.shape)}")
    for key, probe_result in probes.items():
        print(
            f"{key}: acc={_metric_value(probe_result.metrics, 'accuracy'):.4f} "
            f"f1={_metric_value(probe_result.metrics, 'f1'):.4f} "
            f"auroc={_metric_value(probe_result.metrics, 'auroc'):.4f}"
        )


if __name__ == "__main__":
    main()
