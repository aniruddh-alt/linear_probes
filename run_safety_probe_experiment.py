"""Minimal safety probe experiment using config-driven APIs only."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from activation import ActivationExtractor
from core.configs import ExtractionParams, ModelParams, ProbeParams, SweepParams
from dataset import ProbingDataset, ProbingSampleBuilder
from probes import LayerProbeSweepRunner
from probes.analyze import ProbeAnalyzer

if __name__ == "__main__":
    torch.manual_seed(args.seed)

    model_name = ""
    batch_size = 2

    train_val_records = splits.train_records + splits.val_records
    train_val_bundle = ProbingSampleBuilder.from_iterable(train_val_records).to_samples(
        text_key="text"
    )
    train_val_labels = [
        int(label) for label in train_val_bundle.labels if label is not None
    ]

    train_indices = list(range(len(splits.train_records)))
    val_indices = list(range(len(splits.train_records), len(train_val_records)))

    extractor = ActivationExtractor(
        model=ModelParams(
            model_name=model_name,
            dtype="float32",
            attn_implementation="eager",
        ),
        extraction=ExtractionParams(
            save_path="artifacts/qwen_safety_trainval",
            activations=["layers_output:*"],
            batch_size=batch_size,
            token_index=-1,
            to_cpu=True,
        ),
    )

    print("Extracting train+val activations...")
    train_val_extraction = extractor.extract(train_val_bundle)
    train_val_extraction["labels"] = train_val_labels

    sweep = LayerProbeSweepRunner(
        probe=ProbeParams(
            epochs=60,
            learning_rate=1e-3,
            weight_decay=0.05,
            max_grad_norm=1.0,
            early_stopping_patience=5,
            early_stopping_min_delta=1e-4,
            seed=args.seed,
            device=args.device,
        ),
        sweep=SweepParams(
            activation_targets=list(train_val_extraction["requested"]),
            batch_size=batch_size,
            selection_metric="auroc",
        ),
    )

    sweep_result = sweep.run(
        train_val_extraction,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=val_indices,
    )
    probes = list(sweep_result.probes.values())
    analyzer = ProbeAnalyzer(probes)
    best_probe = analyzer.auroc_analysis()
    analyzer.cosine_similarity_analysis()

    print(
        f"Best probe: {best_probe.activation_key} "
        f"(val_auroc={float(best_probe.val_metrics['auroc']):.4f}, "
        f"val_accuracy={float(best_probe.val_metrics['accuracy']):.4f})"
    )

    eval_harm_bundle = ProbingSampleBuilder.from_iterable(
        splits.eval_harmful_records
    ).to_samples(text_key="text")
    eval_benign_bundle = ProbingSampleBuilder.from_iterable(
        splits.eval_harmless_records
    ).to_samples(text_key="text")
    eval_harm_labels = [
        int(label) for label in eval_harm_bundle.labels if label is not None
    ]
    eval_benign_labels = [
        int(label) for label in eval_benign_bundle.labels if label is not None
    ]

    print("Extracting eval activations for best layer...")
    eval_harm = extractor.extract(
        eval_harm_bundle, activations=[best_probe.activation_key]
    )
    eval_harm["labels"] = eval_harm_labels
    eval_benign = extractor.extract(
        eval_benign_bundle, activations=[best_probe.activation_key]
    )
    eval_benign["labels"] = eval_benign_labels

    harm_ds = ProbingDataset.from_extraction_result(
        eval_harm, activation_key=best_probe.activation_key
    )
    harm_metrics = best_probe.trainer.evaluate(
        DataLoader(harm_ds, batch_size=batch_size, shuffle=False)
    )

    balanced_features = torch.cat(
        (
            eval_harm["activations"][best_probe.activation_key],
            eval_benign["activations"][best_probe.activation_key],
        ),
        dim=0,
    )
    balanced_labels = eval_harm_labels + eval_benign_labels
    balanced_ds = ProbingDataset(features=balanced_features, labels=balanced_labels)
    balanced_metrics = best_probe.trainer.evaluate(
        DataLoader(balanced_ds, batch_size=batch_size, shuffle=False)
    )

    print(
        "Eval harmful-only:",
        f"accuracy={float(harm_metrics['accuracy']):.4f}",
        f"recall={float(harm_metrics['recall']):.4f}",
    )
    print(
        "Eval balanced:",
        f"accuracy={float(balanced_metrics['accuracy']):.4f}",
        f"auroc={float(balanced_metrics['auroc']):.4f}",
    )
