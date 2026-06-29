"""YAML-driven experiment runner: the single entry-point for API and CLI."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from sonde.core.configs.aliases import resolve_config_alias
from sonde.core.configs.base import BaseConfig
from sonde.core.configs.diff_means_config import DiffMeansConfig
from sonde.core.configs.extract_config import ExtractConfig
from sonde.core.configs.generate_config import GenerateConfig
from sonde.core.configs.overrides import apply_dot_overrides
from sonde.core.configs.params.extraction_params import ExtractionParams
from sonde.core.configs.pipeline_config import PipelineConfig
from sonde.core.configs.probe_config import ProbeConfig

StageConfig = (
    GenerateConfig | ExtractConfig | ProbeConfig | DiffMeansConfig | PipelineConfig
)

STAGE_CONFIG_MAP: dict[str, type[BaseConfig]] = {
    "generate": GenerateConfig,
    "extract": ExtractConfig,
    "probe_sweep": ProbeConfig,
    "diff_means": DiffMeansConfig,
    "pipeline": PipelineConfig,
}


@dataclass
class RunResult:
    """Structured result returned by every experiment run."""

    summary: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)


def load_run_config(
    config_path: str | Path,
    *,
    overrides: dict[str, str] | None = None,
    aliases_path: str | Path | None = None,
) -> BaseConfig:
    """Load a stage config from YAML, resolving aliases and applying overrides."""
    resolved_path = resolve_config_alias(str(config_path), aliases_path)
    raw = OmegaConf.load(resolved_path)
    raw_dict: dict = OmegaConf.to_container(raw, resolve=True)  # type: ignore[assignment]
    if overrides:
        raw_dict = apply_dot_overrides(raw_dict, overrides)
    action = raw_dict.get("action", "probe_sweep")
    config_cls = STAGE_CONFIG_MAP.get(action)
    if config_cls is None:
        raise ValueError(
            f"Unknown action '{action}'. "
            f"Available: {', '.join(sorted(STAGE_CONFIG_MAP))}"
        )
    return config_cls.from_dict(raw_dict)


def run_experiment(
    *,
    config_path: str | Path,
    overrides: dict[str, str] | None = None,
    aliases_path: str | Path | None = None,
) -> RunResult:
    """Run an experiment from a YAML config file."""
    cfg = load_run_config(config_path, overrides=overrides, aliases_path=aliases_path)
    return dispatch_action(cfg)


def dispatch_action(cfg: BaseConfig) -> RunResult:
    """Dispatch to the appropriate action handler based on config type."""
    if isinstance(cfg, GenerateConfig):
        return _action_generate(cfg)
    if isinstance(cfg, ExtractConfig):
        return _action_extract(cfg)
    if isinstance(cfg, ProbeConfig):
        return _action_probe_sweep(cfg)
    if isinstance(cfg, DiffMeansConfig):
        return _action_diff_means(cfg)
    if isinstance(cfg, PipelineConfig):
        return _action_pipeline(cfg)
    raise ValueError(f"Unhandled config type: {type(cfg).__name__}")


def _action_generate(cfg: GenerateConfig) -> RunResult:
    """Generate model responses for prompts from a JSONL file."""
    input_path = cfg.io.input_path
    if not input_path:
        raise ValueError("io.input_path is required for the generate action.")

    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    output_dir = Path(cfg.io.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "responses.jsonl"

    from sonde.dataset import ProbingSampleBuilder
    from sonde.generation import ResponseGenerator

    builder = ProbingSampleBuilder.from_file(input_file)
    bundle = builder.to_samples(text_key="prompt", label_key=None, id_key=None)

    generator = ResponseGenerator(
        model=cfg.model,
        generation=cfg.generation,
        steering=cfg.steering,
    )
    result = generator.generate(bundle)
    result.to_jsonl(output_file)

    return RunResult(
        summary={
            "run_name": cfg.run_name,
            "action": cfg.action,
            "model": cfg.model.model_name,
            "num_samples": len(result.prompts),
            "status": "completed",
        },
        artifacts={"responses": str(output_file)},
    )


def _action_extract(cfg: ExtractConfig) -> RunResult:
    """Extract activations from a model for samples in input_path."""
    from sonde.activation import ActivationExtractor
    from sonde.dataset import ProbingSampleBuilder

    input_path = Path(cfg.io.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    builder = ProbingSampleBuilder.from_file(input_path)
    bundle = builder.to_samples(text_key="text")

    # The extract action exists to write an artifact; default a path when the
    # config leaves it unset so the result is always reusable downstream.
    if not cfg.extraction.save_path:
        cfg.extraction.save_path = str(Path(cfg.io.output_dir) / "extraction")

    extractor = ActivationExtractor(model=cfg.model, extraction=cfg.extraction)
    extraction = extractor.extract(bundle)

    storage = extraction.get("storage", {})
    manifest_path = storage.get("manifest_path", cfg.extraction.save_path)

    return RunResult(
        summary={"action": "extract", "num_samples": len(bundle.ids)},
        artifacts={"extraction_path": str(manifest_path)},
        metrics={"num_layers": len(extraction["requested"])},
    )


def _action_probe_sweep(cfg: ProbeConfig) -> RunResult:
    """Run probe sweep on pre-extracted activations."""
    raise NotImplementedError("probe_sweep action not yet wired via experiment runner")


def _action_diff_means(cfg: DiffMeansConfig) -> RunResult:
    """Placeholder for diff-means action wiring."""
    raise NotImplementedError("diff_means action not yet wired via experiment runner")


def _action_pipeline(cfg: PipelineConfig) -> RunResult:
    """End-to-end pipeline: load dataset -> extract -> probe sweep -> OOD eval."""
    import random

    import torch
    from datasets import load_dataset

    from sonde.activation import ActivationExtractor
    from sonde.dataset import ProbingSampleBuilder
    from sonde.dataset.probing_dataset import ProbingDataset
    from sonde.probes import LayerProbeSweepRunner
    from sonde.probes.architectures import build_probe
    from sonde.probes.linear import BinaryProbeTrainer

    output_dir = Path(cfg.io.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load dataset ---
    ds = load_dataset(cfg.dataset.path, cfg.dataset.config, split=cfg.dataset.split)
    rows, labels = [], []
    for row in ds:  # type: ignore[union-attr]
        raw_label = row[cfg.dataset.label_key]  # type: ignore[index]
        label = cfg.dataset.label_map.get(str(raw_label), int(raw_label))
        text = row[cfg.dataset.text_key]  # type: ignore[index]
        row_id = row.get(cfg.dataset.id_key, str(len(rows)))  # type: ignore[union-attr]
        rows.append({"id": str(row_id), "text": text, "label": label})
        labels.append(label)

    if cfg.dataset.max_samples and len(rows) > cfg.dataset.max_samples:
        rng = random.Random(cfg.seed)
        pos = [(r, l) for r, l in zip(rows, labels, strict=True) if l == 1]
        neg = [(r, l) for r, l in zip(rows, labels, strict=True) if l == 0]
        n_each = cfg.dataset.max_samples // 2
        rng.shuffle(pos)
        rng.shuffle(neg)
        sampled = pos[:n_each] + neg[:n_each]
        rng.shuffle(sampled)
        rows = [s[0] for s in sampled]
        labels = [s[1] for s in sampled]

    n_pos = sum(labels)
    print(
        f"Loaded {len(rows)} samples: {n_pos} positive, {len(labels) - n_pos} negative"
    )

    bundle = ProbingSampleBuilder.from_iterable(rows).to_samples(text_key="text")
    train_idx, val_idx, test_idx = bundle.train_val_test_split(
        train_fraction=cfg.split.train_fraction,
        val_fraction=cfg.split.val_fraction,
        test_fraction=cfg.split.test_fraction,
        seed=cfg.seed,
        group_ids=bundle.ids,
    )

    # --- Extract activations ---
    targets = cfg.sweep.activation_targets or [f"layers_output:{i}" for i in cfg.layers]
    cfg.extraction.activations = targets
    cfg.extraction.save_path = cfg.extraction.save_path or str(
        output_dir / "activations"
    )

    extractor = ActivationExtractor(model=cfg.model, extraction=cfg.extraction)
    print(f"Extracting activations ({len(targets)} layers, {len(rows)} samples)...")
    extraction = extractor.extract(bundle)

    # --- Probe sweep ---
    sweep_params = cfg.sweep
    if not sweep_params.activation_targets:
        sweep_params.activation_targets = targets

    runner = LayerProbeSweepRunner(probe=cfg.probe, sweep=sweep_params)
    result = runner.run(
        extraction,
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        labels=labels,
        group_ids=bundle.ids,
    )

    # --- Print results ---
    print(f"\nBest layer: {result.best_key} (val_auroc={result.best_score:.4f})")
    print(f"Test metrics: {result.test_metrics}")
    print(f"Controls: {result.controls}")

    for key, p in result.probes.items():
        auroc = p.val_metrics.get("auroc", 0)
        if isinstance(auroc, tuple):
            auroc = auroc[0]
        print(f"  {key}: val_auroc={auroc:.4f}")

    # Save best probe
    best_probe = result.probes[result.best_key]
    probe_path = output_dir / "best_probe.pt"
    torch.save(best_probe.trainer.model.state_dict(), probe_path)

    # --- OOD evaluation ---
    ood_metrics: dict[str, Any] = {}
    if cfg.dataset.ood_configs:
        print(f"\n{'=' * 60}\nOOD EVALUATION\n{'=' * 60}")
        print(f"{'Dataset':<32} {'AUROC':>8} {'Acc':>8} {'F1':>8}")
        print("-" * 60)

        m = best_probe.trainer.model
        w = getattr(m, "W_q", None)
        if w is None:
            w = m.linear.weight  # type: ignore[union-attr]
        input_dim = int(w.shape[0])  # type: ignore[index]
        best_state = best_probe.trainer.model.state_dict()

        ood_extractor = ActivationExtractor(
            model=cfg.model,
            extraction=ExtractionParams(
                save_path=str(output_dir / "ood_activations"),
                activations=[result.best_key],
                batch_size=cfg.extraction.batch_size,
                token_index=cfg.extraction.token_index,
            ),
        )

        for ood_config in cfg.dataset.ood_configs:
            try:
                ood_ds = load_dataset(
                    cfg.dataset.path, ood_config, split=cfg.dataset.ood_split
                )
            except Exception as e:
                print(f"  {ood_config:<30} SKIPPED: {e}")
                continue

            ood_rows, ood_labels = [], []
            for row in ood_ds:  # type: ignore[union-attr]
                raw_label = row[cfg.dataset.label_key]  # type: ignore[index]
                label = cfg.dataset.label_map.get(str(raw_label), int(raw_label))
                text = row[cfg.dataset.text_key]  # type: ignore[index]
                row_id = row.get(cfg.dataset.id_key, str(len(ood_rows)))  # type: ignore[union-attr]
                ood_rows.append({"id": str(row_id), "text": text, "label": label})
                ood_labels.append(label)

            ood_bundle = ProbingSampleBuilder.from_iterable(ood_rows).to_samples(
                text_key="text"
            )
            ood_extraction = ood_extractor.extract(ood_bundle)

            ood_dataset = ProbingDataset.from_extraction_result(
                ood_extraction,
                activation_key=result.best_key,
                labels=ood_labels,
            )
            collate_fn = None
            if ood_dataset.sequence_mode:
                from sonde.dataset.collate import sequence_collate_fn

                collate_fn = sequence_collate_fn

            loader = torch.utils.data.DataLoader(
                ood_dataset,
                batch_size=128,
                shuffle=False,
                collate_fn=collate_fn,
            )

            probe_model = build_probe(
                cfg.probe.probe_type, input_dim, **cfg.probe.probe_kwargs
            )
            probe_model.load_state_dict(best_state)
            evaluator = BinaryProbeTrainer(model=probe_model, config=cfg.probe)
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
            print(f"  {ood_config:<30} {auroc:>8.4f} {acc:>8.4f} {f1:>8.4f}")
            ood_metrics[ood_config] = {"auroc": auroc, "accuracy": acc, "f1": f1}

    return RunResult(
        summary={
            "run_name": cfg.run_name,
            "action": "pipeline",
            "model": cfg.model.model_name,
            "best_layer": result.best_key,
            "num_samples": len(rows),
        },
        metrics={
            "best_score": result.best_score,
            "test_metrics": result.test_metrics,
            "controls": result.controls,
            "ood": ood_metrics,
        },
        artifacts={"probe": str(probe_path)},
    )
