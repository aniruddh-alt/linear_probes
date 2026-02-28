"""YAML-driven experiment runner: the single entry-point for API and CLI."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

from omegaconf import OmegaConf

from core.configs.base import BaseConfig
from core.configs.extract_config import ExtractConfig
from core.configs.generate_config import GenerateConfig
from core.configs.probe_config import ProbeConfig
from core.configs.aliases import resolve_config_alias
from core.configs.overrides import apply_dot_overrides

StageConfig = Union[GenerateConfig, ExtractConfig, ProbeConfig]

STAGE_CONFIG_MAP: dict[str, type[BaseConfig]] = {
    "generate":    GenerateConfig,
    "extract":     ExtractConfig,
    "probe_sweep": ProbeConfig,
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
    aliases_path: str | Path = "configs/aliases.yaml",
) -> BaseConfig:
    """Load a stage config from YAML, resolving aliases and applying overrides.

    The `action:` field in the YAML determines which config class is loaded.

    Args:
        config_path: Path to a YAML config file, or an alias key.
        overrides: Optional dot-notation overrides to apply on top.
        aliases_path: Path to the alias registry YAML file.

    Returns:
        The appropriate stage config instance (GenerateConfig, ExtractConfig, or ProbeConfig).
    """
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
    aliases_path: str | Path = "configs/aliases.yaml",
) -> RunResult:
    """Run an experiment from a YAML config file.

    This is the primary API entry-point. The CLI delegates here.

    Args:
        config_path: Path to a YAML config file, or an alias key.
        overrides: Optional dot-notation CLI overrides.
        aliases_path: Path to the alias registry.

    Returns:
        RunResult with summary, metrics, and artifact paths.
    """
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

    from dataset import ProbingSampleBuilder
    from generation import ResponseGenerator

    builder = ProbingSampleBuilder.from_file(input_file)
    bundle = builder.to_samples(text_key="prompt", label_key=None, id_key=None)

    generator = ResponseGenerator(model=cfg.model, generation=cfg.generation)
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
    """Placeholder for activation extraction action wiring."""
    return RunResult(
        summary={
            "run_name": cfg.run_name,
            "action": cfg.action,
            "model": cfg.model.model_name,
            "status": "configured",
        },
    )


def _action_probe_sweep(cfg: ProbeConfig) -> RunResult:
    """Placeholder for probe sweep action wiring."""
    return RunResult(
        summary={
            "run_name": cfg.run_name,
            "action": cfg.action,
            "seed": cfg.seed,
            "status": "configured",
        },
    )
