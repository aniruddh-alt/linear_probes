"""YAML-driven experiment runner: the single entry-point for API and CLI."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.configs.aliases import resolve_config_alias
from core.configs.overrides import apply_dot_overrides
from core.configs.run_config import RunConfig
from omegaconf import OmegaConf


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
) -> RunConfig:
    """Load a RunConfig from YAML, resolving aliases and applying overrides.

    Args:
        config_path: Path to a YAML config file, or an alias key.
        overrides: Optional dot-notation overrides to apply on top.
        aliases_path: Path to the alias registry YAML file.

    Returns:
        Fully resolved RunConfig instance.
    """
    resolved_path = resolve_config_alias(str(config_path), aliases_path)
    raw = OmegaConf.load(resolved_path)
    raw_dict: dict = OmegaConf.to_container(raw, resolve=True)  # type: ignore[assignment]
    if overrides:
        raw_dict = apply_dot_overrides(raw_dict, overrides)
    return RunConfig.from_dict(raw_dict)


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


def dispatch_action(cfg: RunConfig) -> RunResult:
    """Dispatch to the appropriate action handler based on cfg.action."""
    handlers: dict[str, Any] = {
        "probe_sweep": _action_probe_sweep,
        "extract": _action_extract,
        "analyze": _action_analyze,
        "generate": _action_generate,
    }
    handler = handlers.get(cfg.action)
    if handler is None:
        raise ValueError(
            f"Unknown action '{cfg.action}'. "
            f"Available: {', '.join(sorted(handlers))}"
        )
    return handler(cfg)


def _action_probe_sweep(cfg: RunConfig) -> RunResult:
    """Placeholder for probe sweep action wiring."""
    return RunResult(
        summary={
            "run_name": cfg.run_name,
            "action": cfg.action,
            "model": cfg.model.model_name,
            "seed": cfg.seed,
            "status": "configured",
        },
    )


def _action_extract(cfg: RunConfig) -> RunResult:
    """Placeholder for activation extraction action wiring."""
    return RunResult(
        summary={
            "run_name": cfg.run_name,
            "action": cfg.action,
            "model": cfg.model.model_name,
            "status": "configured",
        },
    )


def _action_generate(cfg: RunConfig) -> RunResult:
    """Placeholder for response generation action wiring."""
    return RunResult(
        summary={
            "run_name": cfg.run_name,
            "action": cfg.action,
            "model": cfg.model.model_name,
            "max_new_tokens": cfg.generation.max_new_tokens,
            "status": "configured",
        },
    )


def _action_analyze(cfg: RunConfig) -> RunResult:
    """Placeholder for analysis action wiring."""
    return RunResult(
        summary={
            "run_name": cfg.run_name,
            "action": cfg.action,
            "status": "configured",
        },
    )
