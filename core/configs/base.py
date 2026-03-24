"""Base configuration class with OmegaConf-powered YAML serialization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Self

from omegaconf import DictConfig, OmegaConf


@dataclass
class BaseConfig:
    """Base config providing YAML round-trip and merge support via OmegaConf."""

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        """Load a config from a YAML file into a typed dataclass instance."""
        schema = OmegaConf.structured(cls)
        raw = OmegaConf.load(str(path))
        merged = OmegaConf.merge(schema, raw)
        return OmegaConf.to_object(merged)  # type: ignore[return-value]

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Load a config from a plain dict into a typed dataclass instance."""
        schema = OmegaConf.structured(cls)
        raw = OmegaConf.create(data)
        merged = OmegaConf.merge(schema, raw)
        return OmegaConf.to_object(merged)  # type: ignore[return-value]

    def to_yaml(self, path: str | Path) -> None:
        """Serialize this config to a YAML file."""
        cfg: DictConfig = OmegaConf.structured(self)
        Path(path).write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")

    def to_yaml_str(self) -> str:
        """Serialize this config to a YAML string."""
        cfg: DictConfig = OmegaConf.structured(self)
        return OmegaConf.to_yaml(cfg)
