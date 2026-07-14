"""Output configuration parameters."""

from __future__ import annotations

from dataclasses import dataclass

from sonde.core.configs.base import BaseConfig


@dataclass
class OutputParams(BaseConfig):
    """Configuration for output artifacts and paths."""

    output_dir: str = "artifacts"
    save_plots: bool = True
    manifest_path: str | None = None
    overwrite_manifest: bool = False
    unique_manifest_path: bool = False
