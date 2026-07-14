"""Artifact I/O path configuration for stage chaining."""

from __future__ import annotations

from dataclasses import dataclass

from sonde.core.configs.base import BaseConfig


@dataclass
class IOParams(BaseConfig):
    """Input/output paths for stage-based artifact chaining."""

    input_path: str = ""
    output_dir: str = "artifacts"
