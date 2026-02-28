"""Config for the extract stage."""
from __future__ import annotations

from dataclasses import dataclass, field

from core.configs.base import BaseConfig
from core.configs.params.extraction_params import ExtractionParams
from core.configs.params.io_params import IOParams
from core.configs.params.model_params import ModelParams


@dataclass
class ExtractConfig(BaseConfig):
    """Configuration for the extract stage (action: extract)."""

    run_name: str = ""
    seed: int = 0
    action: str = "extract"
    model: ModelParams = field(default_factory=ModelParams)
    extraction: ExtractionParams = field(default_factory=ExtractionParams)
    io: IOParams = field(default_factory=IOParams)
