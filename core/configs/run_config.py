"""Top-level run configuration composing all param sections."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from core.configs.base import BaseConfig
from core.configs.params.extraction_params import ExtractionParams
from core.configs.params.model_params import ModelParams
from core.configs.params.output_params import OutputParams
from core.configs.params.probe_params import ProbeParams
from core.configs.params.split_params import SplitParams
from core.configs.params.sweep_params import SweepParams


@dataclass
class RunConfig(BaseConfig):
    """Top-level experiment configuration.

    Composes all parameter sections into one YAML-serializable config.
    """

    run_name: str = ""
    seed: int = 0
    action: str = "probe_sweep"
    model: ModelParams = field(default_factory=ModelParams)
    extraction: ExtractionParams = field(default_factory=ExtractionParams)
    probe: ProbeParams = field(default_factory=ProbeParams)
    split: SplitParams = field(default_factory=SplitParams)
    sweep: SweepParams = field(default_factory=SweepParams)
    output: OutputParams = field(default_factory=OutputParams)
