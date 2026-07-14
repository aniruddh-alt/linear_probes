"""Config for the full pipeline: load dataset -> extract activations -> train probes -> OOD eval."""

from __future__ import annotations

from dataclasses import dataclass, field

from sonde.core.configs.base import BaseConfig
from sonde.core.configs.params.dataset_params import DatasetParams
from sonde.core.configs.params.extraction_params import ExtractionParams
from sonde.core.configs.params.io_params import IOParams
from sonde.core.configs.params.model_params import ModelParams
from sonde.core.configs.params.probe_params import ProbeParams
from sonde.core.configs.params.split_params import SplitParams
from sonde.core.configs.params.sweep_params import SweepParams


@dataclass
class PipelineConfig(BaseConfig):
    """End-to-end pipeline: dataset -> extraction -> probe sweep -> OOD eval."""

    run_name: str = ""
    seed: int = 42
    action: str = "pipeline"
    dataset: DatasetParams = field(default_factory=DatasetParams)
    model: ModelParams = field(default_factory=ModelParams)
    extraction: ExtractionParams = field(default_factory=ExtractionParams)
    probe: ProbeParams = field(default_factory=ProbeParams)
    split: SplitParams = field(default_factory=SplitParams)
    sweep: SweepParams = field(default_factory=SweepParams)
    io: IOParams = field(default_factory=IOParams)
    layers: list[int] = field(default_factory=list)
