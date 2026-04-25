"""Config for the probe_sweep stage."""

from __future__ import annotations

from dataclasses import dataclass, field

from core.configs.base import BaseConfig
from core.configs.params.io_params import IOParams
from core.configs.params.output_params import OutputParams
from core.configs.params.probe_params import ProbeParams
from core.configs.params.split_params import SplitParams
from core.configs.params.sweep_params import SweepParams


@dataclass
class ProbeConfig(BaseConfig):
    """Configuration for the probe_sweep stage (action: probe_sweep)."""

    run_name: str = ""
    seed: int = 0
    action: str = "probe_sweep"
    probe: ProbeParams = field(default_factory=ProbeParams)
    split: SplitParams = field(default_factory=SplitParams)
    sweep: SweepParams = field(default_factory=SweepParams)
    io: IOParams = field(default_factory=IOParams)
    output: OutputParams = field(default_factory=OutputParams)
