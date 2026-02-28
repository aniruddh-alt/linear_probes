"""Config for the generate stage."""
from __future__ import annotations

from dataclasses import dataclass, field

from core.configs.base import BaseConfig
from core.configs.params.generation_params import GenerationParams
from core.configs.params.io_params import IOParams
from core.configs.params.model_params import ModelParams
from core.configs.params.steering_params import SteeringParams


@dataclass
class GenerateConfig(BaseConfig):
    """Configuration for the generate stage (action: generate)."""

    run_name: str = ""
    seed: int = 0
    action: str = "generate"
    model: ModelParams = field(default_factory=ModelParams)
    generation: GenerationParams = field(default_factory=GenerationParams)
    steering: SteeringParams = field(default_factory=SteeringParams)
    io: IOParams = field(default_factory=IOParams)
