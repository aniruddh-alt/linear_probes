"""Model configuration parameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from core.configs.base import BaseConfig


@dataclass
class ModelParams(BaseConfig):
    """Configuration for model initialization.

    Maps from the existing ModelConfig in configs/types.py.
    """

    model_name: str = ""
    device: Optional[str] = None
    dtype: Optional[str] = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    attn_implementation: Optional[str] = None
    trust_remote_code: bool = False
    low_cpu_mem_usage: bool = True
    output_attentions: bool = False
    output_hidden_states: bool = False
