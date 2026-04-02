"""Model configuration parameters."""

from __future__ import annotations

from dataclasses import dataclass

from core.configs.base import BaseConfig

_VALID_DTYPES = frozenset({
    "float32", "float16", "bfloat16", "float64",
    "int8", "int16", "int32", "int64",
})


@dataclass
class ModelParams(BaseConfig):
    """Configuration for model initialization.

    Typed configuration for model initialization.
    """

    model_name: str = ""
    device: str | None = None
    dtype: str | None = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    attn_implementation: str | None = None
    trust_remote_code: bool = False
    low_cpu_mem_usage: bool = True
    output_attentions: bool = False
    output_hidden_states: bool = False

    def __post_init__(self) -> None:
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError(
                "Cannot use both load_in_8bit and load_in_4bit simultaneously."
            )
        if self.dtype is not None and self.dtype not in _VALID_DTYPES:
            raise ValueError(
                f"Invalid dtype '{self.dtype}'. Expected one of: "
                + ", ".join(sorted(_VALID_DTYPES))
            )
