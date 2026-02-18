"""Configuration for model initialization with nnterp."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class ModelConfig:
    """Typed configuration for model initialization with StandardizedTransformer.

    Attributes:
        model_name: Name or path of the model to load (HuggingFace model ID or local path).
        device: Device to load the model on (e.g., "cuda", "cpu", "cuda:0").
        dtype: String representation of torch dtype (e.g., "float32", "float16", "bfloat16").
        torch_dtype: PyTorch dtype for model weights. Takes precedence over `dtype`.
        enable_attention_probs: Whether to enable attention probability extraction.
        trust_remote_code: Whether to trust remote code when loading models.
        load_in_8bit: Whether to load model weights in 8-bit precision.
        load_in_4bit: Whether to load model weights in 4-bit precision.
        attn_implementation: Attention implementation backend to use.
        low_cpu_mem_usage: Whether to use low CPU memory during model loading.
        device_map: Device map for multi-GPU setups (e.g., "auto", dict).
        max_memory: Maximum memory per device for multi-GPU setups.
        offload_folder: Folder to offload model weights when using device_map.
        revision: Specific model version/revision to use.
        token: HuggingFace token for accessing gated models.
        use_cache: Whether to use key-value cache for generation.
        output_attentions: Whether to output attention weights.
        output_hidden_states: Whether to output hidden states.
        additional_kwargs: Additional keyword arguments to pass to the model.
    """

    model_name: str
    device: str | None = None
    dtype: str | None = None
    torch_dtype: Any | None = None
    enable_attention_probs: bool = False
    trust_remote_code: bool = False
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    attn_implementation: Literal["eager", "sdpa", "flash_attention_2"] | None = None
    low_cpu_mem_usage: bool = True
    device_map: str | dict[str, Any] | None = None
    max_memory: dict[str, Any] | None = None
    offload_folder: str | None = None
    revision: str | None = None
    token: str | None = None
    use_cache: bool = True
    output_attentions: bool = False
    output_hidden_states: bool = False
    additional_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name must be a non-empty string.")

        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError(
                "Cannot use both load_in_8bit and load_in_4bit simultaneously."
            )

        if self.dtype is not None and self.dtype not in {
            "float32",
            "float16",
            "bfloat16",
            "float64",
            "int8",
            "int16",
            "int32",
            "int64",
        }:
            raise ValueError(
                f"Invalid dtype '{self.dtype}'. Expected one of: "
                "float32, float16, bfloat16, float64, int8, int16, int32, int64"
            )
