"""Consolidated configuration types for all packages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
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


@dataclass(frozen=True)
class ActivationConfig:
    """Typed configuration for activation extraction runs.

    Attributes:
        model_config: Configuration for model initialization.
        save_path: Path to save the extracted activations.
        activations: List of activation names to extract.
        batch_size: Batch size for activation extraction.
        token_index: Index of the token to extract activations for.
        remote: Whether to extract activations from a remote model.
        to_cpu: Whether to move activations to CPU.
    """

    model_config: ModelConfig
    save_path: str | Path
    activations: list[str] = field(default_factory=list)
    batch_size: int = 8
    token_index: int | None = -1
    remote: bool = False
    to_cpu: bool = True

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0.")


@dataclass(frozen=True)
class ProbeConfig:
    """Training configuration for linear probes.

    Attributes:
        epochs: Number of training epochs.
        learning_rate: Learning rate for the optimizer.
        weight_decay: L2 regularization coefficient.
        threshold: Classification threshold for binary predictions.
        device: Device to train on (e.g., 'cuda', 'cpu'). If None, auto-detected.
        seed: Optional random seed for deterministic initialization.
        bootstrap_samples: Number of bootstrap resamples for evaluation CIs.
        bootstrap_confidence: Confidence level for bootstrap intervals.
    """

    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    max_grad_norm: float | None = None
    threshold: float = 0.5
    device: str | None = None
    seed: int | None = None
    early_stopping_patience: int | None = 5
    early_stopping_min_delta: float = 1e-4
    bootstrap_samples: int = 0
    bootstrap_confidence: float = 0.95

    def __post_init__(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be > 0.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0.")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be > 0 when provided.")
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("threshold must be in (0, 1).")
        if (
            self.early_stopping_patience is not None
            and self.early_stopping_patience < 0
        ):
            raise ValueError("early_stopping_patience must be >= 0 when provided.")
        if self.early_stopping_min_delta < 0.0:
            raise ValueError("early_stopping_min_delta must be >= 0.")
        if self.bootstrap_samples < 0:
            raise ValueError("bootstrap_samples must be >= 0.")
        if not 0.0 < self.bootstrap_confidence < 1.0:
            raise ValueError("bootstrap_confidence must be in (0, 1).")


@dataclass(frozen=True)
class LayerProbeSweepConfig:
    """High-level settings for layerwise probe sweep training."""

    probe: ProbeConfig = field(default_factory=ProbeConfig)
    activation_targets: list[str | int] | None = None
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    batch_size: int = 32
    split_seed: int | None = 0
    selection_metric: str = "auroc"
    maximize_metric: bool = True
    control_seeds: tuple[int, ...] = (0, 1, 2)
    enforce_control_sanity: bool = True

    def __post_init__(self) -> None:
        if self.activation_targets is not None:
            if not self.activation_targets:
                raise ValueError("activation_targets must be non-empty when provided.")
            for target in self.activation_targets:
                if isinstance(target, int):
                    continue
                if isinstance(target, str) and target.strip():
                    continue
                raise ValueError(
                    "activation_targets items must be int layer indices "
                    "or non-empty activation key strings."
                )
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must be in (0, 1).")
        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError("val_fraction must be in (0, 1).")
        if not 0.0 < self.test_fraction < 1.0:
            raise ValueError("test_fraction must be in (0, 1).")
        if abs(self.train_fraction + self.val_fraction + self.test_fraction - 1.0) > 1e-8:
            raise ValueError("train_fraction + val_fraction + test_fraction must equal 1.0.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0.")
        if not self.selection_metric.strip():
            raise ValueError("selection_metric must be non-empty.")
        if not self.control_seeds:
            raise ValueError("control_seeds must be non-empty.")
        if any(not isinstance(seed, int) for seed in self.control_seeds):
            raise ValueError("control_seeds must contain only integers.")
