"""The probe artifact — the contract between probing and causal intervention.

A trained probe's *direction* (a unit vector in activation space) plus its
layer and a little metadata is everything a steering / ablation pipeline needs.
This module persists that as a single self-describing safetensors file::

    <base>.safetensors
        tensors:   direction (D,)   [, bias (1,)]
        __metadata__["info"] = JSON {activation_key, layer, bias, model,
                                     probe_type, metrics, sonde_version, ...}

It is deliberately framework-light: a consumer (including a separate
``sonde.deploy`` package that must not import nnsight) can read the direction
with nothing but ``safetensors``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SCHEMA_VERSION = 1
_LAYER_RE = re.compile(r":(-?\d+)$")


def _sonde_version() -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("sonde")
        except PackageNotFoundError:
            return "unknown"
    except Exception:  # pragma: no cover - defensive
        return "unknown"


def layer_from_activation_key(activation_key: str) -> int | None:
    """Parse the layer index out of e.g. ``"layers_output:15"`` -> ``15``."""
    match = _LAYER_RE.search(activation_key)
    return int(match.group(1)) if match is not None else None


@dataclass
class ProbeArtifact:
    """A persisted concept direction usable for steering / ablation."""

    direction: torch.Tensor
    activation_key: str
    bias: float | None = None
    layer: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.direction is None:
            raise ValueError("ProbeArtifact requires a direction tensor.")
        self.direction = self.direction.detach().reshape(-1).float()
        if self.layer is None:
            self.layer = layer_from_activation_key(self.activation_key)

    def save(self, path: str | Path, *, overwrite: bool = True) -> str:
        """Write the artifact to ``<path>.safetensors``.

        Probe artifacts are derived outputs, so they overwrite by default; pass
        ``overwrite=False`` to refuse clobbering an existing file.
        """
        out = Path(path)
        out = out if out.suffix == ".safetensors" else out.with_suffix(".safetensors")
        if out.exists() and not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing probe artifact at {out}."
            )
        out.parent.mkdir(parents=True, exist_ok=True)

        tensors = {"direction": self.direction.contiguous()}
        if self.bias is not None:
            tensors["bias"] = torch.tensor([float(self.bias)], dtype=torch.float32)

        info = {
            "schema_version": SCHEMA_VERSION,
            "sonde_version": _sonde_version(),
            "activation_key": self.activation_key,
            "layer": self.layer,
            "bias": self.bias,
            "dim": int(self.direction.numel()),
            "metadata": self.metadata,
        }
        save_file(tensors, str(out), metadata={"info": json.dumps(info)})
        return str(out)

    @classmethod
    def load(cls, path: str | Path) -> ProbeArtifact:
        out = Path(path)
        out = out if out.suffix == ".safetensors" else out.with_suffix(".safetensors")
        with safe_open(str(out), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            direction = handle.get_tensor("direction")
            keys = set(handle.keys())
            bias_tensor = handle.get_tensor("bias") if "bias" in keys else None
        info = json.loads(metadata.get("info", "{}"))
        bias = (
            float(bias_tensor.item()) if bias_tensor is not None else info.get("bias")
        )
        return cls(
            direction=direction,
            activation_key=info.get("activation_key", ""),
            bias=bias,
            layer=info.get("layer"),
            metadata=info.get("metadata", {}),
        )


def save_probe_artifact(
    *,
    direction: torch.Tensor,
    activation_key: str,
    path: str | Path,
    bias: float | None = None,
    metadata: dict[str, Any] | None = None,
    overwrite: bool = True,
) -> str:
    """Convenience constructor + save in one call."""
    artifact = ProbeArtifact(
        direction=direction,
        activation_key=activation_key,
        bias=bias,
        metadata=metadata or {},
    )
    return artifact.save(path, overwrite=overwrite)
