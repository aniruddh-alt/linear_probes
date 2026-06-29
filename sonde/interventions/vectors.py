"""Load steering / probe direction vectors from heterogeneous sources.

The single entry point :func:`load_vector` accepts a path string, a tensor, a
trained probe, or a diff-means layer result, and returns a flat 1-D float
tensor that ``nnterp.StandardizedTransformer.steer`` (and the InterventionContext
write paths in R-2) can consume directly.

Supported inputs:

================================================  ===========================================
Input                                             Source of the vector
================================================  ===========================================
``torch.Tensor``                                  used directly
``str`` ending in ``.safetensors``                ``safetensors.torch.load_file`` + key lookup
``str`` ending in ``.pt`` / ``.pth``              ``torch.load(weights_only=True)``
``BaseProbe`` instance                            ``probe.direction``
``DiffMeansLayerResult`` instance                 ``result.direction``
``LayerProbeSweepResult`` instance                ``result.probes[result.best_key].model.direction``
================================================  ===========================================

If ``normalize=True`` (the default) the returned tensor is L2-normalised. A zero
vector is returned unchanged (norm == 0 ⇒ no division).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file


def load_vector(
    source: Any,
    *,
    key: str = "",
    normalize: bool = True,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Resolve ``source`` to a 1-D float tensor on ``device``."""
    raw = _resolve_raw(source, key=key, device=device)
    if not isinstance(raw, torch.Tensor):
        raise TypeError(
            f"load_vector resolved a non-tensor source to {type(raw).__name__}; "
            "expected a torch.Tensor at the leaf."
        )
    vec = raw.detach().to(device=device).float().squeeze()
    if vec.ndim != 1:
        raise ValueError(
            f"load_vector expected a 1-D vector after squeeze; got shape "
            f"{tuple(vec.shape)}."
        )
    if normalize:
        norm = float(torch.linalg.vector_norm(vec).item())
        if norm > 0.0:
            vec = vec / norm
    return vec


def _resolve_raw(source: Any, *, key: str, device: str | torch.device) -> torch.Tensor:
    # 1) raw tensor
    if isinstance(source, torch.Tensor):
        return source

    # 2) string / Path → file load
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix == ".safetensors":
            tensors = load_file(str(path), device=str(device))
            if key:
                if key not in tensors:
                    available = ", ".join(sorted(tensors)) or "(none)"
                    raise KeyError(
                        f"safetensors file '{path}' has no key '{key}'. "
                        f"Available keys: {available}"
                    )
                return tensors[key]
            if len(tensors) != 1:
                available = ", ".join(sorted(tensors))
                raise ValueError(
                    f"safetensors file '{path}' has multiple keys ({available}); "
                    "pass `key=` to disambiguate."
                )
            return next(iter(tensors.values()))
        if path.suffix in (".pt", ".pth"):
            loaded = torch.load(path, map_location=str(device), weights_only=True)
            if isinstance(loaded, dict):
                if not key:
                    raise ValueError(
                        f"'{path}' is a state-dict-like .pt file; pass `key=` "
                        "to select a single tensor."
                    )
                if key not in loaded:
                    raise KeyError(f"'{path}' has no key '{key}'.")
                return loaded[key]
            return loaded
        raise ValueError(
            f"Unsupported file extension '{path.suffix}' for vector path '{path}'. "
            "Use .safetensors, .pt, or .pth."
        )

    # 3) duck-typed objects exposing a usable attribute
    direction = getattr(source, "direction", None)
    if isinstance(direction, torch.Tensor):
        return direction
    best_key = getattr(source, "best_key", None)
    probes = getattr(source, "probes", None)
    if isinstance(best_key, str) and isinstance(probes, dict):
        best_probe = probes.get(best_key)
        # Sweep returns TrainedLayerProbe with .model.direction
        model = getattr(best_probe, "model", None)
        direction = getattr(model, "direction", None)
        if isinstance(direction, torch.Tensor):
            return direction

    raise TypeError(
        f"Cannot extract a steering vector from object of type "
        f"{type(source).__name__}. Provide a tensor, a path, or an object "
        "exposing `.direction` (BaseProbe / DiffMeansLayerResult) or "
        "`.best_key` + `.probes` (LayerProbeSweepResult)."
    )


__all__ = ["load_vector"]
