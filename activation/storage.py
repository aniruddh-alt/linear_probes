"""Helpers for reading saved activation extraction artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import load_file


def load_extraction_manifest(
    extraction_path: str | Path, *, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    """Load an extraction manifest from disk."""
    loaded = torch.load(Path(extraction_path), map_location=map_location)
    if not isinstance(loaded, dict):
        raise TypeError(
            f"Extraction manifest must be a dict, got {type(loaded).__name__}."
        )
    return loaded


def resolve_activation_key(
    extraction: dict[str, Any], activation_key: str | None = None
) -> str:
    """Resolve activation key with explicit handling for ambiguous manifests."""
    keys = _available_activation_keys(extraction)
    if activation_key is not None:
        if activation_key not in keys:
            available = ", ".join(keys)
            raise KeyError(
                f"activation_key '{activation_key}' not found. Available keys: {available}"
            )
        return activation_key

    if not keys:
        raise KeyError("No activation keys available in extraction manifest.")
    if len(keys) > 1:
        available = ", ".join(keys)
        raise ValueError(
            "Multiple activation keys are available; pass `activation_key` explicitly. "
            f"Available keys: {available}"
        )
    return keys[0]


def load_activation_value(
    extraction: dict[str, Any],
    *,
    activation_key: str,
    map_location: str | torch.device = "cpu",
) -> torch.Tensor | Sequence[torch.Tensor]:
    """Load one activation stream from an extraction manifest."""
    activations = extraction.get("activations", {})
    if isinstance(activations, dict) and activation_key in activations:
        value = activations[activation_key]
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, Sequence):
            return value
        raise TypeError(
            f"Unsupported in-memory activation type for key '{activation_key}': "
            f"{type(value).__name__}."
        )

    storage = extraction.get("storage")
    if not (isinstance(storage, dict) and storage.get("mode") == "safetensors"):
        raise TypeError(
            f"Unsupported activation storage format for key '{activation_key}'."
        )

    safetensors_path = storage.get("safetensors_path")
    if not isinstance(safetensors_path, str):
        raise ValueError("Safetensors storage is missing `safetensors_path`.")

    if isinstance(map_location, torch.device):
        device = str(map_location)
    else:
        device = str(map_location)
    tensors = load_file(safetensors_path, device=device)
    if activation_key not in tensors:
        raise KeyError(
            f"activation_key '{activation_key}' not found in safetensors storage."
        )
    return tensors[activation_key]


def _available_activation_keys(extraction: dict[str, Any]) -> list[str]:
    requested = extraction.get("requested")
    ordered: list[str] = []

    if isinstance(requested, list):
        ordered.extend(str(item) for item in requested)

    activations = extraction.get("activations", {})
    if isinstance(activations, dict):
        for key in activations.keys():
            key_text = str(key)
            if key_text not in ordered:
                ordered.append(key_text)

    storage = extraction.get("storage")
    if isinstance(storage, dict):
        if storage.get("mode") == "safetensors":
            safetensors_path = storage.get("safetensors_path")
            if isinstance(safetensors_path, str):
                with safe_open(
                    safetensors_path, framework="pt", device="cpu"
                ) as handle:
                    for key in handle.keys():
                        key_text = str(key)
                        if key_text not in ordered:
                            ordered.append(key_text)

    return ordered
