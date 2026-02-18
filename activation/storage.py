"""Helpers for reading saved activation extraction artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch


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
    if not (isinstance(storage, dict) and storage.get("mode") == "sharded"):
        raise TypeError(
            f"Unsupported activation storage format for key '{activation_key}'."
        )

    shard_index = storage.get("shard_index", {})
    if not isinstance(shard_index, dict):
        raise ValueError("Sharded storage is missing a valid `shard_index` mapping.")
    key_shards = shard_index.get(activation_key)
    if not key_shards:
        raise KeyError(f"activation_key '{activation_key}' not found in sharded index.")
    if not isinstance(key_shards, list):
        raise ValueError(
            f"Shard index for '{activation_key}' must be a list, got {type(key_shards).__name__}."
        )

    chunks: list[torch.Tensor] = []
    expected_start = 0
    for shard_idx, shard in enumerate(key_shards):
        if not isinstance(shard, dict):
            raise ValueError(
                f"Shard metadata at index {shard_idx} for '{activation_key}' must be a dict."
            )
        if "path" not in shard or "start" not in shard or "end" not in shard:
            raise ValueError(
                f"Shard metadata at index {shard_idx} for '{activation_key}' must contain "
                "`path`, `start`, and `end`."
            )

        start = int(shard["start"])
        end = int(shard["end"])
        if start != expected_start:
            raise ValueError(
                f"Non-contiguous shard index for '{activation_key}': expected start "
                f"{expected_start}, got {start} at shard {shard_idx}."
            )
        if end <= start:
            raise ValueError(
                f"Invalid shard bounds for '{activation_key}' at shard {shard_idx}: "
                f"start={start}, end={end}."
            )

        shard_path = Path(shard["path"])
        tensor = torch.load(shard_path, map_location=map_location)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"Shard file '{shard_path}' for '{activation_key}' did not contain a tensor."
            )

        rows = _resolve_num_items(tensor)
        expected_rows = end - start
        if rows != expected_rows:
            raise ValueError(
                f"Shard row count mismatch for '{activation_key}' at shard {shard_idx}: "
                f"index expects {expected_rows}, tensor has {rows}."
            )

        chunks.append(tensor)
        expected_start = end

    if len(chunks) == 1:
        return chunks[0].contiguous()
    return torch.cat(chunks, dim=0).contiguous()


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
        shard_index = storage.get("shard_index", {})
        if isinstance(shard_index, dict):
            for key in shard_index.keys():
                key_text = str(key)
                if key_text not in ordered:
                    ordered.append(key_text)

    return ordered


def _resolve_num_items(activation: torch.Tensor) -> int:
    if activation.ndim == 0:
        return 1
    return int(activation.shape[0])
