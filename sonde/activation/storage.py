"""Read/write helpers for activation-extraction artifacts.

An extraction artifact is two files that share a base path::

    <base>.safetensors        # the tensors (one entry per activation key)
    <base>_manifest.json      # JSON metadata: model, sample_ids, labels, storage

The manifest is **JSON, not pickle** — it holds no tensors, so loading it is
never code execution, and it can be inspected/diffed without torch. The
``safetensors_path`` it records is stored *relative to the manifest* so the
whole artifact directory can be moved or copied to another machine and still
load.

Variable-length (sequence-mode) activations are persisted by padding each key
to ``(N, S_max, D)`` and storing a companion ``"<key>::lengths"`` int64 tensor;
:func:`load_activation_value` unpads them back into a list of ``(S_i, D)``
tensors so the round-trip is exact.

Legacy ``_manifest.pt`` pickle artifacts written by older versions remain
readable for one release via :func:`load_extraction_manifest`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

SCHEMA_VERSION = 2
_LENGTHS_SUFFIX = "::lengths"


def _sonde_version() -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("sonde")
        except PackageNotFoundError:
            return "unknown"
    except Exception:  # pragma: no cover - defensive
        return "unknown"


def resolve_storage_paths(save_path: str | Path) -> tuple[Path, Path]:
    """Return ``(manifest_path, safetensors_path)`` for an artifact base path."""
    base = Path(save_path)
    base = base.with_suffix("") if base.suffix in {".pt", ".json"} else base
    # Derive BOTH names from the full base.name so a dotted stem (e.g.
    # "run.v2") doesn't split the manifest and tensors onto different stems or
    # collide distinct paths via Path.with_suffix dropping the last segment.
    manifest_path = base.parent / f"{base.name}_manifest.json"
    safetensors_path = base.parent / f"{base.name}.safetensors"
    return manifest_path, safetensors_path


def save_extraction(
    result: Mapping[str, Any],
    save_path: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Persist an extraction result to ``<base>.safetensors`` + ``<base>_manifest.json``.

    Rectangular keys are saved as-is. Ragged keys (a ``list`` of per-sample
    ``(S_i, D)`` tensors) are padded to ``(N, S_max, D)`` with a companion
    lengths tensor so the exact sequences can be recovered on load.

    Args:
        result: extraction result mapping with ``activations`` plus metadata.
        save_path: artifact base path (a ``.pt``/``.json`` suffix is stripped).
        overwrite: if ``False`` (default) and either output file already exists,
            raise :class:`FileExistsError` rather than clobbering it.

    Returns:
        The ``storage`` dict to embed in the in-memory result.
    """
    manifest_path, safetensors_path = resolve_storage_paths(save_path)
    if not overwrite:
        for existing in (manifest_path, safetensors_path):
            if existing.exists():
                raise FileExistsError(
                    f"Refusing to overwrite existing artifact at {existing}. "
                    "Pass overwrite=True to replace it."
                )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    activations: Mapping[str, Any] = result.get("activations", {})
    tensors: dict[str, torch.Tensor] = {}
    key_meta: dict[str, dict[str, Any]] = {}
    for key, value in activations.items():
        if isinstance(value, torch.Tensor):
            tensors[key] = value.contiguous()
            key_meta[key] = {"ragged": False}
        elif isinstance(value, (list, tuple)):
            padded, lengths = _pad_sequence_list(list(value))
            tensors[key] = padded.contiguous()
            lengths_key = f"{key}{_LENGTHS_SUFFIX}"
            tensors[lengths_key] = lengths.contiguous()
            key_meta[key] = {"ragged": True, "lengths_key": lengths_key}
        else:
            raise TypeError(
                f"Cannot persist activation '{key}' of type {type(value).__name__}."
            )

    save_file(tensors, str(safetensors_path))

    storage: dict[str, Any] = {
        "mode": "safetensors",
        "manifest_path": str(manifest_path),
        "safetensors_path": str(safetensors_path),
        "keys": key_meta,
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "sonde_version": _sonde_version(),
        "model": result.get("model"),
        "requested": list(result.get("requested", [])),
        "sample_ids": list(result.get("sample_ids", [])),
        "labels": list(result.get("labels", [])),
        "storage": {
            "mode": "safetensors",
            # Stored relative to the manifest so the artifact is relocatable.
            "safetensors_path": safetensors_path.name,
            "keys": key_meta,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return storage


def _pad_sequence_list(
    sequences: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of ``(S_i, D)`` tensors to ``(N, S_max, D)`` + ``lengths (N,)``."""
    if not sequences:
        return torch.empty((0, 0, 0)), torch.empty((0,), dtype=torch.long)
    for idx, seq in enumerate(sequences):
        if seq.ndim != 2:
            raise ValueError(
                f"Ragged activation entry {idx} must be 2-D (S, D), got shape "
                f"{tuple(seq.shape)}."
            )
    hidden = sequences[0].shape[1]
    if any(seq.shape[1] != hidden for seq in sequences):
        raise ValueError("All ragged activation entries must share a hidden dim.")
    lengths = torch.tensor([seq.shape[0] for seq in sequences], dtype=torch.long)
    max_len = int(lengths.max().item())
    padded = torch.zeros((len(sequences), max_len, hidden), dtype=sequences[0].dtype)
    for i, seq in enumerate(sequences):
        padded[i, : seq.shape[0]] = seq
    return padded, lengths


def load_extraction_manifest(
    extraction_path: str | Path, *, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    """Load an extraction manifest (JSON, or legacy ``.pt`` pickle).

    For JSON manifests the recorded ``safetensors_path`` is resolved to an
    absolute path relative to the manifest file, so the returned dict can be
    handed straight to :func:`load_activation_value` from any directory.
    """
    path = Path(extraction_path)
    if path.suffix == ".json" or (path.suffix != ".pt" and _looks_like_json(path)):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise TypeError("Extraction manifest JSON must decode to an object.")
        storage = manifest.get("storage")
        if isinstance(storage, dict) and isinstance(
            storage.get("safetensors_path"), str
        ):
            resolved = (path.parent / storage["safetensors_path"]).resolve()
            storage["safetensors_path"] = str(resolved)
        manifest.setdefault("activations", {})
        return manifest

    # Legacy pickle manifest (pre-0.1 artifacts). weights_only=False is required
    # to read the dict payload; only ever used for this back-compat path.
    loaded = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(loaded, dict):
        raise TypeError(
            f"Extraction manifest must be a dict, got {type(loaded).__name__}."
        )
    return loaded


def _looks_like_json(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            head = handle.read(64).lstrip()
        return head[:1] in (b"{", b"[")
    except OSError:
        return False


def resolve_activation_key(
    extraction: Mapping[str, Any], activation_key: str | None = None
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
    extraction: Mapping[str, Any],
    *,
    activation_key: str,
    map_location: str | torch.device = "cpu",
) -> torch.Tensor | list[torch.Tensor]:
    """Load one activation stream from an extraction manifest.

    Ragged keys are returned as a list of per-sample ``(S_i, D)`` tensors,
    exactly as they were before persistence.
    """
    activations = extraction.get("activations", {})
    if isinstance(activations, dict) and activation_key in activations:
        value = activations[activation_key]
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, (list, tuple)):
            return list(value)
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

    device = str(map_location)
    tensors = load_file(safetensors_path, device=device)
    if activation_key not in tensors:
        raise KeyError(
            f"activation_key '{activation_key}' not found in safetensors storage."
        )

    key_meta: Mapping[str, Any] = {}
    if isinstance(storage.get("keys"), dict):
        key_meta = storage["keys"].get(activation_key, {})
    if key_meta.get("ragged"):
        lengths_key = key_meta.get("lengths_key", f"{activation_key}{_LENGTHS_SUFFIX}")
        if lengths_key not in tensors:
            raise KeyError(
                f"Ragged key '{activation_key}' is missing its lengths tensor "
                f"'{lengths_key}'."
            )
        padded = tensors[activation_key]
        lengths = tensors[lengths_key].long().tolist()
        return [padded[i, : lengths[i]] for i in range(len(lengths))]

    return tensors[activation_key]


def _available_activation_keys(extraction: Mapping[str, Any]) -> list[str]:
    requested = extraction.get("requested")
    ordered: list[str] = []

    if isinstance(requested, list):
        ordered.extend(str(item) for item in requested)

    activations = extraction.get("activations", {})
    if isinstance(activations, dict):
        for key in activations:
            key_text = str(key)
            if key_text not in ordered:
                ordered.append(key_text)

    storage = extraction.get("storage")
    if isinstance(storage, dict):
        meta_keys = storage.get("keys")
        if isinstance(meta_keys, dict):
            for key in meta_keys:
                if str(key) not in ordered:
                    ordered.append(str(key))
        if storage.get("mode") == "safetensors":
            safetensors_path = storage.get("safetensors_path")
            if isinstance(safetensors_path, str) and Path(safetensors_path).is_file():
                with safe_open(
                    safetensors_path, framework="pt", device="cpu"
                ) as handle:
                    for key in handle.keys():  # noqa: SIM118 - not iterable
                        key_text = str(key)
                        if key_text.endswith(_LENGTHS_SUFFIX):
                            continue
                        if key_text not in ordered:
                            ordered.append(key_text)

    return ordered
