"""Run-manifest and fingerprint helpers for probe experiments."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


def compute_dataset_fingerprint(
    *,
    sample_ids: Sequence[str],
    labels: Sequence[int],
    group_ids: Sequence[str] | None = None,
) -> str:
    """Compute a stable fingerprint for dataset alignment/reproducibility."""
    if len(sample_ids) != len(labels):
        raise ValueError("sample_ids and labels must have the same length.")
    if group_ids is not None and len(group_ids) != len(sample_ids):
        raise ValueError("group_ids and sample_ids must have the same length.")

    hasher = hashlib.sha256()
    for idx, sample_id in enumerate(sample_ids):
        # Safe due to length validation above.
        row = {
            "index": idx,
            "sample_id": str(sample_id),
            "label": int(labels[idx]),
            "group_id": str(group_ids[idx]) if group_ids is not None else None,
        }
        hasher.update(json.dumps(row, sort_keys=True).encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def hash_indices(indices: Sequence[int]) -> str:
    hasher = hashlib.sha256()
    payload = ",".join(str(int(idx)) for idx in sorted(indices))
    hasher.update(payload.encode("utf-8"))
    return hasher.hexdigest()


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {field.name: _to_jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    return str(value)


def write_run_manifest(
    *,
    manifest_path: str | Path,
    config: Any,
    dataset_fingerprint: str,
    selected_key: str,
    selection_metric: str,
    split_indices: dict[str, Sequence[int]],
    split_sizes: tuple[int, int, int],
    test_metrics: dict[str, float | tuple[float, float]],
    controls: dict[str, dict[str, float]],
) -> str:
    """Write immutable JSON manifest for one run."""
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Manifest already exists at {path}. Refusing overwrite.")

    config_payload = _to_jsonable(config)
    payload: dict[str, Any] = {
        "artifact_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": config_payload,
        "dataset_fingerprint": dataset_fingerprint,
        "selected_key": selected_key,
        "selection_metric": selection_metric,
        "split_sizes": {
            "train": int(split_sizes[0]),
            "val": int(split_sizes[1]),
            "test": int(split_sizes[2]),
        },
        "split_index_hashes": {
            split: hash_indices(indices) for split, indices in split_indices.items()
        },
        "test_metrics": test_metrics,
        "controls": controls,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return str(path)
