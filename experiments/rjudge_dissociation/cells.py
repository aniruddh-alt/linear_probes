"""Partition scenarios into TP/FP/FN/TN cells from behavioral vs. ground-truth labels."""

from __future__ import annotations

from typing import TypedDict


class CellsResult(TypedDict):
    per_id: dict[str, str]
    counts: dict[str, int]


_VALID_CELLS = ("TP", "FN", "FP", "TN", "unparseable")


def classify_cells(
    *,
    labels: dict[str, int],
    predictions: dict[str, int],
) -> CellsResult:
    """Partition ids into TP/FP/FN/TN cells.

    Args:
        labels: id -> ground-truth label (0 or 1).
        predictions: id -> behavioral prediction (0 or 1, or -1 for unparseable).
            Predictions for ids not in `labels` raise ValueError.

    Returns:
        {"per_id": {id: cell}, "counts": {cell: count}}.
    """
    unknown_ids = set(predictions) - set(labels)
    if unknown_ids:
        raise ValueError(
            f"predictions contain ids not in labels: {sorted(unknown_ids)}"
        )

    per_id: dict[str, str] = {}
    counts: dict[str, int] = {cell: 0 for cell in _VALID_CELLS}

    for sample_id, gt in labels.items():
        pred = predictions.get(sample_id, -1)
        if pred == -1:
            cell = "unparseable"
        elif gt == 1 and pred == 1:
            cell = "TP"
        elif gt == 1 and pred == 0:
            cell = "FN"
        elif gt == 0 and pred == 1:
            cell = "FP"
        elif gt == 0 and pred == 0:
            cell = "TN"
        else:
            raise ValueError(
                f"Unexpected label/prediction combo for id={sample_id}: "
                f"label={gt}, pred={pred}"
            )
        per_id[sample_id] = cell
        counts[cell] += 1

    return {"per_id": per_id, "counts": counts}
