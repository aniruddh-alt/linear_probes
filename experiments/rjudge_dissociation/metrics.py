"""Dissociation-specific metric helpers."""

from __future__ import annotations

import torch
from torchmetrics.classification import BinaryAUROC  # type: ignore[import-untyped]


def auroc_between_cells(
    *,
    scores: torch.Tensor,
    mask_a: torch.Tensor,
    mask_b: torch.Tensor,
) -> float:
    """AUROC treating rows in mask_a as positive and mask_b as negative.

    Returns float('nan') if either group is empty.
    """
    a_scores = scores[mask_a]
    b_scores = scores[mask_b]
    if a_scores.numel() == 0 or b_scores.numel() == 0:
        return float("nan")

    combined_scores = torch.cat([a_scores, b_scores])
    combined_labels = torch.cat(
        [
            torch.ones(a_scores.numel(), dtype=torch.long),
            torch.zeros(b_scores.numel(), dtype=torch.long),
        ]
    )
    metric = BinaryAUROC()
    return float(metric(combined_scores.float().cpu(), combined_labels.cpu()).item())


def classification_rate_at_threshold(
    *,
    scores: torch.Tensor,
    threshold: float,
) -> float:
    """Fraction of scores at or above the threshold. NaN if empty."""
    if scores.numel() == 0:
        return float("nan")
    return float((scores >= threshold).float().mean().item())
