"""Diff-in-means direction estimator."""
from __future__ import annotations

import torch
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)

from directions.types import DiffMeansLayerResult


class DiffMeansEstimator:
    """Computes the diff-in-means direction between two classes.

    Given activations and binary labels, computes:
        r = mean(H+) - mean(H-)
        r_hat = r / ||r||
    """

    def fit(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        *,
        key: str = "",
        val_metrics: dict[str, float | tuple[float, float]] | None = None,
    ) -> DiffMeansLayerResult:
        labels = labels.long()
        pos_mask = labels == 1
        neg_mask = labels == 0
        n_pos = int(pos_mask.sum().item())
        n_neg = int(neg_mask.sum().item())
        if n_pos == 0 or n_neg == 0:
            raise ValueError(
                f"Need both classes for diff-in-means (got {n_pos} positive, {n_neg} negative)."
            )
        pos_mean = features[pos_mask].float().mean(dim=0)
        neg_mean = features[neg_mask].float().mean(dim=0)
        raw_direction = pos_mean - neg_mean
        raw_norm = float(torch.linalg.vector_norm(raw_direction).item())
        if raw_norm == 0.0:
            direction = raw_direction
        else:
            direction = raw_direction / raw_norm
        return DiffMeansLayerResult(
            key=key,
            direction=direction,
            raw_norm=raw_norm,
            positive_count=n_pos,
            negative_count=n_neg,
            val_metrics=val_metrics or {},
        )

    @staticmethod
    def score(features: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        """Project features onto the direction vector. Returns 1D scores."""
        return features.float() @ direction.float()


def evaluate_projection(
    features: torch.Tensor,
    labels: torch.Tensor,
    direction: torch.Tensor,
    *,
    threshold: float | None = None,
) -> dict[str, float]:
    """Evaluate classification by projecting features onto a direction vector.

    Scores = features @ direction. AUROC computed from raw scores.
    For accuracy/F1: if threshold is None, uses median of scores.
    """
    scores = features.float() @ direction.float()
    labels = labels.long()

    probs = torch.sigmoid(scores)
    auroc_metric = BinaryAUROC()
    auroc = float(auroc_metric(probs, labels).item())

    if threshold is None:
        threshold = float(scores.median().item())
    preds = (scores >= threshold).long()
    accuracy = float(BinaryAccuracy()(preds.float(), labels).item())
    precision = float(BinaryPrecision(zero_division=0)(preds.float(), labels).item())
    recall = float(BinaryRecall(zero_division=0)(preds.float(), labels).item())
    f1 = float(BinaryF1Score(zero_division=0)(preds.float(), labels).item())

    return {
        "auroc": auroc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
