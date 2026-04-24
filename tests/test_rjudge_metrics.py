"""Tests for dissociation-specific metric helpers."""

from __future__ import annotations

import torch

from experiments.rjudge_dissociation.metrics import (
    auroc_between_cells,
    classification_rate_at_threshold,
)


class TestAurocBetweenCells:
    def test_perfect_separation(self) -> None:
        scores = torch.tensor([0.9, 0.8, 0.1, 0.2])
        mask_a = torch.tensor([True, True, False, False])   # group A (pos)
        mask_b = torch.tensor([False, False, True, True])   # group B (neg)
        auroc = auroc_between_cells(scores=scores, mask_a=mask_a, mask_b=mask_b)
        assert auroc == 1.0

    def test_inverted_separation(self) -> None:
        scores = torch.tensor([0.1, 0.2, 0.9, 0.8])
        mask_a = torch.tensor([True, True, False, False])
        mask_b = torch.tensor([False, False, True, True])
        auroc = auroc_between_cells(scores=scores, mask_a=mask_a, mask_b=mask_b)
        assert auroc == 0.0

    def test_indistinguishable(self) -> None:
        scores = torch.tensor([0.5, 0.5, 0.5, 0.5])
        mask_a = torch.tensor([True, True, False, False])
        mask_b = torch.tensor([False, False, True, True])
        auroc = auroc_between_cells(scores=scores, mask_a=mask_a, mask_b=mask_b)
        assert auroc == 0.5

    def test_empty_group_returns_nan(self) -> None:
        scores = torch.tensor([0.5, 0.5])
        mask_a = torch.tensor([False, False])
        mask_b = torch.tensor([True, True])
        auroc = auroc_between_cells(scores=scores, mask_a=mask_a, mask_b=mask_b)
        # Undefined when one group is empty; return float("nan")
        import math
        assert math.isnan(auroc)


class TestClassificationRate:
    def test_all_above_threshold(self) -> None:
        scores = torch.tensor([0.8, 0.9, 0.7])
        rate = classification_rate_at_threshold(scores=scores, threshold=0.5)
        assert rate == 1.0

    def test_none_above_threshold(self) -> None:
        scores = torch.tensor([0.1, 0.2, 0.3])
        rate = classification_rate_at_threshold(scores=scores, threshold=0.5)
        assert rate == 0.0

    def test_half_above_threshold(self) -> None:
        scores = torch.tensor([0.1, 0.2, 0.8, 0.9])
        rate = classification_rate_at_threshold(scores=scores, threshold=0.5)
        assert rate == 0.5

    def test_empty_scores_returns_nan(self) -> None:
        scores = torch.tensor([])
        rate = classification_rate_at_threshold(scores=scores, threshold=0.5)
        import math
        assert math.isnan(rate)
