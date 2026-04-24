"""Tests for 4-cell behavioral-vs-ground-truth classification."""

from __future__ import annotations

from experiments.rjudge_dissociation.cells import classify_cells


class TestClassifyCells:
    def test_all_four_cells_and_unparseable(self) -> None:
        labels = {"a": 1, "b": 1, "c": 0, "d": 0, "e": 1}
        preds = {"a": 1, "b": 0, "c": 1, "d": 0, "e": -1}
        result = classify_cells(labels=labels, predictions=preds)
        assert result["per_id"] == {
            "a": "TP",  # gt=1 pred=1
            "b": "FN",  # gt=1 pred=0
            "c": "FP",  # gt=0 pred=1
            "d": "TN",  # gt=0 pred=0
            "e": "unparseable",  # pred=-1
        }
        assert result["counts"] == {
            "TP": 1, "FN": 1, "FP": 1, "TN": 1, "unparseable": 1,
        }

    def test_empty_inputs(self) -> None:
        result = classify_cells(labels={}, predictions={})
        assert result["per_id"] == {}
        assert result["counts"] == {"TP": 0, "FN": 0, "FP": 0, "TN": 0, "unparseable": 0}

    def test_missing_prediction_is_unparseable(self) -> None:
        # If an id has a ground truth but no prediction entry, treat as unparseable.
        labels = {"x": 1}
        preds: dict[str, int] = {}
        result = classify_cells(labels=labels, predictions=preds)
        assert result["per_id"]["x"] == "unparseable"
        assert result["counts"]["unparseable"] == 1

    def test_raises_on_unknown_prediction_id(self) -> None:
        labels = {"x": 1}
        preds = {"x": 1, "extra": 0}
        import pytest
        with pytest.raises(ValueError, match="extra"):
            classify_cells(labels=labels, predictions=preds)
