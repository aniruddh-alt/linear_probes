"""Tests for dataset-creation hardening (Phase 3 audit fixes)."""

from __future__ import annotations

import logging

import pytest
import torch

from sonde.dataset.probing_dataset import ProbingDataset
from sonde.dataset.samples import ProbingSampleBuilder, StringDataset
from sonde.dataset.types import SampleBundle


class TestSequenceModeIsExplicit:
    def test_equal_length_2d_list_defaults_to_sequence_mode(self):
        # The landmine: equal-length (S, D) tensors used to be silently flattened
        # to (N, S*D). They must now be treated as sequence data.
        features = [torch.randn(4, 8), torch.randn(4, 8), torch.randn(4, 8)]
        ds = ProbingDataset(features=features, labels=[0, 1, 0])
        assert ds.sequence_mode is True
        item = ds[0]
        assert len(item) == 3  # (features, label, mask)
        assert item[0].shape == (4, 8)

    def test_variable_length_2d_list_is_sequence_mode(self):
        features = [torch.randn(4, 8), torch.randn(6, 8)]
        ds = ProbingDataset(features=features, labels=[0, 1])
        assert ds.sequence_mode is True

    def test_sequence_mode_false_forces_pooled_flatten(self):
        # Explicit opt-out: equal-length (S, D) flattened to (N, S*D).
        features = [torch.randn(4, 8), torch.randn(4, 8)]
        ds = ProbingDataset(features=features, labels=[0, 1], sequence_mode=False)
        assert ds.sequence_mode is False
        item = ds[0]
        assert len(item) == 2  # pooled: no mask
        assert item[0].numel() == 32  # 4 * 8 flattened

    def test_one_d_list_stays_pooled(self):
        features = [torch.randn(8) for _ in range(4)]
        ds = ProbingDataset(features=features, labels=[0, 1, 0, 1])
        assert ds.sequence_mode is False
        assert len(ds[0]) == 2


class TestLabelLessRaises:
    def test_missing_labels_raises_instead_of_all_zero(self):
        extraction = {
            "requested": ["layers_output:0"],
            "activations": {"layers_output:0": torch.randn(4, 8)},
        }
        with pytest.raises(ValueError, match="without labels"):
            ProbingDataset.from_extraction_result(
                extraction, activation_key="layers_output:0"
            )

    def test_positive_indices_path_still_works(self):
        extraction = {
            "requested": ["layers_output:0"],
            "activations": {"layers_output:0": torch.randn(4, 8)},
        }
        ds = ProbingDataset.from_extraction_result(
            extraction, activation_key="layers_output:0", positive_indices=[1, 3]
        )
        assert ds.labels.tolist() == [0, 1, 0, 1]

    def test_extraction_labels_used_when_present(self):
        extraction = {
            "requested": ["layers_output:0"],
            "activations": {"layers_output:0": torch.randn(3, 8)},
            "labels": [1, 0, 1],
        }
        ds = ProbingDataset.from_extraction_result(
            extraction, activation_key="layers_output:0"
        )
        assert ds.labels.tolist() == [1, 0, 1]


class TestDroppedRowsReported:
    def test_empty_rows_dropped_and_logged(self, caplog):
        records = [
            {"id": "keep-1", "text": "hello"},
            {"id": "drop-1", "text": "   "},
            {"id": "keep-2", "text": "world"},
            {"id": "drop-2", "text": ""},
        ]
        with caplog.at_level(logging.WARNING):
            bundle = ProbingSampleBuilder.from_iterable(records).to_samples(
                text_key="text"
            )
        assert bundle.ids == ["keep-1", "keep-2"]
        assert any("Dropped 2/4" in rec.message for rec in caplog.records)
        assert any("drop-1" in rec.message for rec in caplog.records)


class TestPromptsIsList:
    def test_to_samples_prompts_is_plain_list(self):
        bundle = ProbingSampleBuilder.from_iterable(
            [{"id": "a", "text": "x"}, {"id": "b", "text": "y"}]
        ).to_samples(text_key="text")
        assert isinstance(bundle.prompts, list)
        assert bundle.prompts == ["x", "y"]

    def test_dataset_input_normalised_to_list(self):
        bundle = SampleBundle(
            prompts=StringDataset(["a", "b"]), labels=[0, 1], ids=["0", "1"]
        )
        assert isinstance(bundle.prompts, list)
        assert bundle.prompts == ["a", "b"]


class TestGroupingIsObservable:
    def test_auto_group_logs_regime(self, caplog):
        bundle = SampleBundle(
            prompts=[f"p{i}" for i in range(12)],
            labels=[0, 1] * 6,
            ids=[f"id-{i}" for i in range(12)],
        )
        with caplog.at_level(logging.INFO, logger="sonde.dataset.types"):
            bundle.train_val_test_split(seed=0)
        assert any("Auto-grouping" in rec.message for rec in caplog.records)

    def test_duplicate_ids_non_grouped_warns_leakage(self, caplog):
        # 12 samples but only 3 unique ids -> falls back to non-grouped and must
        # warn about potential leakage.
        ids = ["a", "b", "c"] * 4
        bundle = SampleBundle(
            prompts=[f"p{i}" for i in range(12)],
            labels=[0, 1] * 6,
            ids=ids,
        )
        with caplog.at_level(logging.WARNING, logger="sonde.dataset.types"):
            bundle.train_val_test_split(seed=0)
        assert any("may leak" in rec.message for rec in caplog.records)
