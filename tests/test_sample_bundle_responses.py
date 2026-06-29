"""Tests for SampleBundle responses field."""

from __future__ import annotations

from sonde.dataset.samples import StringDataset
from sonde.dataset.types import SampleBundle


class TestSampleBundleResponses:
    def test_responses_default_none(self):
        bundle = SampleBundle(
            prompts=StringDataset(["a", "b"]),
            labels=[0, 1],
            ids=["0", "1"],
        )
        assert bundle.responses is None

    def test_responses_set(self):
        bundle = SampleBundle(
            prompts=StringDataset(["a", "b"]),
            labels=[0, 1],
            ids=["0", "1"],
            responses=["resp_a", "resp_b"],
        )
        assert bundle.responses == ["resp_a", "resp_b"]

    def test_existing_code_unaffected(self):
        """SampleBundle without responses still works for splitting etc."""
        bundle = SampleBundle(
            prompts=StringDataset(["a"] * 20 + ["b"] * 20),
            labels=[0] * 20 + [1] * 20,  # type: ignore[arg-type]
            ids=[f"a-{i}" for i in range(20)] + [f"b-{i}" for i in range(20)],
        )
        train, val, test = bundle.train_val_test_split(seed=42)
        assert len(train) + len(val) + len(test) == 40
