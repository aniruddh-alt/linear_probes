"""Tests for GenerationResult type and serialization."""

from __future__ import annotations

import json

from generation.types import GenerationResult


class TestGenerationResult:
    def _make_result(self, n=3, labeled=False):
        return GenerationResult(
            prompts=[f"prompt {i}" for i in range(n)],
            responses=[f"response {i}" for i in range(n)],
            sample_ids=[f"s{i}" for i in range(n)],
            labels=[i % 2 for i in range(n)] if labeled else [None] * n,
        )

    def test_basic_construction(self):
        result = self._make_result()
        assert len(result.prompts) == 3
        assert len(result.responses) == 3
        assert all(label is None for label in result.labels)

    def test_to_jsonl_and_from_jsonl_round_trip(self, tmp_path):
        original = self._make_result(n=4, labeled=True)
        path = tmp_path / "output.jsonl"
        original.to_jsonl(path)

        loaded = GenerationResult.from_jsonl(path)
        assert loaded.prompts == original.prompts
        assert loaded.responses == original.responses
        assert loaded.sample_ids == original.sample_ids
        assert loaded.labels == original.labels

    def test_to_jsonl_unlabeled_round_trip(self, tmp_path):
        original = self._make_result(n=2, labeled=False)
        path = tmp_path / "output.jsonl"
        original.to_jsonl(path)

        loaded = GenerationResult.from_jsonl(path)
        assert loaded.labels == [None, None]

    def test_jsonl_format(self, tmp_path):
        result = self._make_result(n=2, labeled=True)
        path = tmp_path / "output.jsonl"
        result.to_jsonl(path)

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        row = json.loads(lines[0])
        assert set(row.keys()) == {"prompt", "response", "sample_id", "label"}

    def test_to_sample_bundle_prompt_only(self):
        result = self._make_result(n=3, labeled=True)
        bundle = result.to_sample_bundle(include_response=False)
        assert len(bundle.prompts) == 3  # type: ignore[arg-type]
        assert bundle.prompts[0] == "prompt 0"
        assert bundle.labels == [0, 1, 0]
        assert bundle.ids == ["s0", "s1", "s2"]
        assert bundle.responses == ["response 0", "response 1", "response 2"]

    def test_to_sample_bundle_with_response(self):
        result = self._make_result(n=2, labeled=False)
        bundle = result.to_sample_bundle(include_response=True)
        # Prompts should be prompt + response concatenated
        assert "prompt 0" in bundle.prompts[0]
        assert "response 0" in bundle.prompts[0]

    def test_to_sample_bundle_unlabeled(self):
        result = self._make_result(n=2, labeled=False)
        bundle = result.to_sample_bundle()
        assert bundle.labels == [None, None]

    def test_length_mismatch_raises(self):
        import pytest

        with pytest.raises(ValueError):
            GenerationResult(
                prompts=["a", "b"],
                responses=["r"],
                sample_ids=["s0", "s1"],
                labels=[None, None],
            )
