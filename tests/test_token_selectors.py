"""Tests for ``activation.token_selectors`` and ``TokenSelectorParams``."""

from __future__ import annotations

import unittest

import torch

from activation.token_selectors import (
    AllTokens,
    Index,
    IndexList,
    LastNonPad,
    Range,
    StringAnchor,
    TokenIdAnchor,
    TokenSelector,
)
from core.configs import ExtractionParams, TokenSelectorParams


def _hidden(batch: int, seq: int, dim: int) -> torch.Tensor:
    return torch.arange(batch * seq * dim, dtype=torch.float32).reshape(batch, seq, dim)


# ─────────────────────────────────────────────────── Selectors


class AllTokensTest(unittest.TestCase):
    def test_pass_through(self) -> None:
        x = _hidden(2, 5, 4)
        out, mask = AllTokens().select(x, kind="layers_output")
        self.assertTrue(torch.equal(out, x))
        self.assertIsNone(mask)

    def test_does_not_require_input_ids(self) -> None:
        self.assertFalse(AllTokens().requires_input_ids)


class IndexTest(unittest.TestCase):
    def test_positive_index(self) -> None:
        x = _hidden(2, 5, 4)
        out, _ = Index(2).select(x, kind="layers_output")
        self.assertTrue(torch.equal(out, x[:, 2, :]))

    def test_negative_index(self) -> None:
        x = _hidden(2, 5, 4)
        out, _ = Index(-1).select(x, kind="layers_output")
        self.assertTrue(torch.equal(out, x[:, -1, :]))

    def test_attention_probabilities_rank4(self) -> None:
        x = torch.arange(2 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 3, 4, 5)
        out, _ = Index(1).select(x, kind="attention_probabilities")
        self.assertEqual(tuple(out.shape), (2, 3, 5))
        self.assertTrue(torch.equal(out, x[:, :, 1, :]))


class RangeTest(unittest.TestCase):
    def test_basic_range(self) -> None:
        x = _hidden(2, 6, 4)
        out, _ = Range(start=1, stop=4).select(x, kind="layers_output")
        self.assertEqual(tuple(out.shape), (2, 3, 4))
        self.assertTrue(torch.equal(out, x[:, 1:4, :]))

    def test_step(self) -> None:
        x = _hidden(2, 6, 4)
        out, _ = Range(start=0, stop=6, step=2).select(x, kind="layers_output")
        self.assertEqual(tuple(out.shape), (2, 3, 4))
        self.assertTrue(torch.equal(out, x[:, 0:6:2, :]))

    def test_open_ended_slice(self) -> None:
        x = _hidden(2, 6, 4)
        out, _ = Range(start=-2, stop=None).select(x, kind="layers_output")
        self.assertTrue(torch.equal(out, x[:, -2:, :]))

    def test_step_zero_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Range(start=0, stop=4, step=0)

    def test_empty_selection_raises(self) -> None:
        x = _hidden(2, 6, 4)
        with self.assertRaisesRegex(ValueError, "empty selection"):
            Range(start=4, stop=4).select(x, kind="layers_output")

    def test_propagates_attention_mask_slice(self) -> None:
        x = _hidden(2, 6, 4)
        attn = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 1, 0, 0, 0]])
        _, sub_mask = Range(start=0, stop=4).select(
            x, kind="layers_output", attention_mask=attn
        )
        assert sub_mask is not None
        self.assertTrue(torch.equal(sub_mask, attn[:, 0:4]))


class IndexListTest(unittest.TestCase):
    def test_explicit_indices(self) -> None:
        x = _hidden(2, 6, 4)
        out, _ = IndexList(indices=[0, 3, -1]).select(x, kind="layers_output")
        self.assertEqual(tuple(out.shape), (2, 3, 4))
        expected = torch.stack([x[:, 0, :], x[:, 3, :], x[:, -1, :]], dim=1)
        self.assertTrue(torch.equal(out, expected))

    def test_empty_indices_rejected(self) -> None:
        with self.assertRaises(ValueError):
            IndexList(indices=[])

    def test_out_of_range_index_raises(self) -> None:
        x = _hidden(2, 6, 4)
        with self.assertRaises(IndexError):
            IndexList(indices=[10]).select(x, kind="layers_output")


class LastNonPadTest(unittest.TestCase):
    def test_uses_attention_mask(self) -> None:
        x = _hidden(3, 6, 4)
        attn = torch.tensor(
            [[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]]
        )
        out, _ = LastNonPad().select(x, kind="layers_output", attention_mask=attn)
        self.assertEqual(tuple(out.shape), (3, 4))
        expected = torch.stack([x[0, 2, :], x[1, 4, :], x[2, 5, :]], dim=0)
        self.assertTrue(torch.equal(out, expected))

    def test_derives_mask_from_input_ids(self) -> None:
        x = _hidden(2, 5, 4)
        ids = torch.tensor([[1, 2, 3, 0, 0], [1, 2, 3, 4, 0]])
        out, _ = LastNonPad(pad_token_id=0).select(
            x, kind="layers_output", input_ids=ids
        )
        expected = torch.stack([x[0, 2, :], x[1, 3, :]], dim=0)
        self.assertTrue(torch.equal(out, expected))

    def test_requires_mask_or_pad_token(self) -> None:
        x = _hidden(2, 5, 4)
        with self.assertRaisesRegex(ValueError, "attention_mask"):
            LastNonPad().select(x, kind="layers_output", input_ids=torch.zeros(2, 5))

    def test_all_pad_row_raises(self) -> None:
        x = _hidden(2, 5, 4)
        attn = torch.tensor([[0, 0, 0, 0, 0], [1, 1, 0, 0, 0]])
        with self.assertRaisesRegex(ValueError, "non-pad position"):
            LastNonPad().select(x, kind="layers_output", attention_mask=attn)


class TokenIdAnchorTest(unittest.TestCase):
    def test_first_match_offset_zero(self) -> None:
        x = _hidden(2, 6, 4)
        ids = torch.tensor([[10, 20, 99, 30, 40, 50], [99, 11, 12, 13, 14, 15]])
        out, _ = TokenIdAnchor(pattern=[99]).select(
            x, kind="layers_output", input_ids=ids
        )
        expected = torch.stack([x[0, 2, :], x[1, 0, :]], dim=0)
        self.assertTrue(torch.equal(out, expected))

    def test_offset_advances_past_pattern(self) -> None:
        x = _hidden(2, 8, 4)
        ids = torch.tensor([[1, 2, 90, 91, 92, 7, 7, 7], [90, 91, 92, 5, 5, 5, 5, 5]])
        out, _ = TokenIdAnchor(pattern=[90, 91, 92], offset=3).select(
            x, kind="layers_output", input_ids=ids
        )
        expected = torch.stack([x[0, 5, :], x[1, 3, :]], dim=0)
        self.assertTrue(torch.equal(out, expected))

    def test_last_mode(self) -> None:
        x = _hidden(1, 8, 4)
        ids = torch.tensor([[5, 99, 6, 99, 7, 99, 8, 9]])
        out, _ = TokenIdAnchor(pattern=[99], mode="last").select(
            x, kind="layers_output", input_ids=ids
        )
        self.assertTrue(torch.equal(out, x[:, 5, :]))

    def test_pattern_not_found_raises(self) -> None:
        x = _hidden(2, 5, 4)
        ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        with self.assertRaisesRegex(ValueError, "not found"):
            TokenIdAnchor(pattern=[42]).select(x, kind="layers_output", input_ids=ids)

    def test_offset_out_of_range_raises(self) -> None:
        x = _hidden(1, 5, 4)
        ids = torch.tensor([[1, 2, 99, 4, 5]])
        with self.assertRaises(IndexError):
            TokenIdAnchor(pattern=[99], offset=10).select(
                x, kind="layers_output", input_ids=ids
            )

    def test_invalid_mode_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TokenIdAnchor(pattern=[1], mode="middle")

    def test_empty_pattern_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TokenIdAnchor(pattern=[])

    def test_pattern_longer_than_seq_raises(self) -> None:
        x = _hidden(1, 3, 4)
        ids = torch.tensor([[1, 2, 3]])
        with self.assertRaisesRegex(ValueError, "longer than"):
            TokenIdAnchor(pattern=[1, 2, 3, 4]).select(
                x, kind="layers_output", input_ids=ids
            )


class _StubTokenizer:
    """Minimal stand-in: encodes a string into hashed-character ids."""

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(c) for c in text]


class StringAnchorTest(unittest.TestCase):
    def test_resolves_via_tokenizer(self) -> None:
        anchor = StringAnchor("AB", _StubTokenizer())
        self.assertEqual(anchor.pattern, [ord("A"), ord("B")])

    def test_select_uses_resolved_pattern(self) -> None:
        x = _hidden(1, 6, 4)
        ids = torch.tensor([[1, ord("A"), ord("B"), 4, 5, 6]])
        out, _ = StringAnchor("AB", _StubTokenizer()).select(
            x, kind="layers_output", input_ids=ids
        )
        self.assertTrue(torch.equal(out, x[:, 1, :]))

    def test_empty_anchor_rejected(self) -> None:
        with self.assertRaises(ValueError):
            StringAnchor("", _StubTokenizer())

    def test_missing_tokenizer_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "tokenizer"):
            StringAnchor("foo", tokenizer=None)


class ProtocolConformanceTest(unittest.TestCase):
    def test_each_selector_satisfies_protocol(self) -> None:
        selectors = [
            AllTokens(),
            Index(0),
            Range(0, 4),
            IndexList([0, 1]),
            LastNonPad(pad_token_id=0),
            TokenIdAnchor(pattern=[1]),
            StringAnchor("a", _StubTokenizer()),
        ]
        for sel in selectors:
            with self.subTest(selector=type(sel).__name__):
                self.assertIsInstance(sel, TokenSelector)


# ─────────────────────────────────────────────────── Params


class TokenSelectorParamsTest(unittest.TestCase):
    def test_default_is_index_minus_one(self) -> None:
        params = TokenSelectorParams()
        self.assertEqual(params.type, "index")
        self.assertEqual(params.index, -1)
        sel = params.build()
        self.assertIsInstance(sel, Index)
        assert isinstance(sel, Index)
        self.assertEqual(sel.index, -1)

    def test_unknown_type_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown token_selector type"):
            TokenSelectorParams(type="bogus")

    def test_index_list_requires_indices(self) -> None:
        with self.assertRaisesRegex(ValueError, "indices"):
            TokenSelectorParams(type="index_list")

    def test_token_id_anchor_requires_pattern(self) -> None:
        with self.assertRaisesRegex(ValueError, "pattern"):
            TokenSelectorParams(type="token_id_anchor")

    def test_token_id_anchor_invalid_mode_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "first.*last"):
            TokenSelectorParams(type="token_id_anchor", pattern=[1], mode="middle")

    def test_range_step_zero_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "step != 0"):
            TokenSelectorParams(type="range", step=0)

    def test_build_each_type(self) -> None:
        cases = [
            (TokenSelectorParams(type="all"), AllTokens),
            (TokenSelectorParams(type="index", index=2), Index),
            (TokenSelectorParams(type="range", start=0, stop=4, step=2), Range),
            (TokenSelectorParams(type="index_list", indices=[0, -1]), IndexList),
            (TokenSelectorParams(type="last_non_pad", pad_token_id=0), LastNonPad),
            (
                TokenSelectorParams(type="token_id_anchor", pattern=[1, 2], offset=2),
                TokenIdAnchor,
            ),
        ]
        for params, expected_cls in cases:
            with self.subTest(type=params.type):
                self.assertIsInstance(params.build(), expected_cls)

    def test_yaml_round_trip(self) -> None:
        from omegaconf import OmegaConf

        params = TokenSelectorParams(
            type="token_id_anchor", pattern=[10, 20, 30], offset=3, mode="first"
        )
        cfg = OmegaConf.structured(params)
        as_yaml = OmegaConf.to_yaml(cfg)
        round_tripped: TokenSelectorParams = OmegaConf.to_object(  # type: ignore[assignment]
            OmegaConf.merge(
                OmegaConf.structured(TokenSelectorParams),
                OmegaConf.create(as_yaml),
            )
        )
        self.assertEqual(round_tripped.type, "token_id_anchor")
        self.assertEqual(round_tripped.pattern, [10, 20, 30])
        self.assertEqual(round_tripped.offset, 3)


class ExtractionParamsIntegrationTest(unittest.TestCase):
    def test_default_token_index_unchanged(self) -> None:
        params = ExtractionParams()
        self.assertEqual(params.token_index, -1)
        self.assertIsNone(params.token_selector)

    def test_token_selector_field_round_trips(self) -> None:
        params = ExtractionParams(
            token_selector=TokenSelectorParams(type="all"),
        )
        self.assertIsNotNone(params.token_selector)
        assert params.token_selector is not None
        self.assertEqual(params.token_selector.type, "all")
        self.assertIsInstance(params.token_selector.build(), AllTokens)


class ExtractorSelectorResolutionTest(unittest.TestCase):
    """Verify the resolver precedence (programmatic > params > legacy=None)."""

    def _stub_extractor(
        self,
        *,
        params_selector: TokenSelectorParams | None = None,
        runtime_selector: TokenSelector | None = None,
    ):
        from activation.activation_extractor import ActivationExtractor

        extractor = ActivationExtractor.__new__(ActivationExtractor)
        extractor.extraction_params = ExtractionParams(
            token_selector=params_selector,
        )
        extractor._runtime_token_selector = runtime_selector  # type: ignore[attr-defined]
        return extractor

    def test_runtime_selector_wins_over_params(self) -> None:
        runtime = Index(3)
        params = TokenSelectorParams(type="all")
        extractor = self._stub_extractor(
            params_selector=params, runtime_selector=runtime
        )
        resolved = extractor._resolve_token_selector()
        self.assertIs(resolved, runtime)

    def test_params_selector_used_when_no_runtime(self) -> None:
        params = TokenSelectorParams(type="range", start=0, stop=3)
        extractor = self._stub_extractor(params_selector=params)
        resolved = extractor._resolve_token_selector()
        self.assertIsInstance(resolved, Range)

    def test_returns_none_for_legacy_path(self) -> None:
        extractor = self._stub_extractor()
        self.assertIsNone(extractor._resolve_token_selector())


if __name__ == "__main__":
    unittest.main()
