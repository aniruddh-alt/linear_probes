"""Tests for the judgment parser."""

from __future__ import annotations

from experiments.rjudge_dissociation.judgment import parse_first_digit


class TestParseFirstDigit:
    def test_pure_zero(self) -> None:
        assert parse_first_digit("0") == 0

    def test_pure_one(self) -> None:
        assert parse_first_digit("1") == 1

    def test_one_with_trailing_explanation(self) -> None:
        assert parse_first_digit("1. The agent leaked private info.") == 1

    def test_zero_after_leading_whitespace(self) -> None:
        assert parse_first_digit("   0  safe.") == 0

    def test_ignores_digits_other_than_0_1(self) -> None:
        # "2 is invalid; fall through to first valid 0 or 1"
        assert parse_first_digit("2 no; 0 safe.") == 0

    def test_returns_none_when_no_digit(self) -> None:
        assert parse_first_digit("Yes.") is None
        assert parse_first_digit("") is None

    def test_returns_none_when_only_invalid_digits(self) -> None:
        assert parse_first_digit("result: 42, 7, 9") is None

    def test_handles_unicode_and_multiline(self) -> None:
        assert parse_first_digit("Explanation:\nThe answer is:\n1") == 1
