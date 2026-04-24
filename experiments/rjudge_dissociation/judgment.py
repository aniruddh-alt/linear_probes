"""Parse binary safety judgments from subject-model output."""

from __future__ import annotations


def parse_first_digit(text: str) -> int | None:
    """Return the first occurrence of the digit '0' or '1' in text.

    Returns None if neither digit appears. Used to extract the R-Judge
    binary judgment from a model's generation, where the format is
    "<0|1>. Explanation..." per the official judge prompt.
    """
    for ch in text:
        if ch == "0":
            return 0
        if ch == "1":
            return 1
    return None
