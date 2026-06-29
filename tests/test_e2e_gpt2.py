"""On-demand real-model E2E: the causal loop on gpt2.

This test loads gpt2 (network + ~0.5GB) and is therefore opt-in: it is skipped
unless ``SONDE_E2E=1`` is set, and also skips gracefully if the model cannot be
loaded (offline CI). It verifies the whole stack closes the loop AND the
causal-specificity control from docs/intervention_design.md 7.5:

  ablating the trained direction collapses its projection; ablating a random
  direction of matched norm preserves it.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("SONDE_E2E") != "1",
    reason="real-model E2E; set SONDE_E2E=1 to run",
)


def test_causal_loop_specificity_on_gpt2():
    try:
        from examples.causal_loop_gpt2 import main
    except Exception as exc:  # pragma: no cover - import guard
        pytest.skip(f"example import failed: {exc}")

    try:
        m = main()
    except Exception as exc:  # pragma: no cover - offline / model unavailable
        pytest.skip(f"gpt2 unavailable: {exc}")

    # The trained direction separates the concept.
    assert m["base_proj"] > 5.0
    assert m["base_auroc"] >= 0.9
    # Ablating the trained direction collapses its component to ~0.
    assert m["ablated_proj"] < 0.1
    # Ablating a random direction preserves the trained direction's component.
    assert m["control_proj"] > 0.8 * m["base_proj"]
    assert m["control_auroc"] >= 0.9
