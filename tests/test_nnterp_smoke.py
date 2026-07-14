"""Real-nnsight smoke test for InterventionContext.apply().

The mock-based tests in test_intervention_context.py cannot catch nnsight's
source-introspection requirement (the historical WithBlockNotFoundError bug:
applying steers from a wrapper context manager fails under real nnsight). This
test exercises the actual apply()-inside-a-literal-with-block path against a
real StandardizedTransformer so that class of regression is caught.

It is NOT env-gated (so it runs in any environment with network access) but
skips gracefully when the model/deps cannot be loaded, keeping offline CI green.
"""

from __future__ import annotations

import pytest
import torch

_TINY_MODELS = ["sshleifer/tiny-gpt2", "hf-internal-testing/tiny-random-gpt2"]


def _load_tiny_model():
    try:
        from nnterp import StandardizedTransformer
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"nnterp unavailable: {exc}")
    last = None
    for name in _TINY_MODELS:
        try:
            return StandardizedTransformer(name, device_map="cpu")
        except Exception as exc:  # pragma: no cover - network/model unavailable
            last = exc
    pytest.skip(f"no tiny model could be loaded (offline?): {last}")


def test_apply_project_subtract_under_real_nnsight():
    from sonde.interventions import InterventionContext

    model = _load_tiny_model()
    layer = 0
    d = int(model.hidden_size)
    v = torch.randn(d)
    vhat = v / v.norm()
    prompt = "the quick brown fox"

    with model.trace(prompt):
        base = model.layers_input[layer + 1].save()
    base_proj = (base[0] * vhat).sum(-1).abs().max().item()

    ctx = InterventionContext(model).add_steering(
        layers=[layer], vector=v, mode="project_subtract", factor=1.0
    )
    # The critical path: apply() inside a LITERAL with-block (nnsight introspects
    # this block's source). A wrapper-CM regression would raise here.
    with model.trace(prompt):
        ctx.apply()
        abl = model.layers_input[layer + 1].save()
    abl_proj = (abl[0] * vhat).sum(-1).abs().max().item()

    assert base_proj > 0
    assert abl_proj < max(1e-4, base_proj * 0.05)


def test_apply_inside_generate_does_not_raise_under_real_nnsight():
    # Regression guard for the generate() path: apply() inside a literal
    # `with model.generate(...)` block must not raise WithBlockNotFoundError.
    # (Behavioural effect of steering is covered by the gpt2 E2E; a tiny
    # random-weight model is too degenerate to assert token changes on.)
    from sonde.interventions import InterventionContext

    model = _load_tiny_model()
    d = int(model.hidden_size)
    v = torch.randn(d)
    prompt = "the quick brown fox"
    n_prompt = len(model.tokenizer(prompt)["input_ids"])

    ctx = InterventionContext(model).add_steering(layers=[0], vector=v, factor=10.0)
    with model.generate(prompt, max_new_tokens=6, do_sample=False) as _:
        ctx.apply()
        out = model.generator.output.save()
    assert out.shape[1] >= n_prompt + 1  # produced at least one new token
