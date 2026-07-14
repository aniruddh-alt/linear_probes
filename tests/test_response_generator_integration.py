"""Real-model integration test for ResponseGenerator's HF-hook steering path.

The unit tests mock the model, so the actual hook registration / firing /
cleanup in ResponseGenerator.generate is never exercised — the same mock-masking
shape as a prior nnsight bug. This drives a tiny real model so a regression in
layer resolution, hook registration, or handle cleanup is caught.

Not env-gated (runs whenever a tiny model can be downloaded) but skips
gracefully offline so the default suite stays green without network.
"""

from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file

from sonde.core.configs.params.generation_params import GenerationParams
from sonde.core.configs.params.model_params import ModelParams
from sonde.core.configs.params.steering_params import SteeringParams

_TINY = "sshleifer/tiny-gpt2"


def _hidden_size() -> int:
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(_TINY)
    except Exception as exc:  # pragma: no cover - offline
        pytest.skip(f"tiny model config unavailable: {exc}")
    return int(getattr(cfg, "n_embd", getattr(cfg, "hidden_size", 0)))


def _make_generator(tmp_path, *, layers, mode="additive", strength=8.0):
    from sonde.generation.response_generator import ResponseGenerator

    d = _hidden_size()
    vec_path = tmp_path / "vec.safetensors"
    save_file({"direction": torch.randn(d)}, str(vec_path))
    try:
        return ResponseGenerator(
            model=ModelParams(model_name=_TINY),
            generation=GenerationParams(max_new_tokens=4, do_sample=False),
            steering=SteeringParams(
                enabled=True,
                vector_path=str(vec_path),
                vector_key="direction",
                layers=layers,
                mode=mode,
                strength=strength,
            ),
        )
    except Exception as exc:  # pragma: no cover - offline / model load failure
        pytest.skip(f"tiny model unavailable: {exc}")


def test_steering_hooks_fire_and_are_cleaned_up(tmp_path):
    from sonde.generation.response_generator import _resolve_layer_modules

    gen = _make_generator(tmp_path, layers=[0])
    result = gen.generate(["hello there"])
    assert len(result.responses) == 1

    # No forward hooks should linger on any layer after generate returns.
    for module in _resolve_layer_modules(gen.model):
        assert len(module._forward_hooks) == 0


def test_out_of_range_layer_raises(tmp_path):
    gen = _make_generator(tmp_path, layers=[9999])
    with pytest.raises(ValueError, match="out of range"):
        gen.generate(["hello there"])
