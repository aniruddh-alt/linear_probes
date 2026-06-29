"""Tests for sonde.interventions (R-1: additive steering + vector loading).

These tests do not load a real transformer. They use a small mock of nnterp's
``StandardizedTransformer`` surface (``num_layers``, ``trace``, ``steer``,
``generate``) to verify that the InterventionContext buffers, applies, and
unwinds steering operations correctly. Real-model smoke tests belong in a
separate file and are run on demand.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any

import pytest
import torch
from safetensors.torch import save_file

from sonde.core.configs.params.steering_params import SteeringParams
from sonde.interventions import (
    InterventionContext,
    PendingSteer,
    load_vector,
)
from sonde.interventions.steering import apply_pending_steers

# ─────────────────────────── Mock model ───────────────────────────


class _MockTrace(AbstractContextManager):
    def __init__(self, model: _MockModel, prompts: Any):
        self.model = model
        self.prompts = prompts

    def __enter__(self) -> _MockTrace:
        self.model.trace_active = True
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.model.trace_active = False


class _MockGenerate(AbstractContextManager):
    def __init__(self, model: _MockModel, prompts: Any, kwargs: dict[str, Any]):
        self.model = model
        self.prompts = prompts
        self.kwargs = kwargs

    def __enter__(self) -> _MockGenerate:
        self.model.generate_active = True
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.model.generate_active = False


@dataclass
class _SteerCall:
    layers: list[int]
    steering_vector: torch.Tensor
    factor: float
    positions: int | list[int] | None


class _MockModel:
    """Minimal stand-in for nnterp.StandardizedTransformer for unit tests."""

    def __init__(self, num_layers: int = 4):
        self.num_layers = num_layers
        self.steer_calls: list[_SteerCall] = []
        self.trace_active = False
        self.generate_active = False

    def trace(self, prompts: Any) -> _MockTrace:
        return _MockTrace(self, prompts)

    def generate(self, prompts: Any, **kwargs: Any) -> _MockGenerate:
        return _MockGenerate(self, prompts, kwargs)

    def steer(
        self,
        *,
        layers: int | list[int],
        steering_vector: torch.Tensor,
        factor: float = 1.0,
        positions: int | list[int] | None = None,
    ) -> None:
        if not (self.trace_active or self.generate_active):
            raise RuntimeError("steer() called outside of trace/generate context")
        self.steer_calls.append(
            _SteerCall(
                layers=list(layers) if isinstance(layers, list) else [int(layers)],
                steering_vector=steering_vector.detach().clone(),
                factor=float(factor),
                positions=positions,
            )
        )


# ─────────────────────────── load_vector ───────────────────────────


class TestLoadVector:
    def test_from_tensor_returns_unit_norm(self):
        raw = torch.tensor([3.0, 4.0])
        out = load_vector(raw)
        assert out.shape == (2,)
        assert float(torch.linalg.vector_norm(out).item()) == pytest.approx(1.0)

    def test_scaling_invariance_under_normalize(self):
        raw = torch.tensor([1.0, 2.0, 3.0])
        a = load_vector(raw)
        b = load_vector(raw * 10.0)
        assert torch.allclose(a, b, atol=1e-6)

    def test_no_normalize_preserves_magnitude(self):
        raw = torch.tensor([3.0, 4.0])
        out = load_vector(raw, normalize=False)
        assert float(torch.linalg.vector_norm(out).item()) == pytest.approx(5.0)

    def test_zero_vector_returned_unchanged(self):
        raw = torch.zeros(8)
        out = load_vector(raw)
        assert torch.equal(out, torch.zeros(8))

    def test_from_safetensors_single_key(self, tmp_path):
        path = tmp_path / "vec.safetensors"
        save_file({"d": torch.tensor([1.0, 0.0, 0.0])}, str(path))
        out = load_vector(str(path))
        assert out.shape == (3,)
        assert float(torch.linalg.vector_norm(out).item()) == pytest.approx(1.0)

    def test_from_safetensors_multi_key_requires_key(self, tmp_path):
        path = tmp_path / "vec.safetensors"
        save_file(
            {"a": torch.tensor([1.0, 0.0]), "b": torch.tensor([0.0, 1.0])},
            str(path),
        )
        with pytest.raises(ValueError, match="multiple keys"):
            load_vector(str(path))
        out = load_vector(str(path), key="b")
        assert torch.allclose(out, torch.tensor([0.0, 1.0]))

    def test_from_safetensors_missing_key_raises(self, tmp_path):
        path = tmp_path / "vec.safetensors"
        save_file({"d": torch.tensor([1.0, 0.0])}, str(path))
        with pytest.raises(KeyError, match="no key 'missing'"):
            load_vector(str(path), key="missing")

    def test_from_pt_single_tensor(self, tmp_path):
        path = tmp_path / "vec.pt"
        torch.save(torch.tensor([0.0, 1.0, 0.0]), path)
        out = load_vector(str(path))
        assert torch.allclose(out, torch.tensor([0.0, 1.0, 0.0]))

    def test_from_pt_state_dict_requires_key(self, tmp_path):
        path = tmp_path / "vec.pt"
        torch.save({"linear.weight": torch.tensor([[1.0, 2.0]])}, path)
        with pytest.raises(ValueError, match="state-dict-like"):
            load_vector(str(path))
        out = load_vector(str(path), key="linear.weight")
        assert out.shape == (2,)  # squeezed

    def test_from_object_with_direction(self):
        class Direction:
            direction = torch.tensor([1.0, 1.0, 0.0])

        out = load_vector(Direction())
        assert out.shape == (3,)
        assert float(torch.linalg.vector_norm(out).item()) == pytest.approx(1.0)

    def test_from_sweep_result_shape_with_best_key(self):
        class _Probe:
            class model:
                direction = torch.tensor([1.0, 0.0])

        class _Sweep:
            best_key = "layers_output:5"
            probes = {"layers_output:5": _Probe()}

        out = load_vector(_Sweep())
        assert torch.allclose(out.abs(), torch.tensor([1.0, 0.0]))

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="Cannot extract"):
            load_vector(object())

    def test_unsupported_extension_raises(self, tmp_path):
        path = tmp_path / "vec.npy"
        path.write_bytes(b"x")
        with pytest.raises(ValueError, match="Unsupported file extension"):
            load_vector(str(path))


# ─────────────────────────── InterventionContext ───────────────────────────


class TestInterventionContextConfiguration:
    def test_add_steering_records_pending(self):
        model = _MockModel(num_layers=4)
        v = torch.tensor([1.0, 0.0])
        ctx = InterventionContext(model).add_steering(
            layers=[1, 3], vector=v, factor=0.5
        )
        assert len(ctx.pending_steers) == 1
        s = ctx.pending_steers[0]
        assert s.layers == [1, 3]
        assert s.factor == 0.5
        assert s.mode == "additive"

    def test_chain_returns_self(self):
        model = _MockModel(num_layers=4)
        ctx = InterventionContext(model)
        result = ctx.add_steering(layers=0, vector=torch.tensor([1.0, 0.0]))
        assert result is ctx

    def test_layers_all_expands(self):
        model = _MockModel(num_layers=6)
        ctx = InterventionContext(model).add_steering(
            layers="all", vector=torch.tensor([1.0, 0.0])
        )
        assert ctx.pending_steers[0].layers == [0, 1, 2, 3, 4, 5]

    def test_layers_int_normalized_to_list(self):
        model = _MockModel(num_layers=4)
        ctx = InterventionContext(model).add_steering(
            layers=2, vector=torch.tensor([1.0, 0.0])
        )
        assert ctx.pending_steers[0].layers == [2]

    def test_layers_empty_list_raises(self):
        model = _MockModel(num_layers=4)
        with pytest.raises(ValueError, match="cannot be empty"):
            InterventionContext(model).add_steering(
                layers=[], vector=torch.tensor([1.0, 0.0])
            )

    def test_unknown_mode_raises(self):
        model = _MockModel(num_layers=4)
        with pytest.raises(ValueError, match="Unknown steering mode"):
            InterventionContext(model).add_steering(
                layers=[0], vector=torch.tensor([1.0, 0.0]), mode="rotation"
            )

    def test_clear_drops_pending(self):
        model = _MockModel(num_layers=4)
        ctx = InterventionContext(model).add_steering(
            layers=[0], vector=torch.tensor([1.0, 0.0])
        )
        ctx.clear()
        assert ctx.pending_steers == []


class TestInterventionContextApplication:
    def test_trace_applies_pending_steers(self):
        model = _MockModel(num_layers=4)
        v = torch.tensor([1.0, 0.0])
        ctx = InterventionContext(model).add_steering(
            layers=[1, 3], vector=v, factor=0.5
        )
        with ctx.trace("hello"):
            pass
        assert len(model.steer_calls) == 1
        call = model.steer_calls[0]
        assert call.layers == [1, 3]
        assert call.factor == 0.5
        assert torch.allclose(call.steering_vector, v / v.norm())
        assert call.positions is None
        assert model.trace_active is False  # restored on exit

    def test_generate_applies_pending_steers(self):
        model = _MockModel(num_layers=4)
        v = torch.tensor([0.0, 1.0])
        ctx = InterventionContext(model).add_steering(layers=2, vector=v)
        ctx.generate("hello", max_new_tokens=8)
        assert len(model.steer_calls) == 1
        assert model.steer_calls[0].layers == [2]
        assert model.generate_active is False

    def test_positions_forwarded(self):
        model = _MockModel(num_layers=4)
        ctx = InterventionContext(model).add_steering(
            layers=[0], vector=torch.tensor([1.0, 0.0]), positions=[-1]
        )
        with ctx.trace("hi"):
            pass
        assert model.steer_calls[0].positions == [-1]

    def test_multiple_steers_applied_in_order(self):
        model = _MockModel(num_layers=4)
        ctx = (
            InterventionContext(model)
            .add_steering(layers=[0], vector=torch.tensor([1.0, 0.0]), factor=1.0)
            .add_steering(layers=[2], vector=torch.tensor([0.0, 1.0]), factor=-1.0)
        )
        with ctx.trace("hi"):
            pass
        assert [c.layers for c in model.steer_calls] == [[0], [2]]
        assert [c.factor for c in model.steer_calls] == [1.0, -1.0]

    def test_project_subtract_mode_raises_until_r2(self):
        model = _MockModel(num_layers=4)
        ctx = InterventionContext(model).add_steering(
            layers=[0],
            vector=torch.tensor([1.0, 0.0]),
            mode="project_subtract",
        )
        with pytest.raises(NotImplementedError, match="R-2"), ctx.trace("hi"):
            pass

    def test_apply_pending_steers_outside_trace_errors(self):
        """The mock raises if steer() is called outside trace/generate — this
        documents the contract that the context manager guards entry."""
        model = _MockModel(num_layers=4)
        pending = [
            PendingSteer(
                layers=[0],
                vector=torch.tensor([1.0, 0.0]),
                factor=1.0,
                mode="additive",
            )
        ]
        with pytest.raises(RuntimeError, match="outside of trace"):
            apply_pending_steers(model, pending)


class TestInterventionContextFromConfig:
    def test_none_yields_empty_context(self):
        model = _MockModel(num_layers=4)
        ctx = InterventionContext.from_config(model, steering=None)
        assert ctx.pending_steers == []

    def test_disabled_block_skipped(self, tmp_path):
        path = tmp_path / "v.safetensors"
        save_file({"d": torch.tensor([1.0, 0.0])}, str(path))
        block = SteeringParams(
            enabled=False, vector_path=str(path), layers=[0], strength=1.0
        )
        ctx = InterventionContext.from_config(_MockModel(), steering=block)
        assert ctx.pending_steers == []

    def test_single_block_added(self, tmp_path):
        path = tmp_path / "v.safetensors"
        save_file({"d": torch.tensor([1.0, 0.0])}, str(path))
        block = SteeringParams(
            enabled=True,
            vector_path=str(path),
            layers=[0, 2],
            strength=0.5,
            mode="additive",
        )
        ctx = InterventionContext.from_config(_MockModel(num_layers=4), steering=block)
        assert len(ctx.pending_steers) == 1
        assert ctx.pending_steers[0].layers == [0, 2]
        assert ctx.pending_steers[0].factor == 0.5

    def test_factor_property_aliases_strength(self, tmp_path):
        path = tmp_path / "v.safetensors"
        save_file({"d": torch.tensor([1.0, 0.0])}, str(path))
        block = SteeringParams(
            enabled=True, vector_path=str(path), layers=[1], strength=3.0
        )
        assert block.factor == 3.0
        ctx = InterventionContext.from_config(_MockModel(num_layers=4), steering=block)
        assert ctx.pending_steers[0].factor == 3.0

    def test_list_of_blocks_preserves_order(self, tmp_path):
        path = tmp_path / "v.safetensors"
        save_file({"d": torch.tensor([1.0, 0.0])}, str(path))
        blocks = [
            SteeringParams(
                enabled=True, vector_path=str(path), layers=[0], strength=1.0
            ),
            SteeringParams(
                enabled=True, vector_path=str(path), layers=[3], strength=-1.0
            ),
        ]
        ctx = InterventionContext.from_config(_MockModel(num_layers=4), steering=blocks)
        assert [s.layers for s in ctx.pending_steers] == [[0], [3]]
        assert [s.factor for s in ctx.pending_steers] == [1.0, -1.0]
