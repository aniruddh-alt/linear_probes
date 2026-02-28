# Vector Steering Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add inference-time activation steering to ResponseGenerator using PyTorch forward hooks, supporting additive and projection-subtraction modes.

**Architecture:** New `SteeringParams` config + steering logic in `ResponseGenerator`. Forward hooks registered on transformer layer modules intercept the residual stream during `model.generate()`. Steering vector loaded from `.pt` or `.safetensors` files.

**Tech Stack:** PyTorch forward hooks, safetensors, OmegaConf dataclasses

---

### Task 1: SteeringParams Config

**Files:**
- Create: `core/configs/params/steering_params.py`
- Test: `tests/test_steering_params.py`

**Step 1: Write the failing test**

```python
"""Tests for SteeringParams."""
from __future__ import annotations

import pytest

from core.configs.params.steering_params import SteeringParams


class TestSteeringParams:
    def test_defaults(self):
        params = SteeringParams()
        assert params.enabled is False
        assert params.vector_path == ""
        assert params.vector_key == ""
        assert params.layers == []
        assert params.strength == 10.0
        assert params.mode == "project_subtract"
        assert params.normalize is True

    def test_valid_modes(self):
        for mode in ("project_subtract", "additive"):
            p = SteeringParams(mode=mode)
            assert p.mode == mode

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="Invalid steering mode"):
            SteeringParams(mode="invalid")

    def test_enabled_without_vector_path_raises(self):
        with pytest.raises(ValueError, match="vector_path is required"):
            SteeringParams(enabled=True, vector_path="", layers=[0])

    def test_enabled_without_layers_raises(self):
        with pytest.raises(ValueError, match="layers is required"):
            SteeringParams(enabled=True, vector_path="v.pt", layers=[])

    def test_yaml_roundtrip(self, tmp_path):
        params = SteeringParams(
            enabled=True,
            vector_path="weights.pt",
            layers=[14, 15, 16],
            strength=15.0,
            mode="additive",
        )
        path = tmp_path / "steering.yaml"
        params.to_yaml(path)
        loaded = SteeringParams.from_yaml(path)
        assert loaded.enabled is True
        assert loaded.layers == [14, 15, 16]
        assert loaded.strength == 15.0
        assert loaded.mode == "additive"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_steering_params.py -v`
Expected: FAIL with import error

**Step 3: Write minimal implementation**

```python
"""Steering configuration parameters."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from core.configs.base import BaseConfig

_VALID_MODES = frozenset({"project_subtract", "additive"})


@dataclass
class SteeringParams(BaseConfig):
    """Configuration for activation steering during generation."""

    enabled: bool = False
    vector_path: str = ""
    vector_key: str = ""
    layers: list[int] = field(default_factory=list)
    strength: float = 10.0
    mode: str = "project_subtract"
    normalize: bool = True

    def __post_init__(self) -> None:
        if self.mode not in _VALID_MODES:
            raise ValueError(
                f"Invalid steering mode '{self.mode}'. "
                f"Expected one of: {', '.join(sorted(_VALID_MODES))}"
            )
        if self.enabled:
            if not self.vector_path:
                raise ValueError("vector_path is required when steering is enabled.")
            if not self.layers:
                raise ValueError("layers is required when steering is enabled.")
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_steering_params.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add core/configs/params/steering_params.py tests/test_steering_params.py
git commit -m "feat: add SteeringParams config"
```

---

### Task 2: Add SteeringParams to GenerateConfig

**Files:**
- Modify: `core/configs/generate_config.py`
- Modify: `tests/test_generate_config.py`

**Step 1: Write the failing test**

Add to `tests/test_generate_config.py`:

```python
    def test_steering_defaults(self):
        cfg = GenerateConfig()
        assert cfg.steering.enabled is False
        assert cfg.steering.mode == "project_subtract"

    def test_steering_from_dict(self):
        cfg = GenerateConfig.from_dict({
            "model": {"model_name": "test"},
            "steering": {
                "enabled": True,
                "vector_path": "v.pt",
                "layers": [14, 15],
                "strength": 20.0,
                "mode": "additive",
            },
        })
        assert cfg.steering.enabled is True
        assert cfg.steering.layers == [14, 15]
        assert cfg.steering.strength == 20.0

    def test_steering_yaml_roundtrip(self, tmp_path):
        cfg = GenerateConfig.from_dict({
            "steering": {
                "enabled": True,
                "vector_path": "w.pt",
                "layers": [10],
                "mode": "project_subtract",
            },
        })
        path = tmp_path / "gen_steer.yaml"
        cfg.to_yaml(path)
        loaded = GenerateConfig.from_yaml(path)
        assert loaded.steering.enabled is True
        assert loaded.steering.layers == [10]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_generate_config.py -v`
Expected: FAIL with AttributeError

**Step 3: Add steering field to GenerateConfig**

In `core/configs/generate_config.py`, add import and field:

```python
from core.configs.params.steering_params import SteeringParams
```

Add field to the dataclass:

```python
    steering: SteeringParams = field(default_factory=SteeringParams)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_generate_config.py -v`
Expected: all PASS

**Step 5: Commit**

```bash
git add core/configs/generate_config.py tests/test_generate_config.py
git commit -m "feat: add steering field to GenerateConfig"
```

---

### Task 3: Steering Logic in ResponseGenerator

**Files:**
- Modify: `generation/response_generator.py`
- Modify: `tests/test_response_generator.py`

**Step 1: Write failing tests for vector loading**

Add to `tests/test_response_generator.py`:

```python
import torch
from safetensors.torch import save_file

from core.configs.params.steering_params import SteeringParams


class TestSteeringVectorLoading:
    @patch("generation.response_generator.AutoModelForCausalLM")
    @patch("generation.response_generator.AutoTokenizer")
    def test_load_pt_vector(self, mock_tok_cls, mock_model_cls, tmp_path):
        mock_tok = MagicMock()
        mock_tok.pad_token_id = 0
        mock_tok_cls.from_pretrained.return_value = mock_tok
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model_cls.from_pretrained.return_value = mock_model

        vec = torch.randn(64)
        vec_path = tmp_path / "v.pt"
        torch.save(vec, vec_path)

        gen = ResponseGenerator(
            model=ModelParams(model_name="test"),
            steering=SteeringParams(
                enabled=True, vector_path=str(vec_path), layers=[0],
            ),
        )
        assert gen._steering_vector is not None
        assert gen._steering_vector.shape == (64,)
        # Should be normalized by default
        assert torch.allclose(gen._steering_vector.norm(), torch.tensor(1.0), atol=1e-5)

    @patch("generation.response_generator.AutoModelForCausalLM")
    @patch("generation.response_generator.AutoTokenizer")
    def test_load_safetensors_vector(self, mock_tok_cls, mock_model_cls, tmp_path):
        mock_tok = MagicMock()
        mock_tok.pad_token_id = 0
        mock_tok_cls.from_pretrained.return_value = mock_tok
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model_cls.from_pretrained.return_value = mock_model

        vec = torch.randn(64)
        vec_path = tmp_path / "v.safetensors"
        save_file({"direction": vec}, str(vec_path))

        gen = ResponseGenerator(
            model=ModelParams(model_name="test"),
            steering=SteeringParams(
                enabled=True, vector_path=str(vec_path),
                vector_key="direction", layers=[0],
            ),
        )
        assert gen._steering_vector is not None
        assert gen._steering_vector.shape == (64,)

    @patch("generation.response_generator.AutoModelForCausalLM")
    @patch("generation.response_generator.AutoTokenizer")
    def test_no_steering_by_default(self, mock_tok_cls, mock_model_cls):
        mock_tok = MagicMock()
        mock_tok.pad_token_id = 0
        mock_tok_cls.from_pretrained.return_value = mock_tok
        mock_model_cls.from_pretrained.return_value = MagicMock(device="cpu")

        gen = ResponseGenerator(model=ModelParams(model_name="test"))
        assert gen._steering_vector is None
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_response_generator.py::TestSteeringVectorLoading -v`
Expected: FAIL

**Step 3: Write failing tests for hook behavior**

Add to `tests/test_response_generator.py`:

```python
class TestSteeringHooks:
    def test_project_subtract_hook(self):
        """Test projection subtraction: h' = h - (h . w_hat) * w_hat"""
        from generation.response_generator import _make_steering_hook

        vec = torch.tensor([1.0, 0.0, 0.0])  # unit vector along dim 0
        hook = _make_steering_hook(vec, mode="project_subtract", strength=1.0)

        h = torch.tensor([[[3.0, 4.0, 5.0]]])  # (1, 1, 3)
        output = (h,)
        result = hook(None, None, output)

        # Should remove component along vec: h' = [3,4,5] - 3*[1,0,0] = [0,4,5]
        expected = torch.tensor([[[0.0, 4.0, 5.0]]])
        assert torch.allclose(result[0], expected, atol=1e-5)

    def test_additive_hook(self):
        """Test additive steering: h' = h + alpha * w_hat"""
        from generation.response_generator import _make_steering_hook

        vec = torch.tensor([1.0, 0.0, 0.0])
        hook = _make_steering_hook(vec, mode="additive", strength=5.0)

        h = torch.tensor([[[3.0, 4.0, 5.0]]])
        output = (h,)
        result = hook(None, None, output)

        expected = torch.tensor([[[8.0, 4.0, 5.0]]])
        assert torch.allclose(result[0], expected, atol=1e-5)

    def test_hook_preserves_extra_outputs(self):
        """Hook should only modify first element, pass rest through."""
        from generation.response_generator import _make_steering_hook

        vec = torch.tensor([1.0, 0.0])
        hook = _make_steering_hook(vec, mode="additive", strength=1.0)

        h = torch.tensor([[[1.0, 2.0]]])
        extra = {"attention": "value"}
        result = hook(None, None, (h, extra))

        assert len(result) == 2
        assert result[1] == extra
```

**Step 4: Implement steering in ResponseGenerator**

Modify `generation/response_generator.py`:

```python
"""Generate text responses from a model."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.configs.params.generation_params import GenerationParams
from core.configs.params.model_params import ModelParams
from core.configs.params.steering_params import SteeringParams
from dataset.types import SampleBundle
from generation.types import GenerationResult


def _make_steering_hook(
    vector: torch.Tensor,
    mode: str,
    strength: float,
):
    """Create a forward hook that steers the residual stream."""
    def hook(module, input, output):
        h = output[0]  # (batch, seq, hidden)
        if mode == "project_subtract":
            # h' = h - (h . w_hat) * w_hat
            proj = (h @ vector) @ vector.unsqueeze(0)
            h = h - proj
        elif mode == "additive":
            # h' = h + alpha * w_hat
            h = h + strength * vector
        return (h, *output[1:])
    return hook


def _load_vector(path: str, key: str, normalize: bool, device: str) -> torch.Tensor:
    """Load a steering vector from .pt or .safetensors file."""
    p = Path(path)
    if p.suffix == ".safetensors":
        from safetensors.torch import load_file
        tensors = load_file(str(p))
        if not key:
            keys = list(tensors.keys())
            if len(keys) != 1:
                raise ValueError(
                    f"safetensors file has {len(keys)} keys {keys}; "
                    "specify vector_key to select one."
                )
            key = keys[0]
        vec = tensors[key]
    else:
        vec = torch.load(str(p), map_location="cpu", weights_only=True)

    if vec.ndim != 1:
        raise ValueError(f"Steering vector must be 1D, got shape {vec.shape}")
    if normalize:
        vec = vec / vec.norm()
    return vec.to(device)


def _resolve_layer_modules(model):
    """Find the sequential layer modules of a transformer model."""
    # Llama/Qwen/Mistral/Gemma style
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    # GPT-2/GPT-Neo style
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise ValueError(
        "Cannot auto-detect layer structure. "
        "Expected model.model.layers or model.transformer.h"
    )


class ResponseGenerator:
    """Generates text responses from a loaded causal LM."""

    def __init__(
        self,
        model: ModelParams,
        generation: GenerationParams | None = None,
        steering: SteeringParams | None = None,
    ):
        self.model_params = model
        self.generation_params = generation or GenerationParams()
        self.steering_params = steering or SteeringParams()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model.model_name,
            trust_remote_code=model.trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model_kwargs = {}
        if model.dtype is not None:
            model_kwargs["torch_dtype"] = getattr(torch, model.dtype)
        if model.load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        if model.load_in_4bit:
            model_kwargs["load_in_4bit"] = True
        self.model = AutoModelForCausalLM.from_pretrained(
            model.model_name,
            trust_remote_code=model.trust_remote_code,
            low_cpu_mem_usage=model.low_cpu_mem_usage,
            **model_kwargs,
        )

        self._steering_vector: torch.Tensor | None = None
        if self.steering_params.enabled:
            self._steering_vector = _load_vector(
                self.steering_params.vector_path,
                self.steering_params.vector_key,
                self.steering_params.normalize,
                str(self.model.device),
            )

    def generate(
        self,
        samples: SampleBundle | Sequence[str],
    ) -> GenerationResult:
        if isinstance(samples, SampleBundle):
            prompts = [samples.prompts[i] for i in range(len(samples.prompts))]
            sample_ids = list(samples.ids)
            labels = list(samples.labels)
        else:
            prompts = list(samples)
            sample_ids = [str(i) for i in range(len(prompts))]
            labels = [None] * len(prompts)

        gen_params = self.generation_params
        responses: list[str] = []

        # Register steering hooks
        hooks = []
        if self._steering_vector is not None:
            layer_modules = _resolve_layer_modules(self.model)
            for idx in self.steering_params.layers:
                handle = layer_modules[idx].register_forward_hook(
                    _make_steering_hook(
                        self._steering_vector,
                        self.steering_params.mode,
                        self.steering_params.strength,
                    )
                )
                hooks.append(handle)

        try:
            for batch_start in range(0, len(prompts), gen_params.batch_size):
                batch = prompts[batch_start : batch_start + gen_params.batch_size]
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                input_ids = inputs["input_ids"].to(self.model.device)
                attention_mask = inputs["attention_mask"].to(self.model.device)
                input_len = input_ids.shape[1]

                with torch.no_grad():
                    output_ids = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=gen_params.max_new_tokens,
                        temperature=gen_params.temperature,
                        top_p=gen_params.top_p,
                        do_sample=gen_params.do_sample,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

                new_tokens = output_ids[:, input_len:]
                decoded = self.tokenizer.batch_decode(
                    new_tokens, skip_special_tokens=True
                )
                responses.extend(decoded)
        finally:
            for h in hooks:
                h.remove()

        return GenerationResult(
            prompts=prompts,
            responses=responses,
            sample_ids=sample_ids,
            labels=labels,
        )
```

**Step 5: Run all tests**

Run: `pytest tests/test_response_generator.py -v`
Expected: all PASS

**Step 6: Commit**

```bash
git add generation/response_generator.py tests/test_response_generator.py
git commit -m "feat: add activation steering to ResponseGenerator"
```

---

### Task 4: Wire Steering in Experiment Runner

**Files:**
- Modify: `runners/experiment_runner.py:122`
- Modify: `tests/test_experiment_runner_smoke.py`

**Step 1: Write the failing test**

Add to test file a test that verifies steering params are passed through:

```python
def test_action_generate_passes_steering(self, ...):
    # Verify ResponseGenerator is constructed with steering param
```

**Step 2: Update _action_generate**

Change line 122 in `runners/experiment_runner.py`:

```python
# Before:
generator = ResponseGenerator(model=cfg.model, generation=cfg.generation)

# After:
generator = ResponseGenerator(
    model=cfg.model, generation=cfg.generation, steering=cfg.steering,
)
```

**Step 3: Run tests**

Run: `pytest tests/ -v`
Expected: all PASS

**Step 4: Commit**

```bash
git add runners/experiment_runner.py tests/test_experiment_runner_smoke.py
git commit -m "feat: wire steering params through experiment runner"
```

---

### Task 5: Run Full Test Suite

**Step 1: Run all tests**

Run: `pytest tests/ -v`
Expected: all 106+ tests PASS

**Step 2: Final commit if any fixups needed**
