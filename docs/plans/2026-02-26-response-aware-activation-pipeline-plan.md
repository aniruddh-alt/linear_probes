# Response-Aware Activation Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `generation/` module that generates text responses from a model, enabling post-hoc labeling workflows where labels depend on model behavior.

**Architecture:** New `ResponseGenerator` class produces `GenerationResult` (prompts, responses, sample_ids, labels). Results export to JSONL for external labeling and convert back to `SampleBundle` for activation extraction. `SampleBundle` gains an optional `responses` field. Config and runner updated with `GenerationParams` and `"generate"` action.

**Tech Stack:** Python dataclasses, transformers `model.generate()`, JSON, OmegaConf configs, pytest

---

### Task 1: GenerationParams Config

**Files:**
- Create: `core/configs/params/generation_params.py`
- Modify: `core/configs/__init__.py`
- Modify: `core/configs/run_config.py`
- Test: `tests/test_generation_params.py`

**Step 1: Write the failing test**

Create `tests/test_generation_params.py`:

```python
"""Tests for GenerationParams config."""

from __future__ import annotations

from core.configs.params.generation_params import GenerationParams


class TestGenerationParams:
    def test_defaults(self):
        params = GenerationParams()
        assert params.max_new_tokens == 256
        assert params.temperature == 1.0
        assert params.top_p == 1.0
        assert params.do_sample is False
        assert params.batch_size == 8

    def test_round_trip_yaml(self, tmp_path):
        params = GenerationParams(max_new_tokens=128, temperature=0.7)
        path = tmp_path / "gen.yaml"
        params.to_yaml(path)
        loaded = GenerationParams.from_yaml(path)
        assert loaded.max_new_tokens == 128
        assert loaded.temperature == 0.7

    def test_run_config_includes_generation(self):
        from core.configs.run_config import RunConfig
        cfg = RunConfig()
        assert hasattr(cfg, "generation")
        assert cfg.generation.max_new_tokens == 256

    def test_run_config_yaml_with_generation(self, tmp_path):
        from core.configs.run_config import RunConfig
        yaml_text = """
run_name: gen-test
action: generate
model:
  model_name: test-model
generation:
  max_new_tokens: 64
  temperature: 0.5
"""
        path = tmp_path / "run.yaml"
        path.write_text(yaml_text, encoding="utf-8")
        cfg = RunConfig.from_yaml(path)
        assert cfg.generation.max_new_tokens == 64
        assert cfg.generation.temperature == 0.5
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_generation_params.py -v`
Expected: FAIL (module not found)

**Step 3: Write minimal implementation**

Create `core/configs/params/generation_params.py`:

```python
"""Generation configuration parameters."""

from __future__ import annotations

from dataclasses import dataclass

from core.configs.base import BaseConfig


@dataclass
class GenerationParams(BaseConfig):
    """Configuration for text generation."""

    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    do_sample: bool = False
    batch_size: int = 8
```

Modify `core/configs/__init__.py` — add the import and `__all__` entry:

```python
from core.configs.params.generation_params import GenerationParams
```

Add `"GenerationParams"` to the `__all__` list.

Modify `core/configs/run_config.py` — add `generation` field:

```python
from core.configs.params.generation_params import GenerationParams
```

Add to the `RunConfig` dataclass:

```python
    generation: GenerationParams = field(default_factory=GenerationParams)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_generation_params.py -v`
Expected: PASS (4 tests)

**Step 5: Run full test suite to check no regressions**

Run: `pytest -v`
Expected: All existing tests still pass

**Step 6: Commit**

```bash
git add core/configs/params/generation_params.py core/configs/__init__.py core/configs/run_config.py tests/test_generation_params.py
git commit -m "feat: add GenerationParams config and wire into RunConfig"
```

---

### Task 2: GenerationResult Type and Serialization

**Files:**
- Create: `generation/__init__.py`
- Create: `generation/types.py`
- Test: `tests/test_generation_result.py`

**Step 1: Write the failing test**

Create `tests/test_generation_result.py`:

```python
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
        assert all(l is None for l in result.labels)

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
        assert len(bundle.prompts) == 3
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
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_generation_result.py -v`
Expected: FAIL (module not found)

**Step 3: Write minimal implementation**

Create `generation/__init__.py`:

```python
"""Response generation module."""
```

Create `generation/types.py`:

```python
"""Data types for the generation module."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from dataset.samples import StringDataset
from dataset.types import SampleBundle


@dataclass
class GenerationResult:
    """Holds prompt-response pairs with optional labels."""

    prompts: list[str]
    responses: list[str]
    sample_ids: list[str]
    labels: list[int | None]

    def __post_init__(self) -> None:
        n = len(self.prompts)
        if len(self.responses) != n:
            raise ValueError(
                f"prompts ({n}) and responses ({len(self.responses)}) must have equal length."
            )
        if len(self.sample_ids) != n:
            raise ValueError(
                f"prompts ({n}) and sample_ids ({len(self.sample_ids)}) must have equal length."
            )
        if len(self.labels) != n:
            raise ValueError(
                f"prompts ({n}) and labels ({len(self.labels)}) must have equal length."
            )

    def to_jsonl(self, path: str | Path) -> None:
        """Export prompt-response pairs to JSONL for external labeling."""
        with Path(path).open("w", encoding="utf-8") as f:
            for prompt, response, sid, label in zip(
                self.prompts, self.responses, self.sample_ids, self.labels
            ):
                row = {
                    "prompt": prompt,
                    "response": response,
                    "sample_id": sid,
                    "label": label,
                }
                f.write(json.dumps(row) + "\n")

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "GenerationResult":
        """Load a GenerationResult from a JSONL file."""
        prompts, responses, sample_ids, labels = [], [], [], []
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                prompts.append(row["prompt"])
                responses.append(row["response"])
                sample_ids.append(row["sample_id"])
                labels.append(row.get("label"))
        return cls(
            prompts=prompts,
            responses=responses,
            sample_ids=sample_ids,
            labels=labels,
        )

    def to_sample_bundle(
        self, *, include_response: bool = False
    ) -> SampleBundle:
        """Convert to SampleBundle for activation extraction.

        Args:
            include_response: If True, prompts become prompt+response
                concatenated. If False, original prompts only.
        """
        if include_response:
            texts = [f"{p}{r}" for p, r in zip(self.prompts, self.responses)]
        else:
            texts = list(self.prompts)
        return SampleBundle(
            prompts=StringDataset(texts),
            labels=list(self.labels),
            ids=list(self.sample_ids),
            responses=list(self.responses),
        )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_generation_result.py -v`
Expected: PASS (8 tests)

**Step 5: Commit**

```bash
git add generation/__init__.py generation/types.py tests/test_generation_result.py
git commit -m "feat: add GenerationResult type with JSONL serialization"
```

---

### Task 3: Add `responses` Field to SampleBundle

**Files:**
- Modify: `dataset/types.py`
- Test: `tests/test_sample_bundle_responses.py`

**Step 1: Write the failing test**

Create `tests/test_sample_bundle_responses.py`:

```python
"""Tests for SampleBundle responses field."""

from __future__ import annotations

from dataset.samples import StringDataset
from dataset.types import SampleBundle


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
            labels=[0] * 20 + [1] * 20,
            ids=[f"a-{i}" for i in range(20)] + [f"b-{i}" for i in range(20)],
        )
        train, val, test = bundle.train_val_test_split(seed=42)
        assert len(train) + len(val) + len(test) == 40
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_sample_bundle_responses.py -v`
Expected: FAIL (`__init__() got an unexpected keyword argument 'responses'` for the second test)

**Step 3: Write minimal implementation**

Modify `dataset/types.py` — add `responses` field to `SampleBundle`:

```python
    responses: list[str] | None = None
```

Add it after the `ids` field.

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_sample_bundle_responses.py -v`
Expected: PASS (3 tests)

**Step 5: Run full suite**

Run: `pytest -v`
Expected: All tests pass (no regressions — `responses=None` is the default)

**Step 6: Commit**

```bash
git add dataset/types.py tests/test_sample_bundle_responses.py
git commit -m "feat: add optional responses field to SampleBundle"
```

---

### Task 4: ResponseGenerator Class

**Files:**
- Create: `generation/response_generator.py`
- Modify: `generation/__init__.py`
- Test: `tests/test_response_generator.py`

**Step 1: Write the failing test**

Create `tests/test_response_generator.py`:

```python
"""Tests for ResponseGenerator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.configs.params.generation_params import GenerationParams
from core.configs.params.model_params import ModelParams
from dataset.samples import StringDataset
from dataset.types import SampleBundle
from generation.response_generator import ResponseGenerator
from generation.types import GenerationResult


class TestResponseGenerator:
    @pytest.fixture
    def mock_model(self):
        model = MagicMock()
        model.tokenizer = MagicMock()
        model.tokenizer.return_value = {"input_ids": MagicMock()}
        model.tokenizer.batch_decode.return_value = ["response 0", "response 1"]
        model.tokenizer.pad_token_id = 0
        model.generate.return_value = MagicMock()
        return model

    @patch("generation.response_generator.AutoModelForCausalLM")
    @patch("generation.response_generator.AutoTokenizer")
    def test_generate_from_strings(self, mock_tok_cls, mock_model_cls, mock_model):
        mock_tok = MagicMock()
        mock_tok.pad_token_id = 0
        mock_tok.pad_token = "<pad>"
        mock_tok.eos_token_id = 1
        mock_tok.return_value = {"input_ids": MagicMock(to=MagicMock(return_value=MagicMock())), "attention_mask": MagicMock(to=MagicMock(return_value=MagicMock()))}
        mock_tok.batch_decode.return_value = ["response 0", "response 1"]
        mock_tok_cls.from_pretrained.return_value = mock_tok
        mock_model_obj = MagicMock()
        mock_model_obj.generate.return_value = MagicMock()
        mock_model_obj.device = "cpu"
        mock_model_cls.from_pretrained.return_value = mock_model_obj

        gen = ResponseGenerator(
            model=ModelParams(model_name="test-model"),
            generation=GenerationParams(max_new_tokens=32, batch_size=2),
        )
        result = gen.generate(["prompt 0", "prompt 1"])

        assert isinstance(result, GenerationResult)
        assert result.prompts == ["prompt 0", "prompt 1"]
        assert len(result.responses) == 2
        assert result.labels == [None, None]

    @patch("generation.response_generator.AutoModelForCausalLM")
    @patch("generation.response_generator.AutoTokenizer")
    def test_generate_from_sample_bundle(self, mock_tok_cls, mock_model_cls):
        mock_tok = MagicMock()
        mock_tok.pad_token_id = 0
        mock_tok.pad_token = "<pad>"
        mock_tok.eos_token_id = 1
        mock_tok.return_value = {"input_ids": MagicMock(to=MagicMock(return_value=MagicMock())), "attention_mask": MagicMock(to=MagicMock(return_value=MagicMock()))}
        mock_tok.batch_decode.return_value = ["resp"]
        mock_tok_cls.from_pretrained.return_value = mock_tok
        mock_model_obj = MagicMock()
        mock_model_obj.generate.return_value = MagicMock()
        mock_model_obj.device = "cpu"
        mock_model_cls.from_pretrained.return_value = mock_model_obj

        bundle = SampleBundle(
            prompts=StringDataset(["hello"]),
            labels=[1],
            ids=["s0"],
        )
        gen = ResponseGenerator(
            model=ModelParams(model_name="test-model"),
            generation=GenerationParams(batch_size=1),
        )
        result = gen.generate(bundle)

        assert isinstance(result, GenerationResult)
        assert result.prompts == ["hello"]
        assert result.labels == [1]
        assert result.sample_ids == ["s0"]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_response_generator.py -v`
Expected: FAIL (module not found)

**Step 3: Write minimal implementation**

Create `generation/response_generator.py`:

```python
"""Generate text responses from a model."""

from __future__ import annotations

from typing import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.configs.params.generation_params import GenerationParams
from core.configs.params.model_params import ModelParams
from dataset.types import SampleBundle
from generation.types import GenerationResult


class ResponseGenerator:
    """Generates text responses from a loaded causal LM."""

    def __init__(
        self,
        model: ModelParams,
        generation: GenerationParams | None = None,
    ):
        self.model_params = model
        self.generation_params = generation or GenerationParams()
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

    def generate(
        self,
        samples: SampleBundle | Sequence[str],
    ) -> GenerationResult:
        """Generate responses for each prompt.

        Args:
            samples: Prompts as a SampleBundle or list of strings.

        Returns:
            GenerationResult with prompts, responses, sample_ids, labels.
        """
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

        return GenerationResult(
            prompts=prompts,
            responses=responses,
            sample_ids=sample_ids,
            labels=labels,
        )
```

Update `generation/__init__.py`:

```python
"""Response generation module."""

from generation.response_generator import ResponseGenerator
from generation.types import GenerationResult

__all__ = ["GenerationResult", "ResponseGenerator"]
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_response_generator.py -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add generation/response_generator.py generation/__init__.py tests/test_response_generator.py
git commit -m "feat: add ResponseGenerator class"
```

---

### Task 5: Wire Runner `"generate"` Action

**Files:**
- Modify: `runners/experiment_runner.py`
- Modify: `tests/test_experiment_runner_smoke.py`

**Step 1: Write the failing test**

Add to `tests/test_experiment_runner_smoke.py`:

```python
    def test_generate_action_dispatches(self, tmp_path):
        config_path = tmp_path / "run.yaml"
        config_path.write_text(
            "run_name: gen\naction: generate\n"
            "model:\n  model_name: test-model\n"
            "generation:\n  max_new_tokens: 64\n",
            encoding="utf-8",
        )
        result = run_experiment(config_path=config_path)
        assert result.summary["action"] == "generate"
        assert result.summary["status"] == "configured"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_experiment_runner_smoke.py::TestExperimentRunner::test_generate_action_dispatches -v`
Expected: FAIL (`Unknown action 'generate'`)

**Step 3: Write minimal implementation**

Add to `runners/experiment_runner.py`:

In `dispatch_action`, add `"generate"` to the handlers dict:

```python
        "generate": _action_generate,
```

Add the handler function:

```python
def _action_generate(cfg: RunConfig) -> RunResult:
    """Placeholder for response generation action wiring."""
    return RunResult(
        summary={
            "run_name": cfg.run_name,
            "action": cfg.action,
            "model": cfg.model.model_name,
            "max_new_tokens": cfg.generation.max_new_tokens,
            "status": "configured",
        },
    )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_experiment_runner_smoke.py -v`
Expected: PASS (all tests including new one)

**Step 5: Run full suite**

Run: `pytest -v`
Expected: All tests pass

**Step 6: Commit**

```bash
git add runners/experiment_runner.py tests/test_experiment_runner_smoke.py
git commit -m "feat: add generate action to runner"
```

---

### Task 6: Full Suite Verification and Cleanup

**Step 1: Run full test suite**

Run: `pytest -v`
Expected: All tests pass (original 72 + new tests)

**Step 2: Verify imports are clean**

Run: `python -c "from generation import ResponseGenerator, GenerationResult; from core.configs import GenerationParams; print('imports OK')"`
Expected: `imports OK`

**Step 3: Commit any cleanup if needed**

Only if there are changes from cleanup.
