# Stage-Based Config Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the `RunConfig` monolith with three stage-specific config classes (`GenerateConfig`, `ExtractConfig`, `ProbeConfig`) plus a shared `IOParams` for artifact chaining, so each YAML only contains the fields relevant to its stage.

**Architecture:** Each stage config is a slim `BaseConfig` dataclass. The runner peeks at the `action:` field in the YAML to determine which config class to load. A new `IOParams` carries `input_path` + `output_dir` so stages chain by convention (each stage writes a predictable artifact; the next stage finds it automatically) with explicit override when needed.

**Tech Stack:** Python dataclasses, OmegaConf (structured configs + merge), pytest. All existing `BaseConfig`, `ModelParams`, `ExtractionParams`, `ProbeParams`, `SplitParams`, `SweepParams`, `GenerationParams` are unchanged — new code only adds wrappers.

---

### Task 1: IOParams

**Files:**
- Create: `core/configs/params/io_params.py`
- Modify: `core/configs/params/__init__.py`
- Test: `tests/test_io_params.py`

**Step 1: Write the failing test**

```python
# tests/test_io_params.py
"""Tests for IOParams config."""
from __future__ import annotations
from core.configs.params.io_params import IOParams


class TestIOParams:
    def test_defaults(self):
        p = IOParams()
        assert p.input_path == ""
        assert p.output_dir == "artifacts"

    def test_from_dict(self):
        p = IOParams.from_dict({"input_path": "data/labeled.jsonl", "output_dir": "out/"})
        assert p.input_path == "data/labeled.jsonl"
        assert p.output_dir == "out/"

    def test_yaml_roundtrip(self, tmp_path):
        p = IOParams(input_path="x/y.jsonl", output_dir="z/")
        path = tmp_path / "io.yaml"
        p.to_yaml(path)
        loaded = IOParams.from_yaml(path)
        assert loaded.input_path == "x/y.jsonl"
        assert loaded.output_dir == "z/"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_io_params.py -v
```
Expected: `ModuleNotFoundError: No module named 'core.configs.params.io_params'`

**Step 3: Write the implementation**

```python
# core/configs/params/io_params.py
"""Artifact I/O path configuration for stage chaining."""
from __future__ import annotations
from dataclasses import dataclass
from core.configs.base import BaseConfig


@dataclass
class IOParams(BaseConfig):
    """Input/output paths for stage-based artifact chaining."""

    input_path: str = ""
    output_dir: str = "artifacts"
```

**Step 4: Export from params `__init__`**

In `core/configs/params/__init__.py`, add:
```python
from core.configs.params.io_params import IOParams
```
And add `"IOParams"` to `__all__` if present.

**Step 5: Run tests to verify they pass**

```bash
pytest tests/test_io_params.py -v
```
Expected: 3 PASSED

**Step 6: Commit**

```bash
git add core/configs/params/io_params.py core/configs/params/__init__.py tests/test_io_params.py
git commit -m "feat: add IOParams for stage artifact chaining"
```

---

### Task 2: GenerateConfig

**Files:**
- Create: `core/configs/generate_config.py`
- Test: `tests/test_generate_config.py`

**Step 1: Write the failing test**

```python
# tests/test_generate_config.py
"""Tests for GenerateConfig."""
from __future__ import annotations
from core.configs.generate_config import GenerateConfig


class TestGenerateConfig:
    def test_defaults(self):
        cfg = GenerateConfig()
        assert cfg.run_name == ""
        assert cfg.seed == 0
        assert cfg.action == "generate"
        assert cfg.model.model_name == ""
        assert cfg.generation.max_new_tokens == 256
        assert cfg.io.output_dir == "artifacts"

    def test_from_dict(self):
        cfg = GenerateConfig.from_dict({
            "run_name": "test_gen",
            "model": {"model_name": "Qwen/Qwen2.5-1.5B-Instruct"},
            "generation": {"max_new_tokens": 128},
            "io": {"output_dir": "artifacts/refusal/"},
        })
        assert cfg.run_name == "test_gen"
        assert cfg.model.model_name == "Qwen/Qwen2.5-1.5B-Instruct"
        assert cfg.generation.max_new_tokens == 128
        assert cfg.io.output_dir == "artifacts/refusal/"

    def test_yaml_roundtrip(self, tmp_path):
        cfg = GenerateConfig(run_name="gen_test")
        cfg.model.model_name = "test-model"
        path = tmp_path / "gen.yaml"
        cfg.to_yaml(path)
        loaded = GenerateConfig.from_yaml(path)
        assert loaded.run_name == "gen_test"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_generate_config.py -v
```
Expected: `ModuleNotFoundError: No module named 'core.configs.generate_config'`

**Step 3: Write the implementation**

```python
# core/configs/generate_config.py
"""Config for the generate stage."""
from __future__ import annotations
from dataclasses import dataclass, field
from core.configs.base import BaseConfig
from core.configs.params.generation_params import GenerationParams
from core.configs.params.io_params import IOParams
from core.configs.params.model_params import ModelParams


@dataclass
class GenerateConfig(BaseConfig):
    """Configuration for the generate stage (action: generate)."""

    run_name: str = ""
    seed: int = 0
    action: str = "generate"
    model: ModelParams = field(default_factory=ModelParams)
    generation: GenerationParams = field(default_factory=GenerationParams)
    io: IOParams = field(default_factory=IOParams)
```

**Step 4: Run tests**

```bash
pytest tests/test_generate_config.py -v
```
Expected: 3 PASSED

**Step 5: Commit**

```bash
git add core/configs/generate_config.py tests/test_generate_config.py
git commit -m "feat: add GenerateConfig stage config"
```

---

### Task 3: ExtractConfig

**Files:**
- Create: `core/configs/extract_config.py`
- Test: `tests/test_extract_config.py`

**Step 1: Write the failing test**

```python
# tests/test_extract_config.py
"""Tests for ExtractConfig."""
from __future__ import annotations
from core.configs.extract_config import ExtractConfig


class TestExtractConfig:
    def test_defaults(self):
        cfg = ExtractConfig()
        assert cfg.run_name == ""
        assert cfg.action == "extract"
        assert cfg.model.model_name == ""
        assert cfg.extraction.batch_size == 8
        assert cfg.extraction.token_index == -1
        assert cfg.io.input_path == ""
        assert cfg.io.output_dir == "artifacts"

    def test_from_dict(self):
        cfg = ExtractConfig.from_dict({
            "run_name": "refusal_extract",
            "model": {"model_name": "Qwen/Qwen2.5-1.5B-Instruct"},
            "extraction": {"batch_size": 4, "token_index": -1},
            "io": {"input_path": "artifacts/labeled.jsonl", "output_dir": "artifacts/"},
        })
        assert cfg.model.model_name == "Qwen/Qwen2.5-1.5B-Instruct"
        assert cfg.extraction.batch_size == 4
        assert cfg.io.input_path == "artifacts/labeled.jsonl"

    def test_yaml_roundtrip(self, tmp_path):
        cfg = ExtractConfig(run_name="extract_test")
        path = tmp_path / "extract.yaml"
        cfg.to_yaml(path)
        loaded = ExtractConfig.from_yaml(path)
        assert loaded.run_name == "extract_test"
        assert loaded.action == "extract"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_extract_config.py -v
```
Expected: `ModuleNotFoundError: No module named 'core.configs.extract_config'`

**Step 3: Write the implementation**

```python
# core/configs/extract_config.py
"""Config for the extract stage."""
from __future__ import annotations
from dataclasses import dataclass, field
from core.configs.base import BaseConfig
from core.configs.params.extraction_params import ExtractionParams
from core.configs.params.io_params import IOParams
from core.configs.params.model_params import ModelParams


@dataclass
class ExtractConfig(BaseConfig):
    """Configuration for the extract stage (action: extract)."""

    run_name: str = ""
    seed: int = 0
    action: str = "extract"
    model: ModelParams = field(default_factory=ModelParams)
    extraction: ExtractionParams = field(default_factory=ExtractionParams)
    io: IOParams = field(default_factory=IOParams)
```

**Step 4: Run tests**

```bash
pytest tests/test_extract_config.py -v
```
Expected: 3 PASSED

**Step 5: Commit**

```bash
git add core/configs/extract_config.py tests/test_extract_config.py
git commit -m "feat: add ExtractConfig stage config"
```

---

### Task 4: ProbeConfig

**Files:**
- Create: `core/configs/probe_config.py`
- Test: `tests/test_probe_config.py`

**Step 1: Write the failing test**

```python
# tests/test_probe_config.py
"""Tests for ProbeConfig."""
from __future__ import annotations
from core.configs.probe_config import ProbeConfig


class TestProbeConfig:
    def test_defaults(self):
        cfg = ProbeConfig()
        assert cfg.run_name == ""
        assert cfg.action == "probe_sweep"
        assert cfg.probe.learning_rate == 1e-3
        assert cfg.split.train_fraction == 0.7
        assert cfg.sweep.selection_metric == "auroc"
        assert cfg.io.input_path == ""
        assert cfg.io.output_dir == "artifacts"
        assert cfg.output.save_plots is True

    def test_from_dict(self):
        cfg = ProbeConfig.from_dict({
            "run_name": "refusal_probe",
            "probe": {"learning_rate": 0.01, "epochs": 20},
            "io": {"input_path": "artifacts/activations/", "output_dir": "artifacts/"},
        })
        assert cfg.run_name == "refusal_probe"
        assert cfg.probe.learning_rate == 0.01
        assert cfg.io.input_path == "artifacts/activations/"

    def test_yaml_roundtrip(self, tmp_path):
        cfg = ProbeConfig(run_name="probe_test")
        path = tmp_path / "probe.yaml"
        cfg.to_yaml(path)
        loaded = ProbeConfig.from_yaml(path)
        assert loaded.run_name == "probe_test"
        assert loaded.action == "probe_sweep"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_probe_config.py -v
```
Expected: `ModuleNotFoundError: No module named 'core.configs.probe_config'`

**Step 3: Write the implementation**

```python
# core/configs/probe_config.py
"""Config for the probe_sweep stage."""
from __future__ import annotations
from dataclasses import dataclass, field
from core.configs.base import BaseConfig
from core.configs.params.io_params import IOParams
from core.configs.params.output_params import OutputParams
from core.configs.params.probe_params import ProbeParams
from core.configs.params.split_params import SplitParams
from core.configs.params.sweep_params import SweepParams


@dataclass
class ProbeConfig(BaseConfig):
    """Configuration for the probe_sweep stage (action: probe_sweep)."""

    run_name: str = ""
    seed: int = 0
    action: str = "probe_sweep"
    probe: ProbeParams = field(default_factory=ProbeParams)
    split: SplitParams = field(default_factory=SplitParams)
    sweep: SweepParams = field(default_factory=SweepParams)
    io: IOParams = field(default_factory=IOParams)
    output: OutputParams = field(default_factory=OutputParams)
```

**Step 4: Run tests**

```bash
pytest tests/test_probe_config.py -v
```
Expected: 3 PASSED

**Step 5: Commit**

```bash
git add core/configs/probe_config.py tests/test_probe_config.py
git commit -m "feat: add ProbeConfig stage config"
```

---

### Task 5: Runner dispatch refactor

Replace `RunConfig`-based dispatch with stage-config dispatch. The runner peeks at `action:` in the raw YAML dict, looks up the config class, and loads it typed.

**Files:**
- Modify: `runners/experiment_runner.py`
- Modify: `tests/test_experiment_runner_smoke.py`

**Step 1: Update the smoke tests first (TDD)**

The existing smoke tests use `RunConfig` indirectly. Update them to work with the new stage configs. The key changes:
- `test_load_run_config_from_yaml`: `action: probe_sweep` → returns `ProbeConfig` (no `model` field). Remove `cfg.model.model_name` assertion.
- `test_generate_action_dispatches`: already uses `action: generate` — still works, just returns `GenerateConfig`.

Replace `tests/test_experiment_runner_smoke.py` with:

```python
"""Smoke tests for the experiment runner API."""
from __future__ import annotations
import pytest
from runners.experiment_runner import load_run_config, run_experiment, RunResult
from core.configs.generate_config import GenerateConfig
from core.configs.extract_config import ExtractConfig
from core.configs.probe_config import ProbeConfig


class TestExperimentRunner:
    def test_load_probe_config_from_yaml(self, tmp_path):
        config_path = tmp_path / "run.yaml"
        config_path.write_text(
            "run_name: smoke\nseed: 0\naction: probe_sweep\n"
            "probe:\n  learning_rate: 0.01\n",
            encoding="utf-8",
        )
        cfg = load_run_config(config_path)
        assert isinstance(cfg, ProbeConfig)
        assert cfg.run_name == "smoke"
        assert cfg.probe.learning_rate == 0.01

    def test_load_generate_config_from_yaml(self, tmp_path):
        config_path = tmp_path / "gen.yaml"
        config_path.write_text(
            "run_name: gen\naction: generate\n"
            "model:\n  model_name: test-model\n",
            encoding="utf-8",
        )
        cfg = load_run_config(config_path)
        assert isinstance(cfg, GenerateConfig)
        assert cfg.model.model_name == "test-model"

    def test_load_extract_config_from_yaml(self, tmp_path):
        config_path = tmp_path / "ext.yaml"
        config_path.write_text(
            "run_name: ext\naction: extract\n"
            "model:\n  model_name: test-model\n",
            encoding="utf-8",
        )
        cfg = load_run_config(config_path)
        assert isinstance(cfg, ExtractConfig)

    def test_load_config_with_overrides(self, tmp_path):
        config_path = tmp_path / "run.yaml"
        config_path.write_text(
            "run_name: base\naction: probe_sweep\n"
            "probe:\n  learning_rate: 0.01\n",
            encoding="utf-8",
        )
        cfg = load_run_config(config_path, overrides={"probe.learning_rate": "0.001"})
        assert cfg.probe.learning_rate == 0.001

    def test_run_experiment_returns_structured_result(self, tmp_path):
        config_path = tmp_path / "run.yaml"
        config_path.write_text(
            "run_name: smoke\naction: probe_sweep\n",
            encoding="utf-8",
        )
        result = run_experiment(config_path=config_path, overrides={})
        assert isinstance(result, RunResult)
        assert result.summary["run_name"] == "smoke"

    def test_run_experiment_unknown_action_raises(self, tmp_path):
        config_path = tmp_path / "run.yaml"
        config_path.write_text(
            "run_name: bad\naction: nonexistent\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Unknown action"):
            run_experiment(config_path=config_path)

    def test_run_experiment_with_alias(self, tmp_path):
        aliases_file = tmp_path / "aliases.yaml"
        aliases_file.write_text("quick: " + str(tmp_path / "quick.yaml") + "\n", encoding="utf-8")
        config_file = tmp_path / "quick.yaml"
        config_file.write_text(
            "run_name: aliased\naction: probe_sweep\n",
            encoding="utf-8",
        )
        result = run_experiment(config_path="quick", aliases_path=aliases_file)
        assert result.summary["run_name"] == "aliased"

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

**Step 2: Run tests to verify they fail**

```bash
pytest tests/test_experiment_runner_smoke.py -v
```
Expected: `test_load_probe_config_from_yaml` FAIL (cfg is RunConfig, not ProbeConfig), others mixed.

**Step 3: Refactor the runner**

Replace `runners/experiment_runner.py` with:

```python
"""YAML-driven experiment runner: the single entry-point for API and CLI."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

from core.configs.base import BaseConfig
from core.configs.extract_config import ExtractConfig
from core.configs.generate_config import GenerateConfig
from core.configs.probe_config import ProbeConfig
from core.configs.aliases import resolve_config_alias
from core.configs.overrides import apply_dot_overrides
from omegaconf import OmegaConf

StageConfig = Union[GenerateConfig, ExtractConfig, ProbeConfig]

STAGE_CONFIG_MAP: dict[str, type[BaseConfig]] = {
    "generate":    GenerateConfig,
    "extract":     ExtractConfig,
    "probe_sweep": ProbeConfig,
}


@dataclass
class RunResult:
    """Structured result returned by every experiment run."""

    summary: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)


def load_run_config(
    config_path: str | Path,
    *,
    overrides: dict[str, str] | None = None,
    aliases_path: str | Path = "configs/aliases.yaml",
) -> BaseConfig:
    """Load a stage config from YAML, resolving aliases and applying overrides.

    The `action:` field in the YAML determines which config class is loaded.

    Args:
        config_path: Path to a YAML config file, or an alias key.
        overrides: Optional dot-notation overrides to apply on top.
        aliases_path: Path to the alias registry YAML file.

    Returns:
        The appropriate stage config instance (GenerateConfig, ExtractConfig, or ProbeConfig).
    """
    resolved_path = resolve_config_alias(str(config_path), aliases_path)
    raw = OmegaConf.load(resolved_path)
    raw_dict: dict = OmegaConf.to_container(raw, resolve=True)  # type: ignore[assignment]
    if overrides:
        raw_dict = apply_dot_overrides(raw_dict, overrides)
    action = raw_dict.get("action", "probe_sweep")
    config_cls = STAGE_CONFIG_MAP.get(action)
    if config_cls is None:
        raise ValueError(
            f"Unknown action '{action}'. "
            f"Available: {', '.join(sorted(STAGE_CONFIG_MAP))}"
        )
    return config_cls.from_dict(raw_dict)


def run_experiment(
    *,
    config_path: str | Path,
    overrides: dict[str, str] | None = None,
    aliases_path: str | Path = "configs/aliases.yaml",
) -> RunResult:
    """Run an experiment from a YAML config file.

    Args:
        config_path: Path to a YAML config file, or an alias key.
        overrides: Optional dot-notation CLI overrides.
        aliases_path: Path to the alias registry.

    Returns:
        RunResult with summary, metrics, and artifact paths.
    """
    cfg = load_run_config(config_path, overrides=overrides, aliases_path=aliases_path)
    return dispatch_action(cfg)


def dispatch_action(cfg: BaseConfig) -> RunResult:
    """Dispatch to the appropriate action handler based on config type."""
    if isinstance(cfg, GenerateConfig):
        return _action_generate(cfg)
    if isinstance(cfg, ExtractConfig):
        return _action_extract(cfg)
    if isinstance(cfg, ProbeConfig):
        return _action_probe_sweep(cfg)
    raise ValueError(f"Unhandled config type: {type(cfg).__name__}")


def _action_generate(cfg: GenerateConfig) -> RunResult:
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


def _action_extract(cfg: ExtractConfig) -> RunResult:
    """Placeholder for activation extraction action wiring."""
    return RunResult(
        summary={
            "run_name": cfg.run_name,
            "action": cfg.action,
            "model": cfg.model.model_name,
            "status": "configured",
        },
    )


def _action_probe_sweep(cfg: ProbeConfig) -> RunResult:
    """Placeholder for probe sweep action wiring."""
    return RunResult(
        summary={
            "run_name": cfg.run_name,
            "action": cfg.action,
            "seed": cfg.seed,
            "status": "configured",
        },
    )
```

**Step 4: Run tests**

```bash
pytest tests/test_experiment_runner_smoke.py -v
```
Expected: 8 PASSED

**Step 5: Commit**

```bash
git add runners/experiment_runner.py tests/test_experiment_runner_smoke.py
git commit -m "refactor: replace RunConfig dispatch with stage-specific config classes"
```

---

### Task 6: Export stage configs and retire RunConfig

**Files:**
- Modify: `core/configs/__init__.py`
- Delete: `core/configs/run_config.py`
- Modify: `configs/recipes/quickstart_probe.yaml`

**Step 1: Update `core/configs/__init__.py`**

Replace the contents of `core/configs/__init__.py`:

```python
"""Oumi-style typed configuration modules."""

from core.configs.base import BaseConfig
from core.configs.extract_config import ExtractConfig
from core.configs.generate_config import GenerateConfig
from core.configs.probe_config import ProbeConfig
from core.configs.params.extraction_params import ExtractionParams
from core.configs.params.generation_params import GenerationParams
from core.configs.params.io_params import IOParams
from core.configs.params.model_params import ModelParams
from core.configs.params.output_params import OutputParams
from core.configs.params.probe_params import ProbeParams
from core.configs.params.split_params import SplitParams
from core.configs.params.sweep_params import SweepParams

__all__ = [
    "BaseConfig",
    "ExtractConfig",
    "ExtractionParams",
    "GenerateConfig",
    "GenerationParams",
    "IOParams",
    "ModelParams",
    "OutputParams",
    "ProbeConfig",
    "ProbeParams",
    "SplitParams",
    "SweepParams",
]
```

**Step 2: Update the quickstart recipe YAML**

`configs/recipes/quickstart_probe.yaml` already works as a `ProbeConfig` YAML — it just needs `action: probe_sweep` added at the top (it currently has it), and the `model:` section removed (ProbeConfig has no model field). Open the file and verify it has `action: probe_sweep` at the top. If the `model:` section is present, remove it. The updated file:

```yaml
# Quickstart: refusal probe sweep on Qwen2.5-1.5B
run_name: quickstart_probe
seed: 0
action: probe_sweep

extraction:
  batch_size: 4
  token_index: -1
  to_cpu: true

probe:
  epochs: 20
  learning_rate: 0.01
  weight_decay: 0.01
  threshold: 0.5
  early_stopping_patience: 5
  bootstrap_samples: 0

split:
  train_fraction: 0.7
  val_fraction: 0.15
  test_fraction: 0.15
  split_seed: 0

sweep:
  batch_size: 32
  selection_metric: auroc
  maximize_metric: true
  enforce_control_sanity: true

output:
  output_dir: artifacts
  save_plots: true
```

Note: `extraction:` is kept for now since ProbeConfig doesn't have it yet — remove it from the recipe to avoid unexpected fields. If OmegaConf raises on unknown fields, remove it. Actually, since `BaseConfig.from_dict` uses `OmegaConf.merge(schema, raw)`, extra fields in the YAML that don't exist in the schema will raise a `ConfigAttributeError`. So remove the `extraction:` section from the recipe.

**Step 3: Delete RunConfig**

```bash
git rm core/configs/run_config.py
```

**Step 4: Run the full test suite**

```bash
pytest --tb=short -q
```

Expected: all tests pass. If any test imports `RunConfig` directly, update that import to use the appropriate stage config.

**Step 5: Commit**

```bash
git add core/configs/__init__.py configs/recipes/quickstart_probe.yaml
git commit -m "refactor: retire RunConfig, export stage configs, update quickstart recipe"
```

---

### Task 7: Full suite verification

**Step 1: Run the full test suite**

```bash
pytest --tb=short -q
```
Expected: all tests pass (was 90 before; should be ~100+ with new stage config tests).

**Step 2: Verify the quickstart recipe loads cleanly**

```python
# Quick sanity check (run in Python REPL or as a one-liner)
from runners.experiment_runner import load_run_config
cfg = load_run_config("configs/recipes/quickstart_probe.yaml")
print(type(cfg).__name__, cfg.run_name)
# Expected: ProbeConfig quickstart_probe
```

**Step 3: Commit if any final fixes were needed**

```bash
git add -p  # stage only what changed
git commit -m "fix: final suite cleanup after stage config migration"
```
