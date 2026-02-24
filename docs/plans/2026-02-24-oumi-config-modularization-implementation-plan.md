# Oumi-Style Config Modularization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Introduce an Oumi-style YAML-driven config spine with a robust API core and thin CLI wrapper, while keeping existing probe/extraction behavior stable through incremental migration.

**Architecture:** Add a new `core/configs` + `runners` + `cli` layer over existing domain modules (`activation`, `dataset`, `probes`). Use composed typed config classes, YAML loading, dot-override merging, and optional aliases. Keep compatibility re-exports from existing config surfaces during migration.

**Tech Stack:** Python 3.13, dataclasses/pydantic-compatible typed configs, PyYAML, pytest, existing repository modules.

---

### Task 1: Create Base Config Spine and Params Modules

**Files:**
- Create: `core/configs/base.py`
- Create: `core/configs/params/model_params.py`
- Create: `core/configs/params/extraction_params.py`
- Create: `core/configs/params/probe_params.py`
- Create: `core/configs/params/split_params.py`
- Create: `core/configs/params/output_params.py`
- Create: `core/configs/__init__.py`
- Test: `tests/test_core_config_base.py`

**Step 1: Write the failing test**

```python
from core.configs.base import BaseConfig
from core.configs.params.model_params import ModelParams


def test_base_config_round_trip_yaml(tmp_path):
    cfg = ModelParams(model_name="Qwen/Qwen2.5-1.5B-Instruct")
    path = tmp_path / "model.yaml"
    cfg.to_yaml(path)
    loaded = ModelParams.from_yaml(path)
    assert loaded == cfg
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_core_config_base.py::test_base_config_round_trip_yaml -v`  
Expected: FAIL with import/module-not-found errors for `core.configs.*`.

**Step 3: Write minimal implementation**

```python
# core/configs/base.py
@dataclass(frozen=True)
class BaseConfig:
    @classmethod
    def from_yaml(cls, path: str | Path):
        ...
    def to_yaml(self, path: str | Path) -> None:
        ...
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_core_config_base.py::test_base_config_round_trip_yaml -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add core/configs tests/test_core_config_base.py
git commit -m "feat: add base config spine and param modules"
```

### Task 2: Add RunConfig + YAML Schema Composition

**Files:**
- Create: `core/configs/run_config.py`
- Modify: `core/configs/__init__.py`
- Create: `tests/test_run_config_schema.py`

**Step 1: Write the failing test**

```python
from core.configs.run_config import RunConfig


def test_run_config_loads_composed_sections(tmp_path):
    yaml_text = """
run_name: smoke
seed: 0
action: probe_sweep
model:
  model_name: Qwen/Qwen2.5-1.5B-Instruct
split:
  train_fraction: 0.7
  val_fraction: 0.15
  test_fraction: 0.15
"""
    path = tmp_path / "run.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    cfg = RunConfig.from_yaml(path)
    assert cfg.run_name == "smoke"
    assert cfg.model.model_name.startswith("Qwen/")
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_run_config_schema.py::test_run_config_loads_composed_sections -v`  
Expected: FAIL because `RunConfig` does not exist.

**Step 3: Write minimal implementation**

```python
@dataclass(frozen=True)
class RunConfig(BaseConfig):
    run_name: str
    seed: int
    action: str
    model: ModelParams
    extraction: ExtractionParams = field(default_factory=ExtractionParams)
    probe: ProbeParams = field(default_factory=ProbeParams)
    split: SplitParams = field(default_factory=SplitParams)
    output: OutputParams = field(default_factory=OutputParams)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_run_config_schema.py -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add core/configs tests/test_run_config_schema.py
git commit -m "feat: add composed run config schema"
```

### Task 3: Implement Dot-Override Merge Utility

**Files:**
- Create: `core/configs/overrides.py`
- Modify: `core/configs/base.py`
- Create: `tests/test_config_overrides.py`

**Step 1: Write the failing test**

```python
from core.configs.overrides import apply_dot_overrides


def test_apply_dot_overrides_nested_values():
    payload = {"probe": {"learning_rate": 1e-2}, "split": {"auto_group_by_id_when_none": True}}
    out = apply_dot_overrides(payload, {"probe.learning_rate": "0.001", "split.auto_group_by_id_when_none": "false"})
    assert out["probe"]["learning_rate"] == 0.001
    assert out["split"]["auto_group_by_id_when_none"] is False
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config_overrides.py::test_apply_dot_overrides_nested_values -v`  
Expected: FAIL because override utility is missing.

**Step 3: Write minimal implementation**

```python
def apply_dot_overrides(config_dict: dict, overrides: dict[str, str]) -> dict:
    # parse booleans/numbers and update nested maps by dot path
    ...
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config_overrides.py -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add core/configs tests/test_config_overrides.py
git commit -m "feat: add dot-notation config override support"
```

### Task 4: Add Alias Resolution for Config Paths

**Files:**
- Create: `configs/aliases.yaml`
- Create: `core/configs/aliases.py`
- Create: `tests/test_config_aliases.py`

**Step 1: Write the failing test**

```python
from core.configs.aliases import resolve_config_alias


def test_resolve_config_alias_from_registry(tmp_path):
    aliases_file = tmp_path / "aliases.yaml"
    aliases_file.write_text("quickstart: configs/recipes/quickstart_probe.yaml\n", encoding="utf-8")
    assert resolve_config_alias("quickstart", aliases_file) == "configs/recipes/quickstart_probe.yaml"
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config_aliases.py::test_resolve_config_alias_from_registry -v`  
Expected: FAIL because alias resolution module is missing.

**Step 3: Write minimal implementation**

```python
def resolve_config_alias(config_or_alias: str, aliases_path: str | Path) -> str:
    # return mapped path when alias key exists; else passthrough
    ...
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config_aliases.py -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add core/configs configs/aliases.yaml tests/test_config_aliases.py
git commit -m "feat: add config alias resolution"
```

### Task 5: Create Runner API and Thin CLI Wrapper

**Files:**
- Create: `runners/experiment_runner.py`
- Create: `cli/main.py`
- Modify: `pyproject.toml` (entrypoint if needed)
- Create: `tests/test_experiment_runner_smoke.py`
- Create: `tests/test_cli_run_command.py`

**Step 1: Write the failing test**

```python
from runners.experiment_runner import run_experiment


def test_run_experiment_returns_structured_result(tmp_path):
    config_path = tmp_path / "run.yaml"
    config_path.write_text("run_name: smoke\nseed: 0\naction: probe_sweep\nmodel:\n  model_name: Qwen/Qwen2.5-1.5B-Instruct\n", encoding="utf-8")
    result = run_experiment(config_path=config_path, overrides={})
    assert "run_name" in result.summary
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_experiment_runner_smoke.py::test_run_experiment_returns_structured_result -v`  
Expected: FAIL because runner API is missing.

**Step 3: Write minimal implementation**

```python
def run_experiment(*, config_path: str | Path, overrides: dict[str, str]) -> RunResult:
    cfg = load_run_config(config_path, overrides=overrides)
    return dispatch_action(cfg)
```

**Step 4: Run tests to verify runner and CLI pass**

Run: `uv run pytest tests/test_experiment_runner_smoke.py tests/test_cli_run_command.py -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add runners cli pyproject.toml tests/test_experiment_runner_smoke.py tests/test_cli_run_command.py
git commit -m "feat: add yaml-driven runner API and thin cli wrapper"
```

### Task 6: Add Compatibility Re-Exports and Migration Docs

**Files:**
- Modify: `configs/__init__.py`
- Modify: `README.md`
- Create: `docs/migration/config-migration.md`
- Create: `tests/test_config_compat_imports.py`

**Step 1: Write the failing test**

```python
def test_legacy_config_imports_still_work():
    from configs import LayerProbeSweepConfig, ProbeConfig  # noqa: F401
```

**Step 2: Run test to verify it fails (if imports broke)**

Run: `uv run pytest tests/test_config_compat_imports.py::test_legacy_config_imports_still_work -v`  
Expected: FAIL if refactor breaks public imports.

**Step 3: Write minimal implementation**

```python
# configs/__init__.py
# re-export from new core/configs modules while preserving old names
```

**Step 4: Run full targeted suite**

Run: `uv run pytest tests/test_config_compat_imports.py tests/test_probe_* tests/test_linear_probe_trainer.py -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add configs/__init__.py README.md docs/migration/config-migration.md tests/test_config_compat_imports.py
git commit -m "docs: add config migration guide and preserve legacy config imports"
```

### Task 7: End-to-End Usability Validation

**Files:**
- Create: `configs/recipes/quickstart_probe.yaml`
- Create: `tests/test_quickstart_yaml_e2e.py`
- Modify: `README.md`

**Step 1: Write the failing test**

```python
def test_quickstart_yaml_executes_probe_flow(tmp_path):
    # copies/uses quickstart yaml and validates runner outputs expected manifest and metrics
    ...
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_quickstart_yaml_e2e.py::test_quickstart_yaml_executes_probe_flow -v`  
Expected: FAIL until recipe wiring is complete.

**Step 3: Write minimal implementation**

```yaml
# quickstart_probe.yaml
run_name: quickstart_probe
seed: 0
action: probe_sweep
...
```

**Step 4: Run final verification**

Run: `uv run pytest tests/test_quickstart_yaml_e2e.py tests/test_experiment_runner_smoke.py tests/test_cli_run_command.py -v`  
Expected: PASS.

**Step 5: Commit**

```bash
git add configs/recipes/quickstart_probe.yaml tests/test_quickstart_yaml_e2e.py README.md
git commit -m "feat: add quickstart yaml and e2e usability validation"
```

## Execution Notes

- Use @superpowers/test-driven-development for each implementation task.
- Keep each task small; do not batch multiple features before tests pass.
- Maintain backward compatibility unless a step explicitly deprecates behavior.
- Do not change probe math or extractor behavior unless required by config wiring.

## Definition of Done

- YAML-driven run path exists and is test-covered.
- API and CLI share the same run engine.
- Configs are moved to domain-appropriate modules under `core/configs`.
- Alias and dot-override workflows are validated.
- Existing imports and core probe workflows remain operational.
