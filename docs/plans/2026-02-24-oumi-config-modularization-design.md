# Oumi-Style Config and Modularity Design

Date: 2026-02-24  
Project: `linear_probes`

## Objective

Clean the toolkit for better usability and migrate toward an Oumi-style architecture:

- robust Python API first
- thin CLI wrapper
- YAML-driven typed configs
- modular orchestration boundaries
- incremental migration with low breakage risk

## Context Summary

Current strengths:

- solid activation extraction primitives
- end-to-end probe sweep flow with split controls and run manifests
- existing typed config classes in `configs/types.py`
- improving test coverage around splitting/manifest/analyzer behavior

Current usability/modularity gaps:

- config definitions are centralized in a generic `types` file instead of domain-oriented config modules
- orchestration entrypoints are still script-centric
- no standardized YAML run config contract for API+CLI parity
- no alias/override UX similar to Oumi

DeepWiki/Oumi takeaways used here:

- typed config classes derived from a shared base
- YAML loading + CLI dot-notation overrides
- alias resolution for friendly config selection
- CLI as wrapper over reusable API flows
- composition of smaller params objects into task-level configs

## Design Decisions

### 1) Approach Choice

Chosen: **Option A — Incremental config spine**.

Why:

- preserves current working modules
- minimizes migration risk and user disruption
- delivers immediate usability improvements
- enables phased refactor to deeper modularization

### 2) Modular Architecture (Incremental)

Keep existing domain engines largely intact:

- `activation/*`
- `dataset/*`
- `probes/*`

Add a new config/orchestration layer:

- `core/configs/` for typed YAML configs
- `runners/` for run lifecycle orchestration
- `cli/` for thin command wrappers

Introduce action-oriented orchestration contracts:

- `extract`
- `probe_sweep`
- `analyze`

Later actions can plug in without rewiring core execution.

### 3) Config Placement and Composition

Move away from a monolithic `configs/types.py` model toward Oumi-style config modules:

- `core/configs/base.py` -> shared `BaseConfig`
- `core/configs/run_config.py` -> top-level run config
- `core/configs/params/model_params.py`
- `core/configs/params/extraction_params.py`
- `core/configs/params/probe_params.py`
- `core/configs/params/split_params.py`
- `core/configs/params/output_params.py`

Migration compatibility:

- keep `configs/__init__.py` re-exports for old imports
- deprecate old locations gradually with warning path

### 4) YAML Contract

Define a top-level `RunConfig`:

- `run_name`
- `seed`
- `action`
- `dataset`
- `extraction`
- `probe`
- `split`
- `output`

Every run should be reproducible from one YAML file and optional overrides.

### 5) Overrides and Aliases

Override strategy:

- support dot-notation CLI overrides (nested fields)
- precedence: defaults < YAML < CLI overrides

Alias strategy:

- `configs/aliases.yaml` maps friendly names to canonical config paths
- CLI resolves alias before file load

### 6) Execution Lifecycle

`runner.run(config_source, overrides)`:

1. Resolve config source (path or alias)
2. Load YAML into typed config
3. Apply validated CLI overrides
4. Build run context (seed, paths, logger, metadata)
5. Dispatch to action executor
6. Persist standard artifacts
7. Return typed `RunResult`

### 7) Standard Artifact Contract

For each run:

- `run_manifest.json`
- `metrics.json`
- `selected_artifacts/*` (action-specific)
- optional plots under configured output path

Behavior for manifest path collisions should remain explicit:

- write-once default
- optional overwrite
- optional unique-path mode

### 8) Error Handling and Testing

Error handling:

- config validation errors include field paths
- runtime errors include stage context (`extract`, `train`, `analyze`)

Testing:

- unit: config parsing/validation/override/alias resolution
- integration: single YAML-driven end-to-end run
- compatibility: legacy imports still operational
- usability: CLI quickstart smoke test

## Implementation Phasing

### Phase 1: Config Spine

- introduce `BaseConfig` and `RunConfig`
- create param modules in `core/configs/params`
- add YAML loader and override merge utility
- keep existing domain logic unchanged

### Phase 2: Runner + CLI Shell

- implement `runners/run_experiment.py`
- add thin CLI entrypoint `toolkit run -c config.yaml`
- wire to existing extraction/probe/analyze flow

### Phase 3: Compatibility + Migration

- re-export old config classes from new modules
- add deprecation notes in docs
- provide migration examples and alias usage

### Phase 4: Expanded Modularity

- formalize action interface and registry
- plug in additional interpretability actions without touching runner core

## Non-Goals (for this incremental pass)

- full package relayout in one commit
- immediate replacement of all existing script entrypoints
- large behavior changes in probe math or extraction internals

## Open Implementation Notes

- `run_safety_probe_experiment.py` is currently deferred for dedicated cleanup.
- Existing split/manifest/analyzer improvements remain part of baseline guarantees.

## Success Criteria

- users can run a complete experiment from one YAML config
- API and CLI use the same typed config path and execution engine
- config modules are domain-appropriate, not centralized in generic `types`
- existing workflows keep working during migration
- modular boundaries are explicit and test-covered
