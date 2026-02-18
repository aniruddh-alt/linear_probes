# Mech Interp Toolkit Design Doc

## Goal

Evolve this repository from a "linear probes + activation extraction" library into a modular mechanistic interpretability toolkit that can run reusable, composable interpretability actions:

1. Feature detection
2. Feature comparison
3. Circuit localization
4. Polysemanticity detection
5. Representation collapse detection

The key design principle is to separate:

- **Data and run configuration**
- **Activation collection and caching**
- **Interpretability actions (probes, interventions, analyses)**
- **Reporting and reproducibility**

---

## Current State (What You Already Have)

Strengths:

- `ActivationExtractor` supports multiple activation selectors and model hooks.
- `ProbingSampleBuilder` and `ProbingDataset` are good foundations for task data.
- Binary linear probing flow is now available end-to-end.

Gaps for scaling:

- No standard experiment schema/artifact format.
- Actions are tightly coupled to specific scripts.
- No explicit registry/plugin architecture for new analyses.
- Limited evaluation controls (null baselines, patching pipelines, layer sweeps).
- No first-class representation of components (head, MLP neuron, residual stream slice).

---

## Architecture Overview

Proposed package layout:

```text
linear_probes/
  core/
    schemas.py            # typed run configs, artifact metadata
    registry.py           # action/metric/model-adapter registry
    errors.py
  data/
    tasks.py              # task definitions, splits, label schemas
    samplers.py
  extract/
    extractor.py          # wraps ActivationExtractor API
    selectors.py          # selector parsing/validation
    cache.py              # feature store and retrieval
    backends/
      nnterp_backend.py
  probes/
    linear.py             # existing trainer + multiclass/regression variants
    controls.py           # label-shuffle, random-subspace controls
  actions/
    base.py               # abstract action interface
    feature_detection.py
    feature_comparison.py
    circuit_localization.py
    polysemanticity.py
    repr_collapse.py
    interventions/
      ablation.py
      activation_patching.py
  analysis/
    metrics.py            # accuracy, AUROC, calibration, sparsity, overlap
    statistics.py         # confidence intervals, significance tests
    geometry.py           # CKA, cosine overlap, subspace metrics
  reports/
    tables.py
    plots.py
    export.py             # json/md/html summary
  runners/
    experiment_runner.py  # config-driven orchestration
    cli.py
```

---

## Core Concepts and Interfaces

### 1) Standardized Schemas

Define typed schemas for reproducibility.

- `DatasetSpec`: data source, split policy, task label definition.
- `ExtractionSpec`: model, selector list, token position policy, precision/device.
- `ActionSpec`: action name + action-specific params.
- `ControlSpec`: shuffle/randomization/null tests.
- `RunSpec`: bundles all specs + seed + output directory.

Outputs should always include:

- `artifacts.json` (metadata + hashes + versions)
- `metrics.json`
- `predictions.parquet` (or `.pt`)
- `activations_index.json` (what was cached)

### 2) Action Interface

Each interpretability action implements a common interface:

- `prepare(context) -> PreparedContext`
- `run(prepared) -> ActionResult`
- `summarize(result) -> dict[str, Any]`

`context` includes:

- dataset split references
- cached activation handles
- selected model components
- baseline/control configuration

This makes actions plug-and-play in the runner.

### 3) Feature Store (Activation Cache)

Add a cache layer to avoid repeated model forward passes:

- Keyed by `(model, checkpoint, selector, token_policy, split, preprocessing_hash)`.
- Supports sharded writes for large datasets.
- Stores memory-mapped arrays/tensors + metadata.

This is essential for scaling layer sweeps and repeated controls.

---

## Design for Your 5 Research Actions

### A) Feature Detection

Question: "Does layer X encode feature Y?"

Pipeline:

1. Extract target activations for selected layers/components.
2. Train probe on train split.
3. Validate and test with controls:
   - label-shuffle baseline
   - random feature baseline
4. Report effect size and confidence intervals.

Primary outputs:

- probe performance per layer/component
- control gap (`real - shuffled`)

### B) Feature Comparison

Question: "Does syntax emerge before semantics?"

Pipeline:

1. Define two tasks/features (`syntax_task`, `semantics_task`).
2. Run aligned extraction and probing across same layers.
3. Compare layer-wise trajectories.
4. Run significance tests on peak-layer differences.

Primary outputs:

- layer curves for each feature
- first-layer-above-threshold estimate
- peak and area-under-curve comparison

### C) Circuit Localization

Question: "Is feature stronger in attention or MLP?"

Pipeline:

1. Probe separate streams (`attentions_output`, `mlps_output`, residual).
2. Rank candidate layers/components by probe score.
3. Perform interventions (ablation/patching) on top candidates.
4. Measure both:
   - probe metric drop
   - behavior metric drop

Primary outputs:

- candidate ranking table
- causal impact matrix (component x metric delta)

### D) Polysemanticity Detection

Question: "Does one neuron align strongly with probe direction?"

Pipeline:

1. Train linear probe, get weight vector `w`.
2. Compute neuron-direction alignment statistics:
   - cosine with basis vectors / sparse projection mass
3. Cross-check neuron against multiple unrelated features.
4. Flag neurons with high multi-feature loading as polysemantic candidates.

Primary outputs:

- neuron alignment scores
- multi-feature loading index
- shortlist of candidate polysemantic units

### E) Representation Collapse Detection

Question: "Does final layer destroy earlier feature encoding?"

Pipeline:

1. Probe feature across all layers with fixed protocol.
2. Compute representational similarity (CKA/cosine subspace) layer-to-layer.
3. Identify abrupt drop regions and correlate with architecture blocks.
4. Optionally patch earlier features into later layers to test recoverability.

Primary outputs:

- feature decodability curve
- collapse index (early peak - final layer score)
- similarity heatmap and drop points

---

## Model and Component Abstractions

Create a first-class component selector abstraction:

- `LayerComponent(kind="residual", layer=10)`
- `LayerComponent(kind="attention_head", layer=10, head=3)`
- `LayerComponent(kind="mlp_neuron", layer=10, neuron=187)`

Your current string selectors remain as user-facing syntax, but internally convert to typed selectors for safer operations and cleaner downstream tooling.

---

## Runner and CLI

Introduce a single command surface:

```bash
python -m runners.cli run --config configs/feature_detection_pythia70m.yaml
```

Runner responsibilities:

- Resolve specs
- Materialize/reuse cached activations
- Execute action(s) and controls
- Save versioned artifacts
- Emit one summary report

This turns ad hoc scripts into reproducible experiments.

---

## Metrics and Statistical Rigor

Minimum standard per action:

- Point metrics (accuracy/F1/AUROC or regression metrics)
- Confidence intervals (bootstrap)
- Null/control comparisons
- Multiple seeds support
- Optional permutation tests for layer/component ranking confidence

Without this, probe findings are often unstable or overinterpreted.

---

## Storage and Scalability Plan

### Near term

- Keep `.pt` tensors but shard by split/layer and store metadata index.

### Mid term

- Move feature storage to memory-mapped arrays or parquet + tensor blobs.
- Add lazy loading APIs for subset access (`layer`, `sample range`, `component`).

### Long term

- Distributed extraction runner for larger models/datasets.
- Optional remote object storage backend.

---

## Validation and Testing Strategy

Add tests at 3 levels:

1. **Unit tests**
   - selector parsing
   - schema validation
   - control generation
2. **Integration tests**
   - extract -> cache -> action -> report flow
3. **Regression tests**
   - fixed synthetic benchmark tasks where expected trends are known

Also add "research correctness tests":

- shuffled labels should drop to near chance
- random features should not outperform weak baseline

---

## Suggested Roadmap

### Phase 1 (1-2 weeks): Foundations

- Implement schemas, registry, run spec loader.
- Add activation cache and artifact metadata.
- Refactor existing linear probe flow into action interface.

### Phase 2 (2-3 weeks): Core actions

- Implement:
  - `feature_detection`
  - `feature_comparison`
  - `repr_collapse`
- Add controls and report templates.

### Phase 3 (2-4 weeks): Causal and component tools

- Add typed component selectors.
- Implement circuit localization with ablation/patching support.
- Add polysemanticity action.

### Phase 4 (ongoing): Scale and polish

- Optimize storage/backends.
- Add richer plots and paper-ready export.
- Build benchmark suite across multiple tasks/models.

---

## Example Config (Feature Detection)

```yaml
run_name: pythia70m_subject_number_lweep
seed: 42
dataset:
  name: subject_number_agreement
  train_split: train
  val_split: val
  test_split: test
extraction:
  model_name: EleutherAI/pythia-70m
  selectors:
    - layers_output:0-5
  token_index: -1
  batch_size: 16
action:
  name: feature_detection
  params:
    probe_type: linear_binary
    metric: accuracy
controls:
  - label_shuffle
  - random_subspace
output_dir: runs/pythia70m_subject_number
```

---

## Immediate Changes to This Repo (Practical Next Step)

1. Add `core/schemas.py` and `runners/experiment_runner.py`.
2. Wrap current `ActivationExtractor + ProbingDataset + BinaryLinearProbeTrainer` as a `feature_detection` action.
3. Add activation caching and a run artifact manifest.
4. Add one end-to-end config-driven command for `pythia-70m`.

This gives you a stable spine to incrementally attach all other mech interp actions.
