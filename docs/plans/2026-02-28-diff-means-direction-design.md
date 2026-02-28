# Diff-in-Means Direction Finding

**Date:** 2026-02-28
**Status:** Approved

## Summary

Add diff-in-means as a direction-finding method alongside linear probing. Given contrastive activation pairs (e.g. harmful vs benign prompts), compute `r_l = mean(H+) - mean(H-)` per layer, normalize, and evaluate via projection scoring. Produces output parallel to `LayerProbeSweepResult` for direct comparison.

## Architecture

New top-level module `directions/` as a peer to `probes/`:

```
directions/
  __init__.py
  diff_means.py      # DiffMeansEstimator
  sweep.py           # DiffMeansSweepRunner
  result.py          # DiffMeansSweepResult + DiffMeansLayerResult
```

### DiffMeansEstimator

Core computation class. Given activations and binary labels for a single layer:

1. Split activations by label: `H+ = activations[labels == 1]`, `H- = activations[labels == 0]`
2. Compute `r_l = mean(H+) - mean(H-)`
3. Normalize: `r_hat = r_l / ||r_l||`
4. Return direction + metadata (raw norm, class sizes, class means)

### DiffMeansSweepRunner

Layer-wise sweep orchestrator mirroring `LayerProbeSweepRunner`:

1. Iterate over activation layers
2. For each layer: compute diff-means direction on **train** split only
3. Evaluate on **val** split via projection scoring
4. Select best layer by AUROC (or configurable metric)
5. Evaluate best layer on **test** split
6. Run controls: shuffle labels, recompute direction, evaluate (expect ~0.5 AUROC)

### DiffMeansSweepResult

Parallel to `LayerProbeSweepResult`:

```python
@dataclass
class DiffMeansLayerResult:
    key: str                    # e.g. "layers_output:15"
    direction: torch.Tensor     # normalized direction vector
    raw_norm: float             # ||r_l|| before normalization
    positive_count: int         # |H+|
    negative_count: int         # |H-|
    val_metrics: dict[str, float]

@dataclass
class DiffMeansSweepResult:
    layers: dict[str, DiffMeansLayerResult]
    best_key: str
    best_metric: str
    best_score: float
    best_direction: torch.Tensor
    test_metrics: dict[str, float]
    controls: dict[str, dict[str, float]]
    split_sizes: tuple[int, int, int]
    dataset_fingerprint: str
    manifest_path: str | None
```

## Config Integration

New `DiffMeansConfig` in `core/configs/`:

```python
@dataclass
class DiffMeansConfig(BaseConfig):
    run_name: str = ""
    seed: int = 0
    action: str = "diff_means"
    split: SplitParams = field(default_factory=SplitParams)
    sweep: SweepParams = field(default_factory=SweepParams)
    io: IOParams = field(default_factory=IOParams)
    output: OutputParams = field(default_factory=OutputParams)
```

No `ModelParams` or `ProbeParams` needed. Operates on pre-extracted activations loaded from `io.input_path`.

New `_action_diff_means` handler in `experiment_runner.py`.

## Evaluation Strategy

Projection scoring (no training required):

- **Score:** `s = h . r_hat` (dot product with normalized direction)
- **AUROC:** use raw scores directly (threshold-free)
- **Accuracy/F1:** threshold at mean of train scores
- **Controls:** shuffle labels, recompute direction, evaluate. Expected: ~0.5 AUROC

Metrics are directly comparable to probe AUROC since both produce scalar scores from the same activations.

## Manifest & Reproducibility

Reuse existing `run_manifest.py` with method="diff_means". Same fields: dataset_fingerprint, split hashes, test_metrics, controls.
