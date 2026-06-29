# Changelog

All notable changes to `sonde` are documented here. The format loosely follows
[Keep a Changelog](https://keepachangelog.com/); this project is pre-1.0 and the
API may change between minor versions.

## [Unreleased]

### Packaging
- Moved every module under a single `sonde/` package (`sonde.activation`,
  `sonde.dataset`, `sonde.probes`, `sonde.interventions`, …). Previously the
  modules squatted top-level names (`dataset`, `core`, `configs`) and a
  non-editable install was broken.
- `sonde/__init__.py` re-exports the public API; added `sonde.__version__`.
- Added `py.typed`; `matplotlib` moved to a `viz` extra; `transformers` and
  `pyyaml` declared explicitly.
- Config aliases and recipes load via `importlib.resources` (CWD-independent);
  `sonde run quickstart` works offline from any directory.
- CLI accepts both `sonde <cfg>` and `sonde run -c <cfg>`.

### Tensor storage
- Extraction artifacts are a JSON manifest (`_manifest.json`) + safetensors, not
  a pickle. Loading a manifest is no longer code execution.
- The safetensors path is stored relative to the manifest, so artifact
  directories are relocatable.
- `save_path=None` (new default) skips persistence cleanly instead of crashing.
- Variable-length (sequence-mode) activations now persist (padded + lengths) and
  round-trip exactly.
- Overwrite protection: existing artifacts are not clobbered unless
  `overwrite=True`. Legacy `_manifest.pt` pickles remain readable.

### Datasets
- `ProbingDataset`: a list of equal-length 2-D `(S, D)` tensors is now sequence
  data by default (no silent flatten to `(N, S*D)`); `sequence_mode=False` opts
  into pooled flattening.
- `from_extraction_result` raises when no labels are resolvable instead of
  silently labelling every sample 0.
- `ProbingSampleBuilder.to_samples` logs dropped empty-text rows.
- `SampleBundle.prompts` is a `list[str]`; the split's grouping regime is logged.

### Probes & runner
- Wired the `probe_sweep` and `diff_means` runner actions (previously
  `NotImplementedError`); both read a pre-extracted artifact and save a
  `ProbeArtifact`.
- `ProbeArtifact`: a single safetensors file with the concept direction (+bias,
  layer, metadata) — the contract between probing and intervention/deploy.

### Causal interventions
- `sonde.interventions.InterventionContext` with additive steering and
  directional ablation (`project_subtract`), applied via `ctx.apply()` inside a
  user `with model.trace/generate(...)` block (verified against gpt2).
- `docs/intervention_design.md` updated with verified findings and a required
  scientific-controls section.
- `examples/causal_loop_gpt2.py`: full extract → probe → ablate → measure loop
  with a random-direction specificity control.
