# Stage-Based Config Pipeline Design

**Date:** 2026-02-27
**Status:** Approved

## Problem

The current `RunConfig` monolith contains all params for all experiments in one dataclass. As the toolkit expands to SAE, activation patching, cross-coders, and skip transcoders, this becomes unmanageable — every new experiment bleeds its params into a shared namespace, there is no schema enforcement per action, and YAML files contain spurious fields irrelevant to the current stage.

## Design

### Core Abstraction

Retire `RunConfig` as the single entry point. Replace it with stage-specific config classes, each owning exactly the params it needs. An `action:` field in the YAML acts as the discriminator — the runner peeks at it first, then loads the appropriate config class.

```python
STAGE_CONFIG_MAP = {
    "generate":    GenerateConfig,
    "extract":     ExtractConfig,
    "probe_sweep": ProbeConfig,
    # "sae":       SAEConfig,   ← add here as toolkit grows
}
```

Adding a new experiment type = define a new config class + register one line. No changes to existing configs.

### Config Classes

```python
@dataclass
class GenerateConfig(BaseConfig):
    run_name: str = ""
    seed: int = 0
    action: str = "generate"
    model: ModelParams = field(default_factory=ModelParams)
    generation: GenerationParams = field(default_factory=GenerationParams)
    io: IOParams = field(default_factory=IOParams)

@dataclass
class ExtractConfig(BaseConfig):
    run_name: str = ""
    seed: int = 0
    action: str = "extract"
    model: ModelParams = field(default_factory=ModelParams)
    extraction: ExtractionParams = field(default_factory=ExtractionParams)
    io: IOParams = field(default_factory=IOParams)

@dataclass
class ProbeConfig(BaseConfig):
    run_name: str = ""
    seed: int = 0
    action: str = "probe_sweep"
    probe: ProbeParams = field(default_factory=ProbeParams)
    split: SplitParams = field(default_factory=SplitParams)
    sweep: SweepParams = field(default_factory=SweepParams)
    io: IOParams = field(default_factory=IOParams)
```

### IOParams

A shared `IOParams` section handles artifact chaining in every config. Convention-based defaults mean you only set `output_dir` once; `input_path` overrides when deviating from convention.

```python
@dataclass
class IOParams(BaseConfig):
    input_path: str = ""          # explicit override (empty = use convention)
    output_dir: str = "artifacts" # all stages write here by default
```

### Artifact Conventions

Each stage writes a predictable artifact under `output_dir`:

| Stage | Writes | Next stage reads |
|---|---|---|
| `generate` | `{output_dir}/responses.jsonl` | you label → `{output_dir}/labeled.jsonl` |
| `extract` | `{output_dir}/activations/` | probe/SAE/patch reads from here |
| `probe_sweep` | `{output_dir}/probes/`, `{output_dir}/plots/` | — |

Typical workflow:

```bash
run generate_refusal.yaml   # → artifacts/refusal/responses.jsonl
# (label externally → artifacts/refusal/labeled.jsonl)
run extract_refusal.yaml    # reads labeled.jsonl by convention
run probe_refusal.yaml      # reads activations/ by convention
```

To reuse activations across multiple probe runs (different hyperparams, splits):

```yaml
# probe_refusal_v2.yaml
action: probe_sweep
io:
  input_path: artifacts/refusal/activations/  # explicit override
  output_dir: artifacts/refusal_v2/
```

### ExtractConfig as the Rich Core

`ExtractConfig` is where depth lives long-term. `ExtractionParams` grows to support the full range of mech interp needs without touching any other config class:

```yaml
# extract_circuit.yaml — future
action: extract
model:
  model_name: Qwen/Qwen2.5-1.5B-Instruct
extraction:
  layers: [8, 12, 16, 20]
  hook_types: [resid_pre, attn_out, mlp_out]
  token_positions: [-1, 0]
```

All downstream experiments (probe, SAE, patching) read the same safetensors format — they index into different layers/hook types from the same artifact.

### Future Experiment Types

Each new experiment type follows the same pattern:

```python
# SAE training
@dataclass
class SAEConfig(BaseConfig):
    action: str = "sae"
    sae: SAEParams = field(default_factory=SAEParams)
    io: IOParams = field(default_factory=IOParams)

# Activation patching (needs two extraction runs)
@dataclass
class PatchConfig(BaseConfig):
    action: str = "patch"
    patching: PatchParams = field(default_factory=PatchParams)
    io: IOParams = field(default_factory=IOParams)
    # io extends to input_paths: list[str] for clean + corrupted
```

### Migration from RunConfig

`RunConfig` is retired. The quickstart recipe becomes a `ProbeConfig` YAML (it only uses probe/split/sweep/output fields anyway). The experiment runner loads the config class based on the `action:` discriminator.

### Path to "Everything Under One Roof"

When labeling becomes automated or the pipeline break is no longer needed, a `PipelineConfig` can chain stage configs automatically — the stage configs are already the composable building blocks. This is the natural evolution toward Option C without needing to redesign anything.
