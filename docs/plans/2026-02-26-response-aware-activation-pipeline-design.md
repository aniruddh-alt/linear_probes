# Response-Aware Activation Pipeline Design

**Date:** 2026-02-26
**Status:** Approved

## Problem

The current pipeline assumes labels exist before model inference. In practice (refusal probing, safety analysis, etc.), labels depend on model *behavior* — you need to see what the model responded before labeling. The pipeline needs to support: generate responses → label externally → extract activations → analyze.

## Approach

**Approach A: Response Generator as a Separate Module.** A new `generation/` module handles text generation, decoupled from activation extraction. Two-pass strategy: generate first, extract activations second.

## Design

### New Module: `generation/`

**`generation/response_generator.py`** — `ResponseGenerator` class:
- Constructor takes `ModelParams` and `GenerationParams`
- `generate(samples: SampleBundle | Sequence[str]) -> GenerationResult` — runs `model.generate()` on each prompt, returns prompt-response pairs

**`generation/types.py`** — `GenerationResult` dataclass:
```python
@dataclass
class GenerationResult:
    prompts: list[str]
    responses: list[str]
    sample_ids: list[str]
    labels: list[int | None]  # carried through if present, all None otherwise
```

Methods:
- `to_jsonl(path)` — export for external labeling
- `from_jsonl(path)` — classmethod, re-import labeled data
- `to_sample_bundle(include_response=False)` — convert to SampleBundle for extraction. If `include_response=True`, prompts become prompt+response concatenated.

**`core/configs/params/generation_params.py`** — `GenerationParams` config:
```python
@dataclass
class GenerationParams(BaseConfig):
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 1.0
    do_sample: bool = False
    batch_size: int = 8
```

### Type Changes

**`SampleBundle`** gains optional `responses`:
```python
@dataclass
class SampleBundle:
    prompts: Dataset[str]
    labels: list[int | None]
    ids: list[str]
    responses: list[str] | None = None  # NEW
```

No changes to `ExtractionResult`, `ActivationExtractor`, `ProbingDataset`, or probe training code.

### Runner Integration

`RunConfig` gains `generation: GenerationParams` field. New `"generate"` action in the runner.

### Supported Workflows

**Post-hoc labeling (new):**
1. `ResponseGenerator.generate(prompts)` → `GenerationResult`
2. `result.to_jsonl("responses.jsonl")` — export
3. Label externally (oumi tools, LLM judge, etc.)
4. `GenerationResult.from_jsonl("labeled.jsonl")` — re-import
5. `result.to_sample_bundle(include_response=False)` → `SampleBundle`
6. `ActivationExtractor.extract(bundle)` → `ExtractionResult`
7. Train probes / diff-mean-vectors

**Pre-labeled (existing, unchanged):**
1. `ProbingSampleBuilder` → `SampleBundle` with labels
2. `ActivationExtractor.extract(bundle)` → `ExtractionResult`
3. Train probes

**Unlabeled exploration (existing, unchanged):**
1. `ActivationExtractor.extract(prompts)` → `ExtractionResult` (labels all None)
2. Compute mean activations, visualize, etc.

### File Structure

```
generation/
    __init__.py
    response_generator.py
    types.py
core/configs/params/
    generation_params.py          (NEW)
dataset/
    types.py                      (MODIFIED — add responses field)
core/configs/
    run_config.py                 (MODIFIED — add generation param)
runners/
    experiment_runner.py           (MODIFIED — add generate action)
```
