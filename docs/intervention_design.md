# Causal Intervention Layer — Design

**Scope.** Research-grade causal intervention built on top of nnterp's native
intervention primitives. Production / vLLM-side serving (`sonde.deploy`) is
deferred — see [§9](#9-out-of-scope-deploy) for the explicit boundary.

**Anchors.**
- `docs/toolkit_audit.md` §5 — the proposed `InterventionContext` shape.
- `generation/response_generator.py` — current hook-based steering (to be replaced).
- `core/configs/params/steering_params.py` — existing YAML knobs (kept; extended).
- `activation/token_selectors.py` — already provides per-position selectors used here for
  per-token steering masks.
- `directions/diff_means.py`, `probes/architectures/base.py` — sources of direction
  vectors (`BaseProbe.direction`, `DiffMeansLayerResult.direction`).

---

## 0. Why now

The toolkit has two parallel paths into the same model:

```
read:  HF model → nnterp.StandardizedTransformer → nnsight tracing → .save()
write: HF model → register_forward_hook                            → output[0] mutation
```

This split forces every model-family special case to be solved twice (once in
nnterp for reads, once in `_resolve_layer_modules` for writes), and it blocks
the canonical mech-interp tests — activation patching and per-head intervention
— that need read+write under the same tracing context.

nnterp already exposes intervention as a first-class operation:

```python
import torch
steering_vector = torch.randn(768)         # GPT-2 hidden size
with model.trace("The weather today is"):
    model.steer(layers=[1, 3], steering_vector=steering_vector, factor=0.5)
```

We adopt this as the **primitive** and build a small context layer around it
that adds YAML wiring, directional-ablation mode, per-position masking, and
activation patching.

---

## 1. Module layout

New package: `interventions/`

```
interventions/
  __init__.py
  context.py          # InterventionContext — the main public class
  steering.py         # steering operations (additive, project_subtract)
  patching.py         # activation patching (read site A → write site B)
  heads.py            # per-head intervention (ITI-style)
  vectors.py          # load_vector(path | tensor | BaseProbe | DiffMeansLayerResult)
  types.py            # PendingSteer, PendingPatch, PendingHeadSteer
```

No new dependencies. `nnterp.StandardizedTransformer` is already in
`pyproject.toml`. The package never imports HuggingFace `register_forward_hook`
— all writes go through nnterp/nnsight proxies.

---

## 2. The primitive: `model.steer()`

nnterp's `model.steer(layers=..., steering_vector=..., factor=...)` is
additive: it adds `factor · steering_vector` to the residual stream at the
output of each listed layer, for every token position in the trace.

We keep this as the additive path. The two operations nnterp **does not** give
us natively, which we build on top:

| Operation | Current behaviour | Sonde layer adds |
|---|---|---|
| Additive steering (all tokens) | `model.steer(layers, vec, factor)` | thin wrapper, normalization, vector loading |
| Directional ablation (`h - (h·v̂)v̂`) | not native | per-layer manual write via `model.layers_output[L]` proxy |
| Per-token-position steering | not native (writes to every position) | mask via `TokenSelector` resolved against `input_ids` |
| Per-head steering | not native | write into `model.layers[L].self_attn.o_proj.input` reshaped to heads |
| Activation patching | not native | read site → store → write into other run's proxy |

All four reuse the same `model.trace(...)` context.

---

## 3. `InterventionContext` — public API

```python
# interventions/context.py
from contextlib import AbstractContextManager
from typing import Self, Union
import torch
from nnterp import StandardizedTransformer
from activation.token_selectors import TokenSelector

DirectionLike = Union[torch.Tensor, str, "BaseProbe", "DiffMeansLayerResult"]

class InterventionContext(AbstractContextManager):
    """Unified read+write context over an nnterp StandardizedTransformer.

    Operations are *configured* before entering the context and *applied*
    inside the `model.trace(...)` call. The same context can be reused across
    multiple traces, or rebuilt per run.
    """

    def __init__(self, model: StandardizedTransformer):
        self.model = model
        self._steers: list[PendingSteer] = []
        self._patches: list[PendingPatch] = []
        self._head_steers: list[PendingHeadSteer] = []

    # ── configuration (chainable) ─────────────────────────────────────────

    def add_steering(
        self,
        layers: list[int] | int,
        vector: DirectionLike,
        *,
        mode: str = "additive",          # "additive" | "project_subtract"
        factor: float = 1.0,
        normalize: bool = True,
        positions: TokenSelector | None = None,   # None = all positions
    ) -> Self: ...

    def add_head_steering(
        self,
        layer: int,
        head: int,
        vector: DirectionLike,            # head-dim, not residual-dim
        *,
        factor: float = 1.0,
        positions: TokenSelector | None = None,
    ) -> Self: ...

    def add_patch(
        self,
        *,
        src_activation: torch.Tensor,     # captured from a separate trace
        dst_spec: str,                    # e.g. "layers_output:15"
        positions: TokenSelector,          # required — which positions to overwrite
    ) -> Self: ...

    def clear(self) -> Self: ...          # forget all pending interventions

    # ── execution ─────────────────────────────────────────────────────────

    def trace(self, prompts, *, remote: bool = False):
        """Returns a context manager equivalent to model.trace(prompts) but
        applies all configured interventions inside it."""

    def generate(self, prompts, **generate_kwargs):
        """Convenience: model.generate(...) under the configured interventions.
        Interventions are reapplied for each generation step."""

    # ── construction from YAML ────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        model: StandardizedTransformer,
        steering: "SteeringParams | list[SteeringParams] | None" = None,
        patches: "list[PatchParams] | None" = None,
    ) -> Self:
        """Build a context from declarative config — used by the YAML runner."""
```

The chainable `add_*` methods + `from_config` cover both programmatic
research and YAML-driven runs from one class.

### 3.1 Programmatic usage (research)

```python
from interventions import InterventionContext
from activation.token_selectors import TokenIdAnchor

# Case A — direct, ad-hoc
ctx = InterventionContext(model)
steering_vector = torch.randn(768)
with ctx.trace("The weather today is"):
    ctx.add_steering(layers=[1, 3], vector=steering_vector, factor=0.5)
    logits = model.logits.save()

# Case B — configured once, applied across many traces
ctx = (
    InterventionContext(model)
    .add_steering(
        layers=list(range(model.num_layers)),
        vector=probe.direction,
        mode="project_subtract",
    )
)
with ctx:  # __enter__ wraps model.trace internally on .generate / .trace
    out = ctx.generate(prompts, max_new_tokens=128)
```

### 3.2 Causal probe test (the loop-closer)

```python
# Did the probe find a causal direction?
ctx = InterventionContext(model).add_steering(
    layers=[probe.layer],
    vector=probe.direction,
    mode="project_subtract",
    positions=TokenIdAnchor(
        pattern=tokenizer.encode("<|assistant|>", add_special_tokens=False),
        offset=1,
    ),
)
with ctx:
    ablated = ctx.generate(prompts, max_new_tokens=256)

# Compare refusal rate ablated vs baseline.
```

### 3.3 Activation patching (canonical mech-interp test)

```python
# 1) capture activations from a source run
with model.trace(prompt_A):
    h_src = model.layers_output[15].save()

# 2) write them into a destination run at one position
ctx = InterventionContext(model).add_patch(
    src_activation=h_src.value,
    dst_spec="layers_output:15",
    positions=Index(-1),
)
with ctx.trace(prompt_B):
    out = model.logits.save()
```

---

## 4. YAML config

The user-facing config keeps the existing `SteeringParams` shape and grows two
small additions. The minimum form the user sketched:

```yaml
steering:
  vector_path: artifacts/refusal_direction.safetensors
  layers: [15]
  factor: 1.0
```

The full form (every field optional except `vector_path` + `layers` when
`enabled: true`):

```yaml
steering:
  enabled: true
  vector_path: artifacts/refusal_direction.safetensors
  vector_key: direction              # which tensor inside the safetensors file
  layers: [15]                       # or "all"
  factor: 1.0                        # signed; negative = subtract direction
  mode: project_subtract             # additive | project_subtract
  normalize: true                    # unit-normalise the vector before use
  positions:                         # OPTIONAL; default = all positions
    kind: token_id_anchor            # all | index | range | index_list
                                     # | last_non_pad | token_id_anchor
    pattern: [128009]                # <|eot_id|>
    offset: 1
    mode: first

patches:                             # OPTIONAL
  - src_extraction_path: artifacts/source_extraction.safetensors
    src_key: layers_output:15
    dst_spec: layers_output:15
    positions:
      kind: index
      index: -1
```

`steering` can be a list to layer multiple interventions in one run.

### 4.1 `SteeringParams` changes

`core/configs/params/steering_params.py` (existing):

```python
@dataclass
class SteeringParams(BaseConfig):
    enabled: bool = False
    vector_path: str = ""
    vector_key: str = ""
    layers: list[int] = field(default_factory=list)
    strength: float = 10.0                                    # → rename to `factor`
    mode: str = "project_subtract"                            # unchanged
    normalize: bool = True                                    # unchanged
```

Changes:

1. **Rename `strength` → `factor`** to match nnterp's API. Keep `strength` as a
   deprecated alias for one minor version (warn + map in `__post_init__`).
2. **Add `positions: TokenSelectorParams | None = None`**. A new
   `core/configs/params/token_selector_params.py` already exists (used by
   extraction); reuse it here.
3. **Add `layers: list[int] | Literal["all"]`** — accept the string `"all"` and
   expand to `range(model.num_layers)` at context-build time.

### 4.2 New `PatchParams`

```python
# core/configs/params/patch_params.py (new)
@dataclass
class PatchParams(BaseConfig):
    src_extraction_path: str            # safetensors written by ActivationExtractor
    src_key: str                        # which activation key to read
    dst_spec: str                       # which site to write into
    positions: TokenSelectorParams      # required
```

---

## 5. Integration with existing code

### 5.1 `ResponseGenerator` — re-fronted

`generation/response_generator.py` today (~200 LOC) splits into:

| Today | After |
|---|---|
| `_load_vector(path, key, normalize, device)` | moves to `interventions/vectors.py:load_vector` |
| `_make_steering_hook(vector, mode, strength)` | deleted; replaced by `InterventionContext.add_steering` |
| `_resolve_layer_modules(model)` | deleted; nnterp handles model-family dispatch |
| forward-hook registration + cleanup | replaced by `with ctx: ...` |
| `AutoModelForCausalLM.from_pretrained(...)` | replaced by `StandardizedTransformer(...)` so generation and extraction load the same model class |
| `generate(samples)` loop | unchanged shape; body becomes `ctx.generate(prompts, ...)` |

After the refactor, `ResponseGenerator` is ~80 LOC of tokenizer + batch
plumbing on top of `InterventionContext`. No HF hooks anywhere in the toolkit.

### 5.2 YAML runner — new `intervene` action

`runners/experiment_runner.py` today wires `extract`, `generate`, `pipeline`,
and stubs out `probe_sweep` / `diff_means`. Add:

```python
def _action_intervene(cfg: PipelineConfig) -> InterveneResult:
    """YAML-driven intervention run.
    Loads model, builds InterventionContext from cfg.steering + cfg.patches,
    runs ctx.generate over cfg.dataset, persists responses + per-sample
    intervention metadata to cfg.output."""
```

Recipe (`configs/recipes/refusal_ablation.yaml`):

```yaml
run_name: refusal_ablation
action: intervene
model:
  model_name: meta-llama/Llama-3.1-8B-Instruct
  dtype: bfloat16
dataset:
  prompts_path: experiments/refusal_probing/data/prompts.jsonl
  prompt_key: text
generation:
  max_new_tokens: 256
  do_sample: false
steering:
  vector_path: artifacts/refusal_direction.safetensors
  layers: all
  factor: 1.0
  mode: project_subtract
output:
  output_dir: artifacts/refusal_ablation
```

### 5.3 Probe + direction interop

`vectors.load_vector(...)` accepts any of:

| Argument type | Handling |
|---|---|
| `torch.Tensor` (1D) | use directly |
| `str` ending in `.safetensors` | `safetensors.torch.load_file`, key resolution |
| `str` ending in `.pt` | `torch.load`, single tensor or dict with `vector_key` |
| `BaseProbe` instance | `probe.direction` (already unit-norm) |
| `DiffMeansLayerResult` instance | `.direction` |
| `LayerProbeSweepResult` | `.probes[result.best_key].model.direction` |

This is what makes "train a probe → use as steering vector" a one-liner.

---

## 6. `TokenSelector` integration

The existing `activation/token_selectors.py` selectors are reused unchanged.
For interventions, the contract becomes a *write mask*: a `(B, S)` boolean
tensor indicating which positions get the modification.

```python
# interventions/steering.py
def _resolve_write_mask(
    selector: TokenSelector | None,
    input_ids: torch.Tensor,         # (B, S)
    attention_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    """Returns a (B, S) bool mask, or None for 'all positions'."""
    if selector is None:
        return None
    # Selectors return positions; we expand into a 0/1 mask.
    ...
```

Application sketch for `mode="project_subtract"` with positions:

```python
with model.trace(prompts):
    for layer in pending.layers:
        h = model.layers_output[layer]                  # (B, S, D)
        v = pending.vector                              # (D,) unit norm
        dot = (h * v).sum(dim=-1, keepdim=True)          # (B, S, 1)
        delta = -dot * v                                  # (B, S, D)
        if mask is not None:
            delta = delta * mask.unsqueeze(-1)           # zero out non-mask positions
        model.layers_output[layer][:] = h + pending.factor * delta
```

Same pattern for additive (`delta = +v`). nnterp's `model.steer(...)` covers
the un-masked additive case; we fall back to the manual write only when a
selector is set or `mode != "additive"`.

---

## 7. Testing plan

### 7.1 Unit tests

`tests/test_intervention_context.py` (new):

- **Additive identity:** `factor=0` ⇒ outputs identical to no-steer baseline.
- **Norm invariance:** with `normalize=True`, scaling `vector_path` by 10 produces same effect.
- **Project-subtract correctness:** after `mode="project_subtract"` with `factor=1` on a
  single layer, the projection of layer activation onto `v̂` is < ε at that layer.
- **Position masking:** with `positions=Index(0)`, only position 0's activation
  changes; positions 1..S-1 are bit-equal to baseline.
- **Layer expansion:** `layers="all"` expands to `range(num_layers)`.
- **Vector loading:** `load_vector` accepts tensor, .pt, .safetensors with/without key,
  `BaseProbe` direction, `DiffMeansLayerResult.direction`.

### 7.2 Activation-patching test

`tests/test_intervention_patching.py`:

- Capture `layers_output:5` at the last position of prompt A.
- Patch it into a run on prompt B at the same position.
- Verify that `model.layers_output[5][:, -1, :]` in run B equals the captured tensor.

### 7.3 Smoke test against existing rjudge

`experiments/rjudge_dissociation/run_causal.py` already runs an all-layer
`project_subtract` ablation via the current `ResponseGenerator`. After the
refactor, the same script runs unchanged (same `SteeringParams` API), but the
underlying execution goes through `InterventionContext`. CI gate: produce
identical responses on a 4-scenario subset before/after the refactor (greedy
decoding, fixed seed, bf16).

### 7.4 Type / lint

Existing CI (ruff + pyright) catches the rest. New module added to
`[tool.ruff.lint.isort].known-first-party`.

---

## 8. Migration

PR sequencing (each independently mergeable, no big-bang):

| # | PR | Touches | Adds | Removes |
|---|---|---|---|---|
| 1 | `interventions/` skeleton + `InterventionContext` (additive only) | new package | `add_steering(mode='additive')`, `vectors.load_vector` | nothing yet |
| 2 | Directional ablation + position masking | `interventions/steering.py` | `mode='project_subtract'`, `positions=TokenSelector` | nothing yet |
| 3 | `ResponseGenerator` re-fronted on context | `generation/response_generator.py` | uses `StandardizedTransformer`, calls `InterventionContext` | `_make_steering_hook`, `_resolve_layer_modules`, raw HF hooks |
| 4 | YAML `intervene` action + `PatchParams` | `runners/experiment_runner.py`, `core/configs/params/patch_params.py`, `core/configs/pipeline_config.py` | `_action_intervene` | `NotImplementedError` for intervention path |
| 5 | Activation patching | `interventions/patching.py` | `add_patch(...)` | nothing |
| 6 | Per-head steering (ITI) | `interventions/heads.py` | `add_head_steering(...)` | nothing |

PRs 1–3 are the spine. PRs 4–6 are independent extensions.

---

## 9. Out of scope: deploy

This document does **not** cover:

- vLLM / SGLang / HF native inference integration.
- `ProbeMonitor` / `SteeringHook` for serving (HF forward-hook based).
- Weight orthogonalisation.
- `LogitsProcessor`-based gating.
- Side-channel score collection per request.

Those belong in `sonde.deploy`, a separate subpackage that does not import
nnterp/nnsight. The contract between research-mode `interventions/` and
production-mode `sonde.deploy/` is the **probe artifact** (a safetensors file
with `direction` + `bias` + `layer` + metadata) plus the YAML `interventions`
section, which both sides can consume. Building `sonde.deploy` is the next
chapter; that work re-uses every selector, every vector loader, and every
config dataclass from this design.

---

## 10. Open questions

1. **Does nnterp expose attention input/output proxies that survive a write?**
   Required for per-head steering (PR-6). Need to verify against current
   `StandardizedTransformer` accessors (`model.attentions_output[L]`, plus
   `model.layers[L].self_attn.o_proj.input`).
2. **Should multiple `SteeringParams` blocks be order-significant?**
   E.g. two project_subtracts on different layers — is composition
   commutative in the residual? In general no (later writes overwrite the
   activation the earlier projection was computed against). Decision: apply
   in YAML order; document the non-commutativity.
3. **`generate()` and intervention reapplication.** nnterp's `model.steer`
   inside `model.trace` applies once per forward pass. During generation, each
   token is a forward pass — does our context re-enter `trace` per step
   transparently, or do we need an explicit per-step hook? Needs a tracer
   read of nnterp's generation path. Likely answer: `ctx.generate(...)`
   internally drives a loop of single-step traces.
4. **Bias term.** Probes have a scalar bias; intervention typically uses the
   direction only. Confirm we never need bias on the write side; if we do,
   surface as `SteeringParams.bias_offset` later.
