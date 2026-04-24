# Linear Probes Toolkit — Architectural Audit

**Status:** Draft 1 · Apr 19 2026
**Scope:** End-to-end audit of the activation-extraction → probe-training → steering pipeline.
**Anchors:**
- Primer doc — `docs/linear-probes-primer.md` — the conceptual reference for what probes *should* do.
- Reference paper — Kramár & Engels et al., *Building Production-Ready Probes For Gemini* (arXiv 2601.11516, Jan 2026).
- Comparison codebase — `oumi-ai/oumi` at `/Users/aniruddhanramesh/dev/oumi/projects/oumi`.

---

## 0. Executive Summary

The toolkit covers the **happy path** of binary linear probing well: extract activations from a HF/nnsight model, persist them as safetensors with a manifest, train a small `nn.Module` probe per layer with shuffled-label and random-feature controls, write a JSON run manifest with a stable dataset fingerprint. Six probe architectures are already implemented. Steering exists for response generation. The config layer is OmegaConf-typed and Oumi-shaped.

The **gaps** that block scientifically credible probing on modern problems:

1. **Token positions are a single integer.** No ranges, no lists, no per-sample anchors, no "first response token". This forces every interesting probe (refusal, deception, hallucination, sycophancy) to either (a) hand-roll position resolution before extraction or (b) extract entire sequences and pool. Both are wrong.
2. **Probes do not share a base class.** Each architecture re-implements the `(B,S,D) ↔ (B,D)` dispatch and the mask-aware reduce. There's no `transform/score/aggregate` decomposition to mirror the Kramár et al. 6-stage framework. **MultiMax** (the paper's headline contribution) is not implemented; an MLP-capacity-baseline probe is missing; a regression / multi-class probe is missing.
3. **Datasets only model `(text, 0/1)`.** No contrast-pair container (CCS, RepE, CAA, ITI), no dual-label container (Quirky-LMs / refusal-vs-harmfulness), no token-labeled container (Othello, structural), no conversation container (multi-turn — the Gemini paper's specific failure mode).
4. **Steering and extraction use different model paths.** Extraction uses `nnterp.StandardizedTransformer` (`nnsight` underneath); steering uses raw HF `register_forward_hook`. There is no `InterventionContext`, no activation patching, no per-head intervention, no weight orthogonalization, no per-position steering masks. Hooks are private functions inside `ResponseGenerator`.
5. **Sequence-mode activations aren't persisted to safetensors.** Variable-length extractions silently fall back to in-memory only. Long-context experiments are not resumable.
6. **`BaseConfig` ≠ Oumi `BaseConfig`.** Same name, half the features. Missing `BaseParams`/`BaseConfig` split, missing `from_yaml_and_arg_list`, missing recursive `finalize_and_validate`. Trivial to align — and worth doing if this becomes `oumi.probes`.

The rest of this document spells each of these out, with file-line citations and concrete proposed APIs.

---

## 1. End-to-End Pipeline Map

### 1.1 Data flow (current)

```
ProbingSampleBuilder ──► SampleBundle (prompts, labels, ids)
        │
        ▼
ActivationExtractor (nnterp.StandardizedTransformer wrapping nnsight)
        │
        ├─► trace(prompts) per batch
        ├─► resolve activation specs (layers_output:5, mlps_input:0-4, ...)
        ├─► _select_token_position (int → activation[:, idx])
        └─► save() under nnsight tracing context
        │
        ▼
ExtractionResult (TypedDict)
   {model, requested, activations, sample_ids, labels, storage}
        │
        ├─► RECTANGULAR PATH: torch.cat → .safetensors + _manifest.pt
        └─► VARIABLE-LENGTH PATH: list[Tensor] → in-memory only (storage="in_memory")
        │
        ▼
ProbingDataset.from_extraction_result / from_extraction_path
        │
        ├─► pooled mode: features tensor (N, D)
        └─► sequence mode: list of (S_i, D) tensors + per-sample mask
        │
        ▼
LayerProbeSweepRunner
        │
        ├─► per activation_key: build_probe(probe_type, input_dim, **kwargs)
        ├─► train via BinaryProbeTrainer (BCEWithLogits, AdamW, early-stop)
        ├─► val_metrics → select best layer
        ├─► run_probe_with_controls (shuffled labels + random features)
        ├─► compute_dataset_fingerprint (sha256 over sample_ids/labels/groups)
        └─► write_run_manifest (JSON, immutable)
        │
        ▼
LayerProbeSweepResult + ProbeAnalyzer (matplotlib AUROC + cosine sim)
```

### 1.2 Module ownership

| Concern | Package | Key file | Public surface |
|---|---|---|---|
| Sample → string prompts | `dataset/` | `samples.py` | `ProbingSampleBuilder`, `SampleBundle`, `StringDataset` |
| Train/val/test splits | `dataset/` | `splitting.py` | `stratified_train_val_test_split` (group-aware) |
| Sequence collation | `dataset/` | `collate.py` | `sequence_collate_fn` |
| Probe-ready Dataset | `dataset/` | `probing_dataset.py` | `ProbingDataset` |
| Model + activations | `activation/` | `activation_extractor.py` | `ActivationExtractor` |
| Safetensors I/O | `activation/` | `storage.py` | `load_extraction_manifest`, `load_activation_value`, `resolve_activation_key` |
| Probe nn.Modules | `probes/architectures/` | `linear.py` `mean.py` `max.py` `softmax.py` `attention.py` `max_rolling_mean.py` | `build_probe(type, input_dim, **kw)` |
| Probe trainer | `probes/` | `linear.py` | `BinaryProbeTrainer`, `run_probe_with_controls` |
| Layerwise sweep | `probes/` | `sweep.py` | `LayerProbeSweepRunner` |
| Reproducibility | `probes/` | `run_manifest.py` | `compute_dataset_fingerprint`, `write_run_manifest` |
| Diff-of-means direction | `directions/` | `diff_means.py`, `sweep.py` | `DiffMeansEstimator`, `DiffMeansSweepRunner`, `evaluate_projection` |
| Generation + steering | `generation/` | `response_generator.py` | `ResponseGenerator` (private hook fns inside) |
| Configs | `core/configs/` | `params/*.py`, `*_config.py`, `base.py` | `BaseConfig`, `*Params`, stage configs |
| YAML runner | `runners/` | `experiment_runner.py` | `run_experiment`, `dispatch_action` |
| CLI | `cli/` | `main.py` | `interp <yaml> [-o k=v]` entrypoint |

---

## 2. Activation Extraction Audit

### 2.1 Model loading
`ActivationExtractor.__init__` (`activation/activation_extractor.py:62-96`):
- Wraps `nnterp.StandardizedTransformer` which itself wraps `nnsight.NNsight`.
- Filters `ModelParams` fields to a kwargs dict, with a special-case dtype string → `torch_dtype` mapping.
- Sets `default_activations = ["layers_output:<num_layers-1>"]` if none requested.
- Will raise `ModuleNotFoundError` if `nnterp` isn't installed (graceful).

### 2.2 Activation specifier grammar (this is good)

`_expand_requested_activations` + `_parse_layer_spec` accept:

| Form | Example | Expands to |
|---|---|---|
| Indexed kind, single layer | `layers_output:5` | `[layers_output:5]` |
| Indexed kind, all layers | `layers_output:*` or `layers_output:all` | `[layers_output:0, ..., layers_output:N-1]` |
| Indexed kind, inclusive range | `layers_output:0-4` | `[layers_output:0, 1, 2, 3, 4]` |
| Indexed kind, slice | `layers_output:0:10:2` | `[layers_output:0, 2, 4, 6, 8]` |
| Non-indexed kind | `token_embeddings`, `logits`, `next_token_probs`, `input_ids` | as-is |
| Path kind | `module_output:layers[4].mlp.gate_proj` | as-is, resolved by `_resolve_module_path` |

Indexed kinds: `{layers_input, layers_output, attentions_input, attentions_output, mlps_input, mlps_output, attention_probabilities}`.

This is well-designed and worth preserving as-is.

### 2.3 Token position handling — **the weak spot** (see §4)

Single integer or full sequence, period:

```493:519:activation/activation_extractor.py
    @staticmethod
    def _select_token_position(
        activation,
        token_index: int | None,
        *,
        kind: str,
        allow_2d: bool = False,
    ):
        # Hidden State: [batch, seq, hidden].
        if token_index is None:
            return activation
        if not hasattr(activation, "ndim"):
            return activation
        if activation.ndim == 3:
            return activation[:, token_index]
        if activation.ndim == 4:
            if kind == "attention_probabilities":
                return activation[:, :, token_index, :]
            raise ValueError(...)
```

`ExtractionParams.token_index: int | None = -1` is the only knob.

### 2.4 Persistence

`_persist_result` (`activation/activation_extractor.py:535-569`) writes two files:

```
<save_path>.safetensors          # tensors only, rectangular (N, D) per key
<save_path>_manifest.pt          # torch.save({model, requested, sample_ids, labels, storage})
```

The manifest re-references the safetensors path; activations are *not* duplicated into the manifest. `load_activation_value` lazy-loads on demand.

**Critical limitation** (lines 543-548):

```544:548:activation/activation_extractor.py
        has_list_activations = any(
            isinstance(v, list) for v in result["activations"].values()
        )
        if has_list_activations:
            return {"mode": "in_memory"}
```

If any activation key is a list-of-tensors (variable-length sequence mode), persistence is silently skipped. The manifest is never written. The extraction lives only in the returned dict. This means:
- Long-context probing experiments can't be resumed.
- Sequence-mode probes can't read activations from disk on a second run.
- The asymmetry between rectangular and ragged paths is invisible to callers — they get a `storage={"mode": "in_memory"}` and have to know what that means.

**Fix sketch.** Either:
- **Padded persistence**: store `(N, S_max, D)` + a `lengths: (N,)` int64 tensor in the same safetensors file. Costs memory; trivial to implement.
- **Ragged persistence**: store one safetensors file per sample, indexed by `sample_id`, with a JSON index alongside the manifest. Higher fan-out; cheaper for very-long sequences.

I'd default to padded; expose ragged as an option for >32k contexts.

---

## 3. Probe Architectures Audit

### 3.1 What's there

All in `probes/architectures/`. Registered in a flat dict in `__init__.py`:

```16:23:probes/architectures/__init__.py
_PROBE_REGISTRY: dict[str, type[nn.Module]] = {
    "linear": LinearProbe,
    "mean": MeanProbe,
    "max": MaxProbe,
    "softmax": SoftmaxProbe,
    "attention": AttentionProbe,
    "max_rolling_mean": MaxRollingMeanProbe,
}
```

| Probe | Aggregation | Where it differs |
|---|---|---|
| `LinearProbe` | last-token select (`x[:, -1, :]`) | Simplest |
| `MeanProbe` | mask-aware mean over seq dim | Standard pooling baseline |
| `MaxProbe` | element-wise max along seq, mask = `-inf` | Pooling baseline |
| `SoftmaxProbe` | learned `softmax(φ · w·x)` weighting | Train+infer use softmax |
| `AttentionProbe` | multi-head attention pooling, query is learned | Stores `attention_weights_` for inspection |
| `MaxRollingMeanProbe` | `avg_pool1d` window then `max` | Gemini paper's "max of rolling means" |

### 3.2 Mapping to Kramár & Engels et al. (2026) 6-stage framework

The paper decomposes any probe into:

```
(1) residual stream activations (B, S, D)
(2) per-position TRANSFORMATION  → (B, S, D')
(3) per-position SCORING          → (B, S, num_classes)
(4) AGGREGATION over positions
(5) optional final transform
(6) scalar output
```

Coverage table:

| Paper concept | In codebase | Status |
|---|---|---|
| Linear probe | `LinearProbe` | ✓ |
| Mean-aggregated | `MeanProbe` | ✓ |
| Softmax-aggregated (a.k.a. attention probe in their fig.2) | `SoftmaxProbe` | ✓ |
| Multi-head attention probe | `AttentionProbe` | ✓ |
| Max of Rolling Means | `MaxRollingMeanProbe` | ✓ |
| **MultiMax** (softmax at train, hard-max at inference) | — | **missing** |
| **Per-position MLP transform** (Stage 2 nonlinearity, used as their MLP-probe baseline) | — | **missing** |
| **Cascade probe + LLM classifier** (their cost-optimal monitor) | — | missing (out of scope for `probes/`, belongs in `monitoring/` or similar) |
| AlphaEvolve-discovered architectures | — | future / research-grade |

### 3.3 Other gaps (pre-Gemini-paper)

| Probe family | Provenance | Status |
|---|---|---|
| Logistic regression (BCE) | Alain & Bengio 2016, default in interp | ✓ via `BinaryProbeTrainer` |
| Difference-of-means | Marks & Tegmark 2023 | ✓ via `DiffMeansEstimator` |
| Ridge / regression probe | Conneau et al. 2018, Gurnee & Tegmark 2023 (lat/lon, time) | **missing** |
| Multi-class probe | Tenney et al. 2019 | **missing** |
| LDA / Fisher | Bayes-optimal under shared Σ | **missing** |
| MLP capacity baseline | Hewitt & Liang 2019 (selectivity control) | **missing** |
| k-sparse probe | Gurnee et al. 2023 | **missing** |
| CCS (Contrast-Consistent Search) | Burns et al. 2022 | **missing** |
| Structural probe (PSD metric) | Hewitt & Manning 2019 | **missing** |
| LEACE / nullspace concept erasure | Belrose et al. 2023 | **missing** |

### 3.4 Probe trainer is binary-only

`BinaryProbeTrainer` (`probes/linear.py:26-236`) is hardcoded to:

```35:52:probes/linear.py
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(...)
        self.accuracy_metric = BinaryAccuracy(threshold=self.config.threshold).to(self.device)
        self.precision_metric = BinaryPrecision(...)
        self.recall_metric = BinaryRecall(...)
        self.f1_metric = BinaryF1Score(...)
```

To support regression / multi-class we need either a `task: Literal["binary","multiclass","regression"]` switch in `ProbeParams` or three sibling trainer classes. The trainer itself is otherwise probe-agnostic — it accepts any `nn.Module` that returns `(B, 1)` logits — which is good.

---

## 4. Token-Position Handling — Detailed Gap

### 4.1 What's possible today

| Need | Today | Workaround in codebase |
|---|---|---|
| Last token | `token_index=-1` | default |
| Specific absolute index | `token_index=5` | works |
| Negative index from end | `token_index=-3` | works |
| Whole sequence | `token_index=None` | works (triggers sequence mode) |
| **Range of positions** (e.g., last 10 tokens) | impossible | extract full seq, slice in `ProbingDataset` |
| **List of positions** (e.g., `[0, 5, -1]`) | impossible | three separate extractions |
| **Per-sample anchor** (e.g., "first response token") | impossible per-row | hand-rolled in `main.py:72-74` for *constant* prefix only |
| **Token-id pattern match** (e.g., after `[/INST]`, after `<|assistant|>`) | impossible | tokenize+search before extraction |
| **Pad-aware "last non-pad token"** | impossible | extract whole seq, mask-pool |

The forced-prefix workaround in `main.py`:

```72:74:main.py
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    prefix_tokens = tokenizer.encode(FORCED_PREFIX, add_special_tokens=False)
    token_index = -len(prefix_tokens)
```

works only because *every prompt* uses the *exact same* `FORCED_PREFIX`. The moment prompts have variable structure (real conversations, varying chat templates, jailbreak suffixes of different lengths), this falls apart and you have to extract the whole sequence and mask-pool — which (a) is wasteful, (b) confuses the construct you're measuring (§4 of `docs/linear-probes-primer.md` — prompt-vs-response confounds).

### 4.2 Proposed `TokenSelector` abstraction

```python
# activation/token_selectors.py (new)
from typing import Protocol
import torch

class TokenSelector(Protocol):
    """Selects a subset of token positions from an activation tensor.

    All selectors take an activation of shape (B, S, D) (or (B, H, S, S) for
    attention probabilities) plus the input_ids and attention_mask used to
    produce it, and return either:
      - (B, D)  if a single position is selected per row,
      - (B, K, D) if K positions are selected per row (K may vary by row,
                  in which case rows are padded and a mask is returned),
      - (B, S, D) if all tokens are kept.
    """

    def select(
        self,
        activation: torch.Tensor,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Returns (selected_activation, optional_per_row_mask)."""

class IndexSelector:
    """Single position. Replaces today's int token_index."""
    def __init__(self, index: int): self.index = index

class RangeSelector:
    """Slice of positions (a:b[:step]).  Returns (B, K, D)."""
    def __init__(self, start: int | None, stop: int | None, step: int = 1): ...

class IndexListSelector:
    """Explicit list of positions, e.g. [0, 5, -1]. Returns (B, K, D)."""
    def __init__(self, indices: list[int]): ...

class LastNonPadSelector:
    """Per-row last non-pad position, computed from attention_mask."""

class TokenIdAnchorSelector:
    """Find first occurrence of a token-id pattern per row, then offset.

    Example: after `[/INST]` → pattern=[INST_END_ID], offset=0
             at the assistant's first generated token → offset=1
    """
    def __init__(self, pattern: list[int], offset: int = 0, mode: str = "first"): ...

class StringAnchorSelector(TokenIdAnchorSelector):
    """Convenience wrapper that tokenizes a string pattern with a tokenizer."""
    def __init__(self, anchor: str, tokenizer, *, offset: int = 0, mode: str = "first"): ...

class AllTokensSelector:
    """No-op. Replaces today's None token_index."""
```

### 4.3 Plumbing changes

```python
# core/configs/params/extraction_params.py
@dataclass
class ExtractionParams(BaseConfig):
    ...
    token_index: int | None = -1   # KEPT for back-compat — sugar over IndexSelector / AllTokensSelector
    token_selector: TokenSelector | None = None   # NEW — wins when set

# activation/activation_extractor.py
def _resolve_token_selector(self) -> TokenSelector:
    if self.extraction_params.token_selector is not None:
        return self.extraction_params.token_selector
    if self.extraction_params.token_index is None:
        return AllTokensSelector()
    return IndexSelector(self.extraction_params.token_index)

# In extract():
selector = self._resolve_token_selector()
for batch in loader:
    inputs = self._tokenize(batch)   # need to expose input_ids + attention_mask
    with self.model.trace(prompts):
        activation = self._resolve_activation(spec)
        activation, per_row_mask = selector.select(
            activation, input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
        )
        ...
```

This single change unblocks: real refusal probes (Arditi-style first-generated-token), per-token streaming monitors, Gemini-style long-context aggregation experiments (rolling windows over response tokens), and the §4 prompt-vs-response disambiguation that the primer doc spends a whole section on.

### 4.4 Storage implications

When `RangeSelector` / `IndexListSelector` returns `(B, K, D)`:
- Concatenate to `(N, K, D)` if K is constant across rows → safetensors-friendly.
- If K varies (e.g., `LastNonPadSelector` with variable lengths), fall back to ragged storage (see §2.4 fix).

---

## 5. Steering & Activation Patching Audit

### 5.1 What works

`generation/response_generator.py` does the obvious thing for additive / projection steering during `.generate()`:

- `_load_vector` (lines 21-37): loads `.pt` or `.safetensors`, normalizes.
- `_make_steering_hook` (lines 40-68): forward hook that mutates `output[0]` (residual stream) per the configured mode.
- `_resolve_layer_modules` (lines 71-82): heuristic resolution for Llama-style (`model.model.layers`) or GPT-2-style (`transformer.h`) decoder stacks.
- Hook registration + cleanup in a `try/finally` (lines 150-199).
- Two modes: `project_subtract` (directional ablation along a unit vector) and `additive` (`h + α·v̂`).

Configured via `SteeringParams`:

```12:34:core/configs/params/steering_params.py
@dataclass
class SteeringParams(BaseConfig):
    enabled: bool = False
    vector_path: str = ""
    vector_key: str = ""
    layers: list[int] = field(default_factory=list)
    strength: float = 10.0
    mode: str = "project_subtract"
    normalize: bool = True
```

### 5.2 What's broken or missing

| Capability | Status | Why it matters |
|---|---|---|
| Steering during a *forward* pass (not `.generate()`) | not exposed | Required to evaluate "does this probe's direction cause the predicted behavior?" — i.e., the canonical causal probe test |
| Per-token-range steering | not supported (every position gets `α·v̂`) | Refusal direction should be added at *response* tokens, not the whole prompt |
| Per-attention-head steering | not supported (only at residual stream output of layer) | ITI (Li et al. 2023) needs per-head |
| Activation patching (read from prompt A, write into prompt B at layer L, position t) | not implemented | Canonical mech-interp causal test |
| Weight orthogonalization (Arditi 2024) | not implemented | The "no-hook-needed" deployment story for refusal ablation |
| Models beyond Llama/GPT-2 (Gemma, Mixtral with experts, vision-language stacks) | silently breaks at `_resolve_layer_modules` | Future-proofing |
| Steer using nnterp/nnsight (the same path extraction uses) | no | We have **two** different read/write paths for the same model |
| `project_subtract` correctness | depends on `_load_vector` having normalized | One bug-fix away from breaking; no test for the projection identity `||h - (h·v̂)v̂|| ≤ ||h||` |

### 5.3 The asymmetry is the headline issue

```
Extraction path: HF model → nnterp.StandardizedTransformer → nnsight tracing → .save()
Steering path:   HF model → register_forward_hook → output[0] mutation
```

These are two completely different APIs over the same underlying model. Two consequences:

1. **Discoverable activations vs intervenable activations diverge.** `nnterp` exposes `mlps_input/output[i]`, `attentions_input/output[i]`, `attention_probabilities[i]`, arbitrary `module_input/module_output:<path>`. The HF hook path can only intervene on the *output* of a registered module — the rest is invisible to it.
2. **Maintenance burden doubles.** Every model-family special case (Gemma's `model.model.layers` vs GPT-2's `transformer.h` vs Mixtral's expert routers) has to be solved twice — once in `nnterp` for reads, once in `_resolve_layer_modules` for writes.

### 5.4 Proposed `InterventionContext`

```python
# interventions/context.py (new package)
from contextlib import AbstractContextManager
from typing import Self
import torch
from nnterp import StandardizedTransformer
from activation.token_selectors import TokenSelector

class InterventionContext(AbstractContextManager):
    """Unified read+write context over an nnterp model.

    Use the same model, the same tracing, the same activation grammar
    (`layers_output:5`, `mlps_input:0`, `module_output:layers[4].mlp`).
    """

    def __init__(self, model: StandardizedTransformer):
        self.model = model
        self._steers: list[_PendingSteer] = []
        self._patches: list[_PendingPatch] = []

    # Read-side (delegates to existing extractor logic)
    def read(self, spec: str, *, positions: TokenSelector | None = None) -> torch.Tensor: ...

    # Write-side: steering
    def add_steering(
        self,
        layer: int,
        vector: torch.Tensor,
        *,
        mode: str = "additive",          # "additive" | "project_subtract"
        alpha: float = 1.0,
        positions: TokenSelector | None = None,   # None = all positions
        sites: str = "layers_output",    # "layers_output" | "mlps_output" | "attentions_output" | "module_output:..."
    ) -> Self: ...

    # Write-side: per-head intervention (ITI-style)
    def add_head_steering(
        self,
        layer: int,
        head: int,
        vector: torch.Tensor,
        *,
        alpha: float = 1.0,
        positions: TokenSelector | None = None,
    ) -> Self: ...

    # Write-side: activation patching
    def patch_activation(
        self,
        *,
        src_activation: torch.Tensor,    # from a separate extraction
        dst_spec: str,                    # e.g., "layers_output:5"
        positions: TokenSelector,         # which positions in dst to overwrite
    ) -> Self: ...

    # Run a forward (or generate) under the configured interventions
    def __enter__(self) -> Self: ...
    def __exit__(self, *exc): ...   # restores everything
```

Usage shapes:

```python
# Causal probe test: "does the direction this probe found actually cause refusal?"
ctx = InterventionContext(model)
ctx.add_steering(
    layer=15, vector=probe.direction, mode="project_subtract",
    positions=TokenIdAnchorSelector(pattern=ASSISTANT_START_IDS, offset=0),
)
with ctx:
    outputs = model.generate(prompts, max_new_tokens=128)
# Did refusal rate drop? That's the causation test §6.4 of docs/linear-probes-primer.md describes.

# Activation patching: "does layer 15 of prompt A's run determine the answer to B?"
src = extractor.extract([prompt_A], activations=["layers_output:15"])["activations"]["layers_output:15"]
ctx = InterventionContext(model)
ctx.patch_activation(src_activation=src, dst_spec="layers_output:15", positions=IndexSelector(-1))
with ctx:
    outputs = model.generate([prompt_B])
```

This single context lifts steering from "thing inside ResponseGenerator" to "first-class API usable in trainers, evaluators, OOD pipelines, and CLI scripts."

### 5.5 Weight orthogonalization (Arditi 2024)

Once `direction` is available on every probe (see §6), this is ~20 lines:

```python
def orthogonalize_residual_writers(model, direction: Tensor, layers: range):
    """Project every matrix that writes to the residual stream to be orthogonal to `direction`.
    For Llama-style: W_O of every attention head and W_down of every MLP."""
    v = direction / direction.norm()
    P = torch.eye(v.numel(), device=v.device) - torch.outer(v, v)
    for L in layers:
        attn_o = model.model.layers[L].self_attn.o_proj.weight     # (D, H*Dh)
        mlp_d  = model.model.layers[L].mlp.down_proj.weight        # (D, Dff)
        attn_o.data = P @ attn_o.data
        mlp_d.data = P @ mlp_d.data
```

The point: with the right model abstraction (nnterp) plus `BaseProbe.direction`, this is trivial. Without them, it's hand-rolling per model family.

---

## 6. Probe Abstraction Proposal

### 6.1 Why a base class

Today every probe re-implements the same dispatch:

```7:17:probes/architectures/linear.py
class LinearProbe(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim == 3:
            x = x[:, -1, :]
        return self.linear(x)
```

```14:22:probes/architectures/mean.py
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim == 2:
            return self.linear(x)
        if mask is None:
            pooled = x.mean(dim=1)
        else:
            mask_expanded = mask.unsqueeze(-1)
            pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        return self.linear(pooled)
```

The `(B,S,D) ↔ (B,D)` dispatch and mask-aware reduce are duplicated across every probe. There's no place to put `direction` extraction (currently `LayerProbeSweepRunner._normalized_direction` reaches into `trainer.model.linear.weight` — only works for probes that have a `.linear` attribute, see `probes/sweep.py:278-287`).

### 6.2 Proposed `BaseProbe` (encodes the Kramár et al. 6-stage framework)

```python
# probes/architectures/base.py (new)
from __future__ import annotations
from abc import ABC, abstractmethod
import torch
from torch import nn

class BaseProbe(nn.Module, ABC):
    """Base class for all probes.

    Encodes the Kramár & Engels et al. (2026) 6-stage framework:
        (1) input  (B, S, D)  or  (B, D)
        (2) per-position transform     -> (B, S, D')
        (3) per-position score          -> (B, S, num_classes)
        (4)+(5) aggregate over positions -> (B, num_classes)
        (6) scalar output for each class

    Subclasses override transform / score / aggregate.
    """

    def __init__(self, input_dim: int, num_classes: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)               # (B, 1, D)
            if mask is None:
                mask = torch.ones(x.shape[0], 1, device=x.device, dtype=torch.float32)
        h = self.transform(x)                # Stage 2
        scores = self.score(h)               # Stage 3        -> (B, S, C)
        pooled = self.aggregate(scores, h, mask)  # Stages 4-5 -> (B, C)
        return pooled                        # Stage 6

    # Stage 2: identity by default (linear probes don't transform)
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return x

    # Stage 3: required
    @abstractmethod
    def score(self, h: torch.Tensor) -> torch.Tensor: ...

    # Stages 4-5: required
    @abstractmethod
    def aggregate(
        self, scores: torch.Tensor, h: torch.Tensor, mask: torch.Tensor | None,
    ) -> torch.Tensor: ...

    # ── Causal interpretability hooks ──
    @property
    def direction(self) -> torch.Tensor | None:
        """Unit-norm direction in activation space, if the probe has one.
        Used by InterventionContext for steering / ablation."""
        return None

    @property
    def bias(self) -> float | None:
        """Scalar bias of the decision function, if defined."""
        return None
```

### 6.3 Concrete probes refactored

```python
# probes/architectures/linear.py
class LinearProbe(BaseProbe):
    """Last-token (or fixed-position) linear probe."""
    def __init__(self, input_dim: int, position: int = -1):
        super().__init__(input_dim)
        self.linear = nn.Linear(input_dim, 1)
        self.position = position
    def score(self, h): return self.linear(h)              # (B, S, 1)
    def aggregate(self, s, h, mask): return s[:, self.position, :]
    @property
    def direction(self):
        w = self.linear.weight.detach().reshape(-1).float()
        return w / w.norm().clamp_min(1e-12)
    @property
    def bias(self):
        return float(self.linear.bias.detach().item()) if self.linear.bias is not None else None


# probes/architectures/mean.py
class MeanProbe(BaseProbe):
    def __init__(self, input_dim): super().__init__(input_dim); self.linear = nn.Linear(input_dim, 1)
    def score(self, h): return self.linear(h)
    def aggregate(self, s, h, mask): return _masked_mean(s, mask)
    @property
    def direction(self): return _normalize(self.linear.weight)


# probes/architectures/multimax.py  (NEW)
class MultiMaxProbe(BaseProbe):
    """Train with softmax weighting; switch to hard-max at inference.
    Kramár & Engels et al. 2026, the paper's headline architecture."""
    def __init__(self, input_dim, phi: float = 5.0):
        super().__init__(input_dim)
        self.linear = nn.Linear(input_dim, 1)
        self.phi = phi
    def score(self, h): return self.linear(h)
    def aggregate(self, s, h, mask):
        if self.training:
            return _softmax_pool(s, h, mask, phi=self.phi)
        return _masked_max(s, mask)
    @property
    def direction(self): return _normalize(self.linear.weight)


# probes/architectures/mlp.py  (NEW)
class MLPProbe(BaseProbe):
    """Capacity baseline. Use with control tasks for selectivity (Hewitt & Liang 2019)."""
    def __init__(self, input_dim, hidden_dim: int = 128, position: int = -1):
        super().__init__(input_dim)
        self.transform_module = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU())
        self.linear = nn.Linear(hidden_dim, 1)
        self.position = position
    def transform(self, x): return self.transform_module(x)
    def score(self, h): return self.linear(h)
    def aggregate(self, s, h, mask): return s[:, self.position, :]
    # No `direction` — MLP probes don't have a single causal direction.
```

### 6.4 Trainer simplifications

`LayerProbeSweepRunner._normalized_direction` (`probes/sweep.py:278-287`) becomes:

```python
@staticmethod
def _normalized_direction(trainer):
    return getattr(trainer.model, "direction", None)
```

`probe.bias` extraction in `sweep.py:100-103` collapses similarly.

### 6.5 Multi-class / regression support

Add a `task` field in `ProbeParams`:

```python
@dataclass
class ProbeParams(BaseParams):   # also: rename BaseConfig→BaseParams here, see §8
    task: Literal["binary", "multiclass", "regression"] = "binary"
    num_classes: int = 1            # ignored when task="binary"
    ...
```

Then `BinaryProbeTrainer` becomes `ProbeTrainer` with the loss/metrics keyed off `task`:

| `task` | Loss | Metrics |
|---|---|---|
| `binary` | `BCEWithLogitsLoss` | accuracy, AUROC, F1, P, R |
| `multiclass` | `CrossEntropyLoss` | accuracy, macro-F1, per-class P/R |
| `regression` | `MSELoss` (or HuberLoss) | MSE, MAE, R² |

Each `BaseProbe` subclass passes `num_classes` through to its final `nn.Linear`.

### 6.6 Registry lift (Oumi-shaped)

Replace the flat dict with:

```python
# probes/registry.py
from oumi.core.registry import Registry, RegistryType   # if integrating; else local copy

# Add probe registry type
class ProbeRegistryType(Enum):
    PROBE = auto()
    PROBE_DATASET = auto()
    TOKEN_SELECTOR = auto()
    INTERVENTION = auto()

REGISTRY = Registry()

def register_probe(name: str):
    def decorator(cls: type[BaseProbe]) -> type[BaseProbe]:
        REGISTRY.register(name, cls, ProbeRegistryType.PROBE)
        return cls
    return decorator

# probes/architectures/linear.py
@register_probe("linear")
class LinearProbe(BaseProbe): ...

# build_probe becomes:
def build_probe(probe_type: str, input_dim: int, **kwargs) -> BaseProbe:
    cls = REGISTRY.get(probe_type, ProbeRegistryType.PROBE)
    return cls(input_dim=input_dim, **kwargs)
```

This is the single largest "we converged with Oumi" signal.

---

## 7. Dataset Abstractions Proposal

### 7.1 What the existing types model

```11:17:dataset/types.py
@dataclass
class SampleBundle:
    prompts: Dataset[str]
    labels: list[int | None]
    ids: list[str]
    responses: list[str] | None = None
```

```19:31:dataset/probing_dataset.py
class ProbingDataset(Dataset[...]):
    """Dataset for binary probes supporting pooled (N, D) and sequence (variable S, D) modes."""

    def __init__(
        self,
        features: Sequence[torch.Tensor] | torch.Tensor,
        labels: Sequence[int],
        sequence_mode: bool | None = None,
    ):
        label_values = [int(label) for label in labels]
        if any(label not in (0, 1) for label in label_values):
            raise ValueError("Binary probe labels must be 0 or 1.")
```

### 7.2 What probes from the literature actually need

| Probe family | Provenance | Required dataset shape |
|---|---|---|
| Logistic / DoM | Alain & Bengio, Marks & Tegmark | `(text, label∈{0,1})` ✓ today |
| Multi-class | Tenney et al. | `(text, label∈{0,…,K-1})` |
| Regression | Conneau et al., Gurnee & Tegmark | `(text, y∈ℝ)` |
| **CCS** | Burns et al. 2022 | `(text⁺, text⁻)` *paired*, no labels |
| **RepE / CAA / ITI** | Zou et al., Panickssery et al., Li et al. | `(prompt, behavior_pos, behavior_neg)` triples |
| **Quirky LMs** | Mallen & Belrose 2023 | `(text, behavior_label, ground_truth_label)` — *two* labels |
| **Refusal vs harmfulness** disambiguation | implied by §4 of `docs/linear-probes-primer.md` | dual-label like Quirky |
| **Othello / world models** | Li et al. 2023 | `(state_seq, per_token_state_label)` |
| **Structural** | Hewitt & Manning 2019 | `(tokens, tree_distance_matrix)` per sample |
| **Multi-turn / long-context** | Gemini paper 2026 | `[turn₁, turn₂, …]` + per-turn or final label |

### 7.3 Proposed dataset hierarchy (mirrors `oumi.core.datasets.BaseMapDataset → Base{Sft,Dpo,Grpo,Kto,…}Dataset`)

```python
# dataset/base.py (new)
from abc import ABC, abstractmethod
from torch.utils.data import Dataset

class BaseProbingDataset(Dataset, ABC):
    """Abstract base class for all probe-ready datasets.
    Subclasses define their own __getitem__ contract and collator."""

    sequence_mode: bool

    @abstractmethod
    def __len__(self) -> int: ...
    @abstractmethod
    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]: ...
    @classmethod
    @abstractmethod
    def from_extraction_result(cls, extraction, **kwargs) -> Self: ...
```

Concrete subclasses:

```python
class BinaryProbingDataset(BaseProbingDataset):
    """Today's ProbingDataset, renamed and made explicit about its task."""
    # __getitem__ → (features, label∈{0,1})  or  (features, label, mask)

class MultiClassProbingDataset(BaseProbingDataset):
    """Multi-class targets."""
    # __getitem__ → (features, label∈{0,…,K-1})

class RegressionProbingDataset(BaseProbingDataset):
    """Continuous targets (lat/lon, time, tree-distance scalars)."""
    # __getitem__ → (features, target∈ℝ^k)

class ContrastPairDataset(BaseProbingDataset):
    """For CCS, RepE, CAA, ITI. Two activations per sample."""
    # __getitem__ → (features_pos, features_neg)

class DualLabelDataset(BaseProbingDataset):
    """For Quirky-LMs and refusal-vs-harmfulness experiments.
    `behavior_label` is what the model did; `truth_label` is what's true."""
    # __getitem__ → (features, behavior_label, truth_label)

class TokenLabeledDataset(BaseProbingDataset):
    """For Othello-style and structural probes — per-token targets."""
    # __getitem__ → (features (S, D), per_token_labels (S,), mask (S,))

class ConversationProbingDataset(BaseProbingDataset):
    """Multi-turn. Holds activations per turn + a label per turn or final."""
    # __getitem__ → ({per_turn_features}, label or per_turn_labels)
```

### 7.4 Collators

`dataset/collate.py` currently has only `sequence_collate_fn` for variable-length single-sequence. Add:

- `contrast_pair_collate_fn` — pad both sides of the pair independently.
- `token_labeled_collate_fn` — pad features + labels + mask consistently.
- `conversation_collate_fn` — pad per-turn, return per-turn masks.

### 7.5 `SampleBundle` extensions

`SampleBundle` should grow sibling builders for the non-binary cases — but stays as the single source of truth for "string prompts that an extractor consumes." Add classmethods:

```python
@classmethod
def from_contrast_pairs(cls, pairs: list[tuple[str, str]]) -> SampleBundle: ...
# yields prompts = [pos_0, neg_0, pos_1, neg_1, ...] with paired ids that
# downstream ContrastPairDataset can decode.
```

---

## 8. Comparison to `oumi-ai/oumi`

`oumi` lives locally at `/Users/aniruddhanramesh/dev/oumi/projects/oumi`. The patterns to converge on:

| Concern | sonde today | oumi today | Recommendation |
|---|---|---|---|
| Config base | One `BaseConfig` (`core/configs/base.py`) used for both top-level configs and nested params; `from_yaml`/`to_yaml`/`from_dict` only | `BaseConfig` (top-level only) + `BaseParams` (nested), with recursive `finalize_and_validate`, CLI dotlist merge (`from_yaml_and_arg_list`), interpolation control, callable-source-aware equality | Split into `BaseConfig` (top-level) and `BaseParams` (nested). Every `*Params` becomes `BaseParams`. Adopt `finalize_and_validate` recursion |
| `ModelParams` | `model_name: str = ""`, basic flags | `model_name: str = MISSING` (omegaconf MISSING), tokenizer fields, adapter_model, processor_kwargs, hardware exception handling | Use `MISSING` for required fields; add tokenizer/adapter/processor fields — needed for chat templates and multimodal probing |
| Dataset abstractions | `ProbingDataset` (concrete), `StringDataset` (concrete) | `BaseMapDataset` (ABC) → `BaseSftDataset`, `BaseDpoDataset`, `BaseGrpoDataset`, `BaseKtoDataset`, `BaseRubricDataset`, `VisionLanguageDataset`, `BasePretrainingDataset`, all ABCs → concrete classes registered via `@register_dataset(...)` | Adopt `BaseProbingDataset` ABC (§7) with concrete subclasses per task |
| Registry | Flat `_PROBE_REGISTRY` dict in `probes/architectures/__init__.py` | Typed `Registry` singleton with `RegistryType` enum (`DATASET`, `MODEL`, `MODEL_CONFIG`, `METRICS_FUNCTION`, `JUDGE_CONFIG`, `EVALUATION_FUNCTION`, `SAMPLE_ANALYZER`, …); decorators `@register_dataset(name)`, `@register_judge_config(...)` | Lift to a `Registry` with new types: `PROBE`, `PROBE_DATASET`, `TOKEN_SELECTOR`, `INTERVENTION`. Decorator-based registration |
| Validation | `__post_init__` per dataclass | `__finalize_and_validate__()` recursively (one level of nesting auto-walked) | Same effect today, but adopting the recursion pattern keeps you compatible if you upstream into Oumi |
| Configs directory | `configs/recipes/quickstart_probe.yaml` (one file) | `configs/recipes/{model}/{task}/...` deep tree | Mirror layout once you grow past one recipe |
| Analyze namespace | `probes/analyze.py` (matplotlib for AUROC + cosine sim) | `oumi.analyze` framework (`base.py`, `pipeline.py`, `discovery.py`); `oumi.core.analyze.SampleAnalyzer` ABC; built-in analyzers: `length`, `quality`, `turn_stats` (text-quality only — NO probe-based analyzer exists in Oumi) | **Biggest integration win:** register your probe-based monitors as `SAMPLE_ANALYZER`s. You'd be the first probe-as-analyzer contributor. Concretely: `ProbeMonitorAnalyzer` that loads a trained probe + its safetensors direction and produces per-sample scores |
| CLI | argparse, single `interp <yaml>` entrypoint with `-o key=val` | typer-based `oumi train\|infer\|evaluate\|judge\|synth\|analyze\|launch ...` | If you converge into Oumi: register an `oumi probe` subcommand. Until then, current CLI is fine |
| Logging | `print(...)` in `experiment_runner.py` | `oumi.utils.logging.logger` | Switch to Python `logging` |

**Bottom line on alignment.** sonde is structurally Oumi-compatible already (omegaconf, dataclass configs, base-config pattern, `core/configs/params/` layout) but consciously not yet merged. The fastest path to real integration is:

1. Split `BaseConfig` / `BaseParams`.
2. Adopt the `Registry` system.
3. Register the toolkit's probe-monitor as a `SAMPLE_ANALYZER`.

Each is a small PR. Together they make `sonde` feel like a natural `oumi.probes` package.

---

## 9. Roadmap — Ranked by Leverage

| # | Item | Files touched | Estimated effort | Unblocks |
|---|---|---|---|---|
| 1 | **`TokenSelector` abstraction** (§4) | `core/configs/params/extraction_params.py`, `activation/activation_extractor.py`, new `activation/token_selectors.py`, `main.py`, tests | 1-2 days | Real refusal / deception / sycophancy probes; per-token streaming; long-context aggregation |
| 2 | **`BaseProbe` ABC + 6-stage decomposition** (§6) | `probes/architectures/*` (all), `probes/sweep.py` (`_normalized_direction`), tests | 2-3 days | MultiMax, MLP capacity baseline, regression/multi-class probes; `direction`/`bias` for any probe |
| 3 | **`InterventionContext` for steering + patching** (§5) | New `interventions/` package; `generation/response_generator.py` becomes a thin user; tests | 3-5 days | Causal probe tests, activation patching, weight orthogonalization, per-head ITI, extraction/intervention symmetry |
| 4 | **`BaseProbingDataset` hierarchy** (§7) | New `dataset/base.py`; `dataset/probing_dataset.py` becomes `BinaryProbingDataset`; new sibling classes; collators | 3-4 days | CCS, RepE, CAA, ITI, Quirky-LMs, Othello, structural, multi-turn probes |
| 5 | **Sequence-mode safetensors persistence** (§2.4) | `activation/activation_extractor.py:_persist_result`, `activation/storage.py`, tests | 1-2 days | Long-context experiments are resumable; sequence-mode probes work from disk |
| 6 | **Registry lift** (§6.6, §8) | New `probes/registry.py`, decorator usage in `probes/architectures/*`, `dataset/*`, `interventions/*` | 1-2 days | `@register_probe("linear")`, `@register_dataset("ccs_pairs")` — Oumi-shaped extension points |
| 7 | **Split `BaseConfig` / `BaseParams` + `from_yaml_and_arg_list`** (§8) | `core/configs/base.py`, every `core/configs/params/*.py`, runner | 0.5-1 day | Direct compatibility with `oumi.core.configs.BaseConfig` |
| 8 | **`ProbeMonitorAnalyzer` registered as Oumi `SAMPLE_ANALYZER`** | New module that imports from `oumi.core.analyze.SampleAnalyzer` | 1 day | Probes plug into the rest of Oumi's analysis pipeline; integration story for upstreaming |

**Items 1-4 are the substantive ones.** They map directly to the four user questions in the kickoff (token positions, probe abstractions, dataset abstractions, steering/patching/hooks).

**Items 5-8 are alignment work** that pays off when (not if) this becomes `oumi.probes`.

### 9.1 Suggested PR sequencing

To keep changes reviewable and avoid one mega-PR, the dependency order is:

```
PR-1: TokenSelector                     (independent)
PR-2: BaseProbe ABC + refactor probes   (independent)
PR-3: BaseProbingDataset hierarchy      (depends on PR-1 for token ranges in token-labeled datasets)
PR-4: Sequence-mode safetensors         (independent; helps PR-3)
PR-5: Registry lift                     (depends on PR-2, PR-3 — covers the new types)
PR-6: InterventionContext               (depends on PR-1 for positions, PR-2 for `direction`)
PR-7: BaseConfig / BaseParams split     (independent; cosmetic)
PR-8: SAMPLE_ANALYZER registration      (depends on PR-6)
```

PR-1 and PR-2 can land in parallel. Everything else gates on those two.

---

## 10. Quick-Reference — Mapping `docs/linear-probes-primer.md` Concepts to Code

| Primer doc concept | Code home today | After roadmap |
|---|---|---|
| Probe weight `w ∈ ℝᵈ` | `trainer.model.linear.weight` (only for probes with `.linear`) | `probe.direction` on `BaseProbe` |
| Layer sweep | `LayerProbeSweepRunner` | unchanged |
| Token position `t` (§4) | `ExtractionParams.token_index: int \| None` | `ExtractionParams.token_selector: TokenSelector` |
| Read site (residual / MLP / attn-head) | `layers_output:i`, `mlps_output:i`, `attentions_output:i` | unchanged + per-head selectors via `module_output:layers[L].self_attn.o_proj` |
| Difference of means (§2.2, §6.2) | `DiffMeansEstimator` | unchanged |
| Refusal probe site disambiguation (§4.3) | hand-rolled in `main.py` for forced prefix | `TokenIdAnchorSelector` for any anchor pattern |
| Probes stored as safetensors (§2.4) | partial — extractions yes, probe weights no | also save `probe.state_dict()` + `direction` + manifest as a single `.safetensors` |
| Steering direction `w / ||w||` (§6.1) | manual via `_load_vector` + `SteeringParams` | `InterventionContext.add_steering(layer, probe.direction, ...)` |
| Directional ablation (§6.1) | `mode="project_subtract"` in `SteeringParams` | `InterventionContext.add_steering(..., mode="project_subtract")` |
| Weight orthogonalization (Arditi 2024, §6.1) | not implemented | `interventions/weight_orthogonalize.py` |
| Activation steering during generation (§6.5) | `ResponseGenerator` with hooks | `ResponseGenerator` uses `InterventionContext` |
| Activation patching | not implemented | `InterventionContext.patch_activation` |
| MDL probing / V-information (§2.4) | not implemented | future probe-evaluation module |
| Selectivity vs control tasks (Hewitt & Liang) (§2.4) | partial — shuffled labels + random features in `run_probe_with_controls` | add proper word-type-randomized control tasks |
| Probe + LLM-classifier cascade (Gemini paper §7.1) | not implemented | new `monitoring/cascade.py` |

---

## Appendix A — File-by-file inventory

### `activation/`
- `activation_extractor.py` (579 lines) — `ActivationExtractor` class, all extraction logic
- `storage.py` (118 lines) — manifest I/O, key resolution, lazy safetensors load
- `types.py` (33 lines) — `ExtractionResult` TypedDict, `LayerSpec`, `ModelMetadata`
- `__init__.py` (1 line) — exports `ActivationExtractor`

### `dataset/`
- `samples.py` (209 lines) — `ProbingSampleBuilder`, `StringDataset`, file-format loaders
- `probing_dataset.py` (224 lines) — `ProbingDataset` (binary, pooled or sequence)
- `splitting.py` (291 lines) — group-aware stratified train/val/test, with hard validation
- `collate.py` (26 lines) — `sequence_collate_fn` only
- `types.py` (66 lines) — `SampleBundle` dataclass + split helper

### `probes/`
- `architectures/` — `linear`, `mean`, `max`, `softmax`, `attention`, `max_rolling_mean`, plus flat registry
- `linear.py` (410 lines) — `BinaryProbeTrainer`, `run_probe_with_controls`, helpers
- `sweep.py` (338 lines) — `LayerProbeSweepRunner`
- `analyze.py` (117 lines) — `ProbeAnalyzer` (matplotlib AUROC + cosine sim heatmap)
- `run_manifest.py` (100 lines) — fingerprint + JSON manifest
- `types.py` (35 lines) — `TrainedLayerProbe`, `LayerProbeSweepResult`

### `directions/`
- `diff_means.py` (95 lines) — `DiffMeansEstimator`, `evaluate_projection`
- `sweep.py` (283 lines) — `DiffMeansSweepRunner`
- `types.py` (30 lines) — `DiffMeansLayerResult`, `DiffMeansSweepResult`

### `generation/`
- `response_generator.py` (207 lines) — `ResponseGenerator` + private hook fns
- `types.py` (90 lines) — `GenerationResult` (JSONL round-trip)

### `core/configs/`
- `base.py` (41 lines) — `BaseConfig` (slim — see §8 for gap)
- `params/` — `extraction_params`, `model_params`, `probe_params`, `dataset_params`, `generation_params`, `io_params`, `output_params`, `split_params`, `steering_params`, `sweep_params`
- `*_config.py` — `ExtractConfig`, `GenerateConfig`, `ProbeConfig`, `DiffMeansConfig`, `PipelineConfig`
- `aliases.py`, `overrides.py` — alias resolution + dot-list overrides

### `runners/`
- `experiment_runner.py` (336 lines) — `run_experiment`, `dispatch_action`, `_action_*` handlers (note: `_action_probe_sweep` and `_action_diff_means` are `NotImplementedError` — only `extract`, `generate`, and `pipeline` are wired)

### `cli/`
- `main.py` (58 lines) — argparse-based `interp <yaml> [-o k=v]`

### Tests
- 26 test files in `tests/` covering configs, extractor, sweep orchestrator, splitting, manifest, generation, etc. The test surface is good and should make the refactors above safe.

---

## Appendix B — Why the Kramár & Engels et al. (2026) paper matters here

The paper is the most direct reference for what a "production-ready" version of this toolkit looks like. Its three operational claims map onto our gaps:

1. **"Probes fail under long-context distribution shift."** — Their MultiMax + Max-of-Rolling-Means architectures address this. We have `MaxRollingMeanProbe` but not `MultiMax`. Adding it is one file.
2. **"A combination of architecture choice + diverse training data is needed."** — The "diverse training data" half requires `ConversationProbingDataset` (multi-turn) and a way to extract activations at *response* tokens specifically (`TokenIdAnchorSelector`). Both are in the roadmap.
3. **"Cascade probes with LLM classifiers for cost-optimal accuracy."** — This is a `monitoring/` module on top of trained probes; it doesn't change the toolkit's core, but it's the natural deployment story.

Their 6-stage framework (transform → score → aggregate) is also the cleanest way to organize `BaseProbe`. Adopting it now means future probes (from this paper or others) drop in by overriding 1-2 methods instead of re-implementing the full forward.

---

*End of audit.*
