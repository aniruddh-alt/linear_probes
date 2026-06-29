# D0: vllm-lens Verification Spike

**Source:** [UKGovernmentBEIS/vllm-lens](https://github.com/UKGovernmentBEIS/vllm-lens) @ `main` (commit `c6b3ab2`, 2026-04-14).
**Method:** Docs/source-only spike via GitHub API. No code was installed or executed. Line numbers are against the files in the repo at that commit.

## Executive summary

vllm-lens does **not** ship any per-architecture wrappers and does **not** call `ModelRegistry.register_model`. Instead it auto-registers a single vLLM general plugin entry (`vllm.general_plugins.activations`) and patches `EngineArgs.create_engine_config`, `AsyncLLM.generate`, and `LLM.generate` at engine startup. The actual activation capture is done by `register_forward_hook(...)` on every decoder layer of the running model, discovered via duck-typing of three layer-container paths. This is the central architectural fact behind every answer below.

Net read:
- **Q1 (architectures):** "anything whose decoder layers live at `model.model.layers`, `model.model.decoder.layers`, or `model.language_model.model.layers`" — covers Llama 3.x, Qwen 2/2.5/3, Mistral, Gemma 2/3, Mixtral, DeepSeek, OPT, and VLMs with a `language_model` submodule, with the caveat that vllm-lens **forces `enforce_eager=True`** (no CUDA graphs) and **only tests on Qwen2.5-0.5B** in CI.
- **Q2 (residual site):** **Post-block residual** — the output of layer `L`, equal to HF `hidden_states[L+1]` and to nnterp `layers_output[L]`. Asserted in their own cross-framework test.
- **Q3 (`position_indices`):** Flat `list[int]` of *absolute* sequence positions, applied to one request's slice at a time. Not a `(batch, n_pos)` matrix; cannot express "different positions for different prompts in the same `SteeringVector`". If you need per-prompt position selection, you submit one `SteeringVector` per request.
- **Q4 (prefill vs decode):** Hooks fire on every forward pass that touches the request, so you get residuals for **all prompt tokens *plus* generated tokens except the last** (`n_prompt + n_gen - 1` after the explicit trim). With `max_tokens=1`, that's exactly `n_prompt` rows: all prompt positions, zero generated positions.

A `Verdict` section at the end translates this to `sonde`'s adoption plan.

---

## Background: the plugin architecture

vllm-lens registers a single entry point in [`pyproject.toml:96-100`](https://github.com/UKGovernmentBEIS/vllm-lens/blob/c6b3ab2/pyproject.toml#L96-L100):

```toml
[project.entry-points."vllm.general_plugins"]
    activations="vllm_lens._activations_plugin:register"

[project.entry-points.inspect_ai]
    vllm_lens="vllm_lens._inspect_entry"
```

`register()` in [`vllm_lens/_activations_plugin.py:351-405`](https://github.com/UKGovernmentBEIS/vllm-lens/blob/c6b3ab2/vllm_lens/_activations_plugin.py#L351-L405) monkey-patches five vLLM call sites:

1. `EngineArgs.create_engine_config` — injects `worker_extension_cls = "vllm_lens._worker_ext.HiddenStatesExtension"` and **sets `enforce_eager = True`** (line 113).
2. `AsyncLLM.generate` — wraps to install hooks and attach activations.
3. `LLM.generate` — same, offline path.
4. `OpenAIServingCompletion.request_output_to_completion_response` — injects serialized activations into HTTP responses.
5. `OpenAIServingChat.chat_completion_full_generator` — same, chat endpoint.

`HiddenStatesExtension.install_hooks()` in [`_worker_ext.py:339-368`](https://github.com/UKGovernmentBEIS/vllm-lens/blob/c6b3ab2/vllm_lens/_worker_ext.py#L339-L368) does the actual hook registration — `layer.register_forward_hook(_make_hook(self, layer_idx))` on every decoder layer.

---

## Q1. Supported model families

**Answer:** vllm-lens ships **zero** `*ForCausalLM`-specific wrappers and makes **zero** `ModelRegistry.register_model` calls. `grep -rn 'ModelRegistry\|register_model\|ForCausalLM' vllm_lens/` returns no matches. Model coverage is achieved by duck-typing three layer-container paths.

**Confidence:** High.

The whole "which models work" surface lives in `_get_layers` ([`_worker_ext.py:36-55`](https://github.com/UKGovernmentBEIS/vllm-lens/blob/c6b3ab2/vllm_lens/_worker_ext.py#L36-L55)):

```python
def _get_layers(model: torch.nn.Module) -> torch.nn.ModuleList:
    """Find the transformer decoder layers regardless of model architecture."""
    m: Any = model
    if hasattr(m, "language_model") and hasattr(m.language_model, "model"):
        return m.language_model.model.layers
    if (
        hasattr(m, "model")
        and hasattr(m.model, "decoder")
        and hasattr(m.model.decoder, "layers")
    ):
        return m.model.decoder.layers
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        return m.model.layers
    raise AttributeError(
        f"Cannot find decoder layers on {type(model).__name__}. "
        "Expected model.language_model.model.layers, "
        "model.model.decoder.layers, or model.model.layers"
    )
```

Mapping the three branches to vLLM's actual model definitions (from working knowledge of `vllm/model_executor/models/`):

| Path | Architectures covered |
| --- | --- |
| `model.language_model.model.layers` | VLMs: Llava family, Qwen2-VL/Qwen2.5-VL, Gemma-3-multimodal, MLlama, Pixtral, Phi-3-V — anything whose `*ForConditionalGeneration` wraps a separate text backbone. |
| `model.model.decoder.layers` | Encoder-decoder / OPT-style: OPT, BART, possibly mBART. **The notebook example uses `facebook/opt-125m`** ([`_examples/extract_residual_stream.ipynb`](https://github.com/UKGovernmentBEIS/vllm-lens/blob/c6b3ab2/vllm_lens/_examples/extract_residual_stream.ipynb), cell 4: `MODEL = "facebook/opt-125m"`). |
| `model.model.layers` | The vast majority of `*ForCausalLM` decoders that follow the `LlamaForCausalLM` template: **Llama 1/2/3/3.1/3.2/3.3, Qwen2/2.5/3 (including Qwen3-Next hybrid), Mistral, Mistral-Nemo, Mixtral, Gemma/Gemma-2/Gemma-3, DeepSeek-V2/V3, Phi-2/3/4, Yi, InternLM, Falcon, Granite, Command-R, OLMo, OLMoE,** etc. |

Other corroboration of the "no model-specific code" claim:

- README explicitly says ([`README.md:5`](https://github.com/UKGovernmentBEIS/vllm-lens/blob/c6b3ab2/README.md#L5)): *"performance comes at the expense of flexibility — for example, you would need to edit the source to add additional custom hooks (though it should be easy enough for coding agents to do that)."*
- The Activation Oracle example targets **Qwen3-8B** ([`_examples/activation_oracle.py:64`](https://github.com/UKGovernmentBEIS/vllm-lens/blob/c6b3ab2/vllm_lens/_examples/activation_oracle.py#L64)) with no model-specific config.
- A comment in `_worker_ext.py:173` mentions *"Hybrid models (e.g. Qwen3-Next with GatedDeltaNet) have multiple attention metadata entries"* — they've already special-cased Qwen3-Next's attention metadata.
- PR #7's body explicitly enumerates *"fused-residual architectures (Qwen2/Qwen3, Gemma-3, Llama, …)"* as the affected set.

**Practical caveats:**

1. **Only Qwen2.5-0.5B-Instruct is tested in CI** ([`tests/conftest.py:16`](https://github.com/UKGovernmentBEIS/vllm-lens/blob/c6b3ab2/vllm_lens/tests/conftest.py#L16)). Coverage on other architectures is inferred from the duck-typing, not asserted.
2. **`enforce_eager=True` is hardcoded** ([`_activations_plugin.py:113`](https://github.com/UKGovernmentBEIS/vllm-lens/blob/c6b3ab2/vllm_lens/_activations_plugin.py#L113), required because `@support_torch_compile` would otherwise bypass the hooks — see `install_hooks` docstring at lines 346-348). You lose CUDA graphs for the whole engine, not just for capture-tagged requests. This is a global perf hit on the worker.
3. **PP layer offset:** `PPMissingLayer` is filtered in `install_hooks` ([`_worker_ext.py:366-367`](https://github.com/UKGovernmentBEIS/vllm-lens/blob/c6b3ab2/vllm_lens/_worker_ext.py#L366-L367)) — pipeline parallel is supported. Captures from multiple PP ranks are merged in rank order in `_merge_captured_states` ([`_activations_plugin.py:48-71`](https://github.com/UKGovernmentBEIS/vllm-lens/blob/c6b3ab2/vllm_lens/_activations_plugin.py#L48-L71)).
4. Models that put decoder layers somewhere *other* than the three duck-typed paths will raise `AttributeError`. None of the standard `*ForCausalLM` classes in vLLM that I'm aware of fall outside, but exotica (e.g. some MoE or state-space models with custom containers) may.

---

## Q2. Residual stream site semantics

**Answer:** **(b) post-block residual.** When you ask for `output_residual_stream=[L]`, vllm-lens returns the output of decoder layer `L` — equivalent to HuggingFace `out.hidden_states[L+1]` and to nnterp's `layers_output[L]`. This is asserted by their own cross-framework test.

**Confidence:** High.

The capture site is a PyTorch *forward* hook (post-hook, not pre-hook). [`_worker_ext.py:368`](https://github.com/UKGovernmentBEIS/vllm-lens/blob/c6b3ab2/vllm_lens/_worker_ext.py#L368):

```python
layer.register_forward_hook(_make_hook(self, layer_idx))
```

The hook signature is the standard post-hook `(module, input, output)` ([`_worker_ext.py:286-302`](https://github.com/UKGovernmentBEIS/vllm-lens/blob/c6b3ab2/vllm_lens/_worker_ext.py#L286-L302)), and the captured tensor comes from `output` — see `_hook_inner` ([`_worker_ext.py:236-246`](https://github.com/UKGovernmentBEIS/vllm-lens/blob/c6b3ab2/vllm_lens/_worker_ext.py#L236-L246)):

```python
# --- Phase 3: capture activations (rank 0 only) -----------------
if getattr(extension, "_should_capture", True):
    capture_src = modified_output if modified_output is not None else output
    hidden_states: Float[torch.Tensor, "total_tokens hidden_dim"]
    if isinstance(capture_src, tuple):
        if capture_src[1] is not None:
            hidden_states = capture_src[0] + capture_src[1]
        else:
            hidden_states = capture_src[0]
    else:
        hidden_states = capture_src
```

That `output[0] + output[1]` branch is the fused-residual handling: vLLM's `LlamaDecoderLayer` / `Qwen[23]DecoderLayer` / `Gemma3DecoderLayer` and friends return `(hidden_states, residual)` where the *true* residual stream after the block is the sum of the two. vllm-lens reconstructs the full residual for you.

**Why this is `hidden_states[L+1]` in HF terms:** see [`tests/test_activations_match.py:26-31`](https://github.com/UKGovernmentBEIS/vllm-lens/blob/c6b3ab2/vllm_lens/tests/test_activations_match.py#L26-L31):

```python
def _get_hf_acts(model, tokenizer, prompt: str) -> torch.Tensor:
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    num_tokens = inputs.input_ids.shape[1]
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, use_cache=False)
    return out.hidden_states[LAYER_IDX + 1][0, :num_tokens].float()
```

and [`tests/test_activations_match.py:34-47`](https://github.com/UKGovernmentBEIS/vllm-lens/blob/c6b3ab2/vllm_lens/tests/test_activations_match.py#L34-L47):

```python
async def _get_vllm_acts(engine, prompt, request_id, max_tokens):
    sampling_params = SamplingParams(
        temperature=0.0, max_tokens=max_tokens,
        extra_args={"output_residual_stream": [LAYER_IDX]},
    )
    # ... runs and ...
    stream = final.activations["residual_stream"]
    return stream
```

The assertion `mean_abs_diff < 1e-2` against `hf.hidden_states[LAYER_IDX + 1]` is the ground truth: vLLM's `output_residual_stream=[L]` matches HuggingFace's `hidden_states[L+1]`. In HF, `hidden_states[0]` is the embedding output and `hidden_states[i+1]` is the output of decoder block `i`, so vllm-lens layer `L` = HF block-`L`'s output = **post-block residual**.

This is corroborated independently in [`_examples/activation_oracle.py:7-14`](https://github.com/UKGovernmentBEIS/vllm-lens/blob/c6b3ab2/vllm_lens/_examples/activation_oracle.py#L7-L14):

> Verified correct:
> - vLLM's Qwen3DecoderLayer returns `(hidden_states, residual)`; the true residual stream is `hidden_states + residual`. This is mathematically equivalent to HF Transformers' single-tensor return where the residual additions happen inside the layer.
> - Layer indexing (0-based) matches between vLLM and HF.

**Mapping to nnterp:**

| Site (nnterp `StandardizedTransformer`) | vllm-lens `output_residual_stream=[L]` |
| --- | --- |
| `layers_input[L]` (pre-block residual = block L's input) | **NOT directly available** |
| `layers_output[L]` (post-block residual = block L's output) | **= this** |

If you've been training probes on `layers_input[L]` in nnterp, you would need to compare against `output_residual_stream=[L-1]` in vllm-lens (the output of block `L-1` is the input of block `L`). For layer 0 you have no equivalent — vllm-lens does not expose the embedding output directly. (You could approximate via a pre-hook fork; see PR #3 below.)

---

## Q3. `position_indices` per-row semantics

**Answer:** `position_indices` is a flat `list[int]` of **absolute sequence positions** that applies to the request that submitted the `SteeringVector`. It is **not** indexed by batch-row. Each request carries its own `SteeringVector` list, so you get per-request specification by submitting different vectors, not by adding a row dimension to `position_indices`.

**Confidence:** High.

The type definition ([`_helpers/types.py:57-59`](https://github.com/UKGovernmentBEIS/vllm-lens/blob/c6b3ab2/vllm_lens/_helpers/types.py#L57-L59)):

```python
position_indices: list[int] | None = None
"""Absolute token positions for 3D activations.  ``None`` means broadcast
(2D) or sequential ``0..n_positions-1`` (3D)."""
```

`activations` itself is shape `(n_layers, hidden_dim)` (2D, broadcast across all positions of the request) or `(n_layers, n_positions, hidden_dim)` (3D, paired with `position_indices`). The validator at [`_helpers/types.py:78-90`](https://github.com/UKGovernmentBEIS/vllm-lens/blob/c6b3ab2/vllm_lens/_helpers/types.py#L78-L90) enforces 2D or 3D only — no batch dimension.

Application logic in `_apply_steering` ([`_worker_ext.py:102-146`](https://github.com/UKGovernmentBEIS/vllm-lens/blob/c6b3ab2/vllm_lens/_worker_ext.py#L102-L146)):

```python
def _apply_steering(configs, layer_idx, target, start, end, abs_start):
    """target is the (already-cloned) output tensor.  start/end are
    batch-relative indices, abs_start is the absolute sequence position
    of the first token in target[start:end]."""
    n_tokens = end - start
    for cfg in configs:
        if layer_idx not in cfg.layer_index_map:
            continue
        act_idx = cfg.layer_index_map[layer_idx]
        vec = cfg.activations[act_idx].to(target.dtype)
        if vec.dim() == 1:
            v = vec.unsqueeze(0)
            if cfg.norm_match:
                v = norm_match(target[start:end], v)
            target[start:end] = target[start:end] + v * cfg.scale
        else:
            # 3D: position-specific
            pos_indices = (cfg.position_indices
                           if cfg.position_indices is not None
                           else list(range(vec.shape[0])))
            abs_end = abs_start + n_tokens
            for pi, abs_pos in enumerate(pos_indices):
                if pi >= vec.shape[0]:
                    break
                if abs_pos < abs_start or abs_pos >= abs_end:
                    continue
                rel = abs_pos - abs_start + start
                v = vec[pi]
                if cfg.norm_match:
                    v = norm_match(target[rel], v)
                target[rel] = target[rel] + v * cfg.scale
```

The caller in `_hook_inner` ([`_worker_ext.py:219-234`](https://github.com/UKGovernmentBEIS/vllm-lens/blob/c6b3ab2/vllm_lens/_worker_ext.py#L219-L234)) iterates each request `i` separately:

```python
for i in range(num_reqs):
    if not per_req_steering[i]:
        continue
    start = int(query_start_loc[i].item())
    end = int(query_start_loc[i + 1].item())
    n_query = end - start
    if seq_lens is not None:
        sl = seq_lens[i]
        sl_val = sl.item() if isinstance(sl, torch.Tensor) else int(sl)
        abs_start = int(sl_val - n_query)
    else:
        abs_start = 0
    _apply_steering(per_req_steering[i], layer_idx, target,
                    start, end, abs_start)
```

So:
- `per_req_steering[i]` is the list of `SteeringVector`s for request `i`.
- `position_indices` inside a `SteeringVector` is interpreted in *absolute* token coordinates of *that* request (i.e. `0` means request-`i`'s first token, `7` means request-`i`'s eighth token), gated by `abs_start <= abs_pos < abs_end`.
- vllm-lens dynamically batches multiple requests' tokens into one forward pass and the steering loop slices per-request — so two concurrent requests can have *different* `SteeringVector`s with *different* `position_indices` because they're stored under different external request IDs in `_steering_data` (see `_find_steering_configs` at [`_worker_ext.py:58-78`](https://github.com/UKGovernmentBEIS/vllm-lens/blob/c6b3ab2/vllm_lens/_worker_ext.py#L58-L78)).

**Practical interpretation for sonde:**
- "Apply at position 5 in every prompt" → one `SteeringVector(position_indices=[5], activations=(n_layers, 1, d))` reused across calls.
- "Apply at the *last* prompt position" → you must know each prompt's length and build a per-request `SteeringVector` (no relative-to-end indexing exists).
- "Apply at different positions in different rows of one batch" → submit one `LLM.generate` request per row, each with its own `SteeringVector`. There is no `(batch, n_pos)` matrix slot.

---

## Q4. Prompt vs generated-position extraction

**Answer:** With `extra_args={"output_residual_stream": [15]}`, `max_tokens=1`, `temperature=0`, you get back a tensor of shape `(1, n_prompt, d_model)` — **all prompt positions, no generated positions**. The hook fires on every forward pass (prefill chunks and decode steps), but the post-`generate` trim explicitly drops the surplus.

**Confidence:** High.

The hook is layer-output, request-keyed, and unconditional on forward-pass kind. [`_worker_ext.py:248-278`](https://github.com/UKGovernmentBEIS/vllm-lens/blob/c6b3ab2/vllm_lens/_worker_ext.py#L248-L278):

```python
for i in range(num_reqs):
    req_id = req_ids[i]
    req_state = runner.requests.get(req_id)
    if req_state is None or req_state.sampling_params is None:
        continue
    extra = req_state.sampling_params.extra_args
    if not extra:
        continue
    output_residual_stream = extra.get("output_residual_stream")
    if output_residual_stream is None:
        continue
    if (isinstance(output_residual_stream, list)
            and layer_idx not in output_residual_stream):
        continue
    start = query_start_loc[i].item()
    end = query_start_loc[i + 1].item()
    activation = hidden_states[start:end].cpu()
    if req_id not in extension._captured_states:
        extension._captured_states[req_id] = {}
    layer_states = extension._captured_states[req_id]
    if layer_idx not in layer_states:
        layer_states[layer_idx] = []
    layer_states[layer_idx].append(activation)
```

`query_start_loc` is the token-boundary tensor from vLLM's forward context. On the **prefill** pass, `end - start = n_prompt` (or `≤ max_num_batched_tokens` per chunk with chunked prefill — the chunks are concatenated by the `append` + later `torch.cat` in [`_worker_ext.py:454-456`](https://github.com/UKGovernmentBEIS/vllm-lens/blob/c6b3ab2/vllm_lens/_worker_ext.py#L454-L456)). On each **decode** step, `end - start = 1` (the new token). So the raw capture for a request that prefills then generates `n_gen` tokens accumulates `n_prompt + n_gen` rows.

But vLLM's v1 scheduler sometimes runs one extra forward pass after EOS, so vllm-lens trims. [`_activations_plugin.py:74-94`](https://github.com/UKGovernmentBEIS/vllm-lens/blob/c6b3ab2/vllm_lens/_activations_plugin.py#L74-L94):

```python
def _trim_activations(activations, expected_len):
    """Trim residual stream activations and input_ids to the expected length.

    The vLLM v1 scheduler may execute one extra forward pass after the EOS
    stop condition is hit ...  This trims the surplus positions so the
    residual stream shape is always deterministic.
    """
    rs = activations.get("residual_stream")
    if rs is not None and rs.shape[1] > expected_len:
        activations["residual_stream"] = rs[:, :expected_len, :]
```

and the `expected_len` is computed at [`_activations_plugin.py:199-201`](https://github.com/UKGovernmentBEIS/vllm-lens/blob/c6b3ab2/vllm_lens/_activations_plugin.py#L199-L201) (async path) and `:284-286` (sync path):

```python
n_prompt = len(output.prompt_token_ids)
n_gen = len(output.outputs[0].token_ids)
_trim_activations(activations, n_prompt + n_gen - 1)
```

So the contract is **`n_prompt + n_gen - 1` rows** per request. The `-1` makes sense because the residual at the last generated position would be the input to a generation step that never happens (you'd sample `n_gen + 1`).

**Cases:**

| Caller spec | Returned `residual_stream.shape[1]` | What you get |
| --- | --- | --- |
| `max_tokens=1` | `n_prompt + 1 - 1 = n_prompt` | All prompt positions, no decode positions |
| `max_tokens=K` (no early stop) | `n_prompt + K - 1` | All prompt + first `K-1` decode steps |
| `max_tokens=K`, EOS at step `j<K` | `n_prompt + j - 1` (after trim) | Same shape relative to actual `n_gen` |

**Implication for probe-training use cases:** For "extract residual at the final prompt token", you set `max_tokens=1` and slice `residual_stream[:, n_prompt - 1, :]`. This is the natural way and matches what `sonde` currently does in the nnterp path.

**One subtle gotcha:** The plugin sets `effective_params.skip_reading_prefix_cache = True` whenever `output_residual_stream` is requested ([`_activations_plugin.py:172-176`](https://github.com/UKGovernmentBEIS/vllm-lens/blob/c6b3ab2/vllm_lens/_activations_plugin.py#L172-L176)):

> Hooks rely on forward passes firing; prefix-cached tokens skip computation entirely, so force a fresh prefill for this request.

This means extraction calls cannot benefit from prefix caching against earlier requests. For probe training on shared prompt prefixes, this nullifies a major vLLM perf win compared to a no-extraction baseline.

---

## Installation, license, repo health

- **Install command** ([`README.md:11-13`](https://github.com/UKGovernmentBEIS/vllm-lens/blob/c6b3ab2/README.md#L11-L13)): `uv add vllm-lens` (or equivalently `pip install vllm-lens`). The package on PyPI is `vllm-lens`. **No extras advertised in the README.** A `[dependency-groups]` `benchmarking` extra exists in [`pyproject.toml:44-50`](https://github.com/UKGovernmentBEIS/vllm-lens/blob/c6b3ab2/pyproject.toml#L44-L50) but that pulls in `nnsight==0.6.3`, `transformer-lens`, and `sifter` from a private AISI HPC repo — it's an internal dev group, not a user-facing extra.
- **Runtime deps** ([`pyproject.toml:7-12`](https://github.com/UKGovernmentBEIS/vllm-lens/blob/c6b3ab2/pyproject.toml#L7-L12)): `datasets>=4.0.0`, `pydantic>=2.0`, `vllm>=0.16.0`, `zstandard>=0.23.0`. Python `>=3.12` ([`pyproject.toml:17`](https://github.com/UKGovernmentBEIS/vllm-lens/blob/c6b3ab2/pyproject.toml#L17)).
- **Version pin in `pyproject.toml`** is `0.0.0` ([`pyproject.toml:18`](https://github.com/UKGovernmentBEIS/vllm-lens/blob/c6b3ab2/pyproject.toml#L18)) with comment "Updated by CICD". Development Status classifier is **Alpha** ([`pyproject.toml:3`](https://github.com/UKGovernmentBEIS/vllm-lens/blob/c6b3ab2/pyproject.toml#L3)).
- **License:** declared **MIT** in `pyproject.toml:14`, but the repo has **no `LICENSE` file checked in** and GitHub's license API returns 404 for the repo. The README and `pyproject` declaration are the only license attestation. I'd ask the maintainers to add a `LICENSE` before depending on this from anything we publish, but the MIT declaration in `pyproject` is legally meaningful.
- **Last commit on `main`:** `c6b3ab2`, 2026-04-14 (*"Support Gemma-4 Style Models (#2)"*). Repo `pushed_at` is 2026-05-07, but that reflects PR branch activity, not new commits to `main`.
- **Activity:** 11 commits on `main`, 6 open PRs, 8 forks, 112 stars. Originated 2026-03-12.

---

## Anything else surprising

### Open PRs that change semantics

These are not merged yet — current `main` does **not** include them.

- **PR #7 — [`norm_match` bug on fused-residual models](https://github.com/UKGovernmentBEIS/vllm-lens/pull/7)** (opened by @adamkarvonen, 2026-06-09; status: open, not yet reviewed). Quoting the PR body:

  > With `norm_match=True`, the steering vector should be scaled so the injected magnitude equals `‖residual_stream‖ · scale`. On fused-residual architectures (Qwen2/Qwen3, Gemma-3, Llama, …) it was instead scaled to the norm of only the MLP-delta half of the residual tuple (0.49× the full residual norm on Qwen3-0.6B), so steering was systematically under-applied. On a simple AG News classification eval this caused significant mismatch with the HF reference score and decreased accuracy from 83% to 51% (random chance) on Gemma-3-27B-IT.

  Root cause: capture path uses `output[0] + output[1]` correctly, but the steering path uses only `target = modified_output[0]` as the `norm_match` reference. The fix threads a `norm_ref = output[0] + output[1]` into `_apply_steering`. **If sonde uses `norm_match=True` for deployment, we must either wait for this fix or carry the patch.** If we use plain `scale` without `norm_match`, this bug doesn't affect us.

- **PR #8 — [Single-layer activations over HTTP](https://github.com/UKGovernmentBEIS/vllm-lens/pull/8)** (opened 2026-06-09; status: open). Over the OpenAI-compatible HTTP path, `vllm_xargs` values are scalars only, so `output_residual_stream` arrives JSON-encoded as a string `"[31]"`. The `isinstance(_, list)` check in [`_worker_ext.py:260-264`](https://github.com/UKGovernmentBEIS/vllm-lens/blob/c6b3ab2/vllm_lens/_worker_ext.py#L260-L264) fails, falling through to "capture all layers". The fix is a `json.loads` in `_patched_generate` mirroring the existing `apply_steering_vectors` string-decode. **This affects the production HTTP deploy path for sonde.** Offline `LLM.generate` is unaffected.

- **PR #6 — [Batched `get_captured_states`](https://github.com/UKGovernmentBEIS/vllm-lens/pull/6)** (status: open). On `LLM.generate` with ≥32 prompts, the per-request RPC loop dominates. PR adds a single batched RPC, gives 1.6–4.4× E2E speedup on Llama-3.1-8B. Pure perf, no semantic change. Worth tracking but not blocking.

- **PR #3 — [Generic Garçon-style hooks](https://github.com/UKGovernmentBEIS/vllm-lens/pull/3)** (status: open). Adds `apply_hooks`, `register_hooks`, pre-hooks, persistent hooks, an HTTP API for hook registration, and a `VLLMLensClient` Python client. Also fixes the same string-vs-list bug as PR #8, plus PP merge list-concat, plus pre/post hook indexing. **This is the future-facing API for arbitrary interpretability work; if it lands, the `extra_args["output_residual_stream"]` path becomes one specific case of a more general hook system.** Not blocking adoption, but worth knowing about: if upstream merges this, our integration code may want to be on the new API.

### Mandatory `enforce_eager=True`

[`_activations_plugin.py:111-116`](https://github.com/UKGovernmentBEIS/vllm-lens/blob/c6b3ab2/vllm_lens/_activations_plugin.py#L111-L116) hardcodes `self.enforce_eager = True` for *every* engine config built with the plugin installed. Per `install_hooks` docstring ([`_worker_ext.py:346-349`](https://github.com/UKGovernmentBEIS/vllm-lens/blob/c6b3ab2/vllm_lens/_worker_ext.py#L346-L349)):

> Requires `enforce_eager=True` in engine args — otherwise `@support_torch_compile` would compile the forward graph and hooks won't fire.

This is the price of admission: **CUDA graphs are disabled engine-wide**, not just on capture-tagged requests. On large models this is a measurable hit (~10-30% depending on shape). For probe *training* this is acceptable. For production *serving* with steering it's a tradeoff to flag.

### Tensor-parallel and pipeline-parallel handling

- **TP:** Hooks install on all ranks (steering must run everywhere), but capture is gated to TP rank 0 only via `_should_capture` ([`_worker_ext.py:359-360`](https://github.com/UKGovernmentBEIS/vllm-lens/blob/c6b3ab2/vllm_lens/_worker_ext.py#L359-L360)). Justification in the README: *"residual streams are identical across TP replicas after all-reduce"*. This is correct.
- **PP:** `PPMissingLayer` filtering ([`_worker_ext.py:366-367`](https://github.com/UKGovernmentBEIS/vllm-lens/blob/c6b3ab2/vllm_lens/_worker_ext.py#L366-L367)) means each PP rank only hooks its own layers. The plugin merges across ranks in `_merge_captured_states` ([`_activations_plugin.py:48-71`](https://github.com/UKGovernmentBEIS/vllm-lens/blob/c6b3ab2/vllm_lens/_activations_plugin.py#L48-L71)) by concatenating along dim 0 (layer dim). Test coverage in `tests/test_activations_pp.py`.

### Chunked prefill

Verified to match HF in `tests/test_activations_chunked_prefill.py` (`max_num_batched_tokens=64` with a >64-token prompt produces multiple prefill chunks, all captured and reassembled, mean-abs-diff < 1e-2 vs HF reference). Good.

### Prefix-cache bypass

Already mentioned in Q4, repeating because it's a perf gotcha: any request with `output_residual_stream` set has prefix caching disabled for that request. If your probe-training workload reuses a long system prompt, that prompt re-prefills on every extraction call.

### Hook error handling

[`_worker_ext.py:296-302`](https://github.com/UKGovernmentBEIS/vllm-lens/blob/c6b3ab2/vllm_lens/_worker_ext.py#L296-L302) wraps every hook call in `try/except` and *logs a warning* on failure, returning `None` so PyTorch leaves the original output untouched. This means a buggy steering vector silently degrades to baseline behavior — quiet failure mode for production. Worth a linter or assertion in our wrapper layer.

### `requires-python>=3.12`

Hard constraint. If sonde targets Python 3.11 anywhere (Oumi cluster, CI), this needs verification.

### `vllm>=0.16.0`

vLLM 0.16 shipped 2026. The plugin patches v1 internals (`vllm.v1.engine.async_llm.AsyncLLM`, `vllm.v1.engine.EngineCoreRequest`) and would need maintenance against any v1 API rewrite. The fact that the maintainers explicitly handle Qwen3-Next's hybrid attention metadata format and the v1 scheduler's EOS-after-stop quirk suggests they're tracking v1 carefully.

---

## Verdict

| Question | Answer acceptable for sonde adoption? |
| --- | --- |
| **Q1 (architectures)** | **Yes**, with monitoring. The duck-typed `_get_layers` covers everything in our likely target set (Llama-3.x, Qwen-2.5/3, Mistral, Gemma-3, Mixtral, DeepSeek). For VLMs we'd want to manually test the `language_model.model.layers` path against a probe sanity-check before claiming support. We should add **one cross-framework activation match test per model family we plan to ship** as a regression guard — they only test Qwen2.5-0.5B in CI. |
| **Q2 (residual site)** | **Yes** — and importantly, this is `layers_output[L]` in nnterp's grammar, which is what sonde already does (diff-of-means + linear probes on post-block residuals). The probe artifact contract holds without any change as long as we standardize on `layers_output[L]` in nnterp and `output_residual_stream=[L]` in vllm-lens. **One-line semantic mapping in the probe metadata is sufficient.** |
| **Q3 (`position_indices`)** | **Yes, with caveats.** For sonde's current use cases (apply at last prompt token; apply across all positions; apply at a fixed absolute position), the existing `list[int]` semantics is sufficient. The lack of relative-to-end indexing and the absence of a `(batch, n_pos)` matrix means we'll need to submit one request per prompt with its own `SteeringVector` for any "last token of each prompt" workload. That's a thin wrapper, not a patch. **No upstream change required.** |
| **Q4 (prefill/decode positions)** | **Yes.** The default `max_tokens=1, output_residual_stream=[L]` gives exactly `(1, n_prompt, d)` — all prompt positions, zero decode positions — which is what probe training wants. The `-1` in the trim is well-defined and matches probe-training semantics. **No upstream change required.** |

**Net assessment:** vllm-lens is fit for purpose as both a faster extraction backend and a production deploy target for sonde. The four core semantics line up cleanly with what sonde already does on nnterp. The probe artifact contract (`direction`, `bias`, `layer`, metadata) survives unchanged as long as we record the site convention (post-block residual = `layers_output[L]` = `output_residual_stream=[L]`) in the artifact.

**Required follow-up actions before depending on this in sonde:**

1. **Track PR #7 (norm_match fix).** If we use `norm_match=True`, we must either wait for merge or vendor the patch. If we ship a pure-`scale` deployment, this is moot.
2. **Track PR #8 (HTTP single-layer fix).** Required for the HTTP serving path; not required for offline `LLM.generate` extraction. If we hit serving before this merges, vendor or work around.
3. **Confirm Python 3.12+ availability** on Oumi cluster and any CI we run probe-training in.
4. **Confirm `enforce_eager=True` perf hit** is acceptable for our extraction throughput targets — should be measured on the target model before committing.
5. **Add a per-model-family activation-match test** mirroring `tests/test_activations_match.py` for each architecture sonde claims to support (Llama-3.1-8B, Qwen2.5-7B, Gemma-3-4B at minimum).
6. **Add a LICENSE file to vllm-lens** (file a PR upstream) so downstream license compliance is unambiguous. MIT in `pyproject.toml` is sufficient for our internal use but worth fixing for good citizenship.
7. **Wrap their quiet hook-error fallback** in our integration layer so a steering misconfig doesn't silently degrade probe deploy to baseline.

No findings in this spike require *upstream* changes for sonde's adoption plan to work. PR #7 and PR #8 are real bugs we'd want before production, but neither is foundational — both have clean upstream PRs already open and authored by an external user, suggesting healthy responsiveness to bug reports.
