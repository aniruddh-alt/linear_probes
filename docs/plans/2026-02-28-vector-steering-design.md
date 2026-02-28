# Vector Steering Design

## Goal

Extend `ResponseGenerator` to support inference-time activation steering using vectors from probe weights or arbitrary tensor files. Two modes: additive steering (`h' = h + alpha * w_hat`) and projection subtraction (`h' = h - (h . w_hat) * w_hat`).

## Decisions

- **Both modes** (additive + projection subtraction), configurable per-run
- **Vector source**: arbitrary tensor file (.pt or .safetensors)
- **Architecture**: extend existing `ResponseGenerator` with forward hooks
- **Multi-layer**: same vector applied at multiple layers
- **Approach**: PyTorch `register_forward_hook` on layer modules during `model.generate()`

## New Config: SteeringParams

`core/configs/params/steering_params.py`:

```python
@dataclass
class SteeringParams:
    enabled: bool = False
    vector_path: str = ""          # path to .pt or .safetensors file
    vector_key: str = ""           # key within safetensors file
    layers: list[int] = []         # which layers to steer at
    strength: float = 10.0         # alpha scaling factor (additive mode)
    mode: str = "project_subtract" # "project_subtract" or "additive"
    normalize: bool = True         # normalize vector to unit norm
```

## GenerateConfig Change

Add `steering: SteeringParams` field to `GenerateConfig`.

## ResponseGenerator Changes

### Init

Accept optional `SteeringParams`. If enabled, load and prepare the steering vector (load from file, optionally normalize, move to device).

### Vector Loading

- `.pt` files: `torch.load(path)` -> expects 1D tensor `(hidden_dim,)`
- `.safetensors` files: `safetensors.torch.load_file(path)[vector_key]`

### Hook Creation

```python
def _make_hook(vector, mode, strength):
    def hook(module, input, output):
        h = output[0]  # (batch, seq, hidden)
        if mode == "project_subtract":
            proj = (h @ vector.unsqueeze(-1)) * vector
            h = h - proj
        elif mode == "additive":
            h = h + strength * vector
        return (h, *output[1:])
    return hook
```

### Layer Resolution

Auto-detect layer structure:
- `model.model.layers[i]` for Llama/Qwen/Mistral/Gemma
- `model.transformer.h[i]` for GPT-2/GPT-Neo

### Generate Flow

1. Register hooks on target layers before `model.generate()`
2. Run generation as normal (hooks fire every forward pass, every token)
3. Remove hooks in `finally` block

## Example YAML

```yaml
action: generate
model:
  model_name: meta-llama/Llama-3.1-8B-Instruct
  dtype: bfloat16
generation:
  max_new_tokens: 256
  batch_size: 8
  do_sample: true
  temperature: 0.7
steering:
  enabled: true
  vector_path: artifacts/probe_weights.pt
  layers: [14, 15, 16, 17, 18, 19, 20]
  strength: 15.0
  mode: project_subtract
  normalize: true
io:
  input_path: data/prompts.jsonl
  output_dir: data/steered
```

## Data Flow

```
probe sweep -> probe weights -> save as .pt
                                     |
YAML config + steering params -> ResponseGenerator loads vector
                                     |
register_forward_hook on layers -> model.generate()
                                     |
each forward pass: intervention at hooked layers
                                     |
output: steered responses -> responses.jsonl
```
