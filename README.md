# Mech Interp Toolkit

This repository is a mechanistic interpretability toolkit.

Current focus:
- activation extraction utilities for transformer models
- linear probing workflows on extracted representations

## Probe Training Usage

```python
from torch.utils.data import DataLoader

from dataset import ProbingDataset
from probes import BinaryLinearProbeTrainer, ProbeTrainConfig

dataset = ProbingDataset.from_extraction_result(
    extraction_result,
    activation_key="layers_output:0",
)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

trainer = BinaryLinearProbeTrainer(
    input_dim=dataset[0][0].numel(),
    config=ProbeTrainConfig(epochs=10, learning_rate=1e-2),
)
trainer.fit(loader)
print(trainer.evaluate(loader))
```

Load directly from a saved extraction manifest:

```python
dataset = ProbingDataset.from_extraction_path(
    "artifacts/activations.pt",
    activation_key="layers_output:0",
)
```

If a manifest contains multiple activation streams, `activation_key` is required.

## Activation Extraction Usage

```python
from dataset import ProbingSampleBuilder
from activation import ActivationExtractor

records = [
    {"id": "ex-1", "text": "The capital of France is", "label": 1},
    {"id": "ex-2", "text": "The capital of Japan is", "label": 0},
]

bundle = ProbingSampleBuilder.from_iterable(records).to_samples(text_key="text")

extractor = ActivationExtractor(
    model_name="openai-community/gpt2",
    activations=[
        "layers_output:0-3",     # inclusive range
        "attentions_output:0:6", # slice start:stop
        "mlps_input:*",          # all layers
        "logits",
    ],
)

result = extractor.extract(bundle, token_index=-1)
print(result["sample_ids"])
print(result["labels"])
```

## Activation Selector Formats

Indexed activation kinds:
- `layers_input`
- `layers_output`
- `attentions_input`
- `attentions_output`
- `mlps_input`
- `mlps_output`
- `attention_probabilities` (requires `enable_attention_probs=True`)

Supported selector syntaxes for indexed kinds:
- single layer: `layers_output:5`
- all layers: `layers_output`, `layers_output:*`, `layers_output:all`
- inclusive range: `layers_output:0-4`
- slice: `layers_output:0:5`
- slice with step: `layers_output:0:12:2`

Non-indexed activation kinds:
- `token_embeddings`
- `logits`
- `next_token_probs`
- `input_ids`

Custom module hooks:
- `module_input:layers[3].mlp.down_proj`
- `module_output:layers[3].mlp.gate_proj`
