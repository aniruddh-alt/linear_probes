"""Geometry of Truth Replication"""

import huggingface_hub as hf
import pandas as pd
import torch as t
from dotenv import load_dotenv  # type: ignore[import-untyped]
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()
device = t.device("cuda" if t.cuda.is_available() else "cpu")
dtype = t.bfloat16
hf.login()
DATASET = ["cities", "neg_cities"]
MODEL_NAME = "meta-llama/Llama-2-13b-hf"
PROBE_LAYER = 14
INTERVENE_LAYER = 8

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)  # type: ignore[assignment]
model = AutoModelForCausalLM.from_pretrained(  # type: ignore[assignment]
    MODEL_NAME,
    dtype=dtype,
    device_map="auto",
)
NUM_LAYERS = len(model.model.layers)
D_MODEL = model.config.hidden_size
tokenizer.pad_token = tokenizer.eos_token  # type: ignore[attr-defined]
tokenizer.padding_side = "right"  # type: ignore[attr-defined]


def load_dataset(dataset_name: str):
    dataset = pd.read_csv(f"experiments/geometry_of_truth/{dataset_name}.csv")
    return dataset[["statement", "label"]]


datasets = {}
for name in DATASET:
    datasets[name] = load_dataset(name)


def extract_activations(
    statements: list[str],
    model: AutoModelForCausalLM,  # type: ignore[type-arg]
    tokenizer: AutoTokenizer,  # type: ignore[type-arg]
    layers: list[int],
    batch_size: int = 25,
) -> dict[int, Tensor]:
    """
    Extract last-token hidden state activations from specified layers for a list of statements.

    Args:
        statements: List of text statements to process.
        model: A HuggingFace causal language model.
        tokenizer: The corresponding tokenizer.
        layers: List of layer indices (0-indexed) to extract activations from.
        batch_size: Number of statements to process at once.

    Returns:
        Dictionary mapping layer index to tensor of activations, shape [n_statements, d_model].
    """
    tokenizer.padding_side = "left"  # type: ignore[attr-defined]
    if tokenizer.pad_token is None:  # type: ignore[attr-defined]
        tokenizer.pad_token = tokenizer.eos_token  # type: ignore[attr-defined]

    all_activations: dict[int, list[Tensor]] = {layer: [] for layer in layers}
    for i in range(0, len(statements), batch_size):
        batch = statements[i : i + batch_size]
        batch_inputs = tokenizer(batch, return_tensors="pt", padding=True)  # type: ignore[operator]
        batch_activations = model(**batch_inputs, output_hidden_states=True)  # type: ignore[operator]
        for layer in layers:
            activations = batch_activations.hidden_states[layer + 1].detach()
            last_token = activations[:, -1, :]
            all_activations[layer].append(last_token)

    return {layer: t.cat(all_activations[layer]) for layer in layers}
