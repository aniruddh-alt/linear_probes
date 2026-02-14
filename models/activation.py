"""Utilities to extract activations from a transformer model."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, cast

import torch
from nnterp import StandardizedTransformer
from torch.utils.data import DataLoader, Dataset
from typing_extensions import TypedDict

from dataset.samples import SampleBundle, StringDataset


class BaseModel(TypedDict):
    name: str
    num_layers: int
    hidden_size: int
    num_heads: int
    vocab_size: int


@dataclass(frozen=True)
class LayerSpec:
    kind: str
    value: int | str | None = None


class ExtractionResult(TypedDict):
    model: BaseModel
    requested: list[str]
    activations: dict[str, list[torch.Tensor]]
    sample_ids: list[str]
    labels: list[int | None]


class ActivationExtractor:
    """Thin nnsight/nnterp wrapper for extracting model activations."""

    _LAYER_PATTERN = re.compile(r"^(?P<kind>[a-z_]+)(?:(?:\[|:)(?P<value>[^\]]+)\]?)?$")
    _PATH_TOKEN_PATTERN = re.compile(r"^(?P<name>[A-Za-z_]\w*)(?:\[(?P<idx>-?\d+)\])?$")
    _INDEXED_KINDS = {
        "layers_input",
        "layers_output",
        "attentions_input",
        "attentions_output",
        "mlps_input",
        "mlps_output",
        "attention_probabilities",
    }
    _NON_INDEXED_KINDS = {"token_embeddings", "logits", "next_token_probs", "input_ids"}
    _PATH_KINDS = {"module_input", "module_output"}
    _ALL_LAYER_TOKENS = {"*", "all"}
    _INT_PATTERN = re.compile(r"^-?\d+$")
    _INCLUSIVE_RANGE_PATTERN = re.compile(r"^(?P<start>-?\d+)-(?P<end>-?\d+)$")

    def __init__(
        self,
        model_name: str,
        activations: list[str] | None = None,
        batch_size: int = 8,
        **model_kwargs: Any,
    ):
        self.model_name = model_name
        self.model = StandardizedTransformer(model_name, **model_kwargs)
        self.batch_size = batch_size
        self.default_activations = activations or [
            f"layers_output:{self.model.num_layers - 1}"
        ]

    @property
    def info(self) -> BaseModel:
        return {
            "name": self.model_name,
            "num_layers": int(self.model.num_layers),
            "hidden_size": int(self.model.hidden_size),
            "num_heads": int(self.model.num_heads),
            "vocab_size": int(self.model.vocab_size),
        }

    def extract(
        self,
        samples: Sequence[str] | Dataset[str] | DataLoader[str] | SampleBundle,
        *,
        activations: list[str] | None = None,
        layers: list[int] | None = None,
        batch_size: int | None = None,
        token_index: int | None = -1,
        remote: bool = False,
        to_cpu: bool = True,
        save_path: str | Path | None = None,
    ) -> ExtractionResult:
        """Extract activations from string prompts in batches.

        `activations` accepts entries such as:
        - layers_output:0
        - layers_input:5
        - mlps_input:10
        - mlps_output:10
        - attentions_input:3
        - attentions_output:3
        - attention_probabilities:3
        - token_embeddings
        - logits
        - next_token_probs
        - module_output:layers[4].mlp.gate_proj
        - module_input:layers[4].mlp.down_proj
        """
        requested = self._resolve_requested_activations(
            activations=activations, layers=layers
        )
        parsed_specs = [self._parse_layer_spec(spec) for spec in requested]
        for spec in parsed_specs:
            self._validate_spec(spec)

        prompts_source, sample_ids, labels = self._resolve_samples_metadata(samples)
        loader = self._as_dataloader(
            prompts_source, batch_size=batch_size or self.batch_size
        )
        outputs: dict[str, list[torch.Tensor]] = {name: [] for name in requested}

        for batch in loader:
            prompts = self._normalize_batch(batch)
            with self.model.trace(prompts, remote=remote):
                for name, spec in zip(requested, parsed_specs):
                    activation = self._resolve_activation(spec)
                    activation = self._select_token_position(activation, token_index)
                    if to_cpu:
                        activation = activation.cpu()
                    tensor = activation.save()
                    self._extend_outputs(outputs[name], tensor, len(prompts))

        result: ExtractionResult = {
            "model": self.info,
            "requested": list(requested),
            "activations": {name: outputs[name] for name in requested},
            "sample_ids": sample_ids,
            "labels": labels,
        }
        if save_path is not None:
            torch.save(result, Path(save_path))
        return result

    @classmethod
    def supported_activation_kinds(cls) -> list[str]:
        return sorted(cls._INDEXED_KINDS | cls._NON_INDEXED_KINDS | cls._PATH_KINDS)

    def _resolve_requested_activations(
        self, activations: list[str] | None, layers: list[int] | None
    ) -> list[str]:
        if activations is not None and layers is not None:
            raise ValueError("Pass either `activations` or `layers`, not both.")
        if activations is not None:
            return self._expand_requested_activations(activations)
        if layers is not None:
            return [f"layers_output:{layer}" for layer in layers]
        return self._expand_requested_activations(self.default_activations)

    def _expand_requested_activations(self, activations: Sequence[str]) -> list[str]:
        expanded: list[str] = []
        for raw in activations:
            text = raw.strip()
            match = self._LAYER_PATTERN.match(text)
            if match is None:
                raise ValueError(
                    f"Invalid activation spec '{raw}'. Expected '<kind>' or '<kind>:<value>'."
                )
            kind = match.group("kind")
            value_text = match.group("value")

            if kind in self._INDEXED_KINDS:
                expanded.extend(self._expand_indexed_activation(kind, value_text))
            else:
                expanded.append(text)
        return expanded

    def _expand_indexed_activation(
        self, kind: str, value_text: str | None
    ) -> list[str]:
        if value_text is None or value_text in self._ALL_LAYER_TOKENS:
            return [f"{kind}:{layer_idx}" for layer_idx in range(self.model.num_layers)]

        if self._INT_PATTERN.fullmatch(value_text):
            return [f"{kind}:{int(value_text)}"]

        range_match = self._INCLUSIVE_RANGE_PATTERN.fullmatch(value_text)
        if range_match is not None:
            start = int(range_match.group("start"))
            end = int(range_match.group("end"))
            step = 1 if end >= start else -1
            indices = list(range(start, end + step, step))
            return [f"{kind}:{idx}" for idx in indices]

        if ":" in value_text:
            indices = self._expand_slice_indices(value_text)
            return [f"{kind}:{idx}" for idx in indices]

        raise ValueError(
            f"Unsupported layer selector '{value_text}' for '{kind}'. "
            "Use an integer (e.g. 5), inclusive range (e.g. 0-4), "
            "slice (e.g. 0:5 or 0:10:2), or all/*."
        )

    def _expand_slice_indices(self, value_text: str) -> list[int]:
        parts = value_text.split(":")
        if len(parts) not in (2, 3):
            raise ValueError(
                f"Invalid slice selector '{value_text}'. Use start:stop[:step]."
            )

        def parse_or_none(text: str) -> int | None:
            if text == "":
                return None
            if not self._INT_PATTERN.fullmatch(text):
                raise ValueError(
                    f"Invalid slice selector '{value_text}'. "
                    "Slice values must be integers."
                )
            return int(text)

        start = parse_or_none(parts[0])
        stop = parse_or_none(parts[1])
        step = parse_or_none(parts[2]) if len(parts) == 3 else None
        if step == 0:
            raise ValueError("Slice step cannot be 0.")

        indices = list(range(self.model.num_layers))[slice(start, stop, step)]
        if not indices:
            raise ValueError(
                f"Slice selector '{value_text}' produced no layers for model with "
                f"{self.model.num_layers} layers."
            )
        return indices

    def _validate_layer_index(self, index: int) -> None:
        lower_bound = -self.model.num_layers
        upper_bound = self.model.num_layers - 1
        if not (lower_bound <= index <= upper_bound):
            raise IndexError(
                f"Layer index {index} out of range for model with {self.model.num_layers} layers."
            )

    @staticmethod
    def _select_token_position(activation, token_index: int | None):
        # Typical hidden states are [batch, seq, hidden].
        if token_index is None:
            return activation
        if hasattr(activation, "ndim") and activation.ndim >= 3:
            return activation[:, token_index]
        return activation

    @staticmethod
    def _as_dataloader(
        samples: Sequence[str] | Dataset[str] | DataLoader[str], batch_size: int
    ) -> DataLoader[str]:
        if isinstance(samples, DataLoader):
            return samples
        if isinstance(samples, Dataset):
            return DataLoader(samples, batch_size=batch_size, shuffle=False)
        return DataLoader(
            StringDataset(list(samples)), batch_size=batch_size, shuffle=False
        )

    @staticmethod
    def _resolve_samples_metadata(
        samples: Sequence[str] | Dataset[str] | DataLoader[str] | SampleBundle,
    ) -> tuple[
        Sequence[str] | Dataset[str] | DataLoader[str], list[str], list[int | None]
    ]:
        if isinstance(samples, SampleBundle):
            return samples.prompts, list(samples.ids), list(samples.labels)
        if isinstance(samples, DataLoader):
            raise ValueError(
                "When passing a DataLoader directly, sample_ids/labels are unavailable. "
                "Pass a SampleBundle or Sequence/Dataset instead."
            )
        if isinstance(samples, Dataset):
            if not hasattr(samples, "__len__"):
                raise ValueError(
                    "Dataset does not expose __len__, cannot build stable sample_ids."
                )
            n = len(cast(Sequence[str], samples))
            return samples, [str(i) for i in range(n)], [None] * n
        n = len(samples)
        return samples, [str(i) for i in range(n)], [None] * n

    @staticmethod
    def _normalize_batch(batch: Any) -> list[str]:
        if isinstance(batch, str):
            return [batch]
        if isinstance(batch, (list, tuple)):
            if not all(isinstance(item, str) for item in batch):
                raise TypeError("All prompts in a batch must be strings.")
            return list(batch)
        raise TypeError("Each batch must be a string or a list/tuple of strings.")

    @staticmethod
    def _extend_outputs(
        store: list[torch.Tensor], tensor: torch.Tensor, batch_size: int
    ) -> None:
        if tensor.ndim > 0 and tensor.shape[0] == batch_size:
            for item in tensor.unbind(dim=0):
                store.append(item)
        else:
            store.append(tensor)

    def _parse_layer_spec(self, raw: str) -> LayerSpec:
        match = self._LAYER_PATTERN.match(raw.strip())
        if match is None:
            raise ValueError(
                f"Invalid activation spec '{raw}'. Expected '<kind>' or '<kind>:<value>'."
            )
        kind = match.group("kind")
        value_text = match.group("value")
        if kind in self._INDEXED_KINDS:
            if value_text is None:
                raise ValueError(f"Activation '{raw}' requires an integer index.")
            try:
                value = int(value_text)
            except ValueError as exc:
                raise ValueError(
                    f"Activation '{raw}' has non-integer layer index."
                ) from exc
            return LayerSpec(kind=kind, value=value)
        if kind in self._PATH_KINDS:
            if value_text is None:
                raise ValueError(f"Activation '{raw}' requires a module path.")
            return LayerSpec(kind=kind, value=value_text)
        if kind in self._NON_INDEXED_KINDS:
            if value_text is not None:
                raise ValueError(f"Activation '{raw}' must not include an index/path.")
            return LayerSpec(kind=kind, value=None)

        allowed = ", ".join(self.supported_activation_kinds())
        raise ValueError(
            f"Unknown activation kind '{kind}'. Supported kinds: {allowed}."
        )

    def _validate_spec(self, spec: LayerSpec) -> None:
        if spec.kind in self._INDEXED_KINDS:
            assert isinstance(spec.value, int)
            self._validate_layer_index(spec.value)
            if (
                spec.kind == "attention_probabilities"
                and not self.model.attn_probs_available
            ):
                raise ValueError(
                    "attention_probabilities requested but not enabled. "
                    "Initialize with `enable_attention_probs=True`."
                )
            return
        if spec.kind in self._PATH_KINDS:
            assert isinstance(spec.value, str)
            self._resolve_module_path(spec.value)

    def _resolve_activation(self, spec: LayerSpec):
        if spec.kind == "token_embeddings":
            return self.model.token_embeddings
        if spec.kind == "logits":
            return self.model.logits
        if spec.kind == "next_token_probs":
            return self.model.next_token_probs
        if spec.kind == "input_ids":
            return self.model.input_ids

        if spec.kind == "module_input":
            module = self._resolve_module_path(str(spec.value))
            return module.input
        if spec.kind == "module_output":
            module = self._resolve_module_path(str(spec.value))
            return module.output

        if not isinstance(spec.value, int):
            raise ValueError(
                f"Activation kind '{spec.kind}' requires an integer index, got {spec.value!r}."
            )
        layer = spec.value
        if spec.kind == "layers_input":
            return self.model.layers_input[layer]
        if spec.kind == "layers_output":
            return self.model.layers_output[layer]
        if spec.kind == "attentions_input":
            return self.model.attentions_input[layer]
        if spec.kind == "attentions_output":
            return self.model.attentions_output[layer]
        if spec.kind == "mlps_input":
            return self.model.mlps_input[layer]
        if spec.kind == "mlps_output":
            return self.model.mlps_output[layer]
        if spec.kind == "attention_probabilities":
            return self.model.attention_probabilities[layer]
        raise ValueError(f"Unsupported activation kind: {spec.kind}")

    def _resolve_module_path(self, path: str):
        current = self.model
        for part in path.split("."):
            token = part.strip()
            match = self._PATH_TOKEN_PATTERN.match(token)
            if match is None:
                raise ValueError(
                    f"Invalid module path token '{token}' in '{path}'. "
                    "Use tokens like 'layers[0]' or 'mlp'."
                )
            name = match.group("name")
            idx_text = match.group("idx")
            if not hasattr(current, name):
                raise ValueError(
                    f"Module path '{path}' is invalid: missing attribute '{name}'."
                )
            current = getattr(current, name)
            if idx_text is not None:
                current = current[int(idx_text)]
        return current
