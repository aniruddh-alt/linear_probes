"""Utilities to extract activations from a transformer model."""

from __future__ import annotations

import re
from dataclasses import fields
from pathlib import Path
from typing import Any, Sequence, cast

import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader, Dataset

from activation.types import ExtractionResult, LayerSpec, ModelMetadata
from configs import ActivationConfig
from dataset.samples import SampleBundle

try:
    from nnterp import StandardizedTransformer
except ModuleNotFoundError:  # pragma: no cover - exercised when optional dep missing.
    StandardizedTransformer = None  # type: ignore[assignment]


class SequenceDataset(Dataset[str]):
    """Dataset wrapper over an existing in-memory string sequence."""

    def __init__(self, samples: Sequence[str]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> str:
        return self.samples[index]


class ActivationExtractor:
    """Thin nnsight/nnterp wrapper for extracting model activations."""

    _KIND_PATTERN = re.compile(r"^[a-z_]+$")
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

    # ========== Public API ==========

    _BOOL_FLAGS = frozenset(
        {"enable_attention_probs", "trust_remote_code", "load_in_8bit", "load_in_4bit"}
    )

    def __init__(self, config: ActivationConfig):
        transformer_cls = StandardizedTransformer
        if transformer_cls is None:
            raise ModuleNotFoundError(
                "ActivationExtractor requires optional dependency 'nnterp'. "
                "Install project dependencies before constructing the extractor."
            )
        self.config = config
        mc = config.model_config
        self.model_name = mc.model_name
        model_kwargs: dict[str, Any] = {
            f.name: getattr(mc, f.name)
            for f in fields(mc)
            if f.name not in ("model_name", "additional_kwargs")
            and getattr(mc, f.name) is not None
            and (f.name not in self._BOOL_FLAGS or getattr(mc, f.name))
        }
        model_kwargs.update(mc.additional_kwargs)
        self.model = transformer_cls(mc.model_name, **model_kwargs)
        self.batch_size = config.batch_size
        self.default_activations = (
            list(config.activations)
            if config.activations
            else [f"layers_output:{self.model.num_layers - 1}"]
        )

    @property
    def info(self) -> ModelMetadata:
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
        resolved_token_index = self.config.token_index
        resolved_remote = self.config.remote
        resolved_to_cpu = self.config.to_cpu
        loader = self._as_dataloader(prompts_source, batch_size=self.batch_size)
        outputs_chunks: dict[str, list[torch.Tensor]] = {name: [] for name in requested}

        for batch in loader:
            prompts = self._normalize_batch(batch)
            saved: dict[str, Any] = {}
            with self.model.trace(prompts, remote=resolved_remote):
                for name, spec in zip(requested, parsed_specs):
                    activation = self._resolve_activation(spec)
                    activation = self._select_token_position(
                        activation,
                        resolved_token_index,
                        kind=spec.kind,
                        allow_2d=spec.kind == "input_ids",
                    )
                    if resolved_to_cpu:
                        activation = activation.cpu()
                    saved[name] = activation.save()
            for name in requested:
                outputs_chunks[name].append(
                    self._normalize_saved_tensor(saved[name], batch_size=len(prompts))
                )

        outputs: dict[str, torch.Tensor] = {}
        for name in requested:
            chunks = outputs_chunks[name]
            if not chunks:
                outputs[name] = torch.empty((0,))
                continue
            try:
                outputs[name] = torch.cat(chunks, dim=0)
            except RuntimeError as exc:
                shapes = ", ".join(str(tuple(c.shape)) for c in chunks)
                raise ValueError(
                    f"Cannot concatenate activation chunks for '{name}'. "
                    f"Encountered shapes: {shapes}."
                ) from exc

        if requested:
            total_items = (
                int(outputs[requested[0]].shape[0])
                if outputs[requested[0]].ndim > 0
                else 1
            )
            for name in requested[1:]:
                key_items = int(outputs[name].shape[0]) if outputs[name].ndim > 0 else 1
                if key_items != total_items:
                    raise ValueError(
                        "Activation outputs have inconsistent sample counts across keys: "
                        f"'{requested[0]}' has {total_items} rows but '{name}' has {key_items}."
                    )
            if not sample_ids:
                sample_ids = [str(i) for i in range(total_items)]
                labels: list[int | None] = [None] * total_items
            if len(sample_ids) != total_items:
                raise ValueError(
                    "Number of extracted activations does not match resolved metadata "
                    f"length: {total_items} vs {len(sample_ids)}."
                )
            if len(labels) != total_items:
                raise ValueError(
                    "Number of labels does not match extracted activation rows: "
                    f"{len(labels)} vs {total_items}."
                )

        result: ExtractionResult = {
            "model": self.info,
            "requested": list(requested),
            "activations": outputs,
            "sample_ids": sample_ids,
            "labels": labels,
            "storage": {"mode": "in_memory"},
        }
        result["storage"] = self._persist_result(
            result=result,
            save_path=Path(self.config.save_path),
        )
        return result

    @classmethod
    def supported_activation_kinds(cls) -> list[str]:
        return sorted(cls._INDEXED_KINDS | cls._NON_INDEXED_KINDS | cls._PATH_KINDS)

    # ========== Activation String Parsing ==========

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
            kind, value_text = self._split_kind_value(text)

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

    def _parse_layer_spec(self, raw: str) -> LayerSpec:
        kind, value_text = self._split_kind_value(raw)
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

    @classmethod
    def _split_kind_value(cls, raw: str) -> tuple[str, str | None]:
        text = raw.strip()
        if ":" in text:
            kind, value_text = text.split(":", 1)
        elif text.endswith("]") and "[" in text:
            kind, value_text = text[:-1].split("[", 1)
        else:
            kind, value_text = text, None

        if not cls._KIND_PATTERN.fullmatch(kind):
            raise ValueError(
                f"Invalid activation spec '{raw}'. Expected '<kind>' or '<kind>:<value>'."
            )
        if value_text is not None:
            value_text = value_text.strip()
            if not value_text:
                raise ValueError(
                    f"Invalid activation spec '{raw}'. Value component cannot be empty."
                )
        return kind, value_text

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

    def _validate_layer_index(self, index: int) -> None:
        lower_bound = -self.model.num_layers
        upper_bound = self.model.num_layers - 1
        if not (lower_bound <= index <= upper_bound):
            raise IndexError(
                f"Layer index {index} out of range for model with {self.model.num_layers} layers."
            )

    # ========== Activation Resolution ==========

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

    # ========== Data Pipeline Helpers ==========

    @staticmethod
    def _resolve_samples_metadata(
        samples: Sequence[str] | Dataset[str] | DataLoader[str] | SampleBundle,
    ) -> tuple[
        Sequence[str] | Dataset[str] | DataLoader[str], list[str], list[int | None]
    ]:
        if isinstance(samples, SampleBundle):
            return samples.prompts, list(samples.ids), list(samples.labels)
        if isinstance(samples, DataLoader):
            return samples, [], []
        if isinstance(samples, Dataset):
            if not hasattr(samples, "__len__"):
                raise ValueError(
                    "Dataset does not expose __len__, cannot build stable sample_ids."
                )
            n = len(cast(Sequence[str], samples))
            return (
                samples,
                [str(i) for i in range(n)],
                cast(list[int | None], [None] * n),
            )
        n = len(samples)
        return samples, [str(i) for i in range(n)], cast(list[int | None], [None] * n)

    @staticmethod
    def _as_dataloader(
        samples: Sequence[str] | Dataset[str] | DataLoader[str], batch_size: int
    ) -> DataLoader[str]:
        if isinstance(samples, DataLoader):
            return samples
        if isinstance(samples, Dataset):
            return DataLoader(samples, batch_size=batch_size, shuffle=False)
        return DataLoader(
            SequenceDataset(samples), batch_size=batch_size, shuffle=False
        )

    @staticmethod
    def _normalize_batch(batch: Any) -> list[str]:
        if isinstance(batch, str):
            return [batch]
        if isinstance(batch, (list, tuple)):
            if not all(isinstance(item, str) for item in batch):
                raise TypeError("All prompts in a batch must be strings.")
            return list(batch)
        raise TypeError("Each batch must be a string or a list/tuple of strings.")

    # ========== Tensor Operations ==========

    @staticmethod
    def _select_token_position(
        activation,
        token_index: int | None,
        *,
        kind: str,
        allow_2d: bool = False,
    ):
        # Typical hidden states are [batch, seq, hidden].
        if token_index is None:
            return activation
        if not hasattr(activation, "ndim"):
            return activation
        if activation.ndim == 3:
            return activation[:, token_index]
        if activation.ndim == 4:
            if kind == "attention_probabilities":
                # StandardizedTransformer attention probs are [batch, heads, query, key].
                return activation[:, :, token_index, :]
            raise ValueError(
                f"token_index was set but activation '{kind}' has rank-4 shape "
                f"{tuple(activation.shape)} with no unambiguous sequence axis. "
                "Set token_index=None or choose an activation with a known token axis."
            )
        if allow_2d and hasattr(activation, "ndim") and activation.ndim == 2:
            return activation[:, token_index]
        return activation

    @staticmethod
    def _normalize_saved_tensor(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
        tensor = tensor.detach()
        if tensor.ndim == 0:
            return tensor.reshape(1)
        if tensor.shape[0] == batch_size:
            return tensor
        if batch_size == 1:
            return tensor.unsqueeze(0)
        raise ValueError(
            "Activation tensor does not expose the expected batch dimension. "
            f"Expected first dimension size {batch_size}, got shape {tuple(tensor.shape)}."
        )

    def _persist_result(
        self,
        *,
        result: ExtractionResult,
        save_path: Path,
    ) -> dict[str, Any]:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path, safetensors_path = self._resolve_storage_paths(save_path)
        tensors = {
            key: tensor.contiguous() for key, tensor in result["activations"].items()
        }
        save_file(tensors, str(safetensors_path))
        storage: dict[str, Any] = {
            "mode": "safetensors",
            "manifest_path": str(manifest_path),
            "safetensors_path": str(safetensors_path),
        }
        manifest: ExtractionResult = {
            "model": result["model"],
            "requested": result["requested"],
            "activations": {},
            "sample_ids": result["sample_ids"],
            "labels": result["labels"],
            "storage": storage,
        }
        torch.save(manifest, manifest_path)
        return cast(dict[str, Any], storage)

    @staticmethod
    def _resolve_storage_paths(save_path: Path) -> tuple[Path, Path]:
        base_path = (
            save_path.with_suffix("") if save_path.suffix == ".pt" else save_path
        )
        manifest_path = base_path.parent / f"{base_path.name}_manifest.pt"
        safetensors_path = base_path.with_suffix(".safetensors")
        return manifest_path, safetensors_path
