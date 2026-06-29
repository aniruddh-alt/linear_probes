"""Generate text responses from a model."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from sonde.core.configs.params.generation_params import GenerationParams
from sonde.core.configs.params.model_params import ModelParams
from sonde.core.configs.params.steering_params import SteeringParams
from sonde.dataset.types import SampleBundle
from sonde.generation.types import GenerationResult

logger = logging.getLogger(__name__)


def _load_vector(
    path: str,
    key: str = "",
    normalize: bool = True,
    device: str = "cpu",
) -> torch.Tensor:
    """Load a steering vector from .pt or .safetensors file."""
    if path.endswith(".safetensors"):
        tensors = load_file(path, device=device)
        vec = tensors[key] if key else next(iter(tensors.values()))
    else:
        vec = torch.load(path, map_location=device, weights_only=True)

    vec = vec.float().squeeze()
    if normalize and vec.norm() > 0:
        vec = vec / vec.norm()
    return vec


def _make_steering_hook(
    vector: torch.Tensor,
    mode: str,
    strength: float,
) -> Callable:
    """Create a forward hook that steers activations."""

    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            h = output
        elif isinstance(output, tuple):
            h = output[0]
        else:
            h = output[0]
        v = vector.to(dtype=h.dtype, device=h.device)
        if mode == "project_subtract":
            # Ablation fraction: strength=1.0 fully removes the component along v
            # (matches sonde.interventions.steering.directional_ablation).
            dot = (h * v).sum(dim=-1, keepdim=True)
            h = h - strength * dot * v
        elif mode == "additive":
            h = h + strength * v
        if isinstance(output, torch.Tensor):
            return h
        if isinstance(output, tuple):
            return (h, *output[1:])
        first_key = next(iter(output.keys()))
        output[first_key] = h
        return output

    return hook


def _resolve_layer_modules(model) -> list:
    """Find the transformer layer modules (Llama-style or GPT-style)."""
    # Llama / Mistral / Qwen style
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    # GPT-2 / GPT-Neo style
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    raise ValueError(
        "Cannot resolve layer modules. Model must have "
        "'model.model.layers' or 'model.transformer.h'."
    )


class ResponseGenerator:
    """Generates text responses from a loaded causal LM."""

    def __init__(
        self,
        model: ModelParams,
        generation: GenerationParams | None = None,
        steering: SteeringParams | None = None,
    ):
        self.model_params = model
        self.generation_params = generation or GenerationParams()
        self.steering_params = steering
        self.tokenizer = AutoTokenizer.from_pretrained(
            model.model_name,
            trust_remote_code=model.trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        model_kwargs = {}
        if model.dtype is not None:
            model_kwargs["torch_dtype"] = getattr(torch, model.dtype)
        if model.load_in_8bit:
            model_kwargs["load_in_8bit"] = True
        if model.load_in_4bit:
            model_kwargs["load_in_4bit"] = True
        self.model = AutoModelForCausalLM.from_pretrained(
            model.model_name,
            trust_remote_code=model.trust_remote_code,
            low_cpu_mem_usage=model.low_cpu_mem_usage,
            **model_kwargs,
        )

        self._steering_vector: torch.Tensor | None = None
        if steering and steering.enabled:
            self._steering_vector = _load_vector(
                path=steering.vector_path,
                key=steering.vector_key,
                normalize=steering.normalize,
                device=str(self.model.device),
            )
            logger.info(
                "Loaded steering vector: shape=%s, mode=%s, layers=%s",
                self._steering_vector.shape,
                steering.mode,
                steering.layers,
            )

    def generate(
        self,
        samples: SampleBundle | Sequence[str],
    ) -> GenerationResult:
        """Generate responses for each prompt."""
        if isinstance(samples, SampleBundle):
            prompts = [samples.prompts[i] for i in range(len(samples.ids))]
            sample_ids = list(samples.ids)
            labels = list(samples.labels)
        else:
            prompts = list(samples)
            sample_ids = [str(i) for i in range(len(prompts))]
            labels: list[int | None] = [None] * len(prompts)

        gen_params = self.generation_params
        responses: list[str] = []

        handles = []
        if self._steering_vector is not None and self.steering_params:
            try:
                layer_modules = _resolve_layer_modules(self.model)
            except ValueError:
                logger.warning("Could not resolve layer modules; skipping steering.")
                layer_modules = []

            for layer_idx in self.steering_params.layers:
                if layer_idx < len(layer_modules):
                    hook_fn = _make_steering_hook(
                        vector=self._steering_vector,
                        mode=self.steering_params.mode,
                        strength=self.steering_params.factor,
                    )
                    handle = layer_modules[layer_idx].register_forward_hook(hook_fn)
                    handles.append(handle)

        try:
            for batch_start in range(0, len(prompts), gen_params.batch_size):
                batch = prompts[batch_start : batch_start + gen_params.batch_size]
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                input_ids = inputs["input_ids"].to(self.model.device)
                attention_mask = inputs["attention_mask"].to(self.model.device)
                input_len = input_ids.shape[1]

                with torch.no_grad():
                    output_ids = self.model.generate(  # pyright: ignore[reportAttributeAccessIssue]
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=gen_params.max_new_tokens,
                        temperature=gen_params.temperature,
                        top_p=gen_params.top_p,
                        do_sample=gen_params.do_sample,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

                new_tokens = output_ids[:, input_len:]
                decoded = self.tokenizer.batch_decode(
                    new_tokens, skip_special_tokens=True
                )
                responses.extend(decoded)
        finally:
            for handle in handles:
                handle.remove()

        return GenerationResult(
            prompts=prompts,
            responses=responses,
            sample_ids=sample_ids,
            labels=labels,
        )
