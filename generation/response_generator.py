"""Generate text responses from a model."""

from __future__ import annotations

from typing import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from core.configs.params.generation_params import GenerationParams
from core.configs.params.model_params import ModelParams
from dataset.types import SampleBundle
from generation.types import GenerationResult


class ResponseGenerator:
    """Generates text responses from a loaded causal LM."""

    def __init__(
        self,
        model: ModelParams,
        generation: GenerationParams | None = None,
    ):
        self.model_params = model
        self.generation_params = generation or GenerationParams()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model.model_name,
            trust_remote_code=model.trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
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

    def generate(
        self,
        samples: SampleBundle | Sequence[str],
    ) -> GenerationResult:
        """Generate responses for each prompt.

        Args:
            samples: Prompts as a SampleBundle or list of strings.

        Returns:
            GenerationResult with prompts, responses, sample_ids, labels.
        """
        if isinstance(samples, SampleBundle):
            prompts = [samples.prompts[i] for i in range(len(samples.prompts))]
            sample_ids = list(samples.ids)
            labels = list(samples.labels)
        else:
            prompts = list(samples)
            sample_ids = [str(i) for i in range(len(prompts))]
            labels = [None] * len(prompts)

        gen_params = self.generation_params
        responses: list[str] = []

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
                output_ids = self.model.generate(
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

        return GenerationResult(
            prompts=prompts,
            responses=responses,
            sample_ids=sample_ids,
            labels=labels,
        )
