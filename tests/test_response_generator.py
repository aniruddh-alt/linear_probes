"""Tests for ResponseGenerator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.configs.params.generation_params import GenerationParams
from core.configs.params.model_params import ModelParams
from dataset.samples import StringDataset
from dataset.types import SampleBundle
from generation.response_generator import ResponseGenerator
from generation.types import GenerationResult


class TestResponseGenerator:
    @patch("generation.response_generator.AutoModelForCausalLM")
    @patch("generation.response_generator.AutoTokenizer")
    def test_generate_from_strings(self, mock_tok_cls, mock_model_cls):
        mock_tok = MagicMock()
        mock_tok.pad_token_id = 0
        mock_tok.pad_token = "<pad>"
        mock_tok.eos_token_id = 1
        mock_tok.return_value = {"input_ids": MagicMock(to=MagicMock(return_value=MagicMock())), "attention_mask": MagicMock(to=MagicMock(return_value=MagicMock()))}
        mock_tok.batch_decode.return_value = ["response 0", "response 1"]
        mock_tok_cls.from_pretrained.return_value = mock_tok
        mock_model_obj = MagicMock()
        mock_model_obj.generate.return_value = MagicMock()
        mock_model_obj.device = "cpu"
        mock_model_cls.from_pretrained.return_value = mock_model_obj

        gen = ResponseGenerator(
            model=ModelParams(model_name="test-model"),
            generation=GenerationParams(max_new_tokens=32, batch_size=2),
        )
        result = gen.generate(["prompt 0", "prompt 1"])

        assert isinstance(result, GenerationResult)
        assert result.prompts == ["prompt 0", "prompt 1"]
        assert len(result.responses) == 2
        assert result.labels == [None, None]

    @patch("generation.response_generator.AutoModelForCausalLM")
    @patch("generation.response_generator.AutoTokenizer")
    def test_generate_from_sample_bundle(self, mock_tok_cls, mock_model_cls):
        mock_tok = MagicMock()
        mock_tok.pad_token_id = 0
        mock_tok.pad_token = "<pad>"
        mock_tok.eos_token_id = 1
        mock_tok.return_value = {"input_ids": MagicMock(to=MagicMock(return_value=MagicMock())), "attention_mask": MagicMock(to=MagicMock(return_value=MagicMock()))}
        mock_tok.batch_decode.return_value = ["resp"]
        mock_tok_cls.from_pretrained.return_value = mock_tok
        mock_model_obj = MagicMock()
        mock_model_obj.generate.return_value = MagicMock()
        mock_model_obj.device = "cpu"
        mock_model_cls.from_pretrained.return_value = mock_model_obj

        bundle = SampleBundle(
            prompts=StringDataset(["hello"]),
            labels=[1],
            ids=["s0"],
        )
        gen = ResponseGenerator(
            model=ModelParams(model_name="test-model"),
            generation=GenerationParams(batch_size=1),
        )
        result = gen.generate(bundle)

        assert isinstance(result, GenerationResult)
        assert result.prompts == ["hello"]
        assert result.labels == [1]
        assert result.sample_ids == ["s0"]
