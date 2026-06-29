"""Tests for ResponseGenerator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch
from safetensors.torch import save_file

from sonde.core.configs.params.generation_params import GenerationParams
from sonde.core.configs.params.model_params import ModelParams
from sonde.core.configs.params.steering_params import SteeringParams
from sonde.dataset.samples import StringDataset
from sonde.dataset.types import SampleBundle
from sonde.generation.response_generator import ResponseGenerator, _make_steering_hook
from sonde.generation.types import GenerationResult


class TestResponseGenerator:
    @patch("sonde.generation.response_generator.AutoModelForCausalLM")
    @patch("sonde.generation.response_generator.AutoTokenizer")
    def test_generate_from_strings(self, mock_tok_cls, mock_model_cls):
        mock_tok = MagicMock()
        mock_tok.pad_token_id = 0
        mock_tok.pad_token = "<pad>"
        mock_tok.eos_token_id = 1
        mock_tok.return_value = {
            "input_ids": MagicMock(to=MagicMock(return_value=MagicMock())),
            "attention_mask": MagicMock(to=MagicMock(return_value=MagicMock())),
        }
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

    @patch("sonde.generation.response_generator.AutoModelForCausalLM")
    @patch("sonde.generation.response_generator.AutoTokenizer")
    def test_generate_from_sample_bundle(self, mock_tok_cls, mock_model_cls):
        mock_tok = MagicMock()
        mock_tok.pad_token_id = 0
        mock_tok.pad_token = "<pad>"
        mock_tok.eos_token_id = 1
        mock_tok.return_value = {
            "input_ids": MagicMock(to=MagicMock(return_value=MagicMock())),
            "attention_mask": MagicMock(to=MagicMock(return_value=MagicMock())),
        }
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


class TestSteeringVectorLoading:
    @patch("sonde.generation.response_generator.AutoModelForCausalLM")
    @patch("sonde.generation.response_generator.AutoTokenizer")
    def test_load_pt_vector(self, mock_tok_cls, mock_model_cls, tmp_path):
        mock_tok = MagicMock()
        mock_tok.pad_token_id = 0
        mock_tok_cls.from_pretrained.return_value = mock_tok
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model_cls.from_pretrained.return_value = mock_model

        vec = torch.randn(64)
        vec_path = tmp_path / "v.pt"
        torch.save(vec, vec_path)

        gen = ResponseGenerator(
            model=ModelParams(model_name="test"),
            steering=SteeringParams(
                enabled=True,
                vector_path=str(vec_path),
                layers=[0],
            ),
        )
        assert gen._steering_vector is not None
        assert gen._steering_vector.shape == (64,)
        assert torch.allclose(gen._steering_vector.norm(), torch.tensor(1.0), atol=1e-5)

    @patch("sonde.generation.response_generator.AutoModelForCausalLM")
    @patch("sonde.generation.response_generator.AutoTokenizer")
    def test_load_safetensors_vector(self, mock_tok_cls, mock_model_cls, tmp_path):
        mock_tok = MagicMock()
        mock_tok.pad_token_id = 0
        mock_tok_cls.from_pretrained.return_value = mock_tok
        mock_model = MagicMock()
        mock_model.device = "cpu"
        mock_model_cls.from_pretrained.return_value = mock_model

        vec = torch.randn(64)
        vec_path = tmp_path / "v.safetensors"
        save_file({"direction": vec}, str(vec_path))

        gen = ResponseGenerator(
            model=ModelParams(model_name="test"),
            steering=SteeringParams(
                enabled=True,
                vector_path=str(vec_path),
                vector_key="direction",
                layers=[0],
            ),
        )
        assert gen._steering_vector is not None
        assert gen._steering_vector.shape == (64,)

    @patch("sonde.generation.response_generator.AutoModelForCausalLM")
    @patch("sonde.generation.response_generator.AutoTokenizer")
    def test_no_steering_by_default(self, mock_tok_cls, mock_model_cls):
        mock_tok = MagicMock()
        mock_tok.pad_token_id = 0
        mock_tok_cls.from_pretrained.return_value = mock_tok
        mock_model_cls.from_pretrained.return_value = MagicMock(device="cpu")

        gen = ResponseGenerator(model=ModelParams(model_name="test"))
        assert gen._steering_vector is None


class TestSteeringHooks:
    def test_project_subtract_hook(self):
        vec = torch.tensor([1.0, 0.0, 0.0])
        hook = _make_steering_hook(vec, mode="project_subtract", strength=1.0)

        h = torch.tensor([[[3.0, 4.0, 5.0]]])  # (1, 1, 3)
        output = (h,)
        result = hook(None, None, output)

        expected = torch.tensor([[[0.0, 4.0, 5.0]]])
        assert torch.allclose(result[0], expected, atol=1e-5)

    def test_additive_hook(self):
        vec = torch.tensor([1.0, 0.0, 0.0])
        hook = _make_steering_hook(vec, mode="additive", strength=5.0)

        h = torch.tensor([[[3.0, 4.0, 5.0]]])
        output = (h,)
        result = hook(None, None, output)

        expected = torch.tensor([[[8.0, 4.0, 5.0]]])
        assert torch.allclose(result[0], expected, atol=1e-5)

    def test_hook_preserves_extra_outputs(self):
        vec = torch.tensor([1.0, 0.0])
        hook = _make_steering_hook(vec, mode="additive", strength=1.0)

        h = torch.tensor([[[1.0, 2.0]]])
        extra = {"attention": "value"}
        result = hook(None, None, (h, extra))

        assert len(result) == 2
        assert result[1] == extra
