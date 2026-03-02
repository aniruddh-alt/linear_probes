from __future__ import annotations

import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from probes.architectures import build_probe


class LinearProbeArchTest(unittest.TestCase):
    def test_linear_probe_last_token_3d_input(self) -> None:
        probe = build_probe("linear", input_dim=8)
        x = torch.randn(4, 10, 8)
        logits = probe(x)
        self.assertEqual(logits.shape, (4, 1))

    def test_linear_probe_2d_input_backward_compat(self) -> None:
        probe = build_probe("linear", input_dim=8)
        x = torch.randn(4, 8)
        logits = probe(x)
        self.assertEqual(logits.shape, (4, 1))

    def test_linear_probe_with_mask_ignores_it(self) -> None:
        probe = build_probe("linear", input_dim=8)
        x = torch.randn(4, 10, 8)
        mask = torch.ones(4, 10)
        logits = probe(x, mask=mask)
        self.assertEqual(logits.shape, (4, 1))

    def test_build_probe_raises_on_unknown_type(self) -> None:
        with self.assertRaises(ValueError):
            build_probe("nonexistent", input_dim=8)


class MeanProbeTest(unittest.TestCase):
    def test_mean_probe_3d(self) -> None:
        probe = build_probe("mean", input_dim=8)
        x = torch.randn(4, 10, 8)
        logits = probe(x)
        self.assertEqual(logits.shape, (4, 1))

    def test_mean_probe_with_mask(self) -> None:
        probe = build_probe("mean", input_dim=8)
        x = torch.randn(4, 10, 8)
        mask = torch.ones(4, 10)
        mask[:, 5:] = 0
        logits = probe(x, mask=mask)
        self.assertEqual(logits.shape, (4, 1))

    def test_mean_probe_mask_excludes_padding(self) -> None:
        torch.manual_seed(42)
        probe = build_probe("mean", input_dim=4)
        x = torch.randn(1, 6, 4)
        mask_full = torch.ones(1, 6)
        mask_partial = torch.tensor([[1, 1, 1, 0, 0, 0]], dtype=torch.float32)
        out_full = probe(x, mask=mask_full)
        out_partial = probe(x, mask=mask_partial)
        self.assertFalse(torch.allclose(out_full, out_partial))


class MaxProbeTest(unittest.TestCase):
    def test_max_probe_3d(self) -> None:
        probe = build_probe("max", input_dim=8)
        logits = probe(torch.randn(4, 10, 8))
        self.assertEqual(logits.shape, (4, 1))


class SoftmaxProbeTest(unittest.TestCase):
    def test_softmax_probe_3d(self) -> None:
        probe = build_probe("softmax", input_dim=8)
        logits = probe(torch.randn(4, 10, 8))
        self.assertEqual(logits.shape, (4, 1))

    def test_softmax_probe_custom_phi(self) -> None:
        probe = build_probe("softmax", input_dim=8, phi=10.0)
        logits = probe(torch.randn(4, 10, 8))
        self.assertEqual(logits.shape, (4, 1))


class AttentionProbeTest(unittest.TestCase):
    def test_attention_probe_3d(self) -> None:
        probe = build_probe("attention", input_dim=8)
        logits = probe(torch.randn(4, 10, 8))
        self.assertEqual(logits.shape, (4, 1))

    def test_attention_probe_with_mask(self) -> None:
        probe = build_probe("attention", input_dim=8)
        x = torch.randn(4, 10, 8)
        mask = torch.ones(4, 10)
        mask[:, 7:] = 0
        logits = probe(x, mask=mask)
        self.assertEqual(logits.shape, (4, 1))


class MaxRollingMeanProbeTest(unittest.TestCase):
    def test_max_rolling_mean_probe_3d(self) -> None:
        probe = build_probe("max_rolling_mean", input_dim=8)
        logits = probe(torch.randn(4, 50, 8))
        self.assertEqual(logits.shape, (4, 1))

    def test_max_rolling_mean_short_seq(self) -> None:
        probe = build_probe("max_rolling_mean", input_dim=8, window_size=40)
        logits = probe(torch.randn(4, 5, 8))
        self.assertEqual(logits.shape, (4, 1))


if __name__ == "__main__":
    unittest.main()
