from __future__ import annotations

import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from probes.architectures import (
    AttentionProbe,
    BaseProbe,
    LinearProbe,
    MaxProbe,
    MaxRollingMeanProbe,
    MeanProbe,
    SoftmaxProbe,
    build_probe,
)


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


class BaseProbeContractTest(unittest.TestCase):
    """Verify every registered probe inherits BaseProbe and honours the contract."""

    REGISTERED_TYPES = (
        "linear",
        "mean",
        "max",
        "softmax",
        "attention",
        "max_rolling_mean",
    )

    def test_every_probe_inherits_base_probe(self) -> None:
        for probe_type in self.REGISTERED_TYPES:
            with self.subTest(probe_type=probe_type):
                probe = build_probe(probe_type, input_dim=8)
                self.assertIsInstance(probe, BaseProbe)

    def test_input_dim_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            LinearProbe(input_dim=0)

    def test_num_classes_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            LinearProbe(input_dim=8, num_classes=0)

    def test_invalid_input_rank_raises(self) -> None:
        probe = build_probe("linear", input_dim=8)
        x = torch.randn(2, 3, 4, 8)
        with self.assertRaises(ValueError):
            probe(x)

    def test_2d_input_synthesises_unit_mask(self) -> None:
        probe = build_probe("mean", input_dim=8)
        x = torch.randn(4, 8)
        out_no_mask = probe(x)
        out_with_arbitrary_mask = probe(x, mask=torch.zeros(4, 99))
        self.assertEqual(out_no_mask.shape, (4, 1))
        self.assertTrue(torch.allclose(out_no_mask, out_with_arbitrary_mask))


class BaseProbeDirectionTest(unittest.TestCase):
    """``direction`` and ``bias`` properties must work for binary linear-readout probes."""

    LINEAR_READOUT_TYPES = ("linear", "mean", "max", "softmax", "max_rolling_mean")

    def test_direction_is_unit_norm_for_linear_readout_probes(self) -> None:
        for probe_type in self.LINEAR_READOUT_TYPES:
            with self.subTest(probe_type=probe_type):
                probe = build_probe(probe_type, input_dim=8)
                direction = probe.direction
                self.assertIsNotNone(direction)
                assert direction is not None
                self.assertEqual(direction.shape, (8,))
                self.assertAlmostEqual(float(direction.norm().item()), 1.0, places=5)

    def test_bias_is_scalar_float_for_linear_readout_probes(self) -> None:
        for probe_type in self.LINEAR_READOUT_TYPES:
            with self.subTest(probe_type=probe_type):
                probe = build_probe(probe_type, input_dim=8)
                bias = probe.bias
                self.assertIsInstance(bias, float)

    def test_attention_probe_returns_no_direction(self) -> None:
        probe = build_probe("attention", input_dim=8)
        self.assertIsNone(probe.direction)
        self.assertIsInstance(probe.bias, float)

    def test_zero_weight_direction_falls_back_to_raw_weight(self) -> None:
        probe = build_probe("linear", input_dim=8)
        with torch.no_grad():
            probe.linear.weight.zero_()
        direction = probe.direction
        self.assertIsNotNone(direction)
        assert direction is not None
        self.assertTrue(torch.equal(direction, torch.zeros(8)))


class BaseProbeBackwardCompatTest(unittest.TestCase):
    """Refactor must not change behaviour observable by existing callers."""

    def test_mean_probe_matches_original_pre_refactor_formula(self) -> None:
        torch.manual_seed(0)
        probe = MeanProbe(input_dim=4)
        x = torch.randn(3, 7, 4)
        mask = torch.tensor(
            [[1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1]],
            dtype=torch.float32,
        )
        out_new = probe(x, mask=mask)
        m = mask.unsqueeze(-1)
        pooled = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        out_pre = probe.linear(pooled)
        self.assertTrue(torch.allclose(out_new, out_pre, atol=1e-5))

    def test_max_probe_matches_original_pre_refactor_formula(self) -> None:
        torch.manual_seed(0)
        probe = MaxProbe(input_dim=4)
        x = torch.randn(3, 7, 4)
        mask = torch.tensor(
            [[1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1]],
            dtype=torch.float32,
        )
        out_new = probe(x, mask=mask)
        x_masked = x.masked_fill(mask.unsqueeze(-1) == 0, float("-inf"))
        pooled = x_masked.max(dim=1).values
        out_pre = probe.linear(pooled)
        self.assertTrue(torch.allclose(out_new, out_pre, atol=1e-5))

    def test_softmax_probe_matches_original_pre_refactor_formula(self) -> None:
        torch.manual_seed(0)
        probe = SoftmaxProbe(input_dim=4, phi=5.0)
        x = torch.randn(3, 7, 4)
        mask = torch.tensor(
            [[1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1]],
            dtype=torch.float32,
        )
        out_new = probe(x, mask=mask)
        scores = torch.nn.functional.linear(x, probe.linear.weight) * probe.phi
        scores = scores.squeeze(-1).masked_fill(mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        pooled = (weights * x).sum(dim=1)
        out_pre = probe.linear(pooled)
        self.assertTrue(torch.allclose(out_new, out_pre, atol=1e-5))

    def test_max_rolling_mean_probe_matches_pre_refactor_formula(self) -> None:
        torch.manual_seed(0)
        probe = MaxRollingMeanProbe(input_dim=4, window_size=3)
        x = torch.randn(2, 10, 4)
        out_new = probe(x)
        pooled = torch.nn.functional.avg_pool1d(
            x.transpose(1, 2), kernel_size=3, stride=1
        )
        max_pooled = pooled.max(dim=2).values
        out_pre = probe.linear(max_pooled)
        self.assertTrue(torch.allclose(out_new, out_pre, atol=1e-5))

    def test_attention_probe_records_attention_weights(self) -> None:
        torch.manual_seed(0)
        probe = AttentionProbe(input_dim=8, n_heads=2)
        x = torch.randn(3, 6, 8)
        probe(x)
        self.assertIsNotNone(probe.attention_weights_)
        assert probe.attention_weights_ is not None
        self.assertEqual(probe.attention_weights_.shape, (3, 6, 2))


class EndToEndProbeTest(unittest.TestCase):
    def test_all_probe_types_train_and_evaluate(self) -> None:
        """Smoke test: every registered probe type can train and evaluate."""
        from core.configs import ProbeParams
        from probes.linear import BinaryProbeTrainer

        torch.manual_seed(42)
        for probe_type in [
            "linear",
            "mean",
            "max",
            "softmax",
            "attention",
            "max_rolling_mean",
        ]:
            with self.subTest(probe_type=probe_type):
                features = torch.randn(32, 15, 8)
                labels = (features[:, -1, 0] > 0).long()
                mask = torch.ones(32, 15)
                dataset = TensorDataset(features, labels, mask)
                loader = DataLoader(dataset, batch_size=8, shuffle=True)

                model = build_probe(probe_type, input_dim=8)
                trainer = BinaryProbeTrainer(
                    model=model,
                    config=ProbeParams(
                        epochs=3, learning_rate=0.01, early_stopping_patience=None
                    ),
                )
                trainer.fit(loader, val_loader=loader)
                metrics = trainer.evaluate(loader)
                self.assertIn("auroc", metrics)
                auroc = metrics["auroc"]
                self.assertGreater(
                    auroc if isinstance(auroc, (int, float)) else auroc[0], 0.0
                )
