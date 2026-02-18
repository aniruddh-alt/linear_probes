"""Linear probes and a minimal training workflow."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from statistics import mean, stdev
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from configs.probe_config import ProbeTrainConfig


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class BinaryLinearProbeTrainer:
    """End-to-end trainer/evaluator for binary linear probes."""

    def __init__(self, input_dim: int, config: ProbeTrainConfig | None = None):
        self.config = config or ProbeTrainConfig()
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
        self.device = torch.device(self.config.device or "cpu")
        self.model = LinearProbe(input_dim=input_dim, output_dim=1).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def fit(
        self,
        train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> dict[str, list[float]]:
        history: dict[str, list[float]] = {"train_loss": []}
        if val_loader is not None:
            history["val_loss"] = []
            history["val_accuracy"] = []

        for _ in range(self.config.epochs):
            self.model.train()
            running_loss = 0.0
            total = 0
            for features, labels in train_loader:
                x = features.to(self.device)
                y = labels.to(self.device).float().unsqueeze(1)
                logits = self.model(x)
                loss = self.criterion(logits, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_size = x.shape[0]
                running_loss += loss.item() * batch_size
                total += batch_size
            history["train_loss"].append(running_loss / max(total, 1))

            if val_loader is not None:
                metrics = self.evaluate(val_loader)
                history["val_loss"].append(metrics["loss"])
                history["val_accuracy"].append(metrics["accuracy"])

        return history

    @torch.no_grad()
    def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        logits = self.model(features.to(self.device))
        return torch.sigmoid(logits).squeeze(-1).cpu()

    @torch.no_grad()
    def predict(self, features: torch.Tensor) -> torch.Tensor:
        probs = self.predict_proba(features)
        return (probs >= self.config.threshold).long()

    @torch.no_grad()
    def evaluate(
        self, data_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]]
    ) -> dict[str, float | tuple[float, float]]:
        self.model.eval()
        running_loss = 0.0
        total = 0
        correct = 0
        total_positive = 0
        predicted_positive = 0
        true_positive = 0
        all_probs: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        for features, labels in data_loader:
            x = features.to(self.device)
            y = labels.to(self.device).float().unsqueeze(1)
            logits = self.model(x)
            loss = self.criterion(logits, y)

            probs = torch.sigmoid(logits).squeeze(-1)
            preds = (probs >= self.config.threshold).long()
            batch_labels = labels.to(self.device).long()
            correct += int((preds == batch_labels).sum().item())
            total_positive += int(batch_labels.sum().item())
            predicted_positive += int(preds.sum().item())
            true_positive += int(((preds == 1) & (batch_labels == 1)).sum().item())
            all_probs.append(probs.detach().cpu())
            all_labels.append(batch_labels.detach().cpu())

            batch_size = x.shape[0]
            running_loss += loss.item() * batch_size
            total += batch_size

        precision = true_positive / max(predicted_positive, 1)
        recall = true_positive / max(total_positive, 1)
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        probs_tensor = (
            torch.cat(all_probs, dim=0) if all_probs else torch.empty((0,), dtype=torch.float32)
        )
        labels_tensor = (
            torch.cat(all_labels, dim=0) if all_labels else torch.empty((0,), dtype=torch.long)
        )
        auroc = _binary_auroc(probs_tensor, labels_tensor)
        return {
            "loss": running_loss / max(total, 1),
            "accuracy": correct / max(total, 1),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auroc": auroc,
            **self._bootstrap_ci(probs_tensor, labels_tensor),
        }

    def _bootstrap_ci(
        self, probs: torch.Tensor, labels: torch.Tensor
    ) -> dict[str, tuple[float, float]]:
        if self.config.bootstrap_samples <= 0 or probs.numel() == 0:
            return {}
        n = int(probs.numel())
        generator = torch.Generator()
        if self.config.seed is not None:
            generator.manual_seed(self.config.seed + 10_000)
        ci_low = (1.0 - self.config.bootstrap_confidence) / 2.0
        ci_high = 1.0 - ci_low
        acc_samples: list[float] = []
        auroc_samples: list[float] = []
        for _ in range(self.config.bootstrap_samples):
            sample_idx = torch.randint(0, n, (n,), generator=generator)
            sample_probs = probs[sample_idx]
            sample_labels = labels[sample_idx]
            sample_preds = (sample_probs >= self.config.threshold).long()
            acc_samples.append(float((sample_preds == sample_labels).float().mean().item()))
            auroc_samples.append(_binary_auroc(sample_probs, sample_labels))
        acc_ci = _percentile_interval(acc_samples, ci_low, ci_high)
        auroc_ci = _percentile_interval(auroc_samples, ci_low, ci_high)
        return {
            "accuracy_ci": acc_ci,
            "auroc_ci": auroc_ci,
        }


def run_probe_with_controls(
    *,
    input_dim: int,
    train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    eval_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    config: ProbeTrainConfig | None = None,
    seeds: Sequence[int] = (0,),
) -> dict[str, Any]:
    """Train/evaluate probe with multi-seed and baseline controls.

    Controls:
    - shuffled_labels: train on a random permutation of training labels.
    - random_features: train/eval on Gaussian random features.
    """
    base_config = config or ProbeTrainConfig()
    train_x, train_y = _loader_to_tensors(train_loader)
    eval_x, eval_y = _loader_to_tensors(eval_loader)

    real_runs: list[dict[str, float | tuple[float, float]]] = []
    shuffled_runs: list[dict[str, float | tuple[float, float]]] = []
    random_feature_runs: list[dict[str, float | tuple[float, float]]] = []

    for seed in seeds:
        run_config = replace(base_config, seed=int(seed))
        trainer = BinaryLinearProbeTrainer(input_dim=input_dim, config=run_config)
        trainer.fit(_tensor_loader(train_x, train_y, train_loader.batch_size))
        real_runs.append(
            trainer.evaluate(_tensor_loader(eval_x, eval_y, eval_loader.batch_size))
        )

        generator = torch.Generator().manual_seed(int(seed) + 1_000)
        permutation = torch.randperm(len(train_y), generator=generator)
        shuffled_y = train_y[permutation]
        shuffled_trainer = BinaryLinearProbeTrainer(input_dim=input_dim, config=run_config)
        shuffled_trainer.fit(_tensor_loader(train_x, shuffled_y, train_loader.batch_size))
        shuffled_runs.append(
            shuffled_trainer.evaluate(
                _tensor_loader(eval_x, eval_y, eval_loader.batch_size)
            )
        )

        rand_train_x = torch.randn(
            train_x.shape, generator=generator, dtype=train_x.dtype
        )
        rand_eval_x = torch.randn(eval_x.shape, generator=generator, dtype=eval_x.dtype)
        random_trainer = BinaryLinearProbeTrainer(input_dim=input_dim, config=run_config)
        random_trainer.fit(
            _tensor_loader(rand_train_x, train_y, train_loader.batch_size)
        )
        random_feature_runs.append(
            random_trainer.evaluate(
                _tensor_loader(rand_eval_x, eval_y, eval_loader.batch_size)
            )
        )

    return {
        "real": _aggregate_metrics(real_runs),
        "controls": {
            "shuffled_labels": _aggregate_metrics(shuffled_runs),
            "random_features": _aggregate_metrics(random_feature_runs),
        },
    }


def _binary_auroc(probs: torch.Tensor, labels: torch.Tensor) -> float:
    if probs.numel() == 0:
        return 0.0
    labels = labels.long()
    pos_mask = labels == 1
    neg_mask = labels == 0
    n_pos = int(pos_mask.sum().item())
    n_neg = int(neg_mask.sum().item())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = torch.argsort(torch.argsort(probs)) + 1
    pos_rank_sum = float(ranks[pos_mask].sum().item())
    u_stat = pos_rank_sum - (n_pos * (n_pos + 1) / 2.0)
    return float(u_stat / (n_pos * n_neg))


def _percentile_interval(
    values: Sequence[float], low_q: float, high_q: float
) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    sorted_vals = sorted(values)
    low_idx = int(low_q * (len(sorted_vals) - 1))
    high_idx = int(high_q * (len(sorted_vals) - 1))
    return (float(sorted_vals[low_idx]), float(sorted_vals[high_idx]))


def _loader_to_tensors(
    data_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for batch_features, batch_labels in data_loader:
        features.append(batch_features.detach().cpu().float())
        labels.append(batch_labels.detach().cpu().long())
    if not features:
        return torch.empty((0, 0), dtype=torch.float32), torch.empty((0,), dtype=torch.long)
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def _tensor_loader(
    features: torch.Tensor, labels: torch.Tensor, batch_size: int | None
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    resolved_batch_size = int(batch_size) if isinstance(batch_size, int) and batch_size > 0 else 32
    dataset = torch.utils.data.TensorDataset(features, labels)
    return DataLoader(dataset, batch_size=resolved_batch_size, shuffle=True)


def _aggregate_metrics(
    runs: Sequence[dict[str, float | tuple[float, float]]]
) -> dict[str, float]:
    if not runs:
        return {}
    grouped: dict[str, list[float]] = {}
    for run in runs:
        for key, value in run.items():
            if isinstance(value, tuple):
                continue
            grouped.setdefault(key, []).append(float(value))
    summary: dict[str, float] = {}
    for key, values in grouped.items():
        summary[f"{key}_mean"] = mean(values)
        summary[f"{key}_std"] = stdev(values) if len(values) > 1 else 0.0
    return summary
