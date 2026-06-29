"""Linear probes and a minimal training workflow."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import replace
from statistics import mean, stdev
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.classification import (  # type: ignore[import-untyped]
    BinaryAccuracy,
    BinaryAUROC,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)

from sonde.core.configs import ProbeParams
from sonde.probes.architectures import LinearProbe


class BinaryProbeTrainer:
    """End-to-end trainer/evaluator for binary probes."""

    def __init__(self, model: nn.Module, config: ProbeParams | None = None):
        self.config = config or ProbeParams()
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
        self.device = torch.device(self.config.device or "cpu")
        self.model = model.to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.accuracy_metric = BinaryAccuracy(threshold=self.config.threshold).to(
            self.device
        )
        self.precision_metric = BinaryPrecision(
            threshold=self.config.threshold,
        ).to(self.device)
        self.recall_metric = BinaryRecall(
            threshold=self.config.threshold,
        ).to(self.device)
        self.f1_metric = BinaryF1Score(
            threshold=self.config.threshold,
        ).to(self.device)

    def _unpack_batch(
        self, batch: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if len(batch) == 3:
            features, labels, mask = batch
            return (
                features.to(self.device),
                labels.to(self.device),
                mask.to(self.device),
            )
        features, labels = batch[0], batch[1]
        return features.to(self.device), labels.to(self.device), None

    def _forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is not None:
            return self.model(x, mask=mask)
        return self.model(x)

    def fit(
        self,
        train_loader: DataLoader[tuple[torch.Tensor, ...]],
        val_loader: DataLoader[tuple[torch.Tensor, ...]] | None = None,
    ) -> dict[str, list[float | tuple[float, float]]]:
        history: dict[str, list[float | tuple[float, float]]] = {"train_loss": []}
        if val_loader is not None:
            history["val_loss"] = []
            history["val_accuracy"] = []
        best_val_loss: float | None = None
        best_state_dict: dict[str, torch.Tensor] | None = None
        epochs_without_improvement = 0

        for _ in range(self.config.epochs):
            self.model.train()
            running_loss = 0.0
            total = 0
            for batch in train_loader:
                x, labels_batch, mask = self._unpack_batch(batch)
                y = labels_batch.float().unsqueeze(1)
                logits = self._forward(x, mask)
                loss = self.criterion(logits, y)
                if self.config.l1_weight > 0:
                    l1 = sum(p.abs().sum() for p in self.model.parameters())
                    loss = loss + self.config.l1_weight * l1
                self.optimizer.zero_grad()
                loss.backward()
                if self.config.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.config.max_grad_norm
                    )
                self.optimizer.step()

                batch_size = x.shape[0]
                running_loss += loss.item() * batch_size
                total += batch_size

            history["train_loss"].append(running_loss / max(total, 1))

            if val_loader is not None:
                metrics = self.evaluate(val_loader)
                loss_value = metrics["loss"]
                val_loss = (
                    float(loss_value)
                    if isinstance(loss_value, (int, float))
                    else loss_value[0]
                )
                history["val_loss"].append(val_loss)
                history["val_accuracy"].append(metrics["accuracy"])
                if self.config.early_stopping_patience is not None:
                    if best_val_loss is None or (
                        best_val_loss - val_loss > self.config.early_stopping_min_delta
                    ):
                        best_val_loss = val_loss
                        best_state_dict = deepcopy(self.model.state_dict())
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1
                    if (
                        epochs_without_improvement
                        >= self.config.early_stopping_patience
                    ):
                        break
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)

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
        self, data_loader: DataLoader[tuple[torch.Tensor, ...]]
    ) -> dict[str, float | tuple[float, float]]:
        self.model.eval()
        running_loss = 0.0
        total = 0

        self.accuracy_metric.reset()
        self.precision_metric.reset()
        self.recall_metric.reset()
        self.f1_metric.reset()

        all_probs: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        for batch in data_loader:
            x, labels_batch, mask = self._unpack_batch(batch)
            y = labels_batch.float().unsqueeze(1)
            logits = self._forward(x, mask)
            loss = self.criterion(logits, y)

            probs = torch.sigmoid(logits).squeeze(-1)
            batch_labels = labels_batch.long()
            self.accuracy_metric.update(probs, batch_labels)
            self.precision_metric.update(probs, batch_labels)
            self.recall_metric.update(probs, batch_labels)
            self.f1_metric.update(probs, batch_labels)
            all_probs.append(probs.detach().cpu())
            all_labels.append(batch_labels.detach().cpu())

            batch_size = x.shape[0]
            running_loss += loss.item() * batch_size
            total += batch_size

        probs_tensor = (
            torch.cat(all_probs, dim=0)
            if all_probs
            else torch.empty((0,), dtype=torch.float32)
        )
        labels_tensor = (
            torch.cat(all_labels, dim=0)
            if all_labels
            else torch.empty((0,), dtype=torch.long)
        )
        if probs_tensor.numel() == 0:
            accuracy = 0.0
            precision = 0.0
            recall = 0.0
            f1 = 0.0
        else:
            accuracy = float(self.accuracy_metric.compute().item())
            precision = float(self.precision_metric.compute().item())
            recall = float(self.recall_metric.compute().item())
            f1 = float(self.f1_metric.compute().item())
        auroc = _binary_auroc(probs_tensor, labels_tensor)
        return {
            "loss": running_loss / max(total, 1),
            "accuracy": accuracy,
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
            acc_samples.append(
                float((sample_preds == sample_labels).float().mean().item())
            )
            auroc_samples.append(_binary_auroc(sample_probs, sample_labels))
        acc_ci = _percentile_interval(acc_samples, ci_low, ci_high)
        auroc_ci = _percentile_interval(auroc_samples, ci_low, ci_high)
        return {
            "accuracy_ci": acc_ci,
            "auroc_ci": auroc_ci,
        }


class BinaryLinearProbeTrainer(BinaryProbeTrainer):
    """Backward-compatible wrapper that creates a LinearProbe internally."""

    def __init__(self, input_dim: int, config: ProbeParams | None = None):
        model = LinearProbe(input_dim=input_dim)
        super().__init__(model=model, config=config)


def run_probe_with_controls(
    *,
    input_dim: int,
    train_loader: DataLoader[tuple[torch.Tensor, ...]],
    eval_loader: DataLoader[tuple[torch.Tensor, ...]],
    config: ProbeParams | None = None,
    seeds: Sequence[int] = (0,),
) -> dict[str, Any]:
    """Train/evaluate probe with multi-seed and baseline controls."""
    from sonde.probes.architectures import build_probe

    base_config = config or ProbeParams()
    train_x, train_y, train_mask = _loader_to_tensors(train_loader)
    eval_x, eval_y, eval_mask = _loader_to_tensors(eval_loader)

    real_runs: list[dict[str, float | tuple[float, float]]] = []
    shuffled_runs: list[dict[str, float | tuple[float, float]]] = []
    random_feature_runs: list[dict[str, float | tuple[float, float]]] = []

    for seed in seeds:
        run_config = replace(base_config, seed=int(seed))
        model = build_probe(run_config.probe_type, input_dim, **run_config.probe_kwargs)
        trainer = BinaryProbeTrainer(model=model, config=run_config)
        trainer.fit(
            _tensor_loader(train_x, train_y, train_loader.batch_size, train_mask)
        )
        real_runs.append(
            trainer.evaluate(
                _tensor_loader(eval_x, eval_y, eval_loader.batch_size, eval_mask)
            )
        )

        generator = torch.Generator().manual_seed(int(seed) + 1_000)
        permutation = torch.randperm(len(train_y), generator=generator)
        shuffled_y = train_y[permutation]
        shuffled_model = build_probe(
            run_config.probe_type, input_dim, **run_config.probe_kwargs
        )
        shuffled_trainer = BinaryProbeTrainer(model=shuffled_model, config=run_config)
        shuffled_trainer.fit(
            _tensor_loader(train_x, shuffled_y, train_loader.batch_size, train_mask)
        )
        shuffled_runs.append(
            shuffled_trainer.evaluate(
                _tensor_loader(eval_x, eval_y, eval_loader.batch_size, eval_mask)
            )
        )

        rand_train_x = torch.randn(
            train_x.shape, generator=generator, dtype=train_x.dtype
        )
        rand_eval_x = torch.randn(eval_x.shape, generator=generator, dtype=eval_x.dtype)
        random_model = build_probe(
            run_config.probe_type, input_dim, **run_config.probe_kwargs
        )
        random_trainer = BinaryProbeTrainer(model=random_model, config=run_config)
        random_trainer.fit(
            _tensor_loader(rand_train_x, train_y, train_loader.batch_size, train_mask)
        )
        random_feature_runs.append(
            random_trainer.evaluate(
                _tensor_loader(rand_eval_x, eval_y, eval_loader.batch_size, eval_mask)
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
    metric = BinaryAUROC()
    score = metric(probs.float().cpu(), labels.cpu())
    return float(score.item())


def _percentile_interval(
    values: Sequence[float], low_q: float, high_q: float
) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    sorted_vals = sorted(values)
    low_idx = round(low_q * (len(sorted_vals) - 1))
    high_idx = round(high_q * (len(sorted_vals) - 1))
    return (float(sorted_vals[low_idx]), float(sorted_vals[high_idx]))


def _loader_to_tensors(
    data_loader: DataLoader,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    has_mask = False
    for batch in data_loader:
        if len(batch) == 3:
            has_mask = True
            features.append(batch[0].detach().cpu().float())
            labels.append(batch[1].detach().cpu().long())
            masks.append(batch[2].detach().cpu().float())
        else:
            features.append(batch[0].detach().cpu().float())
            labels.append(batch[1].detach().cpu().long())
    if not features:
        return (
            torch.empty((0, 0), dtype=torch.float32),
            torch.empty((0,), dtype=torch.long),
            None,
        )
    if has_mask and features[0].ndim == 3:
        # Sequence mode: pad all batches to the same max seq length before catting
        max_seq = max(f.shape[1] for f in features)
        padded_features, padded_masks = [], []
        for f, m in zip(features, masks, strict=True):
            pad_len = max_seq - f.shape[1]
            if pad_len > 0:
                f = torch.nn.functional.pad(f, (0, 0, 0, pad_len))
                m = torch.nn.functional.pad(m, (0, pad_len))
            padded_features.append(f)
            padded_masks.append(m)
        cat_features = torch.cat(padded_features, dim=0)
        cat_mask = torch.cat(padded_masks, dim=0)
    else:
        cat_features = torch.cat(features, dim=0)
        cat_mask = torch.cat(masks, dim=0) if has_mask else None
    cat_labels = torch.cat(labels, dim=0)
    return cat_features, cat_labels, cat_mask


def _tensor_loader(
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int | None,
    mask: torch.Tensor | None = None,
    shuffle: bool = True,
) -> DataLoader:
    resolved_batch_size = (
        int(batch_size) if isinstance(batch_size, int) and batch_size > 0 else 32
    )
    if mask is not None:
        dataset = torch.utils.data.TensorDataset(features, labels, mask)
    else:
        dataset = torch.utils.data.TensorDataset(features, labels)
    return DataLoader(dataset, batch_size=resolved_batch_size, shuffle=shuffle)


def _aggregate_metrics(
    runs: Sequence[dict[str, float | tuple[float, float]]],
) -> dict[str, float]:
    if not runs:
        return {}
    grouped: dict[str, list[float]] = {}
    for run in runs:
        for key, value in run.items():
            if isinstance(value, tuple):
                continue
            grouped.setdefault(key, []).append(value)
    summary: dict[str, float] = {}
    for key, values in grouped.items():
        summary[f"{key}_mean"] = mean(values)
        summary[f"{key}_std"] = stdev(values) if len(values) > 1 else 0.0
    return summary
