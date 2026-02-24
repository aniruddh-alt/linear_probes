"""Analyze trained probes for a model."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from probes.types import TrainedLayerProbe


class ProbeAnalyzer:
    def __init__(
        self,
        probes: list[TrainedLayerProbe],
        *,
        output_dir: str | Path | None = None,
        save_plots: bool = True,
    ):
        if not probes:
            raise ValueError("ProbeAnalyzer requires at least one trained probe.")
        self.probes = probes
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.save_plots = save_plots

    def auroc_analysis(self, *, output_path: str | Path | None = None) -> TrainedLayerProbe:
        """Plot AUROC ranking and return the best probe by AUROC."""
        keys = [probe.activation_key for probe in self.probes]
        scores = [self._metric_value(probe, "auroc") for probe in self.probes]
        best_idx = max(range(len(scores)), key=lambda idx: scores[idx])

        fig, ax = plt.subplots(figsize=(max(8, len(keys) * 0.55), 4.5))
        bars = ax.bar(keys, scores, color="steelblue", alpha=0.85)
        bars[best_idx].set_color("crimson")
        bars[best_idx].set_alpha(0.95)
        ax.axhline(
            0.5, color="gray", linestyle="--", linewidth=1.0, label="Chance (0.5)"
        )
        ax.set_xlabel("Activation Key")
        ax.set_ylabel("AUROC")
        ax.set_title("Probe AUROC Ranking")
        ax.set_ylim(0.0, 1.0)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.legend(loc="lower right")
        fig.tight_layout()
        if self.save_plots:
            plt.savefig(self._resolve_output_path("probe_auroc_ranking.png", output_path))

        ranked = sorted(
            self.probes, key=lambda x: self._metric_value(x, "auroc"), reverse=True
        )
        return ranked[0]

    def cosine_similarity_analysis(
        self, *, output_path: str | Path | None = None
    ) -> torch.Tensor:
        """Plot pairwise cosine-similarity heatmap across probe directions."""
        keys = [probe.activation_key for probe in self.probes]
        directions = [
            probe.direction.detach().cpu().reshape(-1).float() for probe in self.probes
        ]
        dims = {int(direction.numel()) for direction in directions}
        if len(dims) != 1:
            raise ValueError(
                "All probe directions must have the same dimensionality for pairwise cosine similarity."
            )

        direction_matrix = torch.stack(directions, dim=0)
        direction_matrix = torch.nn.functional.normalize(direction_matrix, dim=1)
        cosine_matrix = direction_matrix @ direction_matrix.T

        fig, ax = plt.subplots(
            figsize=(max(6, len(keys) * 0.6), max(5, len(keys) * 0.6))
        )
        im = ax.imshow(cosine_matrix.numpy(), vmin=-1.0, vmax=1.0, cmap="coolwarm")
        ax.set_xticks(range(len(keys)))
        ax.set_yticks(range(len(keys)))
        ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(keys, fontsize=8)
        ax.set_title("Pairwise Cosine Similarity of Probe Directions")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Cosine Similarity")
        fig.tight_layout()
        if self.save_plots:
            plt.savefig(
                self._resolve_output_path("probe_cosine_similarity.png", output_path)
            )
        return cosine_matrix

    def _resolve_output_path(self, default_name: str, output_path: str | Path | None) -> Path:
        if output_path is not None:
            resolved = Path(output_path)
        elif self.output_dir is not None:
            resolved = self.output_dir / default_name
        else:
            resolved = Path(default_name)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved

    @staticmethod
    def _metric_value(probe: TrainedLayerProbe, metric_name: str) -> float:
        metric = probe.val_metrics.get(metric_name)
        if metric is None:
            metric = probe.val_metrics.get(metric_name.upper())
        if metric is None:
            metric = probe.val_metrics.get(metric_name.lower())
        if metric is None or isinstance(metric, tuple):
            raise KeyError(
                f"Metric '{metric_name}' is missing or non-scalar for probe '{probe.activation_key}'."
            )
        return float(metric)
