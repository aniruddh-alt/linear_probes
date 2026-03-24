"""Collate functions for variable-length sequence batches."""

from __future__ import annotations

import torch


def sequence_collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad variable-length (S, D) tensors to (S_max, D) with attention mask."""
    if not batch:
        raise ValueError("Cannot collate an empty batch.")
    features, labels, masks = zip(*batch)
    max_len = max(f.shape[0] for f in features)
    dim = features[0].shape[-1]
    padded = torch.zeros(len(features), max_len, dim)
    out_mask = torch.zeros(len(features), max_len)
    for i, f in enumerate(features):
        seq_len = f.shape[0]
        padded[i, :seq_len] = f
        out_mask[i, :seq_len] = 1.0
    return padded, torch.stack(labels), out_mask
