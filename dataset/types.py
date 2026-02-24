"""Shared data types for the dataset package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from torch.utils.data import Dataset


@dataclass
class SampleBundle:
    prompts: Dataset[str]
    labels: list[int | None]
    ids: list[str]

    def train_val_test_split(
        self,
        *,
        train_fraction: float = 0.7,
        val_fraction: float = 0.15,
        test_fraction: float = 0.15,
        seed: int = 0,
        group_ids: Sequence[str] | None = None,
        auto_group_by_id_when_none: bool = True,
    ) -> tuple[list[int], list[int], list[int]]:
        """Build a validated stratified split for this bundle.

        When ``group_ids`` is ``None`` and ``auto_group_by_id_when_none`` is ``True``,
        sample IDs are used as group IDs when at least six unique IDs are present.
        Set ``auto_group_by_id_when_none=False`` to force non-grouped splitting unless
        you pass explicit ``group_ids``.
        """
        from dataset.splitting import stratified_train_val_test_split

        if not self.labels:
            raise ValueError("No labels in SampleBundle; cannot create train/val/test split.")
        if any(label is None for label in self.labels):
            raise ValueError(
                "SampleBundle has unlabeled rows. Add binary labels before splitting."
            )
        labels = [int(label) for label in self.labels if label is not None]
        if len(labels) != len(self.ids):
            raise ValueError("SampleBundle labels and ids must have equal length.")
        if group_ids is not None and len(group_ids) != len(labels):
            raise ValueError("group_ids must match labels length.")

        resolved_group_ids = (
            [str(group_id) for group_id in group_ids]
            if group_ids is not None
            else (
                self.ids
                if auto_group_by_id_when_none and len(set(self.ids)) >= 6
                else None
            )
        )
        return stratified_train_val_test_split(
            labels=labels,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=seed,
            group_ids=resolved_group_ids,
        )
