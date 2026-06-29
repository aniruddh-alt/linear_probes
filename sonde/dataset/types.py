"""Shared data types for the dataset package."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SampleBundle:
    """Aligned prompts, labels, and ids that an extractor consumes.

    ``prompts`` is a plain ``list[str]``. For back-compatibility, a torch
    ``Dataset[str]`` (or any indexable/iterable of strings) passed at
    construction is materialised into a list in ``__post_init__``.
    """

    prompts: list[str]
    labels: list[int | None]
    ids: list[str]
    responses: list[str] | None = field(default=None)

    def __post_init__(self) -> None:
        if not isinstance(self.prompts, list):
            seq = self.prompts
            try:
                self.prompts = [seq[i] for i in range(len(seq))]  # type: ignore[arg-type]
            except TypeError:
                self.prompts = list(seq)  # type: ignore[arg-type]

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

        When ``group_ids`` is ``None`` and ``auto_group_by_id_when_none`` is
        ``True``, sample IDs are used as group IDs (preventing the same prompt
        from leaking across splits) whenever there are at least six unique IDs.
        The regime actually used is logged so the choice is never silent. Set
        ``auto_group_by_id_when_none=False`` to force a non-grouped split.
        """
        from sonde.dataset.splitting import stratified_train_val_test_split

        if not self.labels:
            raise ValueError(
                "No labels in SampleBundle; cannot create train/val/test split."
            )
        if any(label is None for label in self.labels):
            raise ValueError(
                "SampleBundle has unlabeled rows. Add binary labels before splitting."
            )
        labels = [int(label) for label in self.labels if label is not None]
        if len(labels) != len(self.ids):
            raise ValueError("SampleBundle labels and ids must have equal length.")
        if group_ids is not None and len(group_ids) != len(labels):
            raise ValueError("group_ids must match labels length.")

        resolved_group_ids = self.resolve_group_ids(
            group_ids, auto_group_by_id_when_none
        )
        return stratified_train_val_test_split(
            labels=labels,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=seed,
            group_ids=resolved_group_ids,
        )

    def resolve_group_ids(
        self,
        group_ids: Sequence[str] | None,
        auto_group_by_id_when_none: bool,
    ) -> list[str] | None:
        """Decide the grouping regime and log it (never silent).

        Public so callers that split outside ``train_val_test_split`` (e.g. the
        YAML runner) reuse the exact same regime + leakage-warning logic.
        """
        from sonde.dataset.splitting import MIN_STRATIFIED_GROUPS

        if group_ids is not None:
            resolved = [str(group_id) for group_id in group_ids]
            logger.info(
                "Splitting with explicit group_ids (%d unique groups).",
                len(set(resolved)),
            )
            return resolved

        if not auto_group_by_id_when_none:
            self._warn_if_duplicate_ids("auto-grouping disabled")
            logger.info("Auto-grouping disabled; using non-grouped stratified split.")
            return None

        n_unique = len(set(self.ids))
        if n_unique >= MIN_STRATIFIED_GROUPS:
            logger.info(
                "Auto-grouping train/val/test by sample_id (%d unique groups) to "
                "prevent prompt leakage across splits.",
                n_unique,
            )
            return list(self.ids)

        self._warn_if_duplicate_ids(f"only {n_unique} unique ids (< 6 groups)")
        logger.info(
            "Using non-grouped stratified split (%d unique ids < 6 required groups).",
            n_unique,
        )
        return None

    def _warn_if_duplicate_ids(self, context: str) -> None:
        if len(set(self.ids)) != len(self.ids):
            logger.warning(
                "Non-grouped split with repeated sample ids (%s): the same prompt "
                "may leak across train/val/test. Pass group_ids explicitly to avoid this.",
                context,
            )
