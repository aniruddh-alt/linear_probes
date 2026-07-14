"""Split helpers for train/validation/test dataset workflows."""

from __future__ import annotations

import random
from collections.abc import Sequence

# Minimum number of samples (and of groups, in group-aware mode) required for a
# strict stratified train/val/test split. Shared so the threshold is defined once.
MIN_STRATIFIED_GROUPS = 6


def stratified_train_val_test_split(
    *,
    labels: Sequence[int],
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 0,
    group_ids: Sequence[str] | None = None,
) -> tuple[list[int], list[int], list[int]]:
    """Public train/val/test splitter with strict stratification checks."""
    return _stratified_train_val_test_split(
        labels=labels,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
        group_ids=group_ids,
    )


def _stratified_train_val_test_split(
    *,
    labels: Sequence[int],
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 0,
    group_ids: Sequence[str] | None = None,
) -> tuple[list[int], list[int], list[int]]:
    """Build deterministic train/val/test splits with strict safety checks."""
    _validate_fraction_triplet(train_fraction, val_fraction, test_fraction)
    if len(labels) < MIN_STRATIFIED_GROUPS:
        raise ValueError(
            f"Need at least {MIN_STRATIFIED_GROUPS} samples for strict "
            "train/val/test stratification."
        )
    label_values = [int(label) for label in labels]
    if any(label not in (0, 1) for label in label_values):
        raise ValueError("labels must be binary values in {0, 1}.")

    if group_ids is not None:
        if len(group_ids) != len(labels):
            raise ValueError("group_ids length must equal labels length.")
        train_idx, val_idx, test_idx = _group_aware_split(
            labels=label_values,
            group_ids=[str(group) for group in group_ids],
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=seed,
        )
    else:
        train_idx, val_idx, test_idx = _stratified_index_split(
            labels=label_values,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=seed,
        )

    _validate_split_indices(
        total_samples=len(labels),
        labels=label_values,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        group_ids=[str(group) for group in group_ids]
        if group_ids is not None
        else None,
    )
    return train_idx, val_idx, test_idx


def _validate_split_indices(
    *,
    total_samples: int,
    labels: Sequence[int],
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    test_idx: Sequence[int],
    group_ids: Sequence[str] | None = None,
) -> None:
    """Validate split integrity, label coverage, and optional group disjointness."""
    if total_samples <= 0:
        raise ValueError("total_samples must be positive.")
    if len(labels) != total_samples:
        raise ValueError("labels length must equal total_samples.")

    train = list(train_idx)
    val = list(val_idx)
    test = list(test_idx)
    if not train or not val or not test:
        raise ValueError("train_idx, val_idx, and test_idx must each be non-empty.")

    all_idx = train + val + test
    for idx in all_idx:
        if idx < 0 or idx >= total_samples:
            raise IndexError(
                f"Split index {idx} out of range for {total_samples} samples."
            )

    if len(set(train)) != len(train):
        raise ValueError("train_idx contains duplicate indices.")
    if len(set(val)) != len(val):
        raise ValueError("val_idx contains duplicate indices.")
    if len(set(test)) != len(test):
        raise ValueError("test_idx contains duplicate indices.")

    train_set = set(train)
    val_set = set(val)
    test_set = set(test)
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise ValueError("train_idx, val_idx, and test_idx must be pairwise disjoint.")

    covered = train_set | val_set | test_set
    if covered != set(range(total_samples)):
        missing = sorted(set(range(total_samples)) - covered)
        raise ValueError(
            f"Split indices must cover all samples exactly once. Missing: {missing}"
        )

    for split_name, split in (("train", train), ("val", val), ("test", test)):
        split_labels = [int(labels[idx]) for idx in split]
        if not ({0, 1} <= set(split_labels)):
            raise ValueError(
                f"{split_name} split must contain both classes 0 and 1. "
                f"Observed labels: {sorted(set(split_labels))}."
            )

    if group_ids is not None:
        if len(group_ids) != total_samples:
            raise ValueError("group_ids length must equal total_samples.")
        train_groups = {group_ids[idx] for idx in train}
        val_groups = {group_ids[idx] for idx in val}
        test_groups = {group_ids[idx] for idx in test}
        if (
            train_groups & val_groups
            or train_groups & test_groups
            or val_groups & test_groups
        ):
            raise ValueError("Group leakage detected across train/val/test splits.")


def _stratified_index_split(
    *,
    labels: list[int],
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    rng = random.Random(seed)
    pos_indices = [idx for idx, label in enumerate(labels) if label == 1]
    neg_indices = [idx for idx, label in enumerate(labels) if label == 0]
    if len(pos_indices) < 3 or len(neg_indices) < 3:
        raise ValueError(
            "Need at least 3 positive and 3 negative samples for strict split class coverage."
        )
    rng.shuffle(pos_indices)
    rng.shuffle(neg_indices)

    pos_counts = _counts_with_min_one(
        n=len(pos_indices),
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )
    neg_counts = _counts_with_min_one(
        n=len(neg_indices),
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )

    train = pos_indices[: pos_counts[0]] + neg_indices[: neg_counts[0]]
    val = (
        pos_indices[pos_counts[0] : pos_counts[0] + pos_counts[1]]
        + neg_indices[neg_counts[0] : neg_counts[0] + neg_counts[1]]
    )
    test = (
        pos_indices[pos_counts[0] + pos_counts[1] :]
        + neg_indices[neg_counts[0] + neg_counts[1] :]
    )
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def _group_aware_split(
    *,
    labels: list[int],
    group_ids: list[str],
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    grouped: dict[str, list[int]] = {}
    for idx, group in enumerate(group_ids):
        grouped.setdefault(group, []).append(idx)

    if len(grouped) < MIN_STRATIFIED_GROUPS:
        raise ValueError(
            f"Need at least {MIN_STRATIFIED_GROUPS} groups for strict "
            "group-aware splitting."
        )
    pos_groups: list[str] = []
    neg_groups: list[str] = []
    for group, idxs in grouped.items():
        group_labels = [labels[idx] for idx in idxs]
        positives = sum(group_labels)
        negatives = len(group_labels) - positives
        if positives > negatives:
            pos_groups.append(group)
        elif negatives > positives:
            neg_groups.append(group)
        else:
            if len(pos_groups) <= len(neg_groups):
                pos_groups.append(group)
            else:
                neg_groups.append(group)
    if len(pos_groups) < 3 or len(neg_groups) < 3:
        raise ValueError(
            "Need at least 3 positive-majority and 3 negative-majority groups for strict group-aware splitting."
        )

    rng = random.Random(seed)
    rng.shuffle(pos_groups)
    rng.shuffle(neg_groups)
    pos_counts = _counts_with_min_one(
        n=len(pos_groups),
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )
    neg_counts = _counts_with_min_one(
        n=len(neg_groups),
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
    )

    split_groups = {
        "train": pos_groups[: pos_counts[0]] + neg_groups[: neg_counts[0]],
        "val": (
            pos_groups[pos_counts[0] : pos_counts[0] + pos_counts[1]]
            + neg_groups[neg_counts[0] : neg_counts[0] + neg_counts[1]]
        ),
        "test": (
            pos_groups[pos_counts[0] + pos_counts[1] :]
            + neg_groups[neg_counts[0] + neg_counts[1] :]
        ),
    }
    for groups in split_groups.values():
        rng.shuffle(groups)

    train_idx = [idx for group in split_groups["train"] for idx in grouped[group]]
    val_idx = [idx for group in split_groups["val"] for idx in grouped[group]]
    test_idx = [idx for group in split_groups["test"] for idx in grouped[group]]
    return train_idx, val_idx, test_idx


def _counts_with_min_one(
    *,
    n: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> tuple[int, int, int]:
    if n < 3:
        raise ValueError("Cannot allocate class counts with n < 3.")
    base = [1, 1, 1]
    remaining = n - 3
    fractions = [train_fraction, val_fraction, test_fraction]
    fractional_alloc = [remaining * fraction for fraction in fractions]
    extras = [int(value) for value in fractional_alloc]
    used = sum(extras)
    remainders = [value - int(value) for value in fractional_alloc]
    for split_idx in sorted(range(3), key=lambda idx: remainders[idx], reverse=True)[
        : remaining - used
    ]:
        extras[split_idx] += 1
    counts = [base[idx] + extras[idx] for idx in range(3)]
    return counts[0], counts[1], counts[2]


def _validate_fraction_triplet(
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> None:
    for value in (train_fraction, val_fraction, test_fraction):
        if value <= 0.0 or value >= 1.0:
            raise ValueError("train/val/test fractions must each be in (0, 1).")
    total = train_fraction + val_fraction + test_fraction
    if abs(total - 1.0) > 1e-8:
        raise ValueError(
            "train_fraction + val_fraction + test_fraction must equal 1.0."
        )
