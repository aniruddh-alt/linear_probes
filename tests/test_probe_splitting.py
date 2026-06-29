from __future__ import annotations

import unittest

from sonde.dataset import ProbingSampleBuilder
from sonde.dataset.splitting import stratified_train_val_test_split


class ProbeSplittingTests(unittest.TestCase):
    @staticmethod
    def _bundle(labels: list[int], *, grouped: bool = False):
        records = []
        for idx, label in enumerate(labels):
            sample_id = f"group-{idx // 2}" if grouped else f"group-{idx}"
            records.append(
                {
                    "id": sample_id,
                    "text": f"example-{idx}",
                    "label": label,
                }
            )
        return ProbingSampleBuilder.from_iterable(records).to_samples(text_key="text")

    def test_stratified_split_preserves_label_balance(self) -> None:
        labels = [0] * 60 + [1] * 60
        bundle = self._bundle(labels)
        train_idx, val_idx, test_idx = bundle.train_val_test_split(
            train_fraction=0.7,
            val_fraction=0.15,
            test_fraction=0.15,
            seed=7,
            group_ids=bundle.ids,
        )
        base_rate = sum(labels) / len(labels)
        for idxs in (train_idx, val_idx, test_idx):
            rate = sum(labels[idx] for idx in idxs) / len(idxs)
            self.assertLess(abs(rate - base_rate), 0.15)

    def test_group_aware_split_keeps_groups_disjoint(self) -> None:
        labels = [idx % 2 for idx in range(80)]
        bundle = self._bundle(labels, grouped=True)
        train_idx, val_idx, test_idx = bundle.train_val_test_split(
            train_fraction=0.7,
            val_fraction=0.15,
            test_fraction=0.15,
            seed=0,
            group_ids=bundle.ids,
        )
        train_groups = {bundle.ids[idx] for idx in train_idx}
        val_groups = {bundle.ids[idx] for idx in val_idx}
        test_groups = {bundle.ids[idx] for idx in test_idx}
        self.assertEqual(len(train_groups & val_groups), 0)
        self.assertEqual(len(train_groups & test_groups), 0)
        self.assertEqual(len(val_groups & test_groups), 0)

    def test_split_fails_when_bundle_has_missing_labels(self) -> None:
        records = [
            {"id": "a", "text": "x", "label": 0},
            {"id": "b", "text": "y"},
            {"id": "c", "text": "z", "label": 1},
        ]
        bundle = ProbingSampleBuilder.from_iterable(records).to_samples(text_key="text")
        with self.assertRaisesRegex(ValueError, "unlabeled"):
            bundle.train_val_test_split()

    def test_public_splitter_handles_minimum_six_samples(self) -> None:
        labels = [0, 0, 0, 1, 1, 1]
        train_idx, val_idx, test_idx = stratified_train_val_test_split(
            labels=labels,
            train_fraction=0.5,
            val_fraction=0.25,
            test_fraction=0.25,
            seed=3,
        )
        self.assertEqual(len(train_idx) + len(val_idx) + len(test_idx), 6)
        for split in (train_idx, val_idx, test_idx):
            split_labels = {labels[idx] for idx in split}
            self.assertEqual(split_labels, {0, 1})

    def test_split_accepts_fraction_values_with_floating_point_noise(self) -> None:
        labels = ([0, 1] * 12)[:24]
        train_fraction = 0.1 + 0.2
        val_fraction = 0.3
        test_fraction = 1.0 - train_fraction - val_fraction
        train_idx, val_idx, test_idx = stratified_train_val_test_split(
            labels=labels,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            seed=5,
        )
        self.assertEqual(len(train_idx) + len(val_idx) + len(test_idx), len(labels))

    def test_group_aware_split_supports_exactly_six_groups(self) -> None:
        labels = [
            1,
            1,
            0,  # g0 positive-majority
            1,
            1,
            0,  # g1 positive-majority
            1,
            1,
            0,  # g2 positive-majority
            0,
            0,
            1,  # g3 negative-majority
            0,
            0,
            1,  # g4 negative-majority
            0,
            0,
            1,  # g5 negative-majority
        ]
        group_ids = [f"g{idx // 3}" for idx in range(len(labels))]
        train_idx, val_idx, test_idx = stratified_train_val_test_split(
            labels=labels,
            train_fraction=0.5,
            val_fraction=0.25,
            test_fraction=0.25,
            seed=11,
            group_ids=group_ids,
        )
        train_groups = {group_ids[idx] for idx in train_idx}
        val_groups = {group_ids[idx] for idx in val_idx}
        test_groups = {group_ids[idx] for idx in test_idx}
        self.assertEqual(len(train_groups & val_groups), 0)
        self.assertEqual(len(train_groups & test_groups), 0)
        self.assertEqual(len(val_groups & test_groups), 0)


if __name__ == "__main__":
    unittest.main()
