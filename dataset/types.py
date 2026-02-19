"""Shared data types for the dataset package."""

from __future__ import annotations

from dataclasses import dataclass

from torch.utils.data import Dataset


@dataclass
class SampleBundle:
    prompts: Dataset[str]
    labels: list[int | None]
    ids: list[str]
