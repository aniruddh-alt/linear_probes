"""Dataset loading configuration parameters."""

from __future__ import annotations

from dataclasses import dataclass, field

from core.configs.base import BaseConfig


@dataclass
class DatasetParams(BaseConfig):
    """Configuration for loading a HuggingFace dataset."""

    path: str = ""
    config: str = ""
    split: str = "train"
    text_key: str = "text"
    label_key: str = "label"
    id_key: str = "id"
    label_map: dict[str, int] = field(default_factory=dict)
    max_samples: int | None = None
    ood_configs: list[str] = field(default_factory=list)
    ood_split: str = "test"

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("dataset path must be non-empty.")
