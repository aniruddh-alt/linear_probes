"""Utilities to convert arbitrary datasets into probing text samples."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from torch.utils.data import DataLoader, Dataset

Record = dict[str, Any] | str
Preprocessor = Callable[[str], str]


@dataclass
class SampleBundle:
    prompts: Dataset[str]
    labels: list[int | None]
    ids: list[str]


class StringDataset(Dataset[str]):
    """Minimal PyTorch dataset wrapper for string samples."""

    def __init__(self, samples: Sequence[str]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> str:
        return self.samples[index]


class ProbingSampleBuilder:
    """Converts heterogeneous dataset inputs into string samples for probing."""

    def __init__(self, records: Sequence[Record]):
        self.records = list(records)

    @classmethod
    def from_file(cls, path: str | Path) -> "ProbingSampleBuilder":
        file_path = Path(path)
        suffix = file_path.suffix.lower()
        if suffix == ".json":
            data = cls._read_json(file_path)
        elif suffix == ".jsonl":
            data = cls._read_jsonl(file_path)
        elif suffix == ".csv":
            data = cls._read_csv(file_path)
        elif suffix in {".txt", ".md"}:
            data = cls._read_text_lines(file_path)
        else:
            raise ValueError(
                f"Unsupported dataset file type '{suffix}'. "
                "Supported: .json, .jsonl, .csv, .txt, .md"
            )
        return cls(data)

    @classmethod
    def from_iterable(cls, data: Iterable[Record]) -> "ProbingSampleBuilder":
        return cls(list(data))

    def to_samples(
        self,
        *,
        text_key: str | None = None,
        label_key: str | None = "label",
        id_key: str | None = "id",
        include_keys: Sequence[str] | None = None,
        template: str | None = None,
        pair_sep: str = "\n",
        kv_sep: str = ": ",
        strip_whitespace: bool = True,
        preprocess: Preprocessor | None = None,
    ) -> SampleBundle:
        """Create probing-ready prompts with aligned ids and optional labels."""
        prompts: list[str] = []
        labels: list[int | None] = []
        ids: list[str] = []

        for index, record in enumerate(self.records):
            text, label, sample_id = self._record_to_sample(
                record=record,
                index=index,
                text_key=text_key,
                id_key=id_key,
                label_key=label_key,
                include_keys=include_keys,
                template=template,
                pair_sep=pair_sep,
                kv_sep=kv_sep,
            )

            if strip_whitespace:
                text = text.strip()
            if preprocess is not None:
                text = preprocess(text)
            if text:
                prompts.append(text)
                labels.append(label)
                ids.append(sample_id)

        return SampleBundle(prompts=StringDataset(prompts), labels=labels, ids=ids)

    def to_dataloader(
        self,
        *,
        batch_size: int = 8,
        shuffle: bool = False,
        **sample_kwargs: Any,
    ) -> DataLoader:
        bundle = self.to_samples(**sample_kwargs)
        return DataLoader(bundle.prompts, batch_size=batch_size, shuffle=shuffle)

    def _record_to_sample(
        self,
        *,
        record: Record,
        index: int,
        text_key: str | None,
        id_key: str | None,
        label_key: str | None,
        include_keys: Sequence[str] | None,
        template: str | None,
        pair_sep: str,
        kv_sep: str,
    ) -> tuple[str, int | None, str]:
        if isinstance(record, str):
            return record, None, str(index)
        if not isinstance(record, dict):
            return str(record), None, str(index)

        label: int | None = None
        sample_id = str(index)

        if template is not None:
            try:
                text = template.format(**record)
            except KeyError as exc:
                missing = exc.args[0]
                raise KeyError(
                    f"Template references missing key '{missing}' in record: {record}"
                ) from exc
        elif text_key is not None:
            if text_key not in record:
                raise KeyError(f"text_key '{text_key}' not found in record: {record}")
            text = str(record[text_key])
        else:
            fields = include_keys if include_keys is not None else record.keys()
            lines = [f"{key}{kv_sep}{record.get(key, '')}" for key in fields]
            text = pair_sep.join(lines)

        if id_key is not None and id_key in record:
            sample_id = str(record[id_key])

        if label_key is not None and label_key in record:
            label = self._coerce_binary_label(record[label_key])

        return text, label, sample_id

    @staticmethod
    def _coerce_binary_label(value: Any) -> int:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int) and value in (0, 1):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"0", "false", "neg", "negative", "no"}:
                return 0
            if normalized in {"1", "true", "pos", "positive", "yes"}:
                return 1
        raise ValueError(
            f"Label '{value}' is not binary. Expected one of 0/1, true/false, yes/no."
        )

    @staticmethod
    def _read_json(path: Path) -> list[Record]:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            if "data" in data and isinstance(data["data"], list):
                return data["data"]
            return [data]
        return [str(data)]

    @staticmethod
    def _read_jsonl(path: Path) -> list[Record]:
        rows: list[Record] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    @staticmethod
    def _read_csv(path: Path) -> list[Record]:
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    @staticmethod
    def _read_text_lines(path: Path) -> list[Record]:
        with path.open("r", encoding="utf-8") as f:
            return [line.rstrip("\n") for line in f if line.strip()]
