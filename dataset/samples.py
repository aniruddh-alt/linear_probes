"""Utilities to convert arbitrary datasets into probing text samples."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from torch.utils.data import DataLoader

Record = dict[str, Any] | str
Preprocessor = Callable[[str], str]


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
        include_keys: Sequence[str] | None = None,
        template: str | None = None,
        pair_sep: str = "\n",
        kv_sep: str = ": ",
        strip_whitespace: bool = True,
        preprocess: Preprocessor | None = None,
    ) -> list[str]:
        """Create probing strings from records.

        Rules:
        - string records are used directly
        - dict records:
          - if `template` is set, use `template.format(**record)`
          - elif `text_key` is set, use `record[text_key]`
          - else serialize selected keys (or all keys) as key-value lines
        """
        out: list[str] = []
        for record in self.records:
            text = self._record_to_text(
                record=record,
                text_key=text_key,
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
                out.append(text)
        return out

    def to_dataloader(
        self,
        *,
        batch_size: int = 8,
        shuffle: bool = False,
        **sample_kwargs: Any,
    ) -> DataLoader:
        samples = self.to_samples(**sample_kwargs)
        return DataLoader(samples, batch_size=batch_size, shuffle=shuffle)

    def _record_to_text(
        self,
        *,
        record: Record,
        text_key: str | None,
        include_keys: Sequence[str] | None,
        template: str | None,
        pair_sep: str,
        kv_sep: str,
    ) -> str:
        if isinstance(record, str):
            return record
        if not isinstance(record, dict):
            return str(record)

        if template is not None:
            try:
                return template.format(**record)
            except KeyError as exc:
                missing = exc.args[0]
                raise KeyError(
                    f"Template references missing key '{missing}' in record: {record}"
                ) from exc

        if text_key is not None:
            if text_key not in record:
                raise KeyError(f"text_key '{text_key}' not found in record: {record}")
            return str(record[text_key])

        if include_keys is not None:
            fields = include_keys
        else:
            fields = record.keys()

        lines = [f"{key}{kv_sep}{record.get(key, '')}" for key in fields]
        return pair_sep.join(lines)

    @staticmethod
    def _read_json(path: Path) -> list[Record]:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            # common pattern: {"data": [...]}
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
