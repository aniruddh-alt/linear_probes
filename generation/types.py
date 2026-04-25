"""Data types for the generation module."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from dataset.samples import StringDataset
from dataset.types import SampleBundle


@dataclass
class GenerationResult:
    """Holds prompt-response pairs with optional labels."""

    prompts: list[str]
    responses: list[str]
    sample_ids: list[str]
    labels: list[int | None]

    def __post_init__(self) -> None:
        n = len(self.prompts)
        if len(self.responses) != n:
            raise ValueError(
                f"prompts ({n}) and responses ({len(self.responses)}) must have equal length."
            )
        if len(self.sample_ids) != n:
            raise ValueError(
                f"prompts ({n}) and sample_ids ({len(self.sample_ids)}) must have equal length."
            )
        if len(self.labels) != n:
            raise ValueError(
                f"prompts ({n}) and labels ({len(self.labels)}) must have equal length."
            )

    def to_jsonl(self, path: str | Path) -> None:
        """Export prompt-response pairs to JSONL for external labeling."""
        with Path(path).open("w", encoding="utf-8") as f:
            for prompt, response, sid, label in zip(
                self.prompts, self.responses, self.sample_ids, self.labels, strict=True
            ):
                row = {
                    "prompt": prompt,
                    "response": response,
                    "sample_id": sid,
                    "label": label,
                }
                f.write(json.dumps(row) + "\n")

    @classmethod
    def from_jsonl(cls, path: str | Path) -> GenerationResult:
        """Load a GenerationResult from a JSONL file."""
        prompts, responses, sample_ids, labels = [], [], [], []
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                prompts.append(row["prompt"])
                responses.append(row["response"])
                sample_ids.append(row["sample_id"])
                labels.append(row.get("label"))
        return cls(
            prompts=prompts,
            responses=responses,
            sample_ids=sample_ids,
            labels=labels,
        )

    def to_sample_bundle(self, *, include_response: bool = False) -> SampleBundle:
        """Convert to SampleBundle for activation extraction.

        Args:
            include_response: If True, prompts become prompt+response
                concatenated. If False, original prompts only.
        """
        if include_response:
            texts = [
                f"{p}{r}" for p, r in zip(self.prompts, self.responses, strict=True)
            ]
        else:
            texts = list(self.prompts)
        return SampleBundle(
            prompts=StringDataset(texts),
            labels=list(self.labels),
            ids=list(self.sample_ids),
            responses=list(self.responses),
        )
