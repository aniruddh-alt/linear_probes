"""Token-position selectors for activation extraction.

Replaces the single-integer ``ExtractionParams.token_index`` with a small
strategy hierarchy. See ``docs/toolkit_audit.md`` §4 for the design rationale.

Every selector implements :class:`TokenSelector` and is applied to a
materialised activation tensor (post-trace) plus the batch's ``input_ids``
and (optionally) ``attention_mask``. It returns a tensor whose first dimension
is still the batch dimension and an optional per-row mask.

Output cardinality

============ =================================== ====================
Selector     What it returns                     Notes
============ =================================== ====================
AllTokens    ``(B, S, D)`` (or full rank-4)      Pass-through; sequence mode
Index        ``(B, D)``                          Single absolute position
Range        ``(B, K, D)`` with ``K`` constant   ``start:stop[:step]``
IndexList    ``(B, K, D)`` with ``K`` constant   Explicit positions
LastNonPad   ``(B, D)``                          Per-row gather, needs mask
TokenIdAnchor``(B, D)``                          Per-row gather, needs ids
StringAnchor ``(B, D)``                          Tokenises anchor at init time
============ =================================== ====================

Selectors that require ``input_ids`` advertise it via ``requires_input_ids``
so the extractor knows whether to ``.save()`` the model's input id proxy
inside the trace.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import torch

# ────────────────────────────── Protocol ──────────────────────────────


@runtime_checkable
class TokenSelector(Protocol):
    """Selects a subset of token positions from an activation tensor.

    Implementations should:
      - Accept an activation of shape ``(B, S, D)`` (hidden states) or
        ``(B, H, S, S)`` (attention probabilities — typically raise unless
        explicitly supported).
      - Return ``(selected_activation, optional_per_row_mask)`` where
        ``selected_activation``'s first dim is still the batch dim.
    """

    requires_input_ids: bool

    def select(
        self,
        activation: torch.Tensor,
        *,
        kind: str,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        ...


# ────────────────────────────── Helpers ──────────────────────────────


def _ensure_3d(activation: torch.Tensor, kind: str, selector_name: str) -> None:
    if activation.ndim != 3:
        raise ValueError(
            f"{selector_name} expected a rank-3 activation (B, S, D) for kind "
            f"'{kind}', got shape {tuple(activation.shape)}."
        )


def _resolve_index(idx: int, seq_len: int, *, where: str) -> int:
    """Convert a possibly-negative absolute index to a non-negative one and validate."""
    resolved = seq_len + idx if idx < 0 else idx
    if resolved < 0 or resolved >= seq_len:
        raise IndexError(
            f"{where}: index {idx} is out of range for sequence length {seq_len}."
        )
    return resolved


# ────────────────────────────── Concrete selectors ──────────────────────────────


@dataclass
class AllTokens:
    """No-op selector — keep all positions. Triggers sequence-mode storage."""

    requires_input_ids: bool = False

    def select(
        self,
        activation: torch.Tensor,
        *,
        kind: str,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return activation, attention_mask


@dataclass
class Index:
    """Pick a single absolute position. Replaces today's ``token_index: int``."""

    index: int
    requires_input_ids: bool = False

    def select(
        self,
        activation: torch.Tensor,
        *,
        kind: str,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if activation.ndim == 4 and kind == "attention_probabilities":
            return activation[:, :, self.index, :], None
        _ensure_3d(activation, kind, "Index")
        return activation[:, self.index, :], None


@dataclass
class Range:
    """Slice of positions ``[start:stop:step]``. Returns ``(B, K, D)``."""

    start: int | None = None
    stop: int | None = None
    step: int = 1
    requires_input_ids: bool = False

    def __post_init__(self) -> None:
        if self.step == 0:
            raise ValueError("Range.step must be non-zero.")

    def select(
        self,
        activation: torch.Tensor,
        *,
        kind: str,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _ensure_3d(activation, kind, "Range")
        sliced = activation[:, slice(self.start, self.stop, self.step), :]
        if sliced.shape[1] == 0:
            raise ValueError(
                f"Range produced an empty selection for sequence length "
                f"{activation.shape[1]} with start={self.start}, stop={self.stop}, "
                f"step={self.step}."
            )
        sub_mask: torch.Tensor | None = None
        if attention_mask is not None:
            sub_mask = attention_mask[:, slice(self.start, self.stop, self.step)]
        return sliced, sub_mask


@dataclass
class IndexList:
    """Explicit list of positions, e.g. ``[0, 5, -1]``. Returns ``(B, K, D)``."""

    indices: list[int]
    requires_input_ids: bool = False

    def __post_init__(self) -> None:
        if not self.indices:
            raise ValueError("IndexList.indices must be non-empty.")

    def select(
        self,
        activation: torch.Tensor,
        *,
        kind: str,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _ensure_3d(activation, kind, "IndexList")
        seq_len = activation.shape[1]
        resolved = [
            _resolve_index(i, seq_len, where="IndexList") for i in self.indices
        ]
        idx = torch.tensor(resolved, device=activation.device, dtype=torch.long)
        gathered = activation.index_select(dim=1, index=idx)
        sub_mask: torch.Tensor | None = None
        if attention_mask is not None:
            sub_mask = attention_mask.index_select(dim=1, index=idx)
        return gathered, sub_mask


@dataclass
class LastNonPad:
    """Per-row last non-pad position, gathered from the attention mask.

    Either ``attention_mask`` must be provided or ``input_ids`` plus
    ``pad_token_id`` so a mask can be derived on the fly.
    """

    pad_token_id: int | None = None
    requires_input_ids: bool = True

    def select(
        self,
        activation: torch.Tensor,
        *,
        kind: str,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _ensure_3d(activation, kind, "LastNonPad")
        mask = attention_mask
        if mask is None:
            if input_ids is None or self.pad_token_id is None:
                raise ValueError(
                    "LastNonPad requires either `attention_mask` or both `input_ids` "
                    "and `pad_token_id`."
                )
            mask = (input_ids != self.pad_token_id).long()
        if mask.ndim != 2 or mask.shape[0] != activation.shape[0]:
            raise ValueError(
                f"LastNonPad attention_mask shape {tuple(mask.shape)} is incompatible "
                f"with activation batch shape {tuple(activation.shape[:2])}."
            )
        lengths = mask.long().sum(dim=1)
        if (lengths == 0).any():
            raise ValueError("LastNonPad: every row must have at least one non-pad position.")
        last_idx = (lengths - 1).clamp(min=0)
        gather_idx = last_idx.view(-1, 1, 1).expand(-1, 1, activation.shape[-1])
        gathered = activation.gather(dim=1, index=gather_idx).squeeze(1)
        return gathered, None


@dataclass
class TokenIdAnchor:
    """Find an occurrence of a token-id pattern per row, then offset.

    Args:
        pattern: list of token ids to match contiguously.
        offset:  position relative to the *start* of the matched pattern
                 (e.g. ``0`` = first token of the match,
                 ``len(pattern)`` = first token *after* the match).
        mode:    ``"first"`` or ``"last"``.

    Example:
        Resolve "the first generated token" given an instruction template that
        ends with ``[/INST]``::

            anchor = TokenIdAnchor(pattern=tokenizer.encode("[/INST]",
                                  add_special_tokens=False),
                                  offset=len(pattern), mode="first")
    """

    pattern: list[int]
    offset: int = 0
    mode: str = "first"
    requires_input_ids: bool = True

    def __post_init__(self) -> None:
        if not self.pattern:
            raise ValueError("TokenIdAnchor.pattern must be non-empty.")
        if self.mode not in ("first", "last"):
            raise ValueError(
                f"TokenIdAnchor.mode must be 'first' or 'last', got {self.mode!r}."
            )

    def select(
        self,
        activation: torch.Tensor,
        *,
        kind: str,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _ensure_3d(activation, kind, "TokenIdAnchor")
        if input_ids is None:
            raise ValueError("TokenIdAnchor requires `input_ids`.")
        if input_ids.ndim != 2 or input_ids.shape[0] != activation.shape[0]:
            raise ValueError(
                f"TokenIdAnchor input_ids shape {tuple(input_ids.shape)} is "
                f"incompatible with activation batch {tuple(activation.shape[:2])}."
            )

        positions = self._find_positions(input_ids)
        seq_len = activation.shape[1]
        target = positions + self.offset
        if (target < 0).any() or (target >= seq_len).any():
            raise IndexError(
                f"TokenIdAnchor: anchor + offset went out of range for sequence "
                f"length {seq_len}. positions={positions.tolist()}, offset={self.offset}."
            )
        gather_idx = target.view(-1, 1, 1).expand(-1, 1, activation.shape[-1])
        gathered = activation.gather(dim=1, index=gather_idx).squeeze(1)
        return gathered, None

    def _find_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return the start position of the chosen pattern occurrence per row."""
        pattern = torch.tensor(
            self.pattern, device=input_ids.device, dtype=input_ids.dtype
        )
        plen = pattern.numel()
        seq_len = input_ids.shape[1]
        if plen > seq_len:
            raise ValueError(
                f"TokenIdAnchor: pattern of length {plen} is longer than "
                f"sequence length {seq_len}."
            )
        windows = input_ids.unfold(dimension=1, size=plen, step=1)
        matches = (windows == pattern).all(dim=2)  # (B, S - plen + 1)
        if not matches.any(dim=1).all():
            missing = (~matches.any(dim=1)).nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(
                f"TokenIdAnchor: pattern {self.pattern} not found in rows {missing}."
            )
        if self.mode == "first":
            return matches.float().argmax(dim=1)
        flipped = matches.flip(dims=[1])
        last_from_end = flipped.float().argmax(dim=1)
        last_pos_index = matches.shape[1] - 1 - last_from_end
        return last_pos_index


@dataclass
class StringAnchor(TokenIdAnchor):
    """Convenience: tokenises ``anchor`` once at init time, then behaves as
    :class:`TokenIdAnchor`. Pass any object exposing ``encode(str, add_special_tokens=...)``.

    Note: this dataclass is **not** YAML-friendly because it requires a
    tokenizer at construction time. Use ``TokenIdAnchor`` from YAML and
    construct ``StringAnchor`` programmatically.
    """

    anchor: str = ""

    def __init__(
        self,
        anchor: str,
        tokenizer: Any,
        *,
        offset: int = 0,
        mode: str = "first",
        add_special_tokens: bool = False,
    ) -> None:
        if not anchor:
            raise ValueError("StringAnchor.anchor must be non-empty.")
        if tokenizer is None or not hasattr(tokenizer, "encode"):
            raise ValueError(
                "StringAnchor requires a tokenizer with an `encode` method."
            )
        pattern = list(
            tokenizer.encode(anchor, add_special_tokens=add_special_tokens)
        )
        super().__init__(pattern=pattern, offset=offset, mode=mode)
        self.anchor = anchor


__all__ = [
    "AllTokens",
    "Index",
    "IndexList",
    "LastNonPad",
    "Range",
    "StringAnchor",
    "TokenIdAnchor",
    "TokenSelector",
]
