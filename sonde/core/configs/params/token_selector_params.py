"""YAML-serialisable description of a token selector.

Runtime selectors live in :mod:`activation.token_selectors`. This module
exposes a flat dataclass that OmegaConf can round-trip and a factory that
builds the concrete :class:`~activation.token_selectors.TokenSelector`
instance.

Example YAML::

    extraction:
      token_selector:
        type: token_id_anchor
        pattern: [29914, 25580, 29962]   # tokenized "[/INST]"
        offset: 3
        mode: first
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sonde.core.configs.base import BaseConfig

if TYPE_CHECKING:
    from sonde.activation.token_selectors import TokenSelector

_VALID_TYPES = frozenset(
    {
        "all",
        "index",
        "range",
        "index_list",
        "last_non_pad",
        "token_id_anchor",
    }
)


@dataclass
class TokenSelectorParams(BaseConfig):
    """Discriminated description of a TokenSelector.

    The ``type`` field selects which other fields are read; ``__post_init__``
    enforces the per-type field combinations. ``string_anchor`` is *not*
    representable here because it requires a tokenizer at construction time —
    use ``token_id_anchor`` from YAML and resolve the pattern offline, or
    construct ``StringAnchor`` programmatically.
    """

    type: str = "index"
    index: int = -1
    start: int | None = None
    stop: int | None = None
    step: int = 1
    indices: list[int] = field(default_factory=list)
    pattern: list[int] = field(default_factory=list)
    offset: int = 0
    mode: str = "first"
    pad_token_id: int | None = None

    def __post_init__(self) -> None:
        if self.type not in _VALID_TYPES:
            raise ValueError(
                f"Unknown token_selector type '{self.type}'. "
                f"Supported: {sorted(_VALID_TYPES)}."
            )
        if self.type == "range" and self.step == 0:
            raise ValueError("token_selector type=range requires step != 0.")
        if self.type == "index_list" and not self.indices:
            raise ValueError("token_selector type=index_list requires `indices`.")
        if self.type == "token_id_anchor":
            if not self.pattern:
                raise ValueError(
                    "token_selector type=token_id_anchor requires non-empty `pattern`."
                )
            if self.mode not in ("first", "last"):
                raise ValueError(
                    "token_selector type=token_id_anchor: mode must be 'first' or 'last'."
                )

    def build(self) -> TokenSelector:
        """Construct the concrete :class:`TokenSelector` runtime object."""
        # Local import to avoid circulars at module import time.
        from sonde.activation.token_selectors import (
            AllTokens,
            Index,
            IndexList,
            LastNonPad,
            Range,
            TokenIdAnchor,
        )

        if self.type == "all":
            return AllTokens()
        if self.type == "index":
            return Index(index=self.index)
        if self.type == "range":
            return Range(start=self.start, stop=self.stop, step=self.step)
        if self.type == "index_list":
            return IndexList(indices=list(self.indices))
        if self.type == "last_non_pad":
            return LastNonPad(pad_token_id=self.pad_token_id)
        if self.type == "token_id_anchor":
            return TokenIdAnchor(
                pattern=list(self.pattern),
                offset=self.offset,
                mode=self.mode,
            )
        raise AssertionError(f"unreachable: type={self.type!r}")


__all__ = ["TokenSelectorParams"]
