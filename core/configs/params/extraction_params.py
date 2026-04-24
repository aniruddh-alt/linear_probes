"""Activation extraction configuration parameters."""

from __future__ import annotations

from dataclasses import dataclass, field

from core.configs.base import BaseConfig
from core.configs.params.token_selector_params import TokenSelectorParams


@dataclass
class ExtractionParams(BaseConfig):
    """Configuration for activation extraction runs.

    Token positions are resolved in the following order:

      1. ``token_selector`` (if set) — full :class:`TokenSelectorParams` API.
      2. ``token_index`` — back-compat sugar:
         - ``None``      -> equivalent to ``TokenSelectorParams(type="all")``
         - ``int``       -> equivalent to ``TokenSelectorParams(type="index", index=...)``

    See ``activation/token_selectors.py`` for the runtime selector hierarchy
    and ``docs/toolkit_audit.md`` §4 for the design rationale.
    """

    save_path: str = ""
    activations: list[str] = field(default_factory=list)
    batch_size: int = 8
    token_index: int | None = -1
    token_selector: TokenSelectorParams | None = None
    remote: bool = False
    to_cpu: bool = True
