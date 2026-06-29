"""OmegaConf resolvers for sonde configs.

Registers ``${sonde_pkg:<relative>}`` which expands to an absolute path inside
the packaged ``sonde/configs`` directory, so bundled recipes can reference
bundled data without depending on the current working directory.
"""

from __future__ import annotations

from omegaconf import OmegaConf

from sonde.core.configs.aliases import _packaged_configs_dir


def _sonde_pkg(relative: str) -> str:
    return str((_packaged_configs_dir() / relative).resolve())


def register_resolvers() -> None:
    """Idempotently register sonde's OmegaConf resolvers."""
    OmegaConf.register_new_resolver("sonde_pkg", _sonde_pkg, replace=True)


register_resolvers()
