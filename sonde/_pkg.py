"""Package metadata helpers shared across sonde."""

from __future__ import annotations


def sonde_version() -> str:
    """Return the installed sonde version, or ``"unknown"`` if undeterminable."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("sonde")
    except PackageNotFoundError:
        return "unknown"


__all__ = ["sonde_version"]
