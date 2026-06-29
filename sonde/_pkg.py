"""Package metadata helpers shared across sonde."""

from __future__ import annotations


def sonde_version() -> str:
    """Return the installed sonde version, or ``"unknown"`` if undeterminable."""
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("sonde")
        except PackageNotFoundError:
            return "unknown"
    except Exception:  # pragma: no cover - defensive
        return "unknown"


__all__ = ["sonde_version"]
