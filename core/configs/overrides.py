"""Dot-notation override merging for config dictionaries."""

from __future__ import annotations

from omegaconf import OmegaConf


def apply_dot_overrides(
    config_dict: dict, overrides: dict[str, str]
) -> dict:
    """Apply dot-notation overrides to a config dictionary.

    Parses string values into appropriate Python types (bool, int, float, None)
    and merges them into nested config paths.

    Args:
        config_dict: Base configuration dictionary.
        overrides: Mapping of dot-separated keys to string values.

    Returns:
        Updated config dictionary with overrides applied.
    """
    if not overrides:
        return config_dict

    cfg = OmegaConf.create(config_dict)
    dotlist = [f"{k}={v}" for k, v in overrides.items()]
    override_cfg = OmegaConf.from_dotlist(dotlist)
    merged = OmegaConf.merge(cfg, override_cfg)
    return OmegaConf.to_container(merged, resolve=True)  # type: ignore[return-value]
