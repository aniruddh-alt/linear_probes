# Config Migration Guide

## Overview

The linear probes toolkit is migrating from centralized config classes in `configs/types.py`
to domain-specific OmegaConf-powered configs under `core/configs/`.

## What Changed

### New Config Location

| Old Import | New Import |
|-----------|-----------|
| `configs.ModelConfig` | `core.configs.ModelParams` |
| `configs.ProbeConfig` | `core.configs.ProbeParams` |
| `configs.ActivationConfig` | `core.configs.ExtractionParams` |
| `configs.LayerProbeSweepConfig` | `core.configs.RunConfig` (composed) |

### New Features

- **YAML round-trip**: All configs support `to_yaml()` / `from_yaml()`
- **Dot-notation overrides**: CLI overrides like `probe.learning_rate=0.01`
- **Config aliases**: Friendly names for common config recipes
- **Composed configs**: `RunConfig` composes model, extraction, probe, split, and output params

### Backward Compatibility

Old imports continue to work:

```python
# Still works
from configs import ProbeConfig, ModelConfig

# New way
from core.configs import ProbeParams, ModelParams, RunConfig
```

## Migration Steps

1. Replace `configs.ProbeConfig` with `core.configs.ProbeParams`
2. Replace `configs.ModelConfig` with `core.configs.ModelParams`
3. Use `RunConfig.from_yaml("path/to/config.yaml")` for YAML-driven runs
4. Use dot-notation overrides for CLI parameter changes
