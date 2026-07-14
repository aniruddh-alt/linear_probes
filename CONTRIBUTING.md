# Contributing to sonde

Thanks for wanting to poke at the hidden layers with us. This guide covers the short loop for getting a change merged.

## Dev setup

sonde uses [uv](https://docs.astral.sh/uv/) for environment management.

```bash
git clone https://github.com/aniruddh-alt/sonde.git
cd sonde
uv venv
uv pip install -e ".[dev]"
```

Optional — enable pre-commit hooks so lint/format run on every commit:

```bash
uv pip install pre-commit
pre-commit install
```

## The inner loop

```bash
# format + lint
ruff format .
ruff check --fix .

# type check
pyright

# tests
pytest
```

CI runs all four on every PR. Keep them green locally before pushing.

## Style

- **Ruff** handles formatting and lint. Config lives in `pyproject.toml` under `[tool.ruff]`. Don't fight the formatter.
- **Pyright** runs in `basic` mode. Add type hints on public APIs (anything exported from a package `__init__.py`).
- **Imports**: ruff's isort rule groups them. First-party packages are declared in `pyproject.toml`.
- **Tests**: put new tests in `tests/` mirroring the package layout. Prefer small, focused tests over large end-to-end ones.

## Commit + PR flow

1. Branch off `main`: `git checkout -b <kind>/<short-slug>` (kinds: `feat`, `fix`, `chore`, `docs`, `refactor`).
2. Keep commits atomic — one logical change each. Conventional-commit-style messages are appreciated but not enforced.
3. Open a PR against `main`. Describe **what** changed and **why**; link any related issue.
4. Make sure CI passes. Address review comments with follow-up commits (don't force-push after review has started unless asked).

## Adding a new probe architecture

1. Subclass `BaseProbe` in `sonde/probes/architectures/`.
2. Register it in `sonde/probes/architectures/__init__.py`.
3. Add a test in `tests/test_probe_architectures.py` — at minimum, verify it trains and returns metrics on a trivial 2-class dataset.

## Adding a new activation target

Activation targets are parsed in `sonde/activation/`. When adding a new selector syntax or module-hook kind:

1. Update the parser + the docstring / README selector table.
2. Cover the new syntax in `tests/test_activation_extractor.py` or a sibling test file.

## Reporting bugs

Open an issue with: model + revision you were probing, a minimal config or snippet, and the traceback or unexpected output. Reproducibility beats thoroughness.
