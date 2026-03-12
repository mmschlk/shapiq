# AGENTS.md

Guidance for Claude Code when working in this repository.

## Project Overview

`shapiq` is a Python library for computing Shapley Interactions for Machine Learning. The repo hosts **two separate installable packages**:
- `shapiq` in `src/shapiq/` — the core library
- `shapiq_games` in `src/shapiq_games/` — optional benchmark games (requires extra ML deps via `uv sync --group all_ml`)

## Commands

```sh
# Setup
uv sync                        # dev dependencies (test + lint + all_ml)

# Testing
uv run pytest tests/shapiq
uv run pytest tests/shapiq_games -n logical
uv run pytest tests/shapiq/tests_unit/test_interaction_values.py  # single file

# Quality (run before every commit)
uv run pre-commit run --all-files
uv run ty check                # type check src/shapiq/ only

# Docs
uv run sphinx-build -b html docs/source docs/build/html
```

## Package Structure

```
src/
├── shapiq/
│   ├── interaction_values.py       # InteractionValues — central output type
│   ├── game.py                     # Base Game class
│   ├── approximator/               # Approximation algorithms (base.py + subdirs)
│   ├── explainer/
│   │   ├── tabular.py              # TabularExplainer — main user-facing class
│   │   ├── tree/
│   │   └── product_kernel/
│   ├── imputer/                    # MarginalImputer is the default
│   ├── game_theory/
│   │   ├── exact.py
│   │   ├── indices.py              # ALL_AVAILABLE_CONCEPTS — index registry
│   │   └── moebius_converter.py
│   ├── plot/
│   └── utils/
└── shapiq_games/
    ├── benchmark/
    ├── synthetic/
    └── tabular/
```

## Code Style

- **All files** must start with `from __future__ import annotations` — `isort` enforces this
- `ruff` with `black` style, line length 100, Google-style docstrings
- Uppercase `X` in function signatures is allowed (ML convention)

## Key Rules

- **Do not** add ML-heavy dependencies to `shapiq` core — they belong in `shapiq_games`
- Use existing fixtures in `tests/shapiq/fixtures/` rather than duplicating setup
- `ty` type checking runs on `src/shapiq/` only; tests are excluded
