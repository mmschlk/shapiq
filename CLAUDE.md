# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`shapiq` is a Python library for computing Shapley Interactions for Machine Learning. It approximates any-order Shapley interactions, benchmarks game-theoretical algorithms, and explains feature interactions in model predictions. The repo contains two importable packages: `shapiq` (core) and `shapiq_games` (benchmark games).

## Development Setup

This project uses `uv` for package management.

```sh
# Install dev dependencies (test + lint + all_ml)
uv sync

# Install only test dependencies
uv sync --group test

# Install lint tools
uv sync --group lint
```

## Commands

### Testing

```sh
# Run all shapiq unit tests
uv run pytest tests/shapiq

# Run all shapiq_games tests (parallel)
uv run pytest tests/shapiq_games -n logical

# Run a single test file
uv run pytest tests/shapiq/tests_unit/test_interaction_values.py

# Run with coverage
uv run pytest tests/shapiq --cov=shapiq --cov-report=xml -n logical
```

### Linting and Code Quality

```sh
# Run all pre-commit hooks (ruff lint + format + ty)
uv run pre-commit run --all-files

# Run ruff linter only
uv run ruff check src/ tests/ --fix

# Run ruff formatter only
uv run ruff format src/ tests/

# Run type checking
uv run ty check
```

### Documentation

```sh
uv sync --no-dev --group docs
uv run sphinx-build -b html docs/source docs/build/html
```

## Code Architecture

### Package Structure

```
src/
├── shapiq/              # Core package
│   ├── interaction_values.py   # InteractionValues data class (central output type)
│   ├── game.py                 # Base Game class for cooperative games
│   ├── approximator/           # Approximation algorithms
│   │   ├── base.py             # Approximator base class
│   │   ├── marginals/          # Owen, Stratified sampling
│   │   ├── montecarlo/         # SHAPIQ, SVARM, SVARMIQ, UnbiasedKernelSHAP
│   │   ├── permutation/        # Permutation sampling for SII, STII, SV
│   │   ├── regression/         # KernelSHAP, KernelSHAPIQ, RegressionFSII/FBII
│   │   └── sparse/             # SPEX, ProxySPEX (for large feature spaces)
│   ├── explainer/              # High-level explainer interfaces
│   │   ├── tabular.py          # TabularExplainer (main user-facing class)
│   │   └── product_kernel/     # ProductKernelExplainer
│   ├── tree/                   # TreeExplainer and tree model conversions (top-level)
│   │   ├── explainer.py        # TreeExplainer class
│   │   ├── treeshapiq.py       # TreeSHAPIQ algorithm
│   │   ├── base.py             # TreeModel and EdgeTree dataclasses
│   │   ├── validation.py       # SUPPORTED_MODELS, validate_tree_model
│   │   ├── utils.py            # Tree utility functions
│   │   ├── conversion/         # Model-specific converters (sklearn, xgboost, lightgbm)
│   │   │   ├── common.py       # lazydispatch registry, convert_tree_model
│   │   │   ├── sklearn.py      # sklearn DT, RF, ExtraTrees, IsolationForest
│   │   │   ├── boosting.py     # XGBoost and LightGBM converters
│   │   │   ├── edges.py        # create_edge_tree for EdgeTree format
│   │   │   └── cext/           # C++ extension for fast XGBoost/LightGBM parsing
│   │   └── interventional/     # Interventional tree explainer
│   ├── imputer/                # Imputation strategies for missing features
│   │   ├── marginal_imputer.py # MarginalImputer (most common)
│   │   ├── baseline_imputer.py
│   │   ├── gaussian_imputer.py
│   │   └── tabpfn_imputer.py
│   ├── game_theory/            # Mathematical game-theory utilities
│   │   ├── exact.py            # ExactComputer for exact interaction values
│   │   ├── indices.py          # ALL_AVAILABLE_CONCEPTS index registry
│   │   ├── moebius_converter.py
│   │   └── aggregation.py
│   ├── plot/                   # Visualization functions
│   └── utils/                  # Shared utilities (sets, saving, typing)
└── shapiq_games/        # Benchmark games package (separate from shapiq)
    ├── benchmark/       # Pre-defined benchmark games per use-case
    ├── synthetic/       # Synthetic game functions
    └── tabular/         # Tabular ML games
```

### Core Data Flow

1. **Game** (`game.py`): Wraps any callable as a cooperative game. Subclasses implement `value_function(coalitions) -> np.ndarray`. Takes a boolean coalition matrix and returns scalar game values.

2. **Approximator** (`approximator/`): Takes a `Game` and a budget, calls `approximate(budget, game)` → returns `InteractionValues`. All approximators inherit from `Approximator` base class.

3. **InteractionValues** (`interaction_values.py`): Central data class storing interaction scores as a numpy array with an `interaction_lookup` dict mapping coalition tuples → array indices. Supports arithmetic operations between instances.

4. **Explainer** (`explainer/`): High-level interface combining an ML model + data + an `Imputer` into a `Game`, then calling an `Approximator`. `Explainer.explain(x)` → `InteractionValues`.

5. **Imputer** (`imputer/`): Converts ML model + data into a game by handling missing features. `MarginalImputer` is the default for tabular data.

### Interaction Indices

Available indices are defined in `game_theory/indices.py` (`ALL_AVAILABLE_CONCEPTS`). Key ones:
- `SV` – Shapley Values (order 1 only)
- `SII` – Shapley Interaction Index
- `k-SII` – k-Shapley Interaction Index (most common for explanations)
- `STII` – Shapley-Taylor Interaction Index
- `FSII` – Faithful Shapley Interaction Index
- `FBII` – Faithful Banzhaf Interaction Index
- `BV` – Banzhaf Values

### Code Style

- **Formatter/Linter**: `ruff` with `black` style, line length 100, Google-style docstrings
- **Type checking**: `ty` (checks `src/shapiq/`, excluded for tests)
- **All files** must start with `from __future__ import annotations`
- `isort` is configured with `required-imports = ["from __future__ import annotations"]`
- Variable names `X` (uppercase) in functions are allowed (common in ML code)
- Test files live in `tests/shapiq/` and `tests/shapiq_games/` with separate conftest files

### Test Organization

- `tests/shapiq/tests_unit/` – Unit tests per module
- `tests/shapiq/tests_integration_tests/` – Integration tests
- `tests/shapiq/tests_deprecation/` – Deprecation behavior tests
- `tests/shapiq/fixtures/` – Shared pytest fixtures (data, games, models, interaction values)
- `tests/shapiq_games/` – Tests for the `shapiq_games` package

### Two-Package Setup

The repo hosts two installable packages:
- `shapiq` in `src/shapiq/` — the core library
- `shapiq_games` in `src/shapiq_games/` — optional benchmark games requiring extra ML dependencies (`torch`, `transformers`, `tabpfn`)

`shapiq_games` requires `uv sync --group all_ml` for full functionality.
