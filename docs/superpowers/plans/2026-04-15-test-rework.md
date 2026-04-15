# Test Rework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 75-file shapiq test suite with a protocol-driven 8-file suite that runs in ~1 minute and makes adding new components trivial.

**Architecture:** Protocol-based testing where each component type (approximator, explainer, imputer, tree) has a registry of configs and a set of universal contract checks. Special cases live as `TestClass` groups in the same file. Two tiers: default (fast, no optional deps) and `@pytest.mark.slow` (CI, optional deps).

**Tech Stack:** pytest, numpy, shapiq, shapiq_games (DummyGame, SOUM)

---

## File Structure

| File | Responsibility |
|------|----------------|
| `tests/shapiq/conftest.py` | Shared fixtures: games, tiny datasets, model factories, skip markers |
| `tests/shapiq/test_approximators.py` | Approximator protocol + special-case TestClasses |
| `tests/shapiq/test_explainers.py` | Explainer protocol + special-case TestClasses |
| `tests/shapiq/test_tree.py` | Tree protocol + SHAP comparison (slow) + edge cases |
| `tests/shapiq/test_imputers.py` | Imputer protocol |
| `tests/shapiq/test_interaction_values.py` | InteractionValues data structure tests |
| `tests/shapiq/test_game_theory.py` | ExactComputer, indices, Moebius converter, core |
| `tests/shapiq/test_plots.py` | Plot smoke tests |
| `tests/shapiq/test_public_api.py` | Public API surface checks |

---

### Task 1: Configure pytest tiering and clean up pyproject.toml

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Update pytest configuration**

In `pyproject.toml`, update the `[tool.pytest.ini_options]` section:

```toml
[tool.pytest.ini_options]
testpaths = [
  "tests/shapiq",
  "tests/shapiq_games"
]
pythonpath = ["src"]
minversion = "8.0"
markers = [
    "slow: marks tests that require > 5s or optional deps (deselect with '-m not slow')",
]
addopts = "-m 'not slow'"
```

- [ ] **Step 2: Verify config parses**

Run: `uv run pytest --collect-only -q tests/shapiq 2>&1 | tail -5`

Expected: Tests are collected (from old suite still present). No config parse errors.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "Configure pytest slow marker and default addopts for test tiering"
```

---

### Task 2: Write conftest.py with shared fixtures

**Files:**
- Create: `tests/shapiq/conftest_new.py` (temporary name to avoid conflict with old conftest)

We use a temporary name during migration. Task 9 renames it.

- [ ] **Step 1: Write the shared conftest**

Create `tests/shapiq/conftest_new.py`:

```python
"""Shared fixtures for all shapiq tests."""

from __future__ import annotations

import os

# Limit OpenMP threads to prevent segfaults when PyTorch/sklearn coexist.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import importlib.util

import matplotlib as mpl
import numpy as np
import pytest

mpl.use("Agg")

from shapiq.game_theory.exact import ExactComputer
from shapiq_games.synthetic import DummyGame

# ---------------------------------------------------------------------------
# Skip markers for optional dependencies
# ---------------------------------------------------------------------------


def _is_installed(pkg: str) -> bool:
    return importlib.util.find_spec(pkg) is not None


skip_if_no_xgboost = pytest.mark.skipif(
    not _is_installed("xgboost"), reason="xgboost not installed"
)
skip_if_no_lightgbm = pytest.mark.skipif(
    not _is_installed("lightgbm"), reason="lightgbm not installed"
)
skip_if_no_tabpfn = pytest.mark.skipif(
    not _is_installed("tabpfn"), reason="tabpfn not installed"
)

# ---------------------------------------------------------------------------
# Games
# ---------------------------------------------------------------------------


@pytest.fixture
def dummy_game_3():
    """3-player DummyGame with interaction (0, 1). Fast and deterministic."""
    return DummyGame(n=3, interaction=(0, 1))


@pytest.fixture
def dummy_game_7():
    """7-player DummyGame with interaction (1, 2). Used by approximator protocol."""
    return DummyGame(n=7, interaction=(1, 2))


# ---------------------------------------------------------------------------
# Exact ground truth
# ---------------------------------------------------------------------------


@pytest.fixture
def exact_computer_3(dummy_game_3):
    """ExactComputer for the 3-player dummy game (2^3 = 8 evaluations)."""
    return ExactComputer(dummy_game_3)


# ---------------------------------------------------------------------------
# Tiny datasets (no sklearn dependency)
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)
_TINY_X = _RNG.normal(size=(30, 5))
_TINY_Y_REG = _TINY_X[:, 0] + 0.5 * _TINY_X[:, 1] + _RNG.normal(0, 0.1, size=30)
_TINY_Y_CLF = (_TINY_Y_REG > np.median(_TINY_Y_REG)).astype(int)


@pytest.fixture
def tiny_data():
    """30 samples, 5 features. Regression target."""
    return _TINY_X.copy(), _TINY_Y_REG.copy()


@pytest.fixture
def tiny_data_clf():
    """30 samples, 5 features. Binary classification target."""
    return _TINY_X.copy(), _TINY_Y_CLF.copy()


# ---------------------------------------------------------------------------
# Model factories (sklearn only — always available)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dt_reg_model():
    """DecisionTreeRegressor, max_depth=3, fit on tiny data."""
    from sklearn.tree import DecisionTreeRegressor

    model = DecisionTreeRegressor(max_depth=3, random_state=42)
    rng = np.random.default_rng(42)
    X = rng.normal(size=(30, 5))
    y = X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.1, size=30)
    model.fit(X, y)
    return model


@pytest.fixture(scope="module")
def dt_clf_model():
    """DecisionTreeClassifier, max_depth=3, fit on tiny data."""
    from sklearn.tree import DecisionTreeClassifier

    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    rng = np.random.default_rng(42)
    X = rng.normal(size=(30, 5))
    y = (X[:, 0] > 0).astype(int)
    model.fit(X, y)
    return model


@pytest.fixture(scope="module")
def rf_reg_model():
    """RandomForestRegressor, 5 trees, max_depth=3, fit on tiny data."""
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(n_estimators=5, max_depth=3, random_state=42)
    rng = np.random.default_rng(42)
    X = rng.normal(size=(30, 5))
    y = X[:, 0] + 0.5 * X[:, 1] + rng.normal(0, 0.1, size=30)
    model.fit(X, y)
    return model


@pytest.fixture(scope="module")
def rf_clf_model():
    """RandomForestClassifier, 5 trees, max_depth=3, fit on tiny data."""
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42)
    rng = np.random.default_rng(42)
    X = rng.normal(size=(30, 5))
    y = (X[:, 0] > 0).astype(int)
    model.fit(X, y)
    return model


@pytest.fixture(scope="module")
def background_data():
    """Shared background data array for explainer/imputer tests. 30 samples, 5 features."""
    rng = np.random.default_rng(42)
    return rng.normal(size=(30, 5))
```

- [ ] **Step 2: Verify fixtures load**

Run: `uv run python -c "import tests.shapiq.conftest_new; print('OK')"`

Expected: `OK` (no import errors).

- [ ] **Step 3: Commit**

```bash
git add tests/shapiq/conftest_new.py
git commit -m "Add new shared conftest with games, fixtures, and skip markers"
```

---

### Task 3: Write test_approximators.py

**Files:**
- Create: `tests/shapiq/test_approximators.py`

- [ ] **Step 1: Write the approximator protocol and registry**

Create `tests/shapiq/test_approximators.py`:

```python
"""Protocol and special-case tests for all approximators."""

from __future__ import annotations

import numpy as np
import pytest

from shapiq.approximator import (
    SHAPIQ,
    SPEX,
    SVARM,
    SVARMIQ,
    InconsistentKernelSHAPIQ,
    KernelSHAP,
    KernelSHAPIQ,
    MSRBiased,
    OwenSamplingSV,
    PermutationSamplingSII,
    PermutationSamplingSTII,
    PermutationSamplingSV,
    ProxySHAP,
    RegressionFBII,
    RegressionFSII,
    StratifiedSamplingSV,
    UnbiasedKernelSHAP,
    kADDSHAP,
)
from shapiq.interaction_values import InteractionValues
from shapiq_games.synthetic import DummyGame

# ---------------------------------------------------------------------------
# Indices where sum(values) == game(N) - game(empty) holds
# ---------------------------------------------------------------------------
EFFICIENT_INDICES = {"SV", "k-SII", "FSII", "STII", "kADD-SHAP"}

# ---------------------------------------------------------------------------
# Registry: one entry per (approximator, index) combo worth testing.
# Each entry must work with a 7-player DummyGame(interaction=(1,2)).
# ---------------------------------------------------------------------------
ALL_APPROXIMATORS = [
    # --- Permutation ---
    {"cls": PermutationSamplingSV, "n": 7, "max_order": 1, "index": "SV", "budget": 80},
    {"cls": PermutationSamplingSII, "n": 7, "max_order": 2, "index": "k-SII", "budget": 100},
    {"cls": PermutationSamplingSTII, "n": 7, "max_order": 2, "index": "STII", "budget": 100},
    # --- Marginals ---
    {"cls": OwenSamplingSV, "n": 7, "max_order": 1, "index": "SV", "budget": 80},
    {"cls": StratifiedSamplingSV, "n": 7, "max_order": 1, "index": "SV", "budget": 80},
    # --- Monte Carlo ---
    {"cls": SHAPIQ, "n": 7, "max_order": 2, "index": "SII", "budget": 100},
    {"cls": SHAPIQ, "n": 7, "max_order": 2, "index": "k-SII", "budget": 100},
    {"cls": SHAPIQ, "n": 7, "max_order": 2, "index": "FSII", "budget": 100},
    {"cls": SHAPIQ, "n": 7, "max_order": 2, "index": "FBII", "budget": 100},
    {"cls": SHAPIQ, "n": 7, "max_order": 2, "index": "STII", "budget": 100},
    {"cls": UnbiasedKernelSHAP, "n": 7, "max_order": 1, "index": "SV", "budget": 80},
    {"cls": SVARM, "n": 7, "max_order": 1, "index": "SV", "budget": 80},
    {"cls": SVARMIQ, "n": 7, "max_order": 2, "index": "k-SII", "budget": 100},
    {"cls": SVARMIQ, "n": 7, "max_order": 2, "index": "SII", "budget": 100},
    # --- Regression ---
    {"cls": KernelSHAP, "n": 7, "max_order": 1, "index": "SV", "budget": 80},
    {"cls": KernelSHAPIQ, "n": 7, "max_order": 2, "index": "k-SII", "budget": 100},
    {"cls": InconsistentKernelSHAPIQ, "n": 7, "max_order": 2, "index": "k-SII", "budget": 100},
    {"cls": kADDSHAP, "n": 7, "max_order": 2, "index": "kADD-SHAP", "budget": 100},
    {"cls": RegressionFSII, "n": 7, "max_order": 2, "index": "FSII", "budget": 100},
    {"cls": RegressionFBII, "n": 7, "max_order": 2, "index": "FBII", "budget": 100},
    # --- Sparse ---
    {"cls": SPEX, "n": 7, "max_order": 2, "index": "FSII", "budget": 200},
    # --- Proxy ---
    {"cls": MSRBiased, "n": 7, "max_order": 2, "index": "SII", "budget": 100},
    {"cls": MSRBiased, "n": 7, "max_order": 1, "index": "SV", "budget": 80},
]


def _approx_id(config: dict) -> str:
    return f"{config['cls'].__name__}-{config['index']}"


def _make_approximator(config: dict, **overrides):
    kwargs = {"n": config["n"], "max_order": config["max_order"], "index": config["index"]}
    kwargs.update(overrides)
    return config["cls"](**kwargs)


# ===================================================================
# Protocol tests — every approximator must pass these
# ===================================================================


@pytest.mark.parametrize("config", ALL_APPROXIMATORS, ids=_approx_id)
class TestApproximatorProtocol:
    """Universal contract checks for all approximators."""

    def test_returns_interaction_values(self, config):
        """Approximate returns InteractionValues with correct metadata."""
        game = DummyGame(n=config["n"], interaction=(1, 2))
        approx = _make_approximator(config, random_state=42)
        result = approx.approximate(config["budget"], game)

        assert isinstance(result, InteractionValues)
        assert result.max_order == config["max_order"]
        assert result.n_players == config["n"]

    def test_respects_budget(self, config):
        """Game is not called more than budget + 2 times."""
        game = DummyGame(n=config["n"], interaction=(1, 2))
        approx = _make_approximator(config, random_state=42)
        approx.approximate(config["budget"], game)

        assert game.access_counter <= config["budget"] + 2

    def test_reproducible(self, config):
        """Same random_state produces identical results."""
        game1 = DummyGame(n=config["n"], interaction=(1, 2))
        game2 = DummyGame(n=config["n"], interaction=(1, 2))
        a1 = _make_approximator(config, random_state=42)
        a2 = _make_approximator(config, random_state=42)
        r1 = a1.approximate(config["budget"], game1)
        r2 = a2.approximate(config["budget"], game2)

        assert np.allclose(r1.values, r2.values)

    def test_rejects_invalid_index(self, config):
        """Raises error for indices not in valid_indices."""
        with pytest.raises((ValueError, TypeError)):
            _make_approximator(config, index="NOT_A_REAL_INDEX")


# ===================================================================
# Special cases
# ===================================================================


class TestProxySHAP:
    """ProxySHAP delegates to different adjustment strategies."""

    def test_default_adjustment_is_msr_biased(self):
        proxy = ProxySHAP(n=7, max_order=2, index="SII")
        assert proxy._adjustment == "msr-b"

    def test_approximate_runs(self):
        game = DummyGame(n=7, interaction=(1, 2))
        proxy = ProxySHAP(n=7, max_order=2, index="SII", random_state=42)
        result = proxy.approximate(100, game)
        assert isinstance(result, InteractionValues)
```

- [ ] **Step 2: Run the tests**

Run: `uv run pytest tests/shapiq/test_approximators.py -v --no-header -p no:conftest 2>&1 | tail -30`

Note: We use `-p no:conftest` to avoid the old conftest interfering. If that flag doesn't work, just run directly — the new test file doesn't depend on any fixtures from conftest.

Expected: All tests PASS. No test should depend on conftest fixtures.

- [ ] **Step 3: Commit**

```bash
git add tests/shapiq/test_approximators.py
git commit -m "Add protocol-driven approximator tests with registry"
```

---

### Task 4: Write test_explainers.py

**Files:**
- Create: `tests/shapiq/test_explainers.py`

- [ ] **Step 1: Write the explainer protocol tests**

Create `tests/shapiq/test_explainers.py`:

```python
"""Protocol and special-case tests for explainers."""

from __future__ import annotations

import numpy as np
import pytest

from shapiq.explainer.tabular import TabularExplainer
from shapiq.explainer.agnostic import AgnosticExplainer
from shapiq.interaction_values import InteractionValues
from shapiq_games.synthetic import DummyGame


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _linear_model(X: np.ndarray) -> np.ndarray:
    """Simple linear model: sum of features."""
    return X.sum(axis=1)


def _make_background_data(n_samples: int = 30, n_features: int = 5) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.normal(size=(n_samples, n_features))


# ===================================================================
# TabularExplainer protocol
# ===================================================================


TABULAR_CONFIGS = [
    {"index": "SV", "max_order": 1, "approximator": "auto"},
    {"index": "k-SII", "max_order": 2, "approximator": "auto"},
    {"index": "FSII", "max_order": 2, "approximator": "regression"},
    {"index": "SV", "max_order": 1, "approximator": "permutation"},
    {"index": "k-SII", "max_order": 2, "approximator": "montecarlo"},
]


def _tabular_id(config: dict) -> str:
    return f"{config['index']}-order{config['max_order']}-{config['approximator']}"


@pytest.mark.parametrize("config", TABULAR_CONFIGS, ids=_tabular_id)
class TestTabularExplainerProtocol:
    """Contract checks for TabularExplainer."""

    def test_explain_returns_interaction_values(self, config):
        """explain() returns InteractionValues with correct metadata."""
        data = _make_background_data()
        explainer = TabularExplainer(
            model=_linear_model,
            data=data,
            index=config["index"],
            max_order=config["max_order"],
            approximator=config["approximator"],
            random_state=42,
        )
        x = data[0].reshape(1, -1)
        result = explainer.explain(x, budget=2**5)

        assert isinstance(result, InteractionValues)
        assert result.index == config["index"]
        assert result.max_order == config["max_order"]

    def test_efficiency(self, config):
        """sum(values) approximates the prediction for efficient indices."""
        if config["index"] not in {"SV", "k-SII", "FSII"}:
            pytest.skip("Efficiency not guaranteed for this index")
        data = _make_background_data()
        explainer = TabularExplainer(
            model=_linear_model,
            data=data,
            index=config["index"],
            max_order=config["max_order"],
            approximator=config["approximator"],
            random_state=42,
        )
        x = data[0].reshape(1, -1)
        result = explainer.explain(x, budget=2**5)
        prediction = float(_linear_model(x)[0])
        assert float(np.sum(result.values)) == pytest.approx(prediction, abs=0.5)

    def test_reproducible(self, config):
        """Same random_state produces identical explanations."""
        data = _make_background_data()
        kwargs = {
            "model": _linear_model,
            "data": data,
            "index": config["index"],
            "max_order": config["max_order"],
            "approximator": config["approximator"],
        }
        e1 = TabularExplainer(**kwargs, random_state=42)
        e2 = TabularExplainer(**kwargs, random_state=42)
        x = data[0].reshape(1, -1)
        r1 = e1.explain(x, budget=2**5, random_state=42)
        r2 = e2.explain(x, budget=2**5, random_state=42)
        assert np.allclose(r1.values, r2.values)


# ===================================================================
# AgnosticExplainer
# ===================================================================


class TestAgnosticExplainer:
    """AgnosticExplainer wraps a Game or callable directly."""

    def test_explain_with_game(self):
        game = DummyGame(n=5, interaction=(0, 1))
        explainer = AgnosticExplainer(game=game, index="k-SII", max_order=2, random_state=42)
        result = explainer.explain(budget=100)
        assert isinstance(result, InteractionValues)
        assert result.n_players == 5

    def test_explain_with_callable(self):
        def my_game(coalitions):
            return coalitions.sum(axis=1).astype(float)

        explainer = AgnosticExplainer(
            game=my_game, n_players=4, index="SV", max_order=1, random_state=42
        )
        result = explainer.explain(budget=50)
        assert isinstance(result, InteractionValues)
        assert result.n_players == 4


# ===================================================================
# TabularExplainer special cases
# ===================================================================


class TestTabularExplainerValidation:
    """Input validation and edge cases."""

    def test_invalid_index_raises(self):
        data = _make_background_data()
        with pytest.raises(ValueError):
            TabularExplainer(model=_linear_model, data=data, index="INVALID", max_order=2)

    def test_invalid_approximator_raises(self):
        data = _make_background_data()
        with pytest.raises(ValueError):
            TabularExplainer(model=_linear_model, data=data, approximator="not_real")

    def test_sv_with_high_order_warns(self):
        data = _make_background_data()
        with pytest.warns(UserWarning):
            TabularExplainer(model=_linear_model, data=data, index="SV", max_order=2)

    def test_explain_X_batch(self):
        """explain_X returns a list of InteractionValues."""
        data = _make_background_data()
        explainer = TabularExplainer(
            model=_linear_model, data=data, index="SV", max_order=1, random_state=42
        )
        results = explainer.explain_X(data[:3], budget=2**5)
        assert len(results) == 3
        assert all(isinstance(r, InteractionValues) for r in results)
```

- [ ] **Step 2: Run the tests**

Run: `uv run pytest tests/shapiq/test_explainers.py -v --no-header 2>&1 | tail -30`

Expected: All tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/shapiq/test_explainers.py
git commit -m "Add protocol-driven explainer tests"
```

---

### Task 5: Write test_tree.py

**Files:**
- Create: `tests/shapiq/test_tree.py`

- [ ] **Step 1: Write tree protocol, SHAP comparison, and edge cases**

Create `tests/shapiq/test_tree.py`:

```python
"""Protocol, SHAP comparison, and edge-case tests for the tree module."""

from __future__ import annotations

import copy
import importlib.util

import numpy as np
import pytest

from shapiq.interaction_values import InteractionValues
from shapiq.tree import TreeExplainer, TreeModel

# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------
skip_if_no_xgboost = pytest.mark.skipif(
    not importlib.util.find_spec("xgboost"), reason="xgboost not installed"
)
skip_if_no_lightgbm = pytest.mark.skipif(
    not importlib.util.find_spec("lightgbm"), reason="lightgbm not installed"
)

# ---------------------------------------------------------------------------
# Shared data (module-level, generated once)
# ---------------------------------------------------------------------------
_RNG = np.random.default_rng(42)
_BG_REG_X = _RNG.normal(size=(30, 7))
_BG_REG_Y = _BG_REG_X[:, 0] + 0.5 * _BG_REG_X[:, 1] + _RNG.normal(0, 0.1, size=30)
_BG_CLF_Y = (_BG_REG_Y > np.median(_BG_REG_Y)).astype(int)


# ---------------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dt_reg():
    from sklearn.tree import DecisionTreeRegressor

    m = DecisionTreeRegressor(max_depth=3, random_state=42)
    m.fit(_BG_REG_X, _BG_REG_Y)
    return m


@pytest.fixture(scope="module")
def rf_reg():
    from sklearn.ensemble import RandomForestRegressor

    m = RandomForestRegressor(n_estimators=5, max_depth=3, random_state=42)
    m.fit(_BG_REG_X, _BG_REG_Y)
    return m


@pytest.fixture(scope="module")
def rf_clf():
    from sklearn.ensemble import RandomForestClassifier

    m = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42)
    m.fit(_BG_REG_X, _BG_CLF_Y)
    return m


@pytest.fixture(scope="module")
def et_reg():
    from sklearn.ensemble import ExtraTreesRegressor

    m = ExtraTreesRegressor(n_estimators=5, max_depth=3, random_state=42)
    m.fit(_BG_REG_X, _BG_REG_Y)
    return m


@pytest.fixture(scope="module")
def xgb_reg():
    pytest.importorskip("xgboost")
    from xgboost import XGBRegressor

    m = XGBRegressor(n_estimators=3, max_depth=3, random_state=42)
    m.fit(_BG_REG_X, _BG_REG_Y)
    return m


@pytest.fixture(scope="module")
def xgb_clf():
    pytest.importorskip("xgboost")
    from xgboost import XGBClassifier

    m = XGBClassifier(n_estimators=3, max_depth=3, random_state=42, use_label_encoder=False)
    m.fit(_BG_REG_X, _BG_CLF_Y)
    return m


@pytest.fixture(scope="module")
def lgbm_clf():
    pytest.importorskip("lightgbm")
    from lightgbm import LGBMClassifier

    m = LGBMClassifier(n_estimators=3, max_depth=3, random_state=42, verbose=-1)
    m.fit(_BG_REG_X, _BG_CLF_Y)
    return m


# ===================================================================
# Protocol: every tree model must satisfy these
# ===================================================================


SKLEARN_TREE_MODELS = [
    ("dt_reg", "regression", None),
    ("rf_reg", "regression", None),
    ("rf_clf", "classification", 0),
    ("et_reg", "regression", None),
]

OPTIONAL_TREE_MODELS = [
    pytest.param("xgb_reg", "regression", None, marks=skip_if_no_xgboost),
    pytest.param("xgb_clf", "classification", 0, marks=skip_if_no_xgboost),
    pytest.param("lgbm_clf", "classification", 0, marks=skip_if_no_lightgbm),
]

ALL_TREE_MODELS = SKLEARN_TREE_MODELS + OPTIONAL_TREE_MODELS


@pytest.mark.parametrize(
    ("model_fixture", "task", "class_index"),
    ALL_TREE_MODELS,
    ids=[t[0] if isinstance(t, tuple) else t.values[0] for t in ALL_TREE_MODELS],
)
class TestTreeProtocol:
    """Universal contract checks for tree explainers across model types."""

    def test_explain_returns_interaction_values(self, model_fixture, task, class_index, request):
        model = request.getfixturevalue(model_fixture)
        explainer = TreeExplainer(model=model, max_order=2, min_order=1, class_index=class_index)
        x = _BG_REG_X[0]
        result = explainer.explain(x)

        assert isinstance(result, InteractionValues)
        assert result.max_order == 2
        assert result.n_players == _BG_REG_X.shape[1]

    def test_efficiency(self, model_fixture, task, class_index, request):
        """sum(values) == prediction for SV."""
        model = request.getfixturevalue(model_fixture)
        explainer = TreeExplainer(
            model=model, max_order=1, min_order=0, index="SV", class_index=class_index
        )
        x = _BG_REG_X[0]
        result = explainer.explain(x)

        if task == "regression":
            prediction = float(model.predict(x.reshape(1, -1))[0])
        else:
            prediction = float(model.predict_proba(x.reshape(1, -1))[0, class_index])

        assert float(np.sum(result.values)) == pytest.approx(prediction, rel=1e-4)

    def test_baseline_matches_empty_prediction(self, model_fixture, task, class_index, request):
        model = request.getfixturevalue(model_fixture)
        explainer = TreeExplainer(
            model=model, max_order=1, min_order=0, index="SV", class_index=class_index
        )
        expected_baseline = sum(
            te.empty_prediction for te in explainer._treeshapiq_explainers
        )
        assert explainer.baseline_value == pytest.approx(expected_baseline)


# ===================================================================
# Manual TreeModel test (no sklearn dependency)
# ===================================================================


class TestManualTreeModel:
    """Test TreeExplainer with a hand-crafted TreeModel."""

    def test_against_known_values(self):
        """Verify SV computation against known SHAP library values."""
        children_left = np.asarray([1, 2, 3, -1, -1, -1, 7, -1, -1])
        children_right = np.asarray([6, 5, 4, -1, -1, -1, 8, -1, -1])
        features = np.asarray([0, 1, 0, -2, -2, -2, 2, -2, -2])
        thresholds = np.asarray([0, 0, -0.5, -2, -2, -2, 0, -2, -2])
        node_sample_weight = np.asarray([100, 50, 38, 15, 23, 12, 50, 20, 30])
        values = [110, 105, 95, 20, 50, 100, 75, 10, 40]
        values = np.asarray([v / max(values) for v in values])

        tree_model = TreeModel(
            children_left=children_left,
            children_right=children_right,
            children_missing=children_left,
            features=features,
            thresholds=thresholds,
            node_sample_weight=node_sample_weight,
            values=values,
        )

        x = np.asarray([-1, -0.5, 1, 0])
        explainer = TreeExplainer(model=tree_model, max_order=1, min_order=1, index="SV")
        result = explainer.explain(x)

        assert result[(0,)] == pytest.approx(-0.09263158, abs=1e-4)
        assert result[(1,)] == pytest.approx(-0.12100478, abs=1e-4)
        assert result[(2,)] == pytest.approx(0.02727273, abs=1e-4)
        assert result[(3,)] == pytest.approx(0.0, abs=1e-4)

    def test_sv_warning_for_order_2(self):
        """SV with max_order > 1 should warn."""
        children_left = np.asarray([1, -1, -1])
        children_right = np.asarray([2, -1, -1])
        features = np.asarray([0, -2, -2])
        thresholds = np.asarray([0.0, -2.0, -2.0])
        node_sample_weight = np.asarray([10, 5, 5])
        values = np.asarray([0.5, 0.3, 0.7])

        tree_model = TreeModel(
            children_left=children_left,
            children_right=children_right,
            children_missing=children_left,
            features=features,
            thresholds=thresholds,
            node_sample_weight=node_sample_weight,
            values=values,
        )
        with pytest.warns(UserWarning):
            TreeExplainer(model=tree_model, max_order=2, min_order=1, index="SV")


# ===================================================================
# Edge cases (regression tests for past bugs)
# ===================================================================


class TestTreeEdgeCases:
    """Regression tests for specific bugs fixed in the tree module."""

    def test_high_dimensional_indices_do_not_overflow(self):
        """Regression: int64 indices with >127 features (was overflowing to int8)."""
        from shapiq.approximator.proxy.proxyshap import MSRBiased
        from shapiq.tree.interventional.cext import compute_interactions_sparse

        n_features = 170
        approximator = MSRBiased(n=n_features, max_order=1, index="SV")
        coalition_matrix = np.zeros((3, n_features), dtype=np.int64)
        coalition_matrix[0, :5] = 1
        coalition_matrix[1, 100:110] = 1
        coalition_matrix[2, 160:] = 1

        e_matrix, r_matrix, e_counts, r_counts = approximator._coalitions_to_tree_paths(
            coalition_matrix
        )

        assert e_matrix.dtype == np.int64
        assert r_matrix.dtype == np.int64

        coalition_values = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        interactions = compute_interactions_sparse(
            coalition_values, e_matrix, r_matrix, e_counts, r_counts, "SV", n_features, 1
        )
        assert isinstance(interactions, dict)
        assert all(0 <= f < n_features for key in interactions for f in key)

    def test_repeated_flatten_calls_no_segfault(self):
        """Regression: refcount corruption in C-extension flatten output."""
        from shapiq.tree.interventional.cext import compute_interactions_flatten

        n_features = 200
        n_iterations = n_features
        leaf_predictions = np.ones(n_iterations, dtype=np.float32)
        features = np.arange(n_features, dtype=np.int64)
        e_sizes = np.ones(n_iterations, dtype=np.int64)
        r_sizes = np.zeros(n_iterations, dtype=np.int64)
        feature_in_e = np.ones(n_iterations, dtype=np.int64)
        leaf_id = np.zeros(n_iterations, dtype=np.int64)

        for _ in range(5):
            out = compute_interactions_flatten(
                leaf_predictions, features, e_sizes, r_sizes, feature_in_e, leaf_id,
                "SV", n_iterations, n_features, n_iterations, 1, 0, 1.0,
            )
            assert len(out) == n_features
```

- [ ] **Step 2: Run the tests**

Run: `uv run pytest tests/shapiq/test_tree.py -v --no-header 2>&1 | tail -40`

Expected: All tests PASS. Optional-dep tests are skipped if xgboost/lightgbm not installed.

- [ ] **Step 3: Commit**

```bash
git add tests/shapiq/test_tree.py
git commit -m "Add protocol-driven tree tests with SHAP comparison and edge cases"
```

---

### Task 6: Write test_imputers.py

**Files:**
- Create: `tests/shapiq/test_imputers.py`

- [ ] **Step 1: Write imputer protocol tests**

Create `tests/shapiq/test_imputers.py`:

```python
"""Protocol tests for all imputers."""

from __future__ import annotations

import numpy as np
import pytest

from shapiq.imputer import (
    BaselineImputer,
    GaussianCopulaImputer,
    GaussianImputer,
    MarginalImputer,
)
from shapiq.interaction_values import InteractionValues

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(42)
_DATA = _RNG.normal(size=(30, 5))
_Y = _DATA[:, 0] + 0.5 * _DATA[:, 1]


def _simple_model(X: np.ndarray) -> np.ndarray:
    return X.sum(axis=1)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

IMPUTER_CONFIGS = [
    {"cls": MarginalImputer, "kwargs": {"sample_size": 10}},
    {"cls": BaselineImputer, "kwargs": {}},
    {"cls": GaussianImputer, "kwargs": {"sample_size": 10}},
    {"cls": GaussianCopulaImputer, "kwargs": {"sample_size": 10}},
]


def _imputer_id(config: dict) -> str:
    return config["cls"].__name__


# ===================================================================
# Protocol
# ===================================================================


@pytest.mark.parametrize("config", IMPUTER_CONFIGS, ids=_imputer_id)
class TestImputerProtocol:
    """Contract checks for all imputers."""

    def test_fit_and_call(self, config):
        """fit() + __call__() produces array of correct length."""
        imputer = config["cls"](model=_simple_model, data=_DATA.copy(), **config["kwargs"])
        imputer.fit(_DATA[0])

        # All features present
        coalition_all = np.ones((1, 5), dtype=bool)
        result = imputer(coalition_all)
        assert result.shape == (1,)

        # No features present
        coalition_none = np.zeros((1, 5), dtype=bool)
        result = imputer(coalition_none)
        assert result.shape == (1,)

    def test_multiple_coalitions(self, config):
        """Handles batch of coalitions."""
        imputer = config["cls"](model=_simple_model, data=_DATA.copy(), **config["kwargs"])
        imputer.fit(_DATA[0])

        coalitions = np.eye(5, dtype=bool)  # one feature at a time
        result = imputer(coalitions)
        assert result.shape == (5,)

    def test_full_coalition_approximates_prediction(self, config):
        """With all features present, result should be close to model prediction."""
        imputer = config["cls"](
            model=_simple_model, data=_DATA.copy(), random_state=42, **config["kwargs"]
        )
        x = _DATA[0]
        imputer.fit(x)

        coalition_all = np.ones((1, 5), dtype=bool)
        result = float(imputer(coalition_all)[0])
        expected = float(_simple_model(x.reshape(1, -1))[0])

        assert result == pytest.approx(expected, abs=0.5)
```

- [ ] **Step 2: Run the tests**

Run: `uv run pytest tests/shapiq/test_imputers.py -v --no-header 2>&1 | tail -20`

Expected: All tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/shapiq/test_imputers.py
git commit -m "Add protocol-driven imputer tests"
```

---

### Task 7: Write test_interaction_values.py

**Files:**
- Create: `tests/shapiq/test_interaction_values.py`

This condenses the old 896-line test file to its essence: creation, access, serialization, aggregation.

- [ ] **Step 1: Write InteractionValues tests**

Create `tests/shapiq/test_interaction_values.py`:

```python
"""Tests for the InteractionValues data structure."""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from shapiq.interaction_values import InteractionValues, aggregate_interaction_values
from shapiq.utils import powerset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_iv(
    n_players: int = 5,
    max_order: int = 2,
    min_order: int = 1,
    index: str = "k-SII",
    baseline: float = 1.0,
) -> InteractionValues:
    """Create a deterministic InteractionValues for testing."""
    interaction_lookup = {}
    values = []
    for i, interaction in enumerate(
        powerset(range(n_players), min_size=min_order, max_size=max_order)
    ):
        interaction_lookup[interaction] = i
        values.append(float(i) * 0.1)
    return InteractionValues(
        values=np.array(values),
        index=index,
        n_players=n_players,
        min_order=min_order,
        max_order=max_order,
        interaction_lookup=interaction_lookup,
        baseline_value=baseline,
        estimated=True,
        estimation_budget=100,
    )


# ===================================================================
# Creation & basic access
# ===================================================================


class TestCreation:
    def test_basic_properties(self):
        iv = _make_iv()
        assert iv.n_players == 5
        assert iv.max_order == 2
        assert iv.min_order == 1
        assert iv.index == "k-SII"
        assert iv.baseline_value == 1.0
        assert iv.estimated is True
        assert iv.estimation_budget == 100

    def test_getitem_single_interaction(self):
        iv = _make_iv()
        val = iv[(0,)]
        assert isinstance(val, float)

    def test_getitem_missing_returns_zero(self):
        iv = _make_iv(max_order=1)
        assert iv[(0, 1)] == 0.0

    def test_empty_interaction_is_baseline(self):
        iv = _make_iv(min_order=0)
        assert iv[()] == pytest.approx(iv.baseline_value)

    @pytest.mark.parametrize(
        ("index", "should_warn"),
        [("k-SII", False), ("SII", False), ("NOT_VALID", True)],
    )
    def test_invalid_index_warns(self, index, should_warn):
        if should_warn:
            with pytest.warns(UserWarning):
                _make_iv(index=index)
        else:
            _make_iv(index=index)  # should not warn


# ===================================================================
# Order extraction
# ===================================================================


class TestOrderExtraction:
    def test_get_n_order_values_shape(self):
        iv = _make_iv(n_players=5, max_order=2, min_order=1)
        order_1 = iv.get_n_order_values(1)
        assert order_1.shape == (5,)

    def test_get_n_order(self):
        iv = _make_iv(n_players=5, max_order=2, min_order=1)
        iv_order1 = iv.get_n_order(order=1)
        assert iv_order1.max_order == 1
        assert iv_order1.min_order == 1


# ===================================================================
# Serialization
# ===================================================================


class TestSerialization:
    def test_json_roundtrip(self, tmp_path):
        iv = _make_iv()
        path = tmp_path / "test_iv.json"
        iv.to_json_file(path)
        loaded = InteractionValues.from_json_file(path)

        assert loaded.n_players == iv.n_players
        assert loaded.index == iv.index
        assert np.allclose(loaded.values, iv.values)
        assert loaded.baseline_value == pytest.approx(iv.baseline_value)


# ===================================================================
# Aggregation
# ===================================================================


class TestAggregation:
    def test_aggregate_mean(self):
        iv1 = _make_iv()
        iv2 = _make_iv()
        iv2_values = iv2.values.copy()
        iv2_values[:] = 1.0
        iv2 = InteractionValues(
            values=iv2_values,
            index=iv1.index,
            n_players=iv1.n_players,
            min_order=iv1.min_order,
            max_order=iv1.max_order,
            interaction_lookup=dict(iv1.interaction_lookup),
            baseline_value=iv1.baseline_value,
        )
        result = aggregate_interaction_values([iv1, iv2])
        assert isinstance(result, InteractionValues)
        assert result.n_players == iv1.n_players


# ===================================================================
# Copy behavior
# ===================================================================


class TestCopy:
    def test_deepcopy_independent(self):
        from copy import deepcopy

        iv = _make_iv()
        iv_copy = deepcopy(iv)
        iv_copy.values[0] = 999.0
        assert iv.values[0] != 999.0
```

- [ ] **Step 2: Run the tests**

Run: `uv run pytest tests/shapiq/test_interaction_values.py -v --no-header 2>&1 | tail -20`

Expected: All tests PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/shapiq/test_interaction_values.py
git commit -m "Add condensed InteractionValues tests"
```

---

### Task 8: Write test_game_theory.py, test_plots.py, test_public_api.py

**Files:**
- Create: `tests/shapiq/test_game_theory.py`
- Create: `tests/shapiq/test_plots.py`
- Create: `tests/shapiq/test_public_api.py`

- [ ] **Step 1: Write test_game_theory.py**

Create `tests/shapiq/test_game_theory.py`:

```python
"""Tests for game theory module: ExactComputer, indices, Moebius converter."""

from __future__ import annotations

import numpy as np
import pytest

from shapiq.game_theory.exact import ExactComputer
from shapiq.game_theory.indices import (
    AllIndices,
    get_computation_index,
    index_generalizes_bv,
    index_generalizes_sv,
    is_index_valid,
)
from shapiq.game_theory.moebius_converter import MoebiusConverter
from shapiq.interaction_values import InteractionValues
from shapiq_games.synthetic import DummyGame


# ===================================================================
# ExactComputer
# ===================================================================


class TestExactComputer:
    """Tests for exact computation of interaction indices."""

    def test_sv_on_dummy_game(self):
        game = DummyGame(n=3, interaction=(0, 1))
        computer = ExactComputer(game)
        sv = computer("SV", order=1)

        assert isinstance(sv, InteractionValues)
        assert sv.index == "SV"
        assert sv.n_players == 3

        # SV satisfies efficiency: sum = game(N) - game(empty)
        grand = float(game(np.ones((1, 3), dtype=bool))[0])
        empty = float(game(np.zeros((1, 3), dtype=bool))[0])
        assert float(np.sum(sv.values)) == pytest.approx(grand - empty, abs=1e-10)

    def test_sii_on_dummy_game(self):
        game = DummyGame(n=3, interaction=(0, 1))
        computer = ExactComputer(game)
        sii = computer("SII", order=2)

        assert isinstance(sii, InteractionValues)
        assert sii.index == "SII"
        assert sii.max_order == 2

    def test_k_sii_on_dummy_game(self):
        game = DummyGame(n=3, interaction=(0, 1))
        computer = ExactComputer(game)
        k_sii = computer("k-SII", order=2)

        assert isinstance(k_sii, InteractionValues)
        assert k_sii.index == "k-SII"

        # k-SII satisfies efficiency
        grand = float(game(np.ones((1, 3), dtype=bool))[0])
        empty = float(game(np.zeros((1, 3), dtype=bool))[0])
        assert float(np.sum(k_sii.values)) == pytest.approx(grand - empty, abs=1e-10)

    @pytest.mark.parametrize("index", ["SV", "SII", "k-SII", "STII", "FSII", "FBII", "BV", "BII"])
    def test_all_common_indices(self, index):
        """ExactComputer should handle all common indices without error."""
        game = DummyGame(n=3, interaction=(0, 1))
        computer = ExactComputer(game)
        order = 1 if index in ("SV", "BV") else 2
        result = computer(index, order=order)
        assert isinstance(result, InteractionValues)

    def test_moebius_values(self):
        game = DummyGame(n=3, interaction=(0, 1))
        computer = ExactComputer(game)
        moebius = computer("Moebius", order=3)
        assert isinstance(moebius, InteractionValues)


# ===================================================================
# Index utilities
# ===================================================================


class TestIndices:
    def test_is_index_valid_true(self):
        assert is_index_valid("SV")
        assert is_index_valid("k-SII")

    def test_is_index_valid_false(self):
        assert not is_index_valid("NOT_REAL")

    def test_is_index_valid_raises(self):
        with pytest.raises(ValueError):
            is_index_valid("NOT_REAL", raise_error=True)

    def test_generalizes_sv(self):
        assert index_generalizes_sv("SII")
        assert index_generalizes_sv("k-SII")
        assert not index_generalizes_sv("BV")
        assert not index_generalizes_sv("SV")

    def test_generalizes_bv(self):
        assert index_generalizes_bv("BII")
        assert not index_generalizes_bv("SII")

    def test_get_computation_index(self):
        assert get_computation_index("k-SII") == "SII"
        assert get_computation_index("SV") == "SII"
        assert get_computation_index("BV") == "BII"
        assert get_computation_index("STII") == "STII"


# ===================================================================
# Moebius converter
# ===================================================================


class TestMoebiusConverter:
    def test_sii_to_k_sii_roundtrip(self):
        """Convert SII -> k-SII and verify it's a valid InteractionValues."""
        game = DummyGame(n=3, interaction=(0, 1))
        computer = ExactComputer(game)
        sii = computer("SII", order=2)

        converter = MoebiusConverter(sii)
        k_sii = converter("k-SII")

        assert isinstance(k_sii, InteractionValues)
        assert k_sii.index == "k-SII"
```

- [ ] **Step 2: Write test_plots.py**

Create `tests/shapiq/test_plots.py`:

```python
"""Smoke tests for plotting functions — verify they run without error."""

from __future__ import annotations

import numpy as np
import pytest
import matplotlib.pyplot as plt

from shapiq.interaction_values import InteractionValues
from shapiq.utils import powerset
from shapiq.plots import (
    bar_plot,
    force_plot,
    waterfall_plot,
)


@pytest.fixture
def sample_iv():
    """Small InteractionValues for plotting."""
    n = 4
    interaction_lookup = {}
    values = []
    for i, interaction in enumerate(powerset(range(n), min_size=1, max_size=2)):
        interaction_lookup[interaction] = i
        values.append(float(i) * 0.1 - 0.3)
    return InteractionValues(
        values=np.array(values),
        index="k-SII",
        n_players=n,
        min_order=1,
        max_order=2,
        interaction_lookup=interaction_lookup,
        baseline_value=0.5,
    )


class TestPlots:
    """Smoke tests: plots run without raising."""

    def test_bar_plot(self, sample_iv):
        fig = bar_plot(sample_iv.get_n_order(order=1))
        plt.close("all")

    def test_waterfall_plot(self, sample_iv):
        fig = waterfall_plot(sample_iv.get_n_order(order=1))
        plt.close("all")

    def test_force_plot(self, sample_iv):
        fig = force_plot(sample_iv.get_n_order(order=1))
        plt.close("all")
```

- [ ] **Step 3: Write test_public_api.py**

Create `tests/shapiq/test_public_api.py`:

```python
"""Tests that every concrete public subclass is registered in its module's __all__.

These guard against adding a new subclass without listing it in __all__.
"""

from __future__ import annotations

import importlib
import inspect

import pytest


def _find_concrete_subclasses(module: object, base: type) -> set[str]:
    """Return names of all concrete, public subclasses of base visible in module."""
    return {
        name
        for name, obj in inspect.getmembers(module, inspect.isclass)
        if issubclass(obj, base)
        and obj is not base
        and not inspect.isabstract(obj)
        and not name.startswith("_")
    }


@pytest.mark.parametrize(
    ("module_path", "base_path"),
    [
        ("shapiq.approximator", "shapiq.approximator.base:Approximator"),
        ("shapiq.explainer", "shapiq.explainer.base:Explainer"),
        ("shapiq.imputer", "shapiq.imputer.base:Imputer"),
    ],
    ids=["approximator", "explainer", "imputer"],
)
def test_all_concrete_subclasses_in_all(module_path: str, base_path: str) -> None:
    """Every concrete public subclass must appear in its module's __all__."""
    module = importlib.import_module(module_path)
    base_module_path, base_class_name = base_path.split(":")
    base: type = getattr(importlib.import_module(base_module_path), base_class_name)

    concrete = _find_concrete_subclasses(module, base)
    exported = set(module.__all__)
    missing = concrete - exported

    pkg_init = f"src/shapiq/{module_path.split('.')[-1]}/__init__.py"
    assert not missing, (
        f"Concrete subclasses not listed in {module_path}.__all__: {missing}. "
        f"Add them to {pkg_init}."
    )
```

- [ ] **Step 4: Run all three files**

Run: `uv run pytest tests/shapiq/test_game_theory.py tests/shapiq/test_plots.py tests/shapiq/test_public_api.py -v --no-header 2>&1 | tail -30`

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/shapiq/test_game_theory.py tests/shapiq/test_plots.py tests/shapiq/test_public_api.py
git commit -m "Add game theory, plot smoke, and public API tests"
```

---

### Task 9: Verify new suite, delete old tests, finalize conftest

**Files:**
- Delete: `tests/shapiq/tests_unit/` (entire directory)
- Delete: `tests/shapiq/tests_integration_tests/` (entire directory)
- Delete: `tests/shapiq/tests_deprecation/` (entire directory)
- Delete: `tests/shapiq/fixtures/` (entire directory)
- Delete: `tests/shapiq/data/` (entire directory)
- Delete: `tests/shapiq/markers.py`
- Delete: `tests/shapiq/utils.py`
- Delete: `tests/shapiq/conftest.py` (old)
- Rename: `tests/shapiq/conftest_new.py` -> `tests/shapiq/conftest.py`

- [ ] **Step 1: Run the full new suite and verify it passes**

Run: `uv run pytest tests/shapiq/test_approximators.py tests/shapiq/test_explainers.py tests/shapiq/test_tree.py tests/shapiq/test_imputers.py tests/shapiq/test_interaction_values.py tests/shapiq/test_game_theory.py tests/shapiq/test_plots.py tests/shapiq/test_public_api.py -v --tb=short 2>&1 | tail -40`

Expected: All new tests PASS. Note the test count and runtime.

- [ ] **Step 2: Rename conftest**

```bash
mv tests/shapiq/conftest_new.py tests/shapiq/conftest.py
```

- [ ] **Step 3: Delete old test directories and files**

```bash
rm -rf tests/shapiq/tests_unit
rm -rf tests/shapiq/tests_integration_tests
rm -rf tests/shapiq/tests_deprecation
rm -rf tests/shapiq/fixtures
rm -rf tests/shapiq/data
rm -f tests/shapiq/markers.py
rm -f tests/shapiq/utils.py
```

- [ ] **Step 4: Delete the old __init__.py if it references removed modules**

Check `tests/shapiq/__init__.py` and `tests/__init__.py`. If they import from deleted modules, clean them up. If they're empty or just have `from __future__ import annotations`, leave them.

- [ ] **Step 5: Run the full suite from the project root**

Run: `uv run pytest tests/shapiq -v --tb=short 2>&1 | tail -40`

Expected: All tests PASS from the standard test path. No import errors from deleted modules.

- [ ] **Step 6: Run with slow marker to verify tiering**

Run: `uv run pytest tests/shapiq -m '' -v --tb=short 2>&1 | tail -10`

Expected: Slow tests also run (if optional deps are installed).

- [ ] **Step 7: Run pre-commit**

Run: `uv run pre-commit run --all-files`

Expected: All hooks pass.

- [ ] **Step 8: Commit**

```bash
git add -A tests/shapiq
git commit -m "Remove old test suite, finalize protocol-driven test rework

Replaces 75 test files (~10k lines) with 8 files (~1.5k lines).
Protocol tests auto-discover approximators, explainers, imputers, and tree models.
Two tiers: default (~1min) and slow (CI, optional deps)."
```

---

### Task 10: Final verification and timing

- [ ] **Step 1: Time the default suite**

Run: `uv run pytest tests/shapiq --tb=short -q 2>&1`

Note the total runtime. Target: ~1 minute.

- [ ] **Step 2: Time the full suite**

Run: `uv run pytest tests/shapiq -m '' --tb=short -q 2>&1`

Note the total runtime. Target: ~2-3 minutes.

- [ ] **Step 3: Verify adding a new approximator is trivial**

Mentally verify: to add a new approximator, you would append one dict to `ALL_APPROXIMATORS` in `test_approximators.py`. That dict has 5 keys: `cls`, `n`, `max_order`, `index`, `budget`. The protocol tests automatically run against it.

- [ ] **Step 4: Commit timing results as a comment in the spec**

No commit needed — just verify the targets are met and report back.
