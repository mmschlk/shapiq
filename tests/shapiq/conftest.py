"""Shared fixtures for all shapiq tests."""

from __future__ import annotations

import os

# Limit OpenMP threads to prevent segfaults when PyTorch/sklearn coexist.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import importlib.util
from typing import TYPE_CHECKING

import matplotlib as mpl
import numpy as np
import pytest

mpl.use("Agg")

from shapiq.game_theory.exact import ExactComputer
from shapiq_games.synthetic import SOUM, DummyGame

if TYPE_CHECKING:
    from shapiq.interaction_values import InteractionValues

# ---------------------------------------------------------------------------
# Indices for which we have a closed-form ground truth via MoebiusConverter
# ---------------------------------------------------------------------------

GROUND_TRUTH_INDICES: tuple[str, ...] = ("SV", "BV", "k-SII", "STII", "SII", "FSII", "FBII")


def assert_iv_close(
    actual: InteractionValues,
    expected: InteractionValues,
    *,
    atol: float = 1e-8,
    check_baseline: bool = False,
    strict: bool = False,
) -> None:
    """Compare two InteractionValues by aligning their interaction_lookups.

    Only interactions present in *both* lookups are compared — different
    pipelines use different conventions for whether the empty interaction
    ``()`` is carried in the lookup vs. only in ``baseline_value``. Set
    ``check_baseline=True`` to additionally require the ``baseline_value``
    fields to match.

    Set ``strict=True`` to also require the non-empty supports to match
    exactly; use this when both pipelines should populate the same set of
    interactions (e.g. ExactComputer vs. SOUM.exact_values). Leave
    ``strict=False`` when one side may legitimately carry extra
    interactions (e.g. a sampling approximator emitting keys the
    analytical ground truth doesn't).
    """
    actual_non_empty = {k for k in actual.interaction_lookup if len(k) > 0}
    expected_non_empty = {k for k in expected.interaction_lookup if len(k) > 0}
    if strict:
        # Different pipelines have different conventions for zero-valued
        # interactions: ``ExactComputer`` emits them, ``MoebiusConverter``
        # drops them. Accept keys present on only one side *iff* the value
        # on that side is within ``atol`` of zero — this still catches the
        # regression the ``strict`` flag is designed for (an interaction
        # that should be non-zero getting dropped) while tolerating the
        # encoding difference.
        only_actual_bad = {
            k for k in actual_non_empty - expected_non_empty if abs(float(actual[k])) > atol
        }
        only_expected_bad = {
            k for k in expected_non_empty - actual_non_empty if abs(float(expected[k])) > atol
        }
        assert not only_actual_bad and not only_expected_bad, (
            f"Non-zero interaction support mismatch: only_actual={only_actual_bad}, "
            f"only_expected={only_expected_bad}"
        )
    shared = actual_non_empty & expected_non_empty
    assert shared, "No non-empty interactions in common between IVs."
    for interaction in shared:
        expected_value = float(expected[interaction])
        actual_value = float(actual[interaction])
        assert actual_value == pytest.approx(expected_value, abs=atol), (
            f"Interaction {interaction}: expected {expected_value}, got {actual_value}"
        )
    if check_baseline:
        assert float(actual.baseline_value) == pytest.approx(
            float(expected.baseline_value), abs=atol
        )


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
skip_if_no_tabpfn = pytest.mark.skipif(not _is_installed("tabpfn"), reason="tabpfn not installed")

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
# SOUM games — analytically tractable ground-truth for all GROUND_TRUTH_INDICES
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def soum_5():
    """5-player SOUM used for fast cross-checks (2^5 = 32 coalitions).

    Deliberately non-trivial: ``max_interaction_size=n`` forces the game to
    contain interactions of all orders up to ``n``, including orders strictly
    above ``max_order=2``. This is what separates *consistent* approximators
    (which recover exact values at budget ``2**n``) from *inconsistent* ones
    (like ``InconsistentKernelSHAPIQ``) that only look exact when the true
    game happens to sit in a k-additive regime.
    """
    return SOUM(
        n=5,
        n_basis_games=25,
        min_interaction_size=1,
        max_interaction_size=5,
        random_state=42,
    )


@pytest.fixture(scope="module")
def soum_7():
    """7-player SOUM used for slow cross-checks and convergence tests.

    Like :func:`soum_5` but larger; ``max_interaction_size=n`` keeps the game
    genuinely non-k-additive.
    """
    return SOUM(
        n=7,
        n_basis_games=40,
        min_interaction_size=1,
        max_interaction_size=7,
        random_state=42,
    )


@pytest.fixture(scope="module")
def exact_soum_5(soum_5):
    """Cached ``ExactComputer`` for :func:`soum_5` — avoids re-running the
    brute-force transform once per parametrised test.
    """
    return ExactComputer(soum_5)


@pytest.fixture(scope="module")
def exact_soum_7(soum_7):
    """Cached ``ExactComputer`` for :func:`soum_7`."""
    return ExactComputer(soum_7)


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
_TINY_Y_CLF = (np.median(_TINY_Y_REG) < _TINY_Y_REG).astype(int)


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
