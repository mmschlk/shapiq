"""Cross-approximator conformance + numerical convergence tests.

Parametrized tests run any registered Shapley-value approximator that
appears in :mod:`shapiq.approximator` against ``ExactComputer`` ground
truth on small SOUM games. Two layers:

1. Interface conformance: verifies ``index``, ``n_players``, ``max_order``,
   ``min_order``, and ``values`` shape on the returned ``InteractionValues``.

2. Numerical convergence vs ``ExactComputer``: marked ``xfail`` for
   approximators that are still under development; once their algorithm is
   implemented the marker can be removed.

Approximator classes are looked up dynamically by name so this file is
useful as a forward-looking spec — it auto-skips classes that have not
yet been registered (e.g. while a teammate's branch is still in flight)
rather than failing at import.

Run only one approximator's tests::

    pytest tests/shapiq/tests_unit/tests_approximators/test_approximators_vs_exact.py \\
        -k LeverageSHAP -v
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from shapiq import ExactComputer
from shapiq_games.synthetic import SOUM

# Names of the new SV approximators that should each satisfy the interface.
APPROXIMATOR_NAMES = ["LeverageSHAP", "PolySHAP", "OddSHAP"]


def _load_approximator(name: str):
    """Return the named approximator class, skipping the test if absent."""
    module = importlib.import_module("shapiq.approximator")
    cls = getattr(module, name, None)
    if cls is None:
        pytest.skip(f"{name} not yet registered in shapiq.approximator")
    return cls


@pytest.fixture
def seeded_soum_game(request):
    """Provide a deterministic SOUM benchmark game for a given (n, seed)."""
    n, seed = request.param
    return SOUM(n=n, n_basis_games=20, max_interaction_size=3, random_state=seed)


# -----------------------------------------------------------------------------
# Conformance assertions (always required)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("approx_name", APPROXIMATOR_NAMES)
@pytest.mark.parametrize(
    "seeded_soum_game",
    [(4, 0), (6, 42), (8, 1337), (10, 7)],
    indirect=True,
)
@pytest.mark.parametrize("budget_pct", [0.05, 0.25, 1.0])
def test_interface_conformance(approx_name, seeded_soum_game, budget_pct):
    """Interface contract — must hold for any conforming approximator."""
    Approx = _load_approximator(approx_name)
    n = seeded_soum_game.n_players
    estimator = Approx(n=n, random_state=0)
    iv = estimator.approximate(int(budget_pct * 2 ** n), seeded_soum_game)

    assert iv.index == "SV"
    assert iv.n_players == n
    assert iv.max_order == 1
    assert iv.min_order == 0
    assert iv.values.shape == (n + 1,)
    assert iv.values.dtype == float

    # Standard SV interaction_lookup:
    assert iv.interaction_lookup[()] == 0
    for i in range(n):
        assert iv.interaction_lookup[(i,)] == i + 1


# -----------------------------------------------------------------------------
# Numerical convergence vs ExactComputer
#
# Marked xfail with strict=False so it is silently allowed to fail today
# (stub implementations) and silently allowed to pass tomorrow. Drop the
# marker per-approximator as their algorithm lands.
# -----------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="Approximator may still be a stub. "
    "Convergence will hold once the algorithm is implemented; "
    "remove this xfail marker per-approximator as implementations land.",
    strict=False,
)
@pytest.mark.parametrize("approx_name", APPROXIMATOR_NAMES)
@pytest.mark.parametrize(
    "seeded_soum_game",
    [(4, 0), (6, 42), (8, 1337), (10, 7)],
    indirect=True,
)
@pytest.mark.parametrize("budget_pct", [0.05, 0.25, 1.0])
def test_numerical_convergence_vs_exact(approx_name, seeded_soum_game, budget_pct):
    """Numerical accuracy vs ExactComputer ground truth.

    Tolerance schedule:
        budget_pct < 0.25   atol = 0.5
        budget_pct < 1.0    atol = 0.1
        budget_pct == 1.0   atol = 1e-2  (sub-budget regime)
    """
    Approx = _load_approximator(approx_name)
    n = seeded_soum_game.n_players
    estimator = Approx(n=n, random_state=0)
    iv = estimator.approximate(int(budget_pct * 2 ** n), seeded_soum_game)

    exact = ExactComputer(seeded_soum_game, n_players=n)(index="SV")

    if budget_pct < 0.25:
        atol = 0.5
    elif budget_pct < 1.0:
        atol = 0.1
    else:
        atol = 1e-2

    np.testing.assert_allclose(iv.values, exact.values, atol=atol)


# -----------------------------------------------------------------------------
# Determinism (same seed → bit-identical output)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("approx_name", APPROXIMATOR_NAMES)
def test_determinism(approx_name):
    """Identical (n, random_state, budget, game) → identical output."""
    Approx = _load_approximator(approx_name)
    game = SOUM(n=8, n_basis_games=20, max_interaction_size=3, random_state=42)
    a = Approx(n=8, random_state=42).approximate(256, game)
    b = Approx(n=8, random_state=42).approximate(256, game)
    np.testing.assert_array_equal(a.values, b.values)
