"""Shared fixtures for all shapiq tests."""

from __future__ import annotations

import os

# Limit OpenMP threads to prevent segfaults when PyTorch/sklearn coexist.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from typing import TYPE_CHECKING

import matplotlib as mpl
import pytest

mpl.use("Agg")

from shapiq.game_theory.exact import ExactComputer
from shapiq_games.synthetic import SOUM

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
# Seeded SOUM variants — used by analytical cross-checks to exercise
# multiple random game instances. Each fixture is module-scoped and
# parametrised over ``GROUND_TRUTH_SEEDS``; pytest expands it into one
# test per (seed, index) combination. Keeps CI fully deterministic while
# surfacing conditioning edge cases that a single fixed game would hide.
# ---------------------------------------------------------------------------

GROUND_TRUTH_SEEDS: tuple[int, ...] = (42, 1337, 7, 2024, 31415)


@pytest.fixture(scope="module", params=GROUND_TRUTH_SEEDS)
def soum_5_seeded(request):
    """5-player SOUM, parametrised across several fixed seeds."""
    return SOUM(
        n=5,
        n_basis_games=25,
        min_interaction_size=1,
        max_interaction_size=5,
        random_state=request.param,
    )


@pytest.fixture(scope="module")
def exact_soum_5_seeded(soum_5_seeded):
    """Cached ``ExactComputer`` for :func:`soum_5_seeded`."""
    return ExactComputer(soum_5_seeded)


@pytest.fixture(scope="module", params=GROUND_TRUTH_SEEDS)
def soum_7_seeded(request):
    """7-player SOUM, parametrised across several fixed seeds."""
    return SOUM(
        n=7,
        n_basis_games=40,
        min_interaction_size=1,
        max_interaction_size=7,
        random_state=request.param,
    )


@pytest.fixture(scope="module")
def exact_soum_7_seeded(soum_7_seeded):
    """Cached ``ExactComputer`` for :func:`soum_7_seeded`."""
    return ExactComputer(soum_7_seeded)
