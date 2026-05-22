"""End-to-end smoke test for the multi-output ProxySHAP approximator.

These tests exercise the full Phase-3 pipeline: build a multiclass
classification dataset, train a classifier, wrap one instance in a
:class:`MultiOutputMarginalGame`, run
:meth:`MultiOutputProxySHAP.approximate`, and check the structural contract of
the returned per-output :class:`~shapiq.interaction_values.InteractionValues`.

Rigorous correctness against per-column scalar runs is intentionally *not*
checked here -- that is a later phase. The strongest numerical assertion made
is the Shapley-value efficiency axiom (order-1 values + baseline sum to the
grand-coalition value), and only within a loose tolerance because the proxy
tree fitted on a finite coalition budget is an approximation of the game.
"""

from __future__ import annotations

import numpy as np
import pytest

xgboost = pytest.importorskip("xgboost")

from sklearn.datasets import make_classification  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402

from shapiq.approximator.proxy._multioutput import (  # noqa: E402
    MultiOutputMarginalGame,
    MultiOutputProxySHAP,
)
from shapiq.interaction_values import InteractionValues  # noqa: E402


def _make_setup(
    *,
    n_features: int,
    n_classes: int,
    n_samples: int,
    random_state: int,
) -> tuple[MultiOutputMarginalGame, int]:
    """Build a multiclass dataset, train a classifier, and wrap one instance.

    Args:
        n_features: Number of (informative) features.
        n_classes: Number of target classes ``c``.
        n_samples: Number of dataset rows.
        random_state: Seed for the dataset, model and game.

    Returns:
        A ``(game, n_features)`` tuple.
    """
    x_data, y_data = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=random_state,
    )
    clf = RandomForestClassifier(n_estimators=25, random_state=random_state)
    clf.fit(x_data, y_data)

    instance = x_data[0]
    game = MultiOutputMarginalGame(
        clf,
        background_data=x_data,
        x=instance,
        max_background_samples=40,
        random_state=random_state,
    )
    return game, n_features


@pytest.mark.parametrize(
    ("index", "max_order"),
    [("SV", 1), ("SII", 2)],
)
@pytest.mark.parametrize(
    ("n_features", "n_classes", "n_samples", "random_state"),
    [
        (6, 3, 240, 0),
        (7, 4, 300, 1),
    ],
)
def test_multioutput_proxyshap_end_to_end(
    index: str,
    max_order: int,
    n_features: int,
    n_classes: int,
    n_samples: int,
    random_state: int,
) -> None:
    """MultiOutputProxySHAP returns a well-formed InteractionValues per class."""
    game, n = _make_setup(
        n_features=n_features,
        n_classes=n_classes,
        n_samples=n_samples,
        random_state=random_state,
    )
    c = game.n_outputs
    assert c == n_classes

    approximator = MultiOutputProxySHAP(
        n=n,
        max_order=max_order,
        index=index,
        random_state=random_state,
    )
    # Full enumeration of the coalition cube keeps the proxy a faithful fit so
    # the efficiency axiom holds tightly.
    budget = 2**n
    results = approximator.approximate(budget=budget, game=game)

    # --- structural contract ---
    assert isinstance(results, list)
    assert len(results) == c
    for iv in results:
        assert isinstance(iv, InteractionValues)
        assert iv.n_players == n
        assert iv.max_order == max_order
        # The empty-set baseline entry must be populated.
        assert () in iv.interaction_lookup
        assert np.isfinite(iv.baseline_value)
        assert iv[()] == pytest.approx(iv.baseline_value)

    # --- Shapley efficiency axiom (per output) ---
    # sum of order-1 values + baseline ~= v(grand coalition) for that output.
    grand = game.grand_coalition_value()
    for j, iv in enumerate(results):
        order_one_sum = sum(
            value for interaction, value in iv.interactions.items() if len(interaction) == 1
        )
        reconstructed = order_one_sum + iv.baseline_value
        assert reconstructed == pytest.approx(grand[j], abs=5e-2)


def test_multioutput_proxyshap_sub_budget_runs() -> None:
    """A sub-exhaustive budget still produces a structurally valid result."""
    game, n = _make_setup(n_features=8, n_classes=3, n_samples=260, random_state=2)
    approximator = MultiOutputProxySHAP(n=n, max_order=2, index="SII", random_state=2)
    results = approximator.approximate(budget=128, game=game)

    assert len(results) == game.n_outputs
    for iv in results:
        assert iv.n_players == n
        assert iv.max_order == 2
        assert iv.estimated is True
        assert iv.estimation_budget == 128
