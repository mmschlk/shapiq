"""Rigorous correctness tests for the multi-output interventional tree stack.

This module validates the *Python* layer built on top of the (already
C-verified) fused multi-output interaction kernel:

* the dense-offset -> interaction-tuple decoding,
* the per-output interventional baseline handling,
* :class:`~shapiq.interaction_values.InteractionValues` construction, and
* the ``n_players`` path (proxy not splitting on every feature).

Three independent oracles are used:

Oracle A
    The multi-output explainer must agree, for every output column, with the
    *scalar* :class:`~shapiq.tree.interventional.explainer.InterventionalTreeExplainer`
    run on the equivalent scalar tree (slice column ``j`` of every converted
    multi-output tree into a scalar :class:`~shapiq.tree.base.TreeModel`).

Oracle B
    For a small number of players the multi-output explainer must agree with a
    fully independent brute-force exact computation over all ``2**n``
    coalitions, via :class:`~shapiq.game_theory.exact.ExactComputer`.

Oracle C
    :class:`~shapiq.approximator.proxy._multioutput.proxyshap.MultiOutputProxySHAP`
    end-to-end: each output's :class:`InteractionValues` is internally
    consistent (baseline == empty-coalition game value, SV efficiency axiom).
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
from shapiq.approximator.proxy._multioutput.explainer import (  # noqa: E402
    MultiOutputInterventionalTreeExplainer,
)
from shapiq.approximator.proxy._multioutput.tree import (  # noqa: E402
    convert_multioutput_xgboost,
)
from shapiq.game_theory.exact import ExactComputer  # noqa: E402
from shapiq.tree.base import TreeModel  # noqa: E402
from shapiq.tree.interventional.explainer import InterventionalTreeExplainer  # noqa: E402

# Leaf vectors flow through float32 inside XGBoost / the kernel, so do not
# demand a tighter tolerance than the kernel-level test (Phase 2) uses.
ATOL = 1e-5
RTOL = 1e-5


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _fit_multioutput_proxy(
    *,
    n_features: int,
    n_outputs: int,
    n_samples: int,
    random_state: int,
    max_depth: int = 4,
    n_estimators: int = 3,
) -> tuple[object, np.ndarray, np.ndarray]:
    """Fit an XGBoost multi-output proxy on synthetic 0/1 coalition data.

    Args:
        n_features: Number of players ``n``.
        n_outputs: Output dimensionality ``c``.
        n_samples: Number of training rows.
        random_state: Seed for the data and the model.
        max_depth: ``max_depth`` of the proxy.
        n_estimators: ``n_estimators`` of the proxy.

    Returns:
        A ``(fitted_model, X, y)`` tuple.
    """
    rng = np.random.default_rng(random_state)
    x_data = rng.integers(0, 2, size=(n_samples, n_features)).astype(np.float64)
    # A target that genuinely depends on several features + interactions so the
    # trees actually split.
    base = (
        x_data[:, 0]
        + 0.5 * x_data[:, 1] * x_data[:, 2 % n_features]
        - 0.3 * x_data[:, 3 % n_features]
        + 0.7 * x_data[:, 4 % n_features] * x_data[:, 5 % n_features]
    )
    coeffs = rng.normal(size=n_outputs)
    offsets = rng.normal(size=n_outputs)
    y_data = base[:, None] * coeffs[None, :] + offsets[None, :]
    # Add per-output noise so the columns are not exact scalar multiples.
    y_data = y_data + 0.1 * rng.normal(size=y_data.shape)

    model = xgboost.XGBRegressor(
        multi_strategy="multi_output_tree",
        max_depth=max_depth,
        n_estimators=n_estimators,
        objective="reg:squarederror",
        random_state=random_state,
    )
    model.fit(x_data, y_data)
    return model, x_data, y_data


def _scalar_tree_from_column(multi_tree: object, column: int) -> TreeModel:
    """Slice column ``column`` of a :class:`MultiOutputTreeModel` into a scalar tree.

    The resulting scalar :class:`~shapiq.tree.base.TreeModel` is structurally
    identical to the multi-output tree (same topology, same splits) but carries
    a scalar leaf value -- exactly the model the scalar
    :class:`InterventionalTreeExplainer` is the oracle for.

    Args:
        multi_tree: A :class:`MultiOutputTreeModel`.
        column: The output column to extract.

    Returns:
        The equivalent scalar :class:`~shapiq.tree.base.TreeModel`.
    """
    return TreeModel(
        children_left=multi_tree.children_left.astype(np.int64),
        children_right=multi_tree.children_right.astype(np.int64),
        children_missing=multi_tree.children_default.astype(np.int64),
        features=multi_tree.features.astype(np.int64),
        thresholds=multi_tree.thresholds.astype(np.float64),
        values=multi_tree.values[:, column].astype(np.float64).copy(),
        node_sample_weight=np.ones(multi_tree.n_nodes, dtype=np.float64),
        leaf_mask=multi_tree.leaf_mask.copy(),
        decision_type="<",
    )


def _scalar_reference(
    multi_trees: list,
    column: int,
    *,
    n_players: int,
    index: str,
    max_order: int,
) -> dict:
    """Run the scalar :class:`InterventionalTreeExplainer` on one output column.

    Args:
        multi_trees: The converted multi-output trees.
        column: The output column to explain.
        n_players: Number of players ``n``.
        index: Interaction index.
        max_order: Maximum interaction order.

    Returns:
        The scalar explainer's ``interactions`` dict (incl. the ``()`` entry).
    """
    scalar_trees = [_scalar_tree_from_column(t, column) for t in multi_trees]
    explainer = InterventionalTreeExplainer(
        scalar_trees,
        data=np.zeros((1, n_players), dtype=np.float64),
        index=index,
        max_order=max_order,
        bool_tree=True,
    )
    # For a boolean tree the explain point is irrelevant; the kernel enumerates
    # all coalitions in {0, 1}^n itself.
    return explainer.explain_function(np.ones(n_players, dtype=np.float64)).interactions


# --------------------------------------------------------------------------- #
# Oracle A: multi explainer vs per-column scalar InterventionalTreeExplainer
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("index", "max_order"),
    [("SV", 1), ("SII", 2), ("SII", 3)],
)
@pytest.mark.parametrize(
    ("n_features", "n_outputs"),
    [(6, 3), (10, 5), (14, 8)],
)
def test_oracle_a_multi_matches_scalar(
    index: str,
    max_order: int,
    n_features: int,
    n_outputs: int,
) -> None:
    """The multi explainer must equal the scalar oracle for every output column."""
    model, _x, _y = _fit_multioutput_proxy(
        n_features=n_features,
        n_outputs=n_outputs,
        n_samples=256,
        random_state=0,
    )
    multi_trees = convert_multioutput_xgboost(model)

    multi_explainer = MultiOutputInterventionalTreeExplainer(
        model,
        index=index,
        max_order=max_order,
        n_players=n_features,
    )
    multi_results = multi_explainer.explain()

    assert len(multi_results) == n_outputs

    for j in range(n_outputs):
        scalar_interactions = _scalar_reference(
            multi_trees, j, n_players=n_features, index=index, max_order=max_order
        )
        multi_iv = multi_results[j]

        # Every interaction tuple the scalar oracle reports -- including the
        # empty-set baseline -- must match the multi explainer.
        for interaction, scalar_value in scalar_interactions.items():
            multi_value = multi_iv[interaction]
            assert np.allclose(multi_value, scalar_value, atol=ATOL, rtol=RTOL), (
                f"mismatch output={j} interaction={interaction} "
                f"(index={index}, max_order={max_order}): "
                f"multi={multi_value} scalar={scalar_value}"
            )

        # And the reverse: no extra non-zero interaction in the multi result.
        for interaction in multi_iv.interaction_lookup:
            if interaction == ():
                continue
            scalar_value = scalar_interactions.get(interaction, 0.0)
            assert np.allclose(multi_iv[interaction], scalar_value, atol=ATOL, rtol=RTOL), (
                f"extra interaction {interaction} in multi result for output {j}"
            )

        # Structural contract.
        assert multi_iv.n_players == n_features
        assert multi_iv.max_order == max_order
        assert multi_iv[()] == pytest.approx(multi_iv.baseline_value)


def test_oracle_a_n_players_path() -> None:
    """A proxy that does not split on every feature still yields n_players slots.

    A shallow, few-estimator proxy on a wide feature space provably leaves some
    trailing features unused. The explainer must still report ``n_players``
    players (because it is passed explicitly) and assign exactly zero to every
    interaction touching an unused feature.
    """
    n_features = 16
    n_outputs = 4
    # Shallow + few trees on a wide space -> guaranteed unused features.
    model, x_data, _y = _fit_multioutput_proxy(
        n_features=n_features,
        n_outputs=n_outputs,
        n_samples=400,
        random_state=3,
        max_depth=2,
        n_estimators=2,
    )
    multi_trees = convert_multioutput_xgboost(model)

    # Features actually used by any split in any tree.
    used_features: set[int] = set()
    for tree in multi_trees:
        used_features.update(int(f) for f in tree.features if f >= 0)
    unused_features = set(range(n_features)) - used_features
    # The configuration must genuinely leave some feature unused, otherwise the
    # edge case is not exercised.
    assert unused_features, "expected the shallow proxy to leave features unused"

    # If n_players is not passed it is *inferred* from the highest split index
    # and will undercount when a trailing feature is unused.
    inferred = MultiOutputInterventionalTreeExplainer(model, index="SII", max_order=2)
    max_used = max(used_features)
    assert inferred.n_players == max_used + 1

    # Passing n_players explicitly fixes the player count.
    explainer = MultiOutputInterventionalTreeExplainer(
        model, index="SII", max_order=2, n_players=n_features
    )
    results = explainer.explain()
    assert len(results) == n_outputs

    for iv in results:
        assert iv.n_players == n_features
        # Every interaction touching an unused feature must be exactly zero.
        for interaction in iv.interaction_lookup:
            if interaction and unused_features.intersection(interaction):
                assert iv[interaction] == pytest.approx(0.0, abs=ATOL), (
                    f"interaction {interaction} touches unused feature(s) "
                    f"{unused_features.intersection(interaction)} but is non-zero"
                )

    # Cross-check against the scalar oracle (which also gets the explicit n).
    for j in range(n_outputs):
        scalar_interactions = _scalar_reference(
            multi_trees, j, n_players=n_features, index="SII", max_order=2
        )
        for interaction, scalar_value in scalar_interactions.items():
            assert np.allclose(results[j][interaction], scalar_value, atol=ATOL, rtol=RTOL)


# --------------------------------------------------------------------------- #
# Oracle B: brute-force exact cross-check for small n
# --------------------------------------------------------------------------- #
def _forest_column_game(multi_trees: list, column: int) -> callable:
    """Wrap output ``column`` of the proxy forest as a scalar set function.

    The value of a coalition ``S`` is the forest prediction (no base score) on
    the 0/1 indicator vector of ``S`` -- exactly the boolean-tree value function
    the interventional explainer computes interactions for.

    Args:
        multi_trees: The converted multi-output trees.
        column: The output column to wrap.

    Returns:
        A callable ``game(coalition_matrix) -> values`` of the shape expected by
        :class:`~shapiq.game_theory.exact.ExactComputer`.
    """

    def game(coalitions: np.ndarray) -> np.ndarray:
        coalitions = np.asarray(coalitions, dtype=np.float64)
        out = np.zeros(coalitions.shape[0], dtype=np.float64)
        for tree in multi_trees:
            preds = tree.predict(coalitions)  # (n_coalitions, n_outputs)
            out += preds[:, column]
        return out

    return game


@pytest.mark.parametrize(
    ("index", "max_order"),
    [("SV", 1), ("SII", 2)],
)
@pytest.mark.parametrize("n_features", [6, 9])
def test_oracle_b_bruteforce_exact(index: str, max_order: int, n_features: int) -> None:
    """The multi explainer must match an independent brute-force exact computation."""
    n_outputs = 3
    model, _x, _y = _fit_multioutput_proxy(
        n_features=n_features,
        n_outputs=n_outputs,
        n_samples=300,
        random_state=1,
    )
    multi_trees = convert_multioutput_xgboost(model)

    explainer = MultiOutputInterventionalTreeExplainer(
        model,
        index=index,
        max_order=max_order,
        n_players=n_features,
    )
    results = explainer.explain()

    for j in range(n_outputs):
        game = _forest_column_game(multi_trees, j)
        exact_computer = ExactComputer(game=game, n_players=n_features)
        exact_values = exact_computer(index, max_order)
        exact_interactions = exact_values.interactions

        multi_iv = results[j]
        # Compare every interaction the exact computer reports for the
        # explainer's orders (1 .. max_order).
        for interaction, exact_value in exact_interactions.items():
            if not interaction or len(interaction) > max_order:
                continue
            assert np.allclose(multi_iv[interaction], exact_value, atol=ATOL, rtol=RTOL), (
                f"brute-force mismatch output={j} interaction={interaction} "
                f"(index={index}, max_order={max_order}): "
                f"multi={multi_iv[interaction]} exact={exact_value}"
            )

        # The empty-coalition baseline must equal the brute-force v({}) -- the
        # forest prediction on the all-zeros indicator.
        v_empty = float(game(np.zeros((1, n_features)))[0])
        assert multi_iv.baseline_value == pytest.approx(v_empty, abs=ATOL)


# --------------------------------------------------------------------------- #
# Oracle C: MultiOutputProxySHAP end-to-end consistency
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("index", "max_order"),
    [("SV", 1), ("SII", 2)],
)
def test_oracle_c_proxyshap_end_to_end(index: str, max_order: int) -> None:
    """MultiOutputProxySHAP results are internally consistent per output."""
    n_features = 7
    n_classes = 3
    x_data, y_data = make_classification(
        n_samples=300,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=0,
    )
    clf = RandomForestClassifier(n_estimators=25, random_state=0)
    clf.fit(x_data, y_data)

    game = MultiOutputMarginalGame(
        clf,
        background_data=x_data,
        x=x_data[0],
        max_background_samples=40,
        random_state=0,
    )
    assert game.n_outputs == n_classes

    approximator = MultiOutputProxySHAP(
        n=n_features, max_order=max_order, index=index, random_state=0
    )
    # Full enumeration keeps the proxy a faithful fit so the axioms hold tightly.
    budget = 2**n_features
    results = approximator.approximate(budget=budget, game=game)

    assert len(results) == n_classes

    # True per-output empty-coalition value.
    empty_value = game(np.zeros((1, n_features), dtype=np.int64))[0]
    grand = game.grand_coalition_value()

    for j, iv in enumerate(results):
        assert iv.n_players == n_features
        assert iv.max_order == max_order
        # () baseline must equal the true empty-coalition game value.
        assert () in iv.interaction_lookup
        assert iv.baseline_value == pytest.approx(empty_value[j], abs=1e-6)
        assert iv[()] == pytest.approx(iv.baseline_value)

        if index == "SV":
            # Efficiency: order-1 values + baseline ~= v(grand coalition).
            order_one_sum = sum(
                value for interaction, value in iv.interactions.items() if len(interaction) == 1
            )
            reconstructed = order_one_sum + iv.baseline_value
            assert reconstructed == pytest.approx(grand[j], abs=5e-2)
