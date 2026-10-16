"""Does the interventional tree explainer compute the right values?

The oracle is the exact interventional game: ``InterventionalTreeGame`` evaluates the model under
every coalition (coalition features from ``x``, the others from each reference row) and
``ExactComputer`` turns the game into any interaction index. The explainer runs the cohort fill
over a background dataset and the single-reference fill over one row; both must equal the exact
game, and the cohort fill must equal the average of the single-reference fills over its rows.
"""

from __future__ import annotations

from math import comb

import numpy as np
import pytest

from shapiq.game_theory.exact import ExactComputer
from shapiq.tree import InterventionalTreeSHAPIQ
from shapiq_games.tree import InterventionalTreeGame

# every index the explainer supports, at the orders its closed-form weights cover
INDEX_ORDERS = [
    ("SV", 1),
    ("BV", 1),
    ("FBII", 1),
    ("SII", 2),
    ("BII", 2),
    ("CHII", 2),
    ("FBII", 2),
    ("FSII", 2),
    ("STII", 2),
    ("Moebius", 2),
    ("Fourier", 2),
    ("SII", 3),
    ("BII", 3),
    ("CHII", 3),
    ("FBII", 3),
    ("FSII", 3),
    ("STII", 3),
    ("Moebius", 3),
    ("Fourier", 3),
    ("STII", 4),
    ("FSII", 4),
    ("Moebius", 4),
]

# (model fixture, (X_train, X_test, ...) fixture, class_index)
MODEL_FAMILIES = [
    ("dt_clf_model", "cls_data", 1),
    ("rf_reg_model", "reg_data", None),
    ("rf_clf_model", "cls_data", 1),
    ("xgb_reg_model", "reg_data", None),
    ("xgb_clf_model", "cls_data", 1),
    ("lightgbm_reg_model", "reg_data", None),
    ("lightgbm_clf_model", "cls_data", 1),
]


def _exact_interactions(model, reference, x, index, order, class_index=None) -> dict:
    """Every interaction up to ``order`` of the exact interventional game, by ``ExactComputer``.

    ``ExactComputer`` returns the Moebius transform at every order, hence the filter.
    """
    game = InterventionalTreeGame(model, reference, x, class_index=class_index)
    interactions = ExactComputer(game)(index, order).interactions
    return {key: value for key, value in interactions.items() if len(key) <= order}


def _assert_equal_interactions(got: dict, expected: dict, *, atol: float) -> None:
    """Both dicts hold the same values on every non-empty interaction (missing means zero)."""
    for interaction in {*got, *expected}:
        if len(interaction) == 0:
            continue
        assert got.get(interaction, 0.0) == pytest.approx(
            expected.get(interaction, 0.0), abs=atol
        ), interaction


def _average_single_reference(model, reference_data, x, index, order, class_index=None) -> dict:
    """The cohort result computed the slow way: one single-reference explainer per row.

    Both fills share the layout and the leaf update; the cohort fill over a dataset must equal
    the average of the single-reference fill over its rows. Interactions that are exactly zero
    in one explainer are missing from its dict.
    """
    total: dict = {}
    for row in reference_data:
        single = InterventionalTreeSHAPIQ(
            model, row[None, :], index=index, max_order=order, class_index=class_index
        )
        assert single.reference_data.shape[0] == 1
        for key, value in single.explain_function(x).dict_values.items():
            total[key] = total.get(key, 0.0) + value / reference_data.shape[0]
    return total


@pytest.mark.parametrize(("index", "order"), INDEX_ORDERS)
def test_every_index_matches_the_exact_game(dt_reg_model, reg_data, index, order):
    """The cohort fill equals the exact game for every supported index at orders 1 to 4."""
    X_train, X_test, _, _ = reg_data
    x = X_test[0]
    got = InterventionalTreeSHAPIQ(dt_reg_model, X_train, index=index, max_order=order)
    _assert_equal_interactions(
        got.explain_function(x).interactions,
        _exact_interactions(dt_reg_model, X_train, x, index, order),
        atol=1e-6,
    )


@pytest.mark.parametrize(("index", "order"), [("SII", 3), ("STII", 2)])
@pytest.mark.parametrize(("model_fixture", "data_fixture", "class_index"), MODEL_FAMILIES)
def test_every_model_family_matches_the_exact_game(
    request, model_fixture, data_fixture, class_index, index, order
):
    """Converted sklearn, XGBoost and LightGBM trees, classifiers and regressors, equal the exact game."""
    model = request.getfixturevalue(model_fixture)
    X_train, X_test, _, _ = request.getfixturevalue(data_fixture)
    x = X_test[0]
    got = InterventionalTreeSHAPIQ(
        model, X_train, index=index, max_order=order, class_index=class_index
    )
    _assert_equal_interactions(
        got.explain_function(x).interactions,
        _exact_interactions(model, X_train, x, index, order, class_index),
        atol=1e-5 if "xgb" in model_fixture else 1e-6,  # XGBoost leaves are float32
    )


@pytest.mark.parametrize(("index", "order"), [("SII", 3), ("FBII", 2), ("STII", 4)])
def test_single_reference_fill_matches_the_exact_game(dt_reg_model, reg_data, index, order):
    """An explainer holding one reference row (the single-reference fill) equals the exact game."""
    X_train, X_test, _, _ = reg_data
    reference, x = X_train[:1], X_test[0]
    got = InterventionalTreeSHAPIQ(dt_reg_model, reference, index=index, max_order=order)
    assert got.reference_data.shape[0] == 1
    _assert_equal_interactions(
        got.explain_function(x).interactions,
        _exact_interactions(dt_reg_model, reference, x, index, order),
        atol=1e-6,
    )


@pytest.mark.parametrize(
    ("model_fixture", "data_fixture", "order"),
    [
        ("rf_clf_model", "background_clf_data", 3),
        ("hist_gb_cat_reg_model", "background_cat_dataset", 2),  # categorical splits, NaN rows
        ("xgb_cat_reg_model", "background_cat_dataset", 2),
    ],
)
def test_cohort_fill_equals_averaged_single_reference_fills(
    request, model_fixture, data_fixture, order
):
    """The cohort fill over 20 rows equals the average of the single-reference fills over them.

    The categorical models route in-set values and NaN (to the missing child) in both fills.
    """
    model = request.getfixturevalue(model_fixture)
    data = request.getfixturevalue(data_fixture)
    X = np.asarray(data[0] if isinstance(data, tuple) else data, dtype=float)
    reference = X[:20]
    cohort = InterventionalTreeSHAPIQ(model, reference, index="SII", max_order=order)
    for x in X[20:23]:
        got = cohort.explain_function(x).dict_values
        expected = _average_single_reference(model, reference, x, "SII", order)
        assert set(got) == set(expected)
        for key, value in expected.items():
            assert got[key] == pytest.approx(value, rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_custom_weight_fn_reproduces_sii(dt_reg_model, reg_data, order):
    """A ``weight_fn`` carrying the SII discrete-derivative weights equals ``index="SII"``.

    The SII weight of a coalition of size ``t`` for an interaction of size ``s`` among ``n``
    players is ``1 / ((n - s + 1) * C(n - s, t))``.
    """
    X_train, X_test, _, _ = reg_data

    def sii_weight(coalition_size: int, interaction_size: int, n: int) -> float:
        # the Möbius conversion asks for coalitions the interaction does not fit into: weight 0
        if coalition_size < 0 or n < interaction_size:
            return 0.0
        return 1.0 / ((n - interaction_size + 1) * comb(n - interaction_size, coalition_size))

    reference = X_train[:20]
    built_in = InterventionalTreeSHAPIQ(dt_reg_model, reference, index="SII", max_order=order)
    custom = InterventionalTreeSHAPIQ(
        dt_reg_model, reference, max_order=order, weight_fn=sii_weight
    )
    assert custom.index == "CUSTOM"
    for x in X_test[:3]:
        expected = built_in.explain_function(x).dict_values
        got = custom.explain_function(x).dict_values
        assert set(got) == set(expected)
        for key, value in expected.items():
            assert got[key] == pytest.approx(value, rel=1e-9, abs=1e-12)
