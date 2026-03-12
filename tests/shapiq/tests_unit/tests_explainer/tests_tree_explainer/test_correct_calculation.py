from __future__ import annotations

import numpy as np
import pytest

from shapiq.game_theory.exact import ExactComputer
from shapiq.tree import InterventionalGame, InterventionalTreeExplainer

SEED = 1337
np.random.seed(SEED)


@pytest.mark.parametrize(
    ("index", "order"),
    [
        ("SV", 1),
        ("BV", 1),
        ("SII", 2),
        ("BII", 2),
        ("CHII", 2),
        ("FBII", 2),
        ("FBII", 1),
        ("FSII", 2),
        ("STII", 2),
    ],
)
def test_correct_calculation_dt_reg_index_order(dt_reg_model, reg_data, index, order):
    X_train, X_test, _y_train, _y_test = reg_data
    model = dt_reg_model
    point_to_explain = X_test[0:1]

    # Our InterventionalTreeExplainer
    own_interventional_explainer = InterventionalTreeExplainer(
        model, X_train, index=index, max_order=order, debug=False
    )
    explanation = own_interventional_explainer.explain_function(point_to_explain.flatten())
    own_interactions = explanation.interactions

    # Interventional Game with Exact Computer
    interventional_game = InterventionalGame(model, X_train, point_to_explain.flatten())
    exact_computer = ExactComputer(interventional_game)
    exact_values = exact_computer(index, order)
    game_interactions = exact_values.interactions

    # Assertions that own Interventional Implementatoin matches Exact Computer
    for _i, interaction in enumerate(own_interactions.keys()):
        if index in ["FSII", "STII"]:
            if len(interaction) != order:
                continue
            # Only check full interactions for FBII and FSII due to the current code supporting only the discrete derivate formula
            assert np.isclose(
                own_interactions[interaction],
                game_interactions.get(interaction, 0),
                atol=1e-6,
            )
        elif len(interaction) > 0:
            assert np.isclose(
                own_interactions[interaction],
                game_interactions.get(interaction, 0),
                atol=1e-6,
            )


@pytest.mark.parametrize(
    ("index", "order"),
    [
        ("SV", 1),
        ("BV", 1),
        ("FBII", 1),
        ("SII", 2),
        ("BII", 2),
        ("CHII", 2),
        ("FBII", 2),
        ("FSII", 2),
        ("STII", 2),
        ("SII", 3),
        ("BII", 3),
        ("CHII", 3),
        ("FBII", 3),
        ("FSII", 3),
    ],
)
def test_correct_calculation_dt_clas_index_order(dt_clf_model, cls_data, index, order):
    CLASS_INDEX = 1
    X_train, X_test, _, _ = cls_data
    model = dt_clf_model
    point_to_explain = X_test[0:1]

    # Our InterventionalTreeExplainer
    own_interventional_explainer = InterventionalTreeExplainer(
        model,
        X_train,
        index=index,
        max_order=order,
        debug=False,
        class_index=CLASS_INDEX,
    )
    explanation = own_interventional_explainer.explain_function(point_to_explain.flatten())
    own_interactions = explanation.interactions

    # Interventional Game with Exact Computer
    interventional_game = InterventionalGame(
        model, X_train, point_to_explain.flatten(), class_index=CLASS_INDEX
    )
    exact_computer = ExactComputer(interventional_game)
    exact_values = exact_computer(index, order)
    game_interactions = exact_values.interactions

    # Assertions that own Interventional Implementatoin matches Exact Computer
    for _i, interaction in enumerate(own_interactions.keys()):
        if index in ["FSII", "STII"]:
            if len(interaction) != order:
                continue
            # Only check full interactions for FBII and FSII due to the current code supporting only the discrete derivate formula
            assert np.isclose(
                own_interactions[interaction],
                game_interactions.get(interaction, 0),
                atol=1e-6,
            )
        elif len(interaction) > 0:
            assert np.isclose(
                own_interactions[interaction],
                game_interactions.get(interaction, 0),
                atol=1e-6,
            )


@pytest.mark.parametrize(
    ("index", "order"),
    [
        ("SV", 1),
        ("BV", 1),
        ("FBII", 1),
        ("SII", 2),
        ("BII", 2),
        ("CHII", 2),
        ("FBII", 2),
        ("FSII", 2),
        ("STII", 2),
        ("SII", 3),
        ("BII", 3),
        ("CHII", 3),
        ("FBII", 3),
        ("FSII", 3),
    ],
)
def test_correct_calculation_rf_reg_index_order(rf_reg_model, reg_data, index, order):
    X_train, X_test, _y_train, _y_test = reg_data
    model = rf_reg_model
    point_to_explain = X_test[0:1]

    # Our InterventionalTreeExplainer
    own_interventional_explainer = InterventionalTreeExplainer(
        model, X_train, max_order=order, index=index, debug=False
    )
    explanation = own_interventional_explainer.explain_function(point_to_explain.flatten())
    own_interactions = explanation.interactions

    # Interventional Game with Exact Computer
    interventional_game = InterventionalGame(model, X_train, point_to_explain.flatten())
    exact_computer = ExactComputer(interventional_game)
    exact_values = exact_computer(index, order)
    game_interactions = exact_values.interactions

    # Assertions that own Interventional Implementatoin matches Exact Computer
    for _i, interaction in enumerate(own_interactions.keys()):
        if index in ["FSII", "STII"]:
            if len(interaction) != order:
                continue
            # Only check full interactions for FBII and FSII due to the current code supporting only the discrete derivate formula
            assert np.isclose(
                own_interactions[interaction],
                game_interactions.get(interaction, 0),
                atol=1e-6,
            )
        elif len(interaction) > 0:
            assert np.isclose(
                own_interactions[interaction],
                game_interactions.get(interaction, 0),
                atol=1e-6,
            )


@pytest.mark.parametrize(
    ("index", "order"),
    [
        ("SV", 1),
        ("BV", 1),
        ("FBII", 1),
        ("SII", 2),
        ("BII", 2),
        ("CHII", 2),
        ("FBII", 2),
        ("FSII", 2),
        ("STII", 2),
        ("SII", 3),
        ("BII", 3),
        ("CHII", 3),
        ("FBII", 3),
        ("FSII", 3),
    ],
)
def test_correct_calculation_rf_clas_index_order(rf_clf_model, cls_data, index, order):
    CLASS_INDEX = 1
    X_train, X_test, _, _ = cls_data
    model = rf_clf_model
    point_to_explain = X_test[0:1]

    # Our InterventionalTreeExplainer
    own_interventional_explainer = InterventionalTreeExplainer(
        model,
        X_train,
        max_order=order,
        index=index,
        debug=False,
        class_index=CLASS_INDEX,
    )
    explanation = own_interventional_explainer.explain_function(point_to_explain.flatten())
    own_interactions = explanation.interactions

    # Interventional Game with Exact Computer
    interventional_game = InterventionalGame(
        model, X_train, point_to_explain.flatten(), class_index=CLASS_INDEX
    )
    exact_computer = ExactComputer(interventional_game)
    exact_values = exact_computer(index, order)
    game_interactions = exact_values.interactions

    # Assertions that own Interventional Implementatoin matches Exact Computer
    for interaction in own_interactions:
        if index in ["FSII", "STII"]:
            if len(interaction) != order:
                continue
            # Only check full interactions for FBII and FSII due to the current code supporting only the discrete derivate formula
            assert np.isclose(
                own_interactions[interaction],
                game_interactions.get(interaction, 0),
                atol=1e-6,
            )
        elif len(interaction) > 0:
            assert np.isclose(
                own_interactions[interaction],
                game_interactions.get(interaction, 0),
                atol=1e-6,
            )


@pytest.mark.parametrize(
    ("index", "order"),
    [
        ("SV", 1),
        ("BV", 1),
        ("FBII", 1),
        ("SII", 2),
        ("BII", 2),
        ("CHII", 2),
        ("FBII", 2),
        ("FSII", 2),
        ("STII", 2),
        ("SII", 3),
        ("BII", 3),
        ("CHII", 3),
        ("FBII", 3),
        ("FSII", 3),
    ],
)
def test_correct_calculation_xgb_reg_index_order(xgb_reg_model, reg_data, index, order):
    X_train, X_test, _, _ = reg_data
    model = xgb_reg_model
    point_to_explain = X_test[0:1]

    # Our InterventionalTreeExplainer
    own_interventional_explainer = InterventionalTreeExplainer(
        model, X_train, max_order=order, index=index, debug=False
    )
    explanation = own_interventional_explainer.explain_function(point_to_explain.flatten())
    own_interactions = explanation.interactions

    # Interventional Game with Exact Computer
    interventional_game = InterventionalGame(model, X_train, point_to_explain.flatten())
    exact_computer = ExactComputer(interventional_game)
    exact_values = exact_computer(index, order)
    game_interactions = exact_values.interactions

    # Assertions that own Interventional Implementation matches Exact Computer
    for _, interaction in enumerate(own_interactions.keys()):
        if index in ["FSII", "STII"]:
            if len(interaction) != order:
                continue
            # Using 1e-5 tolerance due to float32 vs float64 precision differences in XGBoost calculations
            assert np.isclose(
                own_interactions[interaction],
                game_interactions.get(interaction, 0),
                atol=1e-5,
            )
        elif len(interaction) > 0:
            assert np.isclose(
                own_interactions[interaction],
                game_interactions.get(interaction, 0),
                atol=1e-5,
            )


@pytest.mark.parametrize(
    ("index", "order"),
    [
        ("SV", 1),
        ("BV", 1),
        ("FBII", 1),
        ("SII", 2),
        ("BII", 2),
        ("CHII", 2),
        ("FBII", 2),
        ("FSII", 2),
        ("STII", 2),
        ("SII", 3),
        ("BII", 3),
        ("CHII", 3),
        ("FBII", 3),
        ("FSII", 3),
    ],
)
def test_correct_calculation_xgb_clas_index_order(xgb_clf_model, cls_data, index, order):
    CLASS_INDEX = 1
    X_train, X_test, _y_train, _y_test = cls_data
    model = xgb_clf_model
    point_to_explain = X_test[0:1]

    # Our InterventionalTreeExplainer
    own_interventional_explainer = InterventionalTreeExplainer(
        model,
        X_train,
        max_order=order,
        index=index,
        debug=False,
        class_index=CLASS_INDEX,
    )
    explanation = own_interventional_explainer.explain_function(point_to_explain.flatten())
    own_interactions = explanation.interactions

    # Interventional Game with Exact Computer
    interventional_game = InterventionalGame(
        model, X_train, point_to_explain.flatten(), class_index=CLASS_INDEX
    )
    exact_computer = ExactComputer(interventional_game)
    exact_values = exact_computer(index, order)
    game_interactions = exact_values.interactions

    # Assertions that own Interventional Implementatoin matches Exact Computer
    for _, interaction in enumerate(own_interactions.keys()):
        if index in ["FSII", "STII"]:
            if len(interaction) != order:
                continue
            # Only check full interactions for FBII and FSII due to the current code supporting only the discrete derivate formula
            assert np.isclose(
                own_interactions[interaction],
                game_interactions.get(interaction, 0),
                atol=1e-6,
            )
        elif len(interaction) > 0:
            assert np.isclose(
                own_interactions[interaction],
                game_interactions.get(interaction, 0),
                atol=1e-6,
            )


@pytest.mark.parametrize(
    ("index", "order"),
    [
        ("SV", 1),
        ("BV", 1),
        ("FBII", 1),
        ("SII", 2),
        ("BII", 2),
        ("CHII", 2),
        ("FBII", 2),
        ("FSII", 2),
        ("STII", 2),
        ("SII", 3),
        ("BII", 3),
        ("CHII", 3),
        ("FBII", 3),
        ("FSII", 3),
    ],
)
def test_correct_calculation_lgbm_reg_index_order(lightgbm_reg_model, reg_data, index, order):
    X_train, X_test, _y_train, _y_test = reg_data
    model = lightgbm_reg_model
    point_to_explain = X_test[0:1]

    # Our InterventionalTreeExplainer
    own_interventional_explainer = InterventionalTreeExplainer(
        model, X_train, max_order=order, index=index, debug=False
    )
    explanation = own_interventional_explainer.explain_function(point_to_explain.flatten())
    own_interactions = explanation.interactions

    # Interventional Game with Exact Computer
    interventional_game = InterventionalGame(model, X_train, point_to_explain.flatten())
    exact_computer = ExactComputer(interventional_game)
    exact_values = exact_computer(index, order)
    game_interactions = exact_values.interactions

    # Assertions that own Interventional Implementation matches Exact Computer
    for _, interaction in enumerate(own_interactions.keys()):
        if index in ["FSII", "STII"]:
            if len(interaction) != order:
                continue
            # Only check full interactions for FBII and FSII due to the current code supporting only the discrete derivate formula
            assert np.isclose(
                own_interactions[interaction],
                game_interactions[interaction],
                atol=1e-6,
            )
        elif len(interaction) > 0:
            assert np.isclose(
                own_interactions[interaction],
                game_interactions.get(interaction, 0),
                atol=1e-6,
            )


@pytest.mark.parametrize(
    ("index", "order"),
    [
        ("SV", 1),
        ("BV", 1),
        ("FBII", 1),
        ("SII", 2),
        ("BII", 2),
        ("CHII", 2),
        ("FBII", 2),
        ("FSII", 2),
        ("STII", 2),
        ("SII", 3),
        ("BII", 3),
        ("CHII", 3),
        ("FBII", 3),
        ("FSII", 3),
    ],
)
def test_correct_calculation_lgbm_clas_index_order(lightgbm_clf_model, cls_data, index, order):
    CLASS_INDEX = 1
    X_train, X_test, _, _ = cls_data
    model = lightgbm_clf_model
    point_to_explain = X_test[0:1]

    # Our InterventionalTreeExplainer
    own_interventional_explainer = InterventionalTreeExplainer(
        model,
        X_train,
        max_order=order,
        index=index,
        debug=False,
        class_index=CLASS_INDEX,
    )
    explanation = own_interventional_explainer.explain_function(point_to_explain.flatten())
    own_interactions = explanation.interactions

    # Interventional Game with Exact Computer
    interventional_game = InterventionalGame(
        model, X_train, point_to_explain.flatten(), class_index=CLASS_INDEX
    )
    exact_computer = ExactComputer(interventional_game)
    exact_values = exact_computer(index, order)
    game_interactions = exact_values.interactions

    # Assertions that own Interventional Implementatoin matches Exact Computer
    for _, interaction in enumerate(own_interactions.keys()):
        if index in ["FSII", "STII"]:
            if len(interaction) != order:
                continue
            # Only check full interactions for FBII and FSII due to the current code supporting only the discrete derivate formula
            assert np.isclose(
                own_interactions[interaction],
                game_interactions.get(interaction, 0),
                atol=1e-6,
            )
        elif len(interaction) > 0:
            assert np.isclose(
                own_interactions[interaction],
                game_interactions.get(interaction, 0),
                atol=1e-6,
            )
