"""Tests that check if the explainer gets the correct interaction values on the California Housing dataset."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, get_args

import pytest

from shapiq.explainer.agnostic import AgnosticExplainer
from shapiq.explainer.base import Explainer
from shapiq.explainer.custom_types import (
    ExplainerIndices,
)
from shapiq.explainer.tabular import TabularExplainer, TabularExplainerIndices
from shapiq.explainer.tree import TreeExplainer
from shapiq.explainer.tree.treeshapiq import TreeSHAPIQIndices
from shapiq.interaction_values import InteractionValues
from tests.utils import get_expected_index_or_skip

if TYPE_CHECKING:
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor

    from shapiq.utils.custom_types import IndexType


def _compute_gt(save_dir: pathlib.Path) -> None:
    """Computes the gt if it does not exist."""
    from .compute_test_explanations import compute_tabular_explanations, compute_tree_explanations

    if not save_dir.exists():
        # create the directory if it does not exist
        save_dir.mkdir(parents=True, exist_ok=True)
        compute_tree_explanations(save_path=save_dir)
        compute_tabular_explanations(save_path=save_dir)


def _load_ground_truth_interaction_values_california(
    index: IndexType, order: int, *, tabular: bool, load_from_runner: bool
):
    """Load the ground truth interaction values for the California Housing dataset and the Tabular Explainer.

    Note to developers:
        This function loads the interaction values that were precomputed by `tests/tests_integration_tests/compute_test_explanations.py`.
    """

    if tabular:
        name_part = "iv_california_housing_imputer_9070456741283270540"
    else:
        name_part = "iv_california_housing_tree"

    if load_from_runner:
        save_dir = pathlib.Path(__file__).parent
        save_dir = save_dir / "data" / "interaction_values" / "california_housing"
        _compute_gt(save_dir=save_dir)
    else:
        save_dir = pathlib.Path(__file__).parent.parent
        save_dir = save_dir / "data" / "interaction_values" / "california_housing"

    all_files = list(save_dir.glob(f"{name_part}_index={index}_order={order}.pkl"))
    assert len(all_files) == 1
    file_path = all_files[0]
    return InteractionValues.load(file_path)


def _compare(
    gt: InteractionValues,
    iv: InteractionValues,
    index: IndexType,
    order: int,
    tolerance: float = 0.1,
) -> None:
    """Compare the ground truth interaction values with the computed interaction values."""
    assert isinstance(gt, InteractionValues)
    assert isinstance(iv, InteractionValues)
    assert gt.index == iv.index
    assert gt.max_order == iv.max_order
    assert gt.min_order == iv.min_order

    for key in gt.dict_values:
        assert key in iv.dict_values, f"Key {key} not found in computed interaction values."
        assert gt.dict_values[key] == pytest.approx(iv.dict_values[key], rel=tolerance), (
            f"Interaction value for key {key} does not match ground truth."
        )

    for key in iv.dict_values:
        assert key in gt.dict_values, f"Key {key} not found in ground truth interaction values."
        assert iv.dict_values[key] == pytest.approx(gt.dict_values[key], rel=tolerance), (
            f"Computed interaction value for key {key} does not match ground truth."
        )

    # check baseline value
    if index not in ["BV", "FBII", "BII"]:
        assert gt.baseline_value == pytest.approx(iv.baseline_value, rel=tolerance), (
            f"Baseline value for index {index} and order {order} does not match ground truth."
        )


@pytest.mark.integration
def test_compare_gt_and_new_computed():
    """Computes the ground truth on the test runner and compares it to the old ground truth."""

    indices = get_args(TabularExplainerIndices)
    orders = [1, 2, 3, 4, 5, 6, 7]

    for index in indices:
        for order in orders:
            # Load the ground truth interaction values from the test runner
            try:
                gt_iv_runner = _load_ground_truth_interaction_values_california(
                    index=index,
                    order=order,
                    tabular=True,
                    load_from_runner=True,
                )
                # Load the old ground truth interaction values
                gt_iv_old = _load_ground_truth_interaction_values_california(
                    index=index, order=order, tabular=True, load_from_runner=False
                )
            except AssertionError:
                # If the ground truth interaction values do not exist, skip the test
                continue
            # Compare the two interaction values, here we intentionally use a higher tolerance
            # since the new computations might have more differences due to different random seeds
            # or other external factors ...
            _compare(gt=gt_iv_old, iv=gt_iv_runner, index=index, order=order, tolerance=0.3)


@pytest.mark.integration
@pytest.mark.parametrize("index", get_args(TreeSHAPIQIndices))
@pytest.mark.parametrize("order", [1, 2, 3, 4, 5, 6, 7])
def test_tree_explainer_california_housing(
    index: IndexType,
    order: int,
    california_housing_train_test_explain: tuple[np.ndarray, ...],
    california_housing_rf_model: RandomForestRegressor,
):
    """Test the TreeSHAPIQXAI game on the California Housing dataset."""
    expected_index = get_expected_index_or_skip(index, order)

    # get the data and model
    _, _, _, _, x_explain = california_housing_train_test_explain
    model = california_housing_rf_model

    # get the explainer and explain
    explainer = TreeExplainer(model=model, index=index, max_order=order)
    iv = explainer.explain(x_explain.flatten())
    iv = iv.get_n_order(min_order=1, max_order=order)
    assert iv.index == expected_index

    # load the ground truth interaction values
    gt_iv = _load_ground_truth_interaction_values_california(
        index, order, load_from_runner=True, tabular=False
    )

    # do the comparison of the interaction values
    _compare(gt=gt_iv, iv=iv, index=index, order=order)


@pytest.mark.integration
@pytest.mark.parametrize("index", get_args(ExplainerIndices))
@pytest.mark.parametrize("order", [1, 2, 3, 4, 5, 6, 7])
def test_agnostic_explainer_california_housing(
    index: IndexType,
    order: int,
    california_housing_train_test_explain: tuple[np.ndarray, ...],
    california_housing_imputer,
) -> None:
    """Test AgnosticExplainer on the California Housing dataset."""
    # prepare the expected index based on the order and index
    expected_index = get_expected_index_or_skip(index, order)

    explainer = AgnosticExplainer(
        game=california_housing_imputer,
        index=index,
        max_order=order,
        random_state=42,
    )
    iv = explainer.explain(budget=2**california_housing_imputer.n_players, x=None)
    iv = iv.get_n_order(min_order=1, max_order=order)
    assert iv.index == expected_index

    # load the ground truth interaction values
    gt_iv = _load_ground_truth_interaction_values_california(
        index, order, load_from_runner=True, tabular=True
    )

    # do the comparison of the interaction values
    _compare(gt=gt_iv, iv=iv, index=index, order=order)


@pytest.mark.integration
@pytest.mark.parametrize("explainer", [TabularExplainer, Explainer])
@pytest.mark.parametrize("index", get_args(TabularExplainerIndices))
@pytest.mark.parametrize("order", [1, 2, 3, 4, 5, 6, 7])
def test_tabular_explainer_california_housing(
    explainer: TabularExplainer | Explainer,
    index: IndexType,
    order: int,
    california_housing_train_test_explain: tuple[np.ndarray, ...],
    california_housing_rf_model: RandomForestRegressor,
) -> None:
    """Test the explainer on the California Housing dataset."""
    # prepare the expected index based on the order and index
    expected_index = get_expected_index_or_skip(index, order)

    # get the data and model
    x_train, y_train, x_test, y_test, x_explain = california_housing_train_test_explain
    n_features = x_train.shape[1]
    model = california_housing_rf_model

    # get the explainer and explain
    explainer = explainer(
        model=model.predict,
        data=x_test,
        index=index,
        max_order=order,
        random_state=42,
        verbose=False,
        # imputer params
        imputer="marginal",
        sample_size=100,
        joint_marginal_distribution=True,
    )
    iv = explainer.explain(x_explain, budget=2**n_features)
    iv = iv.get_n_order(min_order=1, max_order=order)
    assert iv.index == expected_index

    # load the ground truth interaction values
    gt_iv = _load_ground_truth_interaction_values_california(
        index, order, load_from_runner=True, tabular=True
    )

    # do the comparison of the interaction values
    _compare(gt=gt_iv, iv=iv, index=index, order=order)
