"""Tests that check if the explainer gets the correct interaction values on the California Housing dataset."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, get_args

import pytest

from shapiq.explainer.agnostic import AgnosticExplainer
from shapiq.explainer.base import Explainer
from shapiq.explainer.custom_types import (
    ExplainerIndices,
    ValidProductKernelExplainerIndices,
)
from shapiq.explainer.product_kernel import ProductKernelExplainer
from shapiq.explainer.tabular import TabularExplainer, TabularExplainerIndices
from shapiq.explainer.tree import TreeExplainer
from shapiq.explainer.tree.treeshapiq import TreeSHAPIQIndices
from shapiq.interaction_values import InteractionValues
from tests.shapiq.utils import get_expected_index_or_skip

if TYPE_CHECKING:
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor

    from shapiq.typing import IndexType

TABULAR_NAME_START = "iv_california_housing_imputer_9070456741283270540"
TREE_NAME_START = "iv_california_housing_tree"
PRODUCT_KERNEL_NAME_START = "iv_california_housing_product_kernel"


@pytest.fixture(scope="module")
def california_interaction_values() -> dict[str, InteractionValues]:
    """Computes the gt if it does not exist."""
    from .compute_test_explanations import (
        compute_tabular_explanations,
        compute_tree_explanations,
    )
    from .integration_test_product_kernel_explainer import compute_product_kernel_explanations

    print("Computing interaction values for California Housing dataset...")  # noqa: T201
    ivs_tabular = compute_tabular_explanations()
    ivs_tree = compute_tree_explanations()
    ivs_product_kernel = compute_product_kernel_explanations()
    return {**ivs_tabular, **ivs_tree, **ivs_product_kernel}


def _load_ground_truth_interaction_values_california(
    index: IndexType, order: int, *, tabular: bool
) -> InteractionValues:
    """Load the ground truth interactions for the California Housing dataset from disk.

    Note to developers:
        This function loads the interaction values that were precomputed by
            `tests/tests_integration_tests/compute_test_explanations.py`.

    Args:
        index: The index of the interaction values to load.
        order: The order of the interaction values to load.
        tabular: If True, load the interaction values for the Tabular Explainer, otherwise for the
            Tree Explainer.

    Returns:
        InteractionValues: The interaction values for the given index and order.

    """
    name_part = TABULAR_NAME_START if tabular else TREE_NAME_START

    save_dir = pathlib.Path(__file__).parent.parent
    save_dir = save_dir / "data" / "interaction_values" / "california_housing"

    all_files = list(save_dir.glob(f"{name_part}_index={index}_order={order}.json"))
    if len(all_files) != 1:
        msg = (
            f"Expected exactly one file for index {index} and order {order}, "
            f"but found {len(all_files)} files in {save_dir}."
        )
        raise ValueError(msg)
    file_path = all_files[0]
    return InteractionValues.load(file_path)


def _compare(
    gt: InteractionValues,
    iv: InteractionValues,
    index: IndexType,
    order: int,
    tolerance: float = 0.01,
) -> None:
    """Compare the ground truth interaction values with the computed interaction values."""
    tolerance = max(abs(gt.get_n_order(min_order=1).values)) * tolerance

    assert isinstance(gt, InteractionValues)
    assert isinstance(iv, InteractionValues)
    assert gt.index == iv.index
    assert gt.max_order == iv.max_order
    assert gt.min_order == iv.min_order

    for key in gt.dict_values:
        assert key in iv.dict_values, f"Key {key} not found in computed interaction values."
        assert gt.dict_values[key] == pytest.approx(iv.dict_values[key], abs=tolerance), (
            f"Interaction value for key {key} does not match ground truth."
        )

    for key in iv.dict_values:
        assert key in gt.dict_values, f"Key {key} not found in ground truth interaction values."
        assert iv.dict_values[key] == pytest.approx(gt.dict_values[key], abs=tolerance), (
            f"Computed interaction value for key {key} does not match ground truth."
        )

    # check baseline value
    if index not in ["BV", "FBII", "BII"]:
        assert gt.baseline_value == pytest.approx(iv.baseline_value, abs=tolerance), (
            f"Baseline value for index {index} and order {order} does not match ground truth."
        )


@pytest.mark.integration
class TestCaliforniaHousingExactComputer:
    """Tests that checks that the ExactComputer yields correct interaction values on California Housing."""

    @pytest.mark.parametrize("index", get_args(TabularExplainerIndices))
    @pytest.mark.parametrize("order", [1, 2, 3, 4, 5, 6, 7])
    def test_with_recompute(self, california_interaction_values, index, order):
        """Computes the ground truth on the test runner and compares it to the old ground truth."""

        _ = get_expected_index_or_skip(index, order)

        name = f"{TABULAR_NAME_START}_index={index}_order={order}.json"
        gt_iv_runner = california_interaction_values[name]
        gt_iv_old = _load_ground_truth_interaction_values_california(
            index=index, order=order, tabular=True
        )

        _compare(gt=gt_iv_old, iv=gt_iv_runner, index=index, order=order, tolerance=0.05)


@pytest.mark.integration
class TestCaliforniaHousingExplainers:
    """Tests that check if the Explainer get the correct interaction values on California Housing.

    This class uses the California Housing dataset and precomputed interaction values to check
    if the explainers with complete budget can compute the precomputed interaction values. This
    test serves as a regression test to ensure that the explainers work correctly in the future and
    also as an integration test to check that the explainers work correctly in a "real-world"
    scenario.
    """

    @pytest.mark.parametrize("index", get_args(TreeSHAPIQIndices))
    @pytest.mark.parametrize("order", [1, 2, 3, 4, 5, 6, 7])
    def test_tree_explainer(
        self,
        index: IndexType,
        order: int,
        california_housing_train_test_explain: tuple[np.ndarray, ...],
        california_housing_rf_model: RandomForestRegressor,
        california_interaction_values: dict[str, InteractionValues],
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
        name = f"{TREE_NAME_START}_index={index}_order={order}.json"
        gt_iv = california_interaction_values[name]

        # do the comparison of the interaction values
        _compare(gt=gt_iv, iv=iv, index=index, order=order)

    @pytest.mark.parametrize("index", get_args(ExplainerIndices))
    @pytest.mark.parametrize("order", [1, 2, 3, 4, 5, 6, 7])
    def test_agnostic_explainer(
        self,
        index: IndexType,
        order: int,
        california_housing_train_test_explain: tuple[np.ndarray, ...],
        california_housing_imputer,
        california_interaction_values: dict[str, InteractionValues],
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
        name = f"{TABULAR_NAME_START}_index={index}_order={order}.json"
        gt_iv = california_interaction_values[name]

        # do the comparison of the interaction values
        _compare(gt=gt_iv, iv=iv, index=index, order=order)

    @pytest.mark.parametrize("explainer", [TabularExplainer, Explainer])
    @pytest.mark.parametrize("index", get_args(TabularExplainerIndices))
    @pytest.mark.parametrize("order", [1, 2, 3, 4, 5, 6, 7])
    def test_tabular_explainer(
        self,
        explainer: TabularExplainer | Explainer,
        index: IndexType,
        order: int,
        california_housing_train_test_explain: tuple[np.ndarray, ...],
        california_housing_rf_model: RandomForestRegressor,
        california_interaction_values: dict[str, InteractionValues],
    ) -> None:
        """Test the explainer on the California Housing dataset."""
        # prepare the expected index based on the order and index
        expected_index = get_expected_index_or_skip(index, order)

        # get the data and model
        x_train, y_train, x_test, y_test, x_explain = california_housing_train_test_explain
        n_features = x_train.shape[1]
        model = california_housing_rf_model

        # get the explainer and explain
        if not issubclass(explainer, TabularExplainer) and not issubclass(explainer, Explainer):
            msg = "The explainer must be a subclass of TabularExplainer or Explainer."
            raise ValueError(msg)

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
        name = f"{TABULAR_NAME_START}_index={index}_order={order}.json"
        gt_iv = california_interaction_values[name]

        # do the comparison of the interaction values
        _compare(gt=gt_iv, iv=iv, index=index, order=order)

    @pytest.mark.parametrize("index", get_args(ValidProductKernelExplainerIndices))
    @pytest.mark.parametrize("order", [1])
    def test_product_kernel_explainer(
        self,
        index: IndexType,
        order: int,
        california_housing_train_test_explain: tuple[np.ndarray, ...],
        california_housing_svr_model,
        california_interaction_values: dict[str, InteractionValues],
    ) -> None:
        """Test ProductKernelExplainer on the California Housing dataset."""
        expected_index = get_expected_index_or_skip(index, order)

        # get the data and model
        _, _, _, _, x_explain = california_housing_train_test_explain
        model = california_housing_svr_model

        # get the explainer and explain
        explainer = ProductKernelExplainer(model=model, index=index, max_order=order)
        iv = explainer.explain(x_explain.flatten())
        iv = iv.get_n_order(min_order=1, max_order=order)
        assert iv.index == expected_index

        # load the ground truth interaction values
        name = f"{PRODUCT_KERNEL_NAME_START}_index={index}_order={order}.json"
        gt_iv = california_interaction_values[name]

        # do the comparison of the interaction values
        _compare(gt=gt_iv, iv=iv, index=index, order=order)
