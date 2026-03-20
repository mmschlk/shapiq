"""Tests for the TreeLocalXAI benchmark and computer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from shapiq.explainer.tree.treeshapiq import TreeSHAPIQIndices
    from sklearn.tree import DecisionTreeRegressor

    from shapiq_benchmark.typing import TabularDataSet


class TestTreeGroundTruthComputer:
    """Tests for the TreeGroundTruthComputer."""

    @pytest.mark.parametrize(("index", "order"), [("SV", 1), ("k-SII", 2), ("k-SII", 3)])
    def test_exact_values(
        self,
        index: TreeSHAPIQIndices,
        order: int,
        california_housing_dt: DecisionTreeRegressor,
        california_housing: TabularDataSet,
    ) -> None:
        """Test the exact values computation."""
        from shapiq import ExactComputer

        from shapiq_benchmark.tree import TreeLocalXAI, TreeSAHPIQComputer

        from .utils import iv_equal_values

        x_explain = california_housing.x_test[0:1]
        game = TreeLocalXAI(
            tree_model=california_housing_dt,
            x_explain=x_explain,
            class_label=None,
            normalize=False,
        )

        # compute exact values using TreeSAHPIQComputer
        tree_computer = TreeSAHPIQComputer(
            tree_model=california_housing_dt,
            x_explain=x_explain,
        )
        sii_tree = tree_computer.exact_values(index=index, order=order)

        # compute exact values using ExactComputer
        exact_computer = ExactComputer(game=game, n_players=game.n_players)
        sii_exact = exact_computer(index=index, order=order)

        assert iv_equal_values(sii_tree, sii_exact)

    def test_exchanging_x_explain(
        self, california_housing_dt: DecisionTreeRegressor, california_housing: TabularDataSet
    ) -> None:
        """Tests that exchanging the public x_explain attribute works correctly."""
        from shapiq_benchmark.tree import TreeSAHPIQComputer

        from .utils import iv_equal_values

        x_explain_first = california_housing.x_test[0]
        tree_computer = TreeSAHPIQComputer(
            tree_model=california_housing_dt,  # Placeholder for the actual model
            x_explain=x_explain_first,
        )
        assert np.array_equal(tree_computer.x_explain, x_explain_first)
        iv_first = tree_computer.exact_values(index="SV", order=1)
        x_explain_second = california_housing.x_test[1]
        tree_computer.x_explain = x_explain_second
        assert np.array_equal(tree_computer.x_explain, x_explain_second)
        iv_second = tree_computer.exact_values(index="SV", order=1)
        assert not iv_equal_values(iv_first, iv_second)

    def test_error_on_invalid_x_explain(
        self, california_housing_dt: DecisionTreeRegressor, california_housing: TabularDataSet
    ) -> None:
        """Test that an error is raised when x_explain is not a valid feature vector."""
        from shapiq_benchmark.tree import TreeSAHPIQComputer

        msg = (
            r"x_explain must be a 1-dimensional or 2-dimensional array of shape \(1, n_features\),"
        )
        with pytest.raises(ValueError, match=msg):
            TreeSAHPIQComputer(
                tree_model=california_housing_dt,
                x_explain=california_housing.x_test[0:10],  # Invalid input
            )
