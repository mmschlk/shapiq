"""Tests for the tabular local XAI module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pytest

from shapiq.imputer import (
    BaselineImputer,
    ConditionalImputer,
    MarginalImputer,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from fixtures.tabular import TabularDataSet
    from sklearn.tree import DecisionTreeRegressor

    from shapiq.typing import NumericArray


@pytest.fixture(scope="module")
def model(
    california_housing_dt: DecisionTreeRegressor,
) -> Callable[[NumericArray], NumericArray]:
    """Returns a callable model for testing."""
    return california_housing_dt.predict


@pytest.fixture(scope="module")
def data(california_housing: TabularDataSet) -> NumericArray:
    """Returns the data for testing."""
    return california_housing.x_test


@pytest.fixture(scope="module")
def x_explain(california_housing: TabularDataSet) -> NumericArray:
    """Returns a specific data point to explain."""
    return california_housing.x_test[0]


class TestGetImputer:
    """Tests for getting different imputers."""

    def test_get_imputer_baseline(
        self,
        model: Callable[[NumericArray], NumericArray],
        data: NumericArray,
        x_explain: NumericArray,
    ) -> None:
        """Test the baseline imputer."""
        from shapiq_games.tabular.game_local_xai import get_imputer

        imputer = get_imputer("baseline", model, data, x_explain)
        assert isinstance(imputer, BaselineImputer)

    def test_get_imputer_marginal(
        self,
        model: Callable[[NumericArray], NumericArray],
        data: NumericArray,
        x_explain: NumericArray,
    ) -> None:
        """Test the marginal imputer."""
        from shapiq_games.tabular.game_local_xai import get_imputer

        imputer = get_imputer("marginal", model, data, x_explain)
        assert isinstance(imputer, MarginalImputer)

    def test_get_imputer_conditional(
        self,
        model: Callable[[NumericArray], NumericArray],
        data: NumericArray,
        x_explain: NumericArray,
    ) -> None:
        """Test the conditional imputer."""
        from shapiq_games.tabular.game_local_xai import get_imputer

        data = data[:200]
        imputer = get_imputer("conditional", model, data, x_explain)
        assert isinstance(imputer, ConditionalImputer)


class TestTabularLocalExplanation:
    """Tests for the TabularLocalExplanation game."""

    @pytest.mark.parametrize("imputer", ["baseline", "marginal", "conditional"])
    def test_game(
        self,
        imputer: Literal["baseline", "marginal", "conditional"],
        model: Callable[[NumericArray], NumericArray],
        data: NumericArray,
        x_explain: NumericArray,
    ) -> None:
        """Test the initialization of the LocalExplanation game."""
        from shapiq_games.tabular.game_local_xai import TabularLocalExplanation

        data = data[:200]  # make this small for efficient tests (conditional takes quite long)

        game = TabularLocalExplanation(x=x_explain, data=data, model=model, imputer=imputer)
        assert game.n_players == data.shape[1]
        assert game.empty_prediction_value is not None

        # call game with a coalition
        game_values = game(game.grand_coalition)
        model_prediction = model(x_explain.reshape(1, -1))[0]
        expected_output = model_prediction - game.empty_prediction_value
        assert game_values[0] == expected_output

    def test_empty_prediction_property(
        self,
        model: Callable[[NumericArray], NumericArray],
        data: NumericArray,
        x_explain: NumericArray,
    ) -> None:
        """Test the empty_prediction_value property."""
        from shapiq_games.tabular.game_local_xai import TabularLocalExplanation

        game = TabularLocalExplanation(x=x_explain, data=data, model=model, normalize=False)
        empty_game_output = float(game(game.empty_coalition)[0])
        assert game.empty_prediction_value == empty_game_output

    def test_initialization_with_imputer(
        self,
        model: Callable[[NumericArray], NumericArray],
        data: NumericArray,
        x_explain: NumericArray,
    ) -> None:
        """Test the initialization of the LocalExplanation game with an imputer instead of a str."""
        from shapiq_games.tabular.game_local_xai import TabularLocalExplanation, get_imputer

        imputer = get_imputer("marginal", model, data, x_explain)
        game = TabularLocalExplanation(x=x_explain, data=data, model=model, imputer=imputer)
        assert isinstance(game.imputer, MarginalImputer)

    def test_inserting_imputer(
        self,
        model: Callable[[NumericArray], NumericArray],
        data: NumericArray,
        x_explain: NumericArray,
    ) -> None:
        """Test that we can insert an imputer into a game after initialization."""
        from shapiq_games.tabular.game_local_xai import TabularLocalExplanation, get_imputer

        imputer = get_imputer("baseline", model, data, x_explain)
        game = TabularLocalExplanation(x=x_explain, data=data, model=model, imputer=imputer)
        assert isinstance(game.imputer, BaselineImputer)

        marginal_imputer = get_imputer("marginal", model, data, x_explain)
        game.imputer = marginal_imputer
        assert isinstance(game.imputer, MarginalImputer)
