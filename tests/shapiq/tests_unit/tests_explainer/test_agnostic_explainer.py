"""Unit tests for the AgnosticExplainer class."""

from __future__ import annotations

from typing import TYPE_CHECKING, get_args

import pytest

from shapiq import InteractionValues, MarginalImputer
from shapiq.explainer.agnostic import AgnosticExplainer, AgnosticExplainerIndices
from shapiq.explainer.base import Explainer
from shapiq.games.benchmark.synthetic import DummyGame, RandomGame

if TYPE_CHECKING:
    import numpy as np

    from shapiq.typing import IndexType


@pytest.mark.parametrize("index", get_args(AgnosticExplainerIndices))
def test_initialize_agnostic_explainer(index: IndexType) -> None:
    """Test the initialization of the AgnosticExplainer."""
    game = DummyGame(n=5, interaction=(1, 2))
    explainer = AgnosticExplainer(
        game=game,
        index=index,
        max_order=2,
        random_state=42,
    )
    assert isinstance(explainer, AgnosticExplainer)
    assert isinstance(explainer, Explainer)


def test_compute_interactions() -> None:
    """Test the computation of interactions using the AgnosticExplainer."""
    game = DummyGame(n=5, interaction=(1, 2))
    explainer = AgnosticExplainer(
        game=game,
        index="k-SII",
        max_order=2,
        random_state=42,
    )

    # Compute interaction values
    iv = explainer.explain(budget=2**game.n_players)

    assert isinstance(iv, InteractionValues)
    assert iv.index == "k-SII"
    assert iv.max_order == 2


def test_compute_interactions_with_x(dt_reg_model, background_reg_data) -> None:
    """Test the computation of interactions with a specific input."""

    imputer = MarginalImputer(
        model=dt_reg_model.predict,
        data=background_reg_data,
        random_state=42,
    )
    x_explain = background_reg_data[0]

    explainer = AgnosticExplainer(
        game=imputer,
        index="k-SII",
        max_order=2,
        random_state=42,
    )

    iv = explainer.explain(x_explain, budget=2**imputer.n_players)

    assert isinstance(iv, InteractionValues)
    assert iv.index == "k-SII"
    assert iv.max_order == 2


def test_compute_random_seed_init(dt_reg_model, background_reg_data) -> None:
    """Test the computation of interactions with a specific random seed."""
    imputer = MarginalImputer(
        model=dt_reg_model.predict,
        data=background_reg_data,
        random_state=42,
        x=background_reg_data[0],
    )

    explainer = AgnosticExplainer(
        game=imputer,
        index="k-SII",
        max_order=2,
        random_state=42,
    )

    iv1 = explainer.explain(budget=10)

    explainer_random = AgnosticExplainer(
        game=imputer,
        index="k-SII",
        max_order=2,
        random_state=42,
    )

    iv2 = explainer_random.explain(budget=10)

    assert iv1 == iv2  # Ensure that the results are consistent with the same random seed

    explainer_third = AgnosticExplainer(
        game=imputer,
        index="k-SII",
        max_order=2,
        random_state=43,  # Different random seed
    )

    iv3 = explainer_third.explain(budget=10)

    assert iv1 != iv3  # Ensure that the results differ with a different random seed


def test_compute_random_seed_in_function_call(dt_reg_model, background_reg_data) -> None:
    """Test the computation of interactions with a specific random seed in function call."""
    imputer = MarginalImputer(
        model=dt_reg_model.predict,
        data=background_reg_data,
        random_state=42,
        x=background_reg_data[0],
    )

    explainer = AgnosticExplainer(
        game=imputer,
        index="k-SII",
        max_order=2,
        random_state=None,
    )

    iv1 = explainer.explain(budget=10, random_state=42)
    iv2 = explainer.explain(budget=10, random_state=42)

    assert iv1 == iv2  # Ensure that the results are consistent with the same random seed

    explainer_third = AgnosticExplainer(game=imputer, index="k-SII", max_order=2, random_state=None)

    iv3 = explainer_third.explain(budget=10, random_state=43)  # Different random seed
    iv4 = explainer_third.explain(budget=10, random_state=43)

    assert iv3 == iv4  # Ensure that the results are consistent with the same random seed
    assert iv1 != iv3  # Ensure that the results differ with a different random seed


def test_with_callable():
    """Test the AgnosticExplainer with a callable game."""
    game_one = RandomGame(n=5, random_state=42)

    def callable_game(coalitions: np.ndarray) -> np.ndarray:
        return game_one(coalitions)

    explainer = AgnosticExplainer(
        game=callable_game,
        index="k-SII",
        max_order=2,
        random_state=42,
        n_players=game_one.n_players,
    )

    iv1 = explainer.explain(budget=2**game_one.n_players)

    game_two = RandomGame(n=5, random_state=42)
    explainer_game = AgnosticExplainer(
        game=game_two, index="k-SII", max_order=2, random_state=42, n_players=game_two.n_players
    )
    iv2 = explainer_game.explain(budget=2**game_two.n_players)
    assert iv1 == iv2  # Ensure that the results are consistent with the callable and game instance


def test_raise_error_n_players_not_specified() -> None:
    """Test that an error is raised if the number of players is not specified."""

    game = DummyGame(n=5, interaction=(1, 2))

    def callable_game(coalitions: np.ndarray) -> np.ndarray:
        return game(coalitions)

    with pytest.raises(ValueError):
        AgnosticExplainer(
            game=callable_game,
            index="k-SII",
            max_order=2,
            n_players=None,  # n_players not specified
        )
