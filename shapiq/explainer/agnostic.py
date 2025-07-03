"""Agnostic Explainer for shapiq."""

from __future__ import annotations

from typing import TYPE_CHECKING

from shapiq.games.base import Game

from .base import Explainer
from .configuration import setup_approximator
from .custom_types import ExplainerIndices

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any, Literal

    import numpy as np

    from shapiq.approximator.base import Approximator
    from shapiq.interaction_values import InteractionValues

    from .tabular import TabularExplainerApproximators


AgnosticExplainerIndices = ExplainerIndices


class AgnosticExplainer(Explainer):
    """Agnostic Explainer for shapiq.

    This explainer is used to explain models that do not have a specific implementation in shapiq.
    It uses the game-based approach to explain the model's predictions.

    """

    game: Game | Callable[[np.ndarray], np.ndarray]
    """The cooperative game to be explained, either as a Game instance or a callable value function."""

    def __init__(
        self,
        game: Game | Callable[[np.ndarray], np.ndarray],
        *,
        n_players: int | None = None,
        index: AgnosticExplainerIndices = "k-SII",
        max_order: int = 2,
        approximator: Approximator | Literal["auto"] | TabularExplainerApproximators = "auto",
        random_state: int | None = None,
    ) -> None:
        """Initialize the AgnosticExplainer.

        Args:
            game: The cooperative game to be explained. This can be an instance of
                :class:`shapiq.games.base.Game` or a callable value function that expects a
                one-hot matrix of coalitions as input and returns the value of the coalition.

            n_players: The number of players in the game. If not provided, it will be inferred from
                the :class:`shapiq.games.base.Game`.

            index: The type of game-theoretic index to be used for the explanation.
                Defaults to "k-SII".

            max_order: The maximum order of interactions to be computed. Defaults to 2.

            approximator: The approximator to use for the game-based approach. Defaults to "auto",
                which will automatically select the appropriate approximator based on the index and
                max_order. Other options include "regression", "spex", "svarm", "montecarlo", and
                "permutation". Can also be an instance of
                :class:`shapiq.approximator._base.Approximator`.

            random_state: The random state to use for reproducibility. Defaults to None.

        Raises:
            ValueError: If the `game` is not an instance of `shapiq.games.base.Game` and `n_players`
                is not specified.

        """
        if not isinstance(game, Game) and n_players is None:
            msg = (
                f"The number of players must be specified with the `n_players` argument if no "
                f"`shapiq.games.base.Game` instance is provided. Got {type(game)}."
            )
            raise ValueError(msg)

        if n_players is None:
            n_players = game.n_players

        super().__init__(model=game, data=None, class_index=None)

        self.game = game
        self.approximator = setup_approximator(
            approximator=approximator,
            max_order=max_order,
            index=index,
            n_players=n_players,
            random_state=random_state,
        )

    def explain_function(
        self,
        budget: int,
        *,
        x: np.ndarray | None = None,
        random_state: int | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> InteractionValues:
        """Explain the function using the game-based approach.

        Args:
            budget: The budget used for the approximation / computation of interaction values.

            x: An optional data point to explain. This is only usable if the game is a
                :class:`~shapiq.games.imputer.base.Imputer`. If provided, the imputer will be fitted
                to this data point before computing the interaction values. Defaults to ``None``.

            random_state: An optional random state for reproducibility. Defaults to ``None``. If
                ``None``, no random state is set for the game or approximator.

            **kwargs: Additional keyword arguments (not used, only for compatibility).

        Returns:
            InteractionValues: The computed interaction values.
        """
        from shapiq.games.imputer.base import Imputer

        if x is not None and isinstance(self.game, Imputer):
            self.game.fit(x=x)
            if random_state is not None:
                self.game.set_random_state(random_state=random_state)
        if random_state is not None:
            self.approximator.set_random_state(random_state=random_state)
        return self.approximator.approximate(game=self.game, budget=budget)
