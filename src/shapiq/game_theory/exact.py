"""ExactComputer class for a plethora of game theoretic concepts like interaction indices or generalized values."""

from __future__ import annotations

import copy
import warnings
from typing import TYPE_CHECKING, Literal, get_args

import numpy as np
from scipy.special import bernoulli, binom

from shapiq.game import Game
from shapiq.interaction_values import InteractionValues
from shapiq.typing import CoalitionMatrix, GameValues, IndexType
from shapiq.utils import powerset

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["ExactComputer", "get_bernoulli_weights"]


class ExactComputer:
    """Computes a variety of game-theoretic values exactly.

    The ExactComputer class computes exact values for cooperative games with a small number of
    players. Most game-theoretic concepts like the Shapley value, Banzhaf value, or the various
    interaction indices require the evaluation of the game on all possible :math:`2^n` coalitions.
    This is feasible for small games (e.g. 5-14 players).

    Currently, the following game-theoretic concepts are supported:
        - The Moebius Transform (Moebius)
        - The Co-Moebius Transform (Co-Moebius)
        - The Shapley Value (SV) [Sha53]_
        - The Banzhaf Value (BV) [Ban64]_
        - Shapley Interaction Index (SII) [Gra99]_
        - Banzhaf Interaction Index (BII) [Gra99]_
        - Chaining Interaction Index (CHII) [Mar99]_
        - k-Shapley Interaction Index (k-SII) [Bor23]_
        - Shapley Taylor Interaction Index (STII) [Sun20]_
        - Faithful Shapley Interaction Index (FSII) [Tsa23]_
        - Faithful Banzhaf Interaction Index (FBII) [Tsa23]_
        - k-additive Shapley Interaction Index (kADD-SHAP) [Pel23]_
        - Shapley Generalized Value (SGV) [Mar00]_
        - Banzhaf Generalized Value (BGV) [Mar00]_
        - Chaining Generalized Value (CHGV) [Mar07]_
        - Internal Generalized Value (IGV) [Mar07]_
        - External Generalized Value (EGV) [Mar07]_
        - Joint Shapley Value (JointSV) [Har22]_
        - Egalitarian Least Core (ELC) [Yan21]_


    Args:
        n_players: The number of players in the game.
        game: A callable game that takes a binary matrix of shape ``(n_coalitions, n_players)``
            and returns a numpy array of shape ``(n_coalitions,)`` containing the game values.
        evaluate_game: whether to compute the values at init (if True) or first call (False)

    Attributes:
        n: The number of players.
        _game_fun: The callable game function.


    Properties:
        baseline_value: The baseline value of the game (empty coalition).
        coalition_lookup: A dictionary mapping coalitions to their indices in the game values.
        game_values: The game values for all coalitions.

    References:
        .. [Sha53] Lloyd S. Shapley (1953). A Value for n-Person Games. Princeton University Press, pp. 307-318. https://doi.org/10.1515/9781400881970-018
        .. [Ban64] John F. Banzhaf III (1964). Weighted voting doesn't work: A mathematical analysis. Rutgers L. Rev., 19, 317.
        .. [Dub81] Pradeep Dubey, Abraham Neyman, Robert James Weber (1981) Value Theory Without Efficiency. In: Mathematics of Operations Research 6(1):122-128. Value Theory Without Efficiency. Mathematics of Operations Research 6(1):122-128. https://doi.org/10.1287/moor.6.1.122
        .. [Gra99] Michel Grabisch, Marc Roubens (1999). An axiomatic approach to the concept of interaction among players in cooperative games. In: Game Theory 28:547-565. https://link.springer.com/article/10.1007/s001820050125
        .. [Mar99] Jean-Luc Marichal, Marc Roubens (1999). The Chaining Interaction Index among Players in Cooperative Games. In: Meskens, N., Roubens, M. (eds) Advances in Decision Analysis. Mathematical Modelling: Theory and Applications, vol 4. Springer, Dordrecht. https://link.springer.com/chapter/10.1007/978-94-017-0647-6_5
        .. [Mar00] Jean-Luc Marichal (2000). The influence of variables on pseudo-Boolean functions with applications to game theory and multicriteria decision making. In: Discrete Applied Mathematics 107(1-3):139-164. https://doi.org/10.1016/S0166-218X(00)00264-X
        .. [Fui06] Katsushige Fujimoto, Ivan Kojadinovic, Jean-Luc Marichal (2006). Axiomatic characterizations of probabilistic and cardinal-probabilistic interaction indices. In: Games and Economic Behavior 55(1):72-99. https://doi.org/10.1016/j.geb.2005.03.002
        .. [Mar07] Jean-Luc Marichal, Ivan Kojadinovic, Katsushige Fujimoto (2007). Axiomatic characterizations of generalized values. In: Discrete Applied Mathematics 155(1):26-43. https://doi.org/10.1016/j.dam.2006.05.002
        .. [Web09] Robert James Weber (2009). Probabilistic values for games. In: Roth AE, ed. The Shapley Value: Essays in Honor of Lloyd S. Shapley. Cambridge University Press. 1988:101-120. https://doi.org/10.1017/CBO9780511528446.008
        .. [Sun20] Mukund Sundararajan, Kedar Dhamdhere, Ashish Agarwal (2020). In: Proceedings of the 37th International Conference on Machine Learning, PMLR 119:9259-9268. https://proceedings.mlr.press/v119/sundararajan20a.html
        .. [Yan21] Tom Yan, Ariel D. Procaccia (2021). If You Like Shapley Then You'll Love the Core. In: Proceedings of the AAAI Conference on Artificial Intelligence 35(6):5751-5759. https://doi.org/10.1609/aaai.v35i6.16721
        .. [Har22] Chris Harris, Richard Pymar, Colin Rowat (2022). Joint Shapley values: a measure of joint feature importance. In: Proceedings of The Tenth International Conference on Learning Representations. https://openreview.net/forum?id=vcUmUvQCloe
        .. [Bor23] Sebastian Bordt, Ulrike von Luxburg (2023). In: Proceedings of The 26th International Conference on Artificial Intelligence and Statistics, PMLR 206:709-745. https://proceedings.mlr.press/v206/bordt23a.html
        .. [Tsa23] Che-Ping Tsai, Chih-Kuan Yeh, Pradeep Ravikumar (2023). Faith-Shap: The Faithful Shapley Interaction Index. In: Journal of Machine Learning Research 24(94):1-42. https://jmlr.org/papers/v24/22-0202.html
        .. [Pel23] Guilherme Dean Pelegrina, Leonardo Tomazeli Duarte, Michel Grabisch (2023). A k-additive Choquet integral-based approach to approximate the SHAP values for local interpretability in machine learning. In: Artificial Intelligence 325:104014. https://doi.org/10.1016/j.artint.2023.104014

    """

    valid_indices: tuple[IndexType, ...] = tuple(get_args(IndexType))
    """The valid indices for the exact computer."""

    def __init__(
        self,
        game: Game | Callable[[CoalitionMatrix], GameValues],
        n_players: int | None = None,
        *,
        evaluate_game: bool = False,
    ) -> None:
        """Initialize the ExactComputer class.

        Args:
            game: A callable cooperative game that expects a binary matrix of shape
                ``(n_coalitions, n_players)`` denoting presence and absence of players and returns
                a numpy array of shape ``(n_coalitions,)`` containing the game values.

            n_players: The number of players in the game (if game is not a :class:`shapiq.game.Game`
                object). Defaults to ``None``.

            evaluate_game: Flag denoting whether to compute the game values at initialization of
                the ExactComputer (``True``) or at its first computation call (``False``).
        """
        # clean inputs
        if n_players is None:
            if isinstance(game, Game):
                n_players = game.n_players
            else:
                msg = "n_players must be specified if game is not a Game object."
                raise ValueError(msg)

        # set parameter attributes
        self._n = n_players
        self._game_fun = game

        # set object attributes
        self._grand_coalition_tuple: tuple[int, ...] = tuple(range(self.n_players))
        self._grand_coalition_set: set[int] = set(self._grand_coalition_tuple)
        self._big_M: float = 10e7
        self._n_interactions: np.ndarray = self.get_n_interactions(self.n_players)
        self._computed: dict[tuple[IndexType, int], InteractionValues] = {}  # cache computed ivs
        self._elc_stability_subsidy: float | None = None
        self._game_is_computed: bool = False
        if evaluate_game:  # evaluate the game on the powerset
            self._evaluate_game()

    def __repr__(self) -> str:
        """String Representation of the ExactComputer class."""
        return f"ExactComputer(n_players={self.n_players}, game_fun={self._game_fun})"

    def __str__(self) -> str:
        """String representation of the ExactComputer class."""
        return f"ExactComputer(n_players={self.n_players})"

    def __call__(self, index: IndexType, order: int | None = None) -> InteractionValues:
        """Calls the computation of the specified index or value.

        Args:
            index: The index or value to compute
            order: The order of the interaction index. If not specified the maximum order
                (i.e. ``n_players``) is used. Defaults to ``None``.

        Returns:
            The desired interaction values or generalized values.

        Raises:
            ValueError: If the index is not supported.

        """
        if order is None:
            order = self.n_players

        if (index, order) in self._computed:
            return copy.deepcopy(self._computed[(index, order)])

        match index:
            case "Moebius":
                # TODO(mmshlk): Why do we not use shapley_base_interaction here?  # noqa: TD003
                result = self.moebius_transform
            case "Co-Moebius":
                # TODO(mmshlk): Why do we not use shapley_base_interaction here?  # noqa: TD003
                result = self.shapley_base_interaction(index, order)
            case "k-SII" | "STII" | "kADD-SHAP":  # Shapley interactions
                result = self.shapley_interactions(index, order)
            case "FSII" | "FBII":  # Faithful Shapley/Banzhaf Interaction Index
                result = self.compute_fii(index, order)
            case "SGV" | "BGV" | "CHGV" | "IGV" | "EGV":  # Base generalized values
                result = self.base_generalized_value(index, order)
            case "SII" | "BII" | "CHII":
                result = self.shapley_base_interaction(index, order)
            case "BV" | "SV":  # probabilistic values
                result = self.probabilistic_value(index)
            case "JointSV":  # Joint Shapley Value
                result = self.shapley_generalized_value(order)
            case "ELC":  # The Core
                result = self.compute_egalitarian_least_core()
            case _:
                msg = "Index or value not supported."
                raise ValueError(msg)
        self._computed[(index, order)] = copy.deepcopy(result)
        return copy.deepcopy(result)

    @property
    def baseline_value(self) -> float:
        """Return the baseline value of the game (empty coalition)."""
        if not self._game_is_computed:
            self._evaluate_game()
        return self._baseline_value

    @property
    def coalition_lookup(self) -> dict[tuple[int, ...], int]:
        """Return the coalition lookup dictionary."""
        if not self._game_is_computed:
            self._evaluate_game()
        return self._coalition_lookup

    @property
    def game_values(self) -> np.ndarray:
        """Return the game values for all possible coalitions."""
        if not self._game_is_computed:
            self._evaluate_game()
        return self._game_values

    @property
    def n_players(self) -> int:
        """Return the number of players in the game."""
        return self._n

    def _evaluate_game(self) -> None:
        """Evaluate the game on the powerset of all coalitions and store the results."""
        baseline_value, game_values, coalition_lookup = self.compute_game_values()
        self._baseline_value = baseline_value
        self._game_values = game_values
        self._coalition_lookup = coalition_lookup
        self._game_is_computed = True

    def compute_game_values(self) -> tuple[float, GameValues, dict[tuple[int], int]]:
        """Evaluates the game on the powerset of all coalitions.

        Returns:
            - The baseline value of the game (empty coalition).
            - A numpy array of shape ``(2**n_players,)`` containing the game values for all
              coalitions.
            - A dictionary mapping coalitions (as tuples of player indices) to their corresponding
                index in the game values array.

        """
        coalition_lookup = {}
        coalition_matrix = np.zeros((2**self.n_players, self.n_players), dtype=bool)
        for i, T in enumerate(
            powerset(self._grand_coalition_set, min_size=0, max_size=self.n_players)
        ):
            coalition_lookup[T] = i  # set lookup for the coalition
            coalition_matrix[i, T] = True  # one-hot-encode the coalition
        game_values = self._game_fun(coalition_matrix)  # compute the game values
        baseline_value = float(game_values[0])  # set the baseline value
        return baseline_value, game_values, coalition_lookup

    @property
    def moebius_transform(self) -> InteractionValues:
        """Computes and returns the Moebius transform of the game."""
        try:
            return self._computed[("Moebius", self.n_players)]
        except KeyError:  # if not computed yet, just continue
            pass

        # compute the Moebius transform
        moebius_transform = np.zeros(2**self.n_players)
        coalition_lookup = {}
        for interaction_pos, interaction in enumerate(powerset(self._grand_coalition_set)):
            coalition_lookup[interaction] = interaction_pos
            interaction_size = len(interaction)
            for coalition in powerset(interaction):
                coalition_pos = self.coalition_lookup[coalition]
                moebius_transform[interaction_pos] += (-1) ** (
                    interaction_size - len(coalition)
                ) * self.game_values[coalition_pos]

        # fill result into InteractionValues object and storage dictionary
        interaction_values = InteractionValues(
            values=moebius_transform,
            index="Moebius",
            max_order=self.n_players,
            min_order=0,
            n_players=self.n_players,
            interaction_lookup=coalition_lookup,
            estimated=False,
            baseline_value=self.baseline_value,
        )
        self._computed[("Moebius", self.n_players)] = copy.deepcopy(interaction_values)
        return copy.deepcopy(interaction_values)

    def _base_weights(
        self,
        coalition_size: int,
        interaction_size: int,
        index: Literal[
            "SII", "SGV", "BII", "BGV", "CHII", "CHGV", "Moebius", "IGV", "Co-Moebius", "EGV"
        ],
    ) -> float:
        """Computes the weight of different indices in their common representation.

        For example, the weight of the discrete derivative of S given T in SII or the weight of the
        marginal contribution of S given T in SGV.

        Args:
            coalition_size: The size of the coalition from ``0,...,n-interaction_size``
            interaction_size: The size of the interaction from ``0,...,order``
            index: The computed index

        Returns:
            The base weight of the interaction index

        Raises:
            ValueError: If the index is not supported

        """
        match index:
            case "SII" | "SGV":
                weight = 1 / (
                    (self.n_players - interaction_size + 1)
                    * binom(self.n_players - interaction_size, coalition_size)
                )
            case "BII" | "BGV":
                weight = 1 / (2 ** (self.n_players - interaction_size))
            case "CHII" | "CHGV":
                weight = interaction_size / (
                    (interaction_size + coalition_size)
                    * binom(self.n_players, interaction_size + coalition_size)
                )
            case "Moebius" | "IGV":
                weight = 0
                if coalition_size == 0:
                    weight = 1
            case "Co-Moebius" | "EGV":
                weight = 0
                if coalition_size == self.n_players - interaction_size:
                    weight = 1
            case _:
                msg = f"Index {index} not supported."
                raise ValueError(msg)
        return weight

    def _stii_weight(self, coalition_size: int, interaction_size: int, order: int) -> float:
        """Sets the weight for the representation of STII as a CII (using discrete derivatives).

        Args:
            coalition_size: Size of the Discrete Derivative
            interaction_size: Interaction size with ``s <= k``
            order: Interaction order

        Returns:
            The weight of STII

        """
        if interaction_size == order:
            return float(order / (self.n_players * binom(self.n_players - 1, coalition_size)))
        if coalition_size == 0:
            return 1.0
        return 0.0

    def _get_fii_weights(self, index: Literal["FSII", "FBII"]) -> np.ndarray:
        """Pre-computes the kernel weight for the least square representation of FSII and FBII.

        Returns:
            An array of the kernel weights for ``0,...,n with "infinite weight"`` on ``0`` and ``n``.

        """
        fii_weights = np.zeros(self.n_players + 1, dtype=float)

        match index:
            case "FSII":
                fii_weights[0] = self._big_M
                fii_weights[-1] = self._big_M
                for coalition_size in range(1, self.n_players):
                    fii_weights[coalition_size] = 1 / (
                        (self.n_players - 1) * binom(self.n_players - 2, coalition_size - 1)
                    )
            case "FBII":
                fii_weights[:] = 2 ** (-self.n_players)
            case _:
                msg = f"Index {index} not supported."
                raise ValueError(msg)
        return fii_weights

    def _get_stii_weights(self, order: int) -> np.ndarray:
        """Pre-computes the STII weights for the CII representation (using discrete derivatives).

        Args:
            order: The interaction order

        Returns:
            An array with pre-computed weights for ``t=0,...,n-k``

        """
        stii_weights = np.zeros(self.n_players - order + 1, dtype=float)
        for t in range(self.n_players - order + 1):
            stii_weights[t] = self._stii_weight(t, order, order)
        return stii_weights

    def _get_discrete_derivative(
        self,
        interaction: set[int] | tuple[int, ...],
        coalition: set[int] | tuple[int, ...],
    ) -> float:
        """Computes the discrete derivative of a coalition with respect to an interaction.

        Args:
            interaction: A subset of the grand coalition as a set or tuple
            coalition: Subset of N as set of tuple

        Returns:
            The discrete derivative of the coalition with respect to the interaction

        """
        discrete_derivative = 0.0
        interaction_size = len(interaction)
        for interaction_subset in powerset(interaction):
            interaction_subset_size = len(interaction_subset)
            pos = self.coalition_lookup[
                tuple(sorted(set(coalition).union(set(interaction_subset))))
            ]
            discrete_derivative += (-1) ** (
                interaction_size - interaction_subset_size
            ) * self.game_values[pos]
        return discrete_derivative

    @staticmethod
    def get_n_interactions(n_players: int) -> np.ndarray:
        """Pre-computes the number of interactions for all coalition sizes.

        Pre-computes an array that contains the number of interactions up to the size of the index
        (e.g. ``n_interactions[4]`` is the number of interactions up to size ``4``).

        Args:
            n_players: The number of players

        Returns:
            A numpy array containing the number of interactions up to the size of the index

        Examples:
            >>> ExactComputer.get_n_interactions(3)
            array([1, 4, 7, 8])  # binom(3, 0), binom(3, 0) + binom(3, 1), etc. ...

        """
        n_interactions = np.zeros(n_players + 1, dtype=int)
        n_interaction = 0
        for interaction_size in range(n_players + 1):
            n_interaction += int(binom(n_players, interaction_size))
            n_interactions[interaction_size] = n_interaction
        return n_interactions

    def _get_base_weights(
        self,
        index: Literal[
            "SII", "SGV", "BII", "BGV", "CHII", "CHGV", "Moebius", "IGV", "Co-Moebius", "EGV"
        ],
        order: int,
    ) -> np.ndarray:
        """Pre-compute all base weights for all coalition and interaction sizes.

        Pre-compute all base weights for all coalition sizes  (i.e. ``0, ..., n-s``) and all
        interaction sizes (i.e. ``1, ..., order``).

        Args:
            index: The interaction index
            order: The interaction order

        Returns:
            A numpy array with all base interaction weights

        """
        base_weights = np.zeros((self.n_players + 1, order + 1), dtype=float)
        for interaction_size in range(order + 1):
            for coalition_size in range(self.n_players - interaction_size + 1):
                base_weights[coalition_size, interaction_size] = self._base_weights(
                    coalition_size,
                    interaction_size,
                    index,
                )
        return base_weights

    def base_interaction(
        self,
        index: Literal[
            "SII",
            "BII",
            "CHII",
            "Moebius",
            "Co-Moebius",
        ],
        order: int,
    ) -> InteractionValues:
        """Computes interactions based on representation with discrete derivatives.

        Interactions based on the discrete derivative are base interactions like SII or BII.

        Args:
            index: The interaction index
            order: The interaction order

        Returns:
            An InteractionValues object containing the base interactions.

        """
        base_interaction_values = np.zeros(self._n_interactions[order])
        base_weights = self._get_base_weights(index, order)
        for coalition in powerset(self._grand_coalition_set):
            coalition_size = len(coalition)
            coalition_pos = self.coalition_lookup[coalition]
            for j, interaction in enumerate(powerset(self._grand_coalition_set, max_size=order)):
                interaction_size = len(interaction)
                coalition_cap_interaction = len(set(coalition).intersection(set(interaction)))
                base_interaction_values[j] += (
                    (-1) ** (interaction_size - coalition_cap_interaction)
                    * base_weights[coalition_size - coalition_cap_interaction, interaction_size]
                    * self.game_values[coalition_pos]
                )

        interaction_lookup = {
            interaction: i
            for i, interaction in enumerate(powerset(self._grand_coalition_set, max_size=order))
        }

        # CHII is un-defined for empty set
        if index == "CHII" and () in interaction_lookup:
            warnings.warn(
                f"CHII is not defined for the empty set. Setting to the baseline value "
                f"{self.baseline_value}.",
                stacklevel=2,
            )
            base_interaction_values[interaction_lookup[()]] = self.baseline_value

        # Transform into InteractionValues object and store in computed dictionary
        base_interaction = InteractionValues(
            values=base_interaction_values,
            index=index,
            max_order=order,
            min_order=0,
            n_players=self.n_players,
            interaction_lookup=interaction_lookup,
            estimated=False,
            baseline_value=self.baseline_value,
        )
        self._computed[(index, order)] = copy.deepcopy(base_interaction)
        return copy.deepcopy(base_interaction)

    def base_generalized_value(
        self,
        index: Literal["SGV", "BGV", "CHGV", "IGV", "EGV"],
        order: int,
    ) -> InteractionValues:
        """Compute Base Generalized Values.

        Base Generalized Values are probabilistic generalized values that do not depend on the
        order. According to the underlying representation using marginal contributions from
        [Mar07]_, the following indices are supported:
            - SGV: Shapley Generalized Value [Mar00]_
            - BGV: Banzhaf Generalized Value [Mar00]_
            - CHGV: Chaining Generalized Value [Mar07]_
            - IGV: Internal Generalized Value [Mar07]_
            - EGV: External Generalized Value [Mar07]_

        Args:
            order: The highest order of interactions
            index: The generalized value index

        Returns:
            An InteractionValues object containing generalized values.

        """
        base_generalized_values = np.zeros(self._n_interactions[order])
        base_weights = self._get_base_weights(index, order)

        interaction_lookup = {
            interaction: i
            for i, interaction in enumerate(powerset(self._grand_coalition_set, max_size=order))
        }

        for i, coalition in enumerate(
            powerset(self._grand_coalition_set, min_size=0, max_size=self.n_players - 1),
        ):
            coalition_val = self.game_values[i]
            for interaction in powerset(
                (self._grand_coalition_set - set(coalition)),
                min_size=1,
                max_size=order,
            ):
                coalition_weight = base_weights[len(coalition), len(interaction)]
                base_generalized_values[interaction_lookup[tuple(sorted(interaction))]] += (
                    coalition_weight
                    * (
                        self.game_values[
                            self.coalition_lookup[tuple(sorted(coalition + interaction))]
                        ]
                        - coalition_val
                    )
                )

        # Transform into InteractionValues object
        base_generalized_values = InteractionValues(
            values=base_generalized_values,
            index=index,
            max_order=order,
            min_order=0,
            n_players=self.n_players,
            interaction_lookup=interaction_lookup,
            estimated=False,
            baseline_value=self.baseline_value,
        )
        self._computed[(index, order)] = copy.deepcopy(base_generalized_values)
        return copy.deepcopy(base_generalized_values)

    def base_aggregation(
        self,
        base_interactions: InteractionValues,
        order: int,
    ) -> InteractionValues:
        """Transform Base Interactions into Interactions satisfying efficiency, e.g. SII to k-SII.

        Args:
            base_interactions: InteractionValues object containing interactions up to order
                ``order``.
            order: The highest order of interactions considered.

        Returns:
            InteractionValues object containing transformed base_interactions

        """
        from .aggregation import aggregate_base_interaction

        transformed_interactions = aggregate_base_interaction(base_interactions, order)
        return copy.deepcopy(transformed_interactions)

    def compute_stii(self, order: int) -> InteractionValues:
        """Compute the STII index up to order ``order`` after [Sun20]_.

        Args:
            order: The highest order of interactions

        Returns:
            InteractionValues object containing STII

        """
        stii_values = np.zeros(self._n_interactions[order])
        stii_values[0] = self.baseline_value  # set baseline value

        # create interaction lookup
        interaction_lookup = {
            interaction: i
            for i, interaction in enumerate(powerset(self._grand_coalition_set, max_size=order))
        }

        # lower-order interactions (size < order) are the Möbius transform, i.e. discrete derivative with empty set
        for interaction in powerset(self._grand_coalition_set, max_size=order - 1):
            stii_values[interaction_lookup[interaction]] = self._get_discrete_derivative(
                interaction,
                (),
            )

        # pre-compute STII weights
        stii_weights = self._get_stii_weights(order)

        # top-order STII interactions
        for interaction in powerset(self._grand_coalition_set, min_size=order, max_size=order):
            interaction_pos = interaction_lookup[interaction]
            for coalition_pos, coalition in enumerate(powerset(self._grand_coalition_set)):
                coalition_size = len(coalition)
                intersection_size = len(set(coalition).intersection(set(interaction)))
                stii_values[interaction_pos] += (
                    (-1) ** (order - intersection_size)
                    * stii_weights[coalition_size - intersection_size]
                    * self.game_values[coalition_pos]
                )

        # transform into InteractionValues object
        stii = InteractionValues(
            values=stii_values,
            index="STII",
            max_order=order,
            min_order=0,
            n_players=self.n_players,
            interaction_lookup=interaction_lookup,
            estimated=False,
            baseline_value=self.baseline_value,
        )
        return copy.deepcopy(stii)

    def compute_fii(self, index: Literal["FSII", "FBII"], order: int) -> InteractionValues:
        """Compute the FSII or FBII indices up to order ``order`` after [Tsa23]_.

        Args:
            order: The highest order of interactions
            index: FSII for Shapley or FBII for Banzhaf

        Returns:
            InteractionValues object containing FSII or FBII

        """
        fii_weights = self._get_fii_weights(index)
        least_squares_weights = np.zeros(2**self.n_players, dtype=float)
        coalition_matrix = np.zeros((2**self.n_players, self._n_interactions[order]), dtype=bool)

        # create interaction lookup
        interaction_lookup = {
            interaction: i
            for i, interaction in enumerate(powerset(self._grand_coalition_set, max_size=order))
        }
        coalition_store = {}

        # Set least squares matrices
        for coalition_pos, coalition in enumerate(powerset(self._grand_coalition_set)):
            least_squares_weights[coalition_pos] = fii_weights[len(coalition)]
            for interaction in powerset(coalition, max_size=order):
                pos = interaction_lookup[interaction]
                coalition_matrix[coalition_pos, pos] = 1
            coalition_store[coalition] = coalition_pos
        weight_matrix_sqrt = np.sqrt(np.diag(least_squares_weights))
        coalition_matrix_weighted_sqrt = np.dot(weight_matrix_sqrt, coalition_matrix)

        # normalization
        regression_response = self.game_values - self.baseline_value

        # solve the weighted least squares (WLSQ) problem
        regression_response_weighted_sqrt = np.dot(regression_response, weight_matrix_sqrt)
        fii_values, _, _, _ = np.linalg.lstsq(
            coalition_matrix_weighted_sqrt,
            regression_response_weighted_sqrt,
            rcond=None,
        )

        # transform into InteractionValues object
        match index:
            case "FSII":
                # For FSII ensure empty set is set to baseline
                baseline_value = self.baseline_value
            case "FBII":
                # For FBII the empty set is computed
                baseline_value = fii_values[0] + self.baseline_value
            case _:
                msg = f"Index {index} not supported."
                raise ValueError(msg)
        fii_values[0] = baseline_value  # set baseline value
        fii = InteractionValues(
            values=fii_values,
            index=index,
            max_order=order,
            min_order=0,
            n_players=self.n_players,
            interaction_lookup=interaction_lookup,
            estimated=False,
            baseline_value=baseline_value,
        )

        # cache and return
        self._computed[(index, order)] = copy.deepcopy(fii)
        return copy.deepcopy(fii)

    def get_jointsv_weights(self, order: int) -> np.ndarray:
        """Pre-compute JointSV weights for all coalition sizes ``(0, ..., n - order)``.

        Args:
            order: The highest order of interactions

        Returns:
            An array of pre-computed weights

        """
        weights = np.zeros(self.n_players, dtype=np.longdouble)
        q0den = sum([binom(self.n_players, s) for s in range(1, order + 1)])
        weights[0] = 1 / q0den
        # Carry out recursion
        for r in range(1, self.n_players):
            limd = min(order, (self.n_players - r))
            limn = max((r - order), 0)
            qden = sum([binom(self.n_players - r, s) for s in range(1, limd + 1)])
            qnum = sum([binom(r, s) * weights[s] for s in range(limn, r)])
            weights[r] = qnum / qden
        # check that the checksum is satisfied
        checksum = sum(
            [
                binom(self.n_players, i) * weights[i]
                for i in range((self.n_players - order), self.n_players)
            ]
        )
        if not np.isclose(checksum, 1.0):
            message = (
                f"JointSV weights do not sum to 1.0. but to {checksum}. This is likely due "
                f"to numerical instability."
            )
            warnings.warn(message, stacklevel=2)
        return weights

    def compute_kadd_shap(self, order: int) -> InteractionValues:
        """Computes the kADD-SHAP index up to order "order".

        The kADD-SHAP index is similar to FSII except that the coalition matrix contains the
        Bernoulli weights. The implementation is according to [Pel23]_.

        Args:
            order: The highest order of interactions

        Returns:
            An InteractionValues object containing kADD-SHAP values

        """
        weights = self._get_fii_weights(index="FSII")
        least_squares_weights = np.zeros(2**self.n_players)
        coalition_matrix = np.zeros((2**self.n_players, self._n_interactions[order]))
        bernoulli_weights = get_bernoulli_weights(order)

        interaction_lookup = {
            interaction: i
            for i, interaction in enumerate(powerset(self._grand_coalition_set, max_size=order))
        }

        for coalition_pos, coalition in enumerate(powerset(self._grand_coalition_set)):
            least_squares_weights[coalition_pos] = weights[len(coalition)]
            for interaction in powerset(self._grand_coalition_set, min_size=1, max_size=order):
                intersection_size = len(set(coalition).intersection(interaction))
                interaction_size = len(interaction)
                # This is different from FSII
                coalition_matrix[coalition_pos, interaction_lookup[interaction]] = (
                    bernoulli_weights[interaction_size, intersection_size]
                )

        weight_matrix_sqrt = np.sqrt(np.diag(least_squares_weights))
        coalition_matrix_weighted_sqrt = np.dot(weight_matrix_sqrt, coalition_matrix)

        regression_response = self.game_values - self.baseline_value  # normalization
        regression_response_weighted_sqrt = np.dot(regression_response, weight_matrix_sqrt)
        kADD_shap_values, residuals, rank, singular_values = np.linalg.lstsq(
            coalition_matrix_weighted_sqrt,
            regression_response_weighted_sqrt,
            rcond=None,
        )

        # Set baseline value
        kADD_shap_values[0] = self.baseline_value

        # Transform into InteractionValues object
        return InteractionValues(
            values=kADD_shap_values,
            index="kADD-SHAP",
            max_order=order,
            min_order=0,
            n_players=self.n_players,
            interaction_lookup=interaction_lookup,
            estimated=False,
            baseline_value=self.baseline_value,
        )

    def compute_joint_sv(self, order: int) -> InteractionValues:
        """Computes the JointSV index up to an order according to [Har22]_.

        Args:
            order: The highest order of interactions

        Returns:
            An InteractionValues object containing kADD-SHAP values

        """
        jointSV_values = np.zeros(self._n_interactions[order])
        # Set baseline value
        jointSV_values[0] = self.baseline_value
        coalition_weights = self.get_jointsv_weights(order)

        interaction_lookup = {
            interaction: i
            for i, interaction in enumerate(powerset(self._grand_coalition_set, max_size=order))
        }

        for coalition_pos, coalition in enumerate(
            powerset(self._grand_coalition_set, min_size=0, max_size=self.n_players - 1),
        ):
            coalition_val = self.game_values[coalition_pos]
            coalition_weight = coalition_weights[len(coalition)]
            for interaction in powerset(
                self._grand_coalition_set - set(coalition),
                min_size=1,
                max_size=order,
            ):
                jointSV_values[interaction_lookup[interaction]] += coalition_weight * (
                    self.game_values[self.coalition_lookup[tuple(sorted(coalition + interaction))]]
                    - coalition_val
                )

        # Transform into InteractionValues object
        return InteractionValues(
            values=jointSV_values,
            index="JointSV",
            max_order=order,
            min_order=0,
            n_players=self.n_players,
            interaction_lookup=interaction_lookup,
            estimated=False,
            baseline_value=self.baseline_value,
        )

    def shapley_generalized_value(self, order: int) -> InteractionValues:
        """Computes Shapley Generalized Values.

        The underlying representation of Shapley Generalized Values (i.e. Generalized Values that
        satisfy efficiency) is presented in [Mar07]_. The following indices are supported:
            - JointSV [Har22]_

        Args:
            order: The highest order of interactions.

        Returns:
            An InteractionValues object containing generalized values

        Raises:
            ValueError: If the index is not supported.

        """
        shapley_generalized_value = self.compute_joint_sv(order)
        self._computed[("JointSV", order)] = shapley_generalized_value
        return shapley_generalized_value

    def shapley_interactions(
        self, index: Literal["k-SII", "STII", "kADD-SHAP", "FSII"], order: int
    ) -> InteractionValues:
        """Computes k-additive Shapley Interactions, i.e. probabilistic interaction indices that depend on the order k.

        According to the underlying representation using discrete derivatives from [Fui06]_, the
        following indices are supported:
            - k-SII: k-Shapley Values [Bor23]_
            - STII:  Shapley-Taylor Interaction Index [Sun20]_
            - kADD-SHAP: k-additive Shapley Values [Pel23]_
            - FSII: Faithful Shapley Interaction Index [Tsa23]_

        Args:
            order: The highest order of interactions
            index: The interaction index

        Returns:
            An InteractionValues object containing interaction values

        Raises:
            ValueError: If the index is not supported

        """
        match index:
            case "k-SII":
                sii = self.base_interaction("SII", order)
                shapley_interaction = self.base_aggregation(sii, order)
            case "STII":
                shapley_interaction = self.compute_stii(order)
            case "FSII":
                shapley_interaction = self.compute_fii(index, order)
            case "kADD-SHAP":
                shapley_interaction = self.compute_kadd_shap(order)
            case _:
                msg = f"Index {index} not supported."
                raise ValueError(msg)
        self._computed[(index, order)] = shapley_interaction
        return copy.copy(shapley_interaction)

    def shapley_base_interaction(
        self, index: Literal["SII", "BII", "CHII", "Moebius", "Co-Moebius"], order: int
    ) -> InteractionValues:
        """Computes Shapley Base Interactions, i.e. probabilistic interaction indices not depending on the order.

        According to the underlying representation using discrete derivatives from [Fui06]_, the
        following indices are supported:
            - SII: Shapley Interaction Index [Gra99]_
            - BII: Banzhaf Interaction Index [Gra99]_
            - CHII: Chaining Interaction Index [Mar99]_
            - Moebius: Moebius Transform [Ras02]_
            - Co-Moebius: Co-Moebius Transform [Mar07]_

        Args:
            order: The highest order of interactions
            index: The interaction index

        Returns:
            An InteractionValues object containing interaction values

        """
        base_interaction = self.base_interaction(index, order)
        self._computed[(index, order)] = base_interaction
        return copy.copy(base_interaction)

    def probabilistic_value(self, index: Literal["SV", "BV"]) -> InteractionValues:
        """Computes common semi-values or probabilistic values depending on the index.

        These semi-values are special forms of interaction indices and generalized values for
        ``order = 1``. According to the underlying representation using marginal contributions; cf.
        semi-values [Dub81]_, or probabilistic values [Web09]_ for the following indices are
        supported:
            - SV: Shapley value [Sha53]_
            - BV: Banzhaf value [Ban64]_

        Args:
            index: The interaction index

        Returns:
            An InteractionValues object containing probabilistic values

        Raises:
            ValueError: If the index is not supported

        """
        order = 1
        match index:
            case "BV":
                probabilistic_value = self.base_interaction(index="BII", order=order)
            case "SV":
                probabilistic_value = self.base_interaction(index="SII", order=order)
            case _:
                msg = f"Index {index} not supported."
                raise ValueError(msg)
        # Change emptyset to baseline value, due to the definitions of players
        probabilistic_value.baseline_value = self.baseline_value
        probabilistic_value.values[probabilistic_value.interaction_lookup[()]] = self.baseline_value
        self._computed[(index, order)] = probabilistic_value
        return copy.copy(probabilistic_value)

    def compute_egalitarian_least_core(self) -> InteractionValues:
        """Computes the egalitarian least core (ELC) of the game.

        The egalitarian least core (ELC) is a solution concept in cooperative game theory that
        distributes the total value of the grand coalition among the players in a way that
        minimizes the maximum excess of any coalition. It is a refinement of the core and
        represents a fair distribution of the total value of the grand coalition. The ELC is
        implemented after [Yan21]_.

        Returns:
            The egalitarian least core of the game.
        """
        from shapiq.game_theory.core import egalitarian_least_core

        order = 1

        # Compute egalitarian least-core
        egalitarian_vector, subsidy = egalitarian_least_core(
            n_players=self.n_players,
            game_values=self.game_values,
            coalition_lookup=self.coalition_lookup,
        )

        # Store results
        self._computed[("ELC", order)] = egalitarian_vector
        self._elc_stability_subsidy = subsidy

        return copy.copy(egalitarian_vector)


def get_bernoulli_weights(order: int) -> np.ndarray:
    """Returns the bernoulli weights in the k-additive approximation via SII.

    For some indices like ``'kADD-SHAP'``, the weights must be scaled with the Bernoulli numbers.

    Args:
        order: The highest order of interactions.

    Returns:
        An array containing the bernoulli weights for the k-additive approximation.

    """
    # TODO(mmshlk): We should import this in the kADD-SHAP approximator from here https://github.com/mmschlk/shapiq/issues/390
    bernoulli_numbers = bernoulli(order)
    weights = np.zeros((order + 1, order + 1))
    for interaction_size in range(1, order + 1):
        for intersection_size in range(interaction_size + 1):
            for sum_index in range(1, intersection_size + 1):
                weights[interaction_size, intersection_size] += (
                    binom(intersection_size, sum_index)
                    * bernoulli_numbers[interaction_size - sum_index]
                )
    return weights
