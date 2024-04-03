"""This module contains the base class for all games in the shapiq package."""

import pickle
import warnings
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from shapiq.utils import powerset, transform_array_to_coalitions, transform_coalitions_to_array


class Game(ABC):
    """Base class for all games in the shapiq package.

    This class implements some common methods and attributes that all games should have.

    Args:
        n_players: The number of players in the game.
        normalize: Whether the game values should be normalized. Defaults to `True`.
        normalization_value: The value to normalize and center the game values with such that the
            value for the empty coalition is zero. Defaults to `None`.  If `normalization` is set
            to `False` this value is not required. Otherwise, the value is needed to normalize and
            center the game. If no value is provided, the game raise a warning.

    Note:
        This class is an abstract base class and should not be instantiated directly. All games
        should inherit from this class and implement the abstract methods.
    """

    @abstractmethod
    def __init__(
        self,
        n_players: int,
        normalize: bool = True,
        normalization_value: Optional[float] = None,
    ) -> None:
        # define storage variables
        self.value_storage: np.ndarray = np.zeros(0, dtype=float)
        self.coalition_lookup: dict[tuple[int, ...], int] = {}

        # define some handy variables describing the game
        self.n_players: int = n_players

        # setup normalization of the game
        self.normalization_value: float = 0.0
        if normalize:
            self.normalization_value = normalization_value
            if self.normalization_value is None:
                warnings.warn(
                    RuntimeWarning(
                        "Normalization value is set to `None`. No normalization value was provided"
                        " at initialization."
                    )
                )

        # define some handy coalition variables
        self.empty_coalition = np.zeros(n_players, dtype=bool)
        self.grand_coalition = np.ones(n_players, dtype=bool)

        # manual flag for choosing precomputed values even if not all values might be stored
        self.precompute_flag: bool = False  # flag to manually override the precomputed check

    @property
    def n_values_stored(self) -> int:
        """The number of values stored in the game."""
        return len(self.coalition_lookup)

    @property
    def precomputed(self) -> bool:
        """Indication whether the game has been precomputed."""
        return self.n_values_stored >= 2**self.n_players or self.precompute_flag

    @property
    def normalize(self) -> bool:
        """Indication whether the game values are normalized."""
        return int(self.normalization_value) != 0

    def __call__(self, coalitions: np.ndarray) -> np.ndarray:
        """Calls the game's value function with the given coalitions and returns the output of the
        value function.

        Args:
            coalitions: The coalitions to evaluate.

        Returns:
            The values of the coalitions.
        """
        # check if coalitions are correct dimensions
        if coalitions.ndim == 1:
            coalitions = coalitions.reshape((1, self.n_players))

        if not self.precomputed:
            values = self.value_function(coalitions)
        else:
            # lookup the values present in the storage
            values = np.zeros(coalitions.shape[0], dtype=float)
            for i, coalition in enumerate(coalitions):
                # convert one-hot vector to tuple
                coalition_tuple = tuple(np.where(coalition)[0])
                index = self.coalition_lookup[coalition_tuple]
                values[i] = self.value_storage[index]

        return values - self.normalization_value

    @abstractmethod
    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """The value function of the game, which models the behavior of the game. The value function
        is the core of the game and should be implemented in the inheriting class.

        Args:
            coalitions: The coalitions to evaluate.

        Returns:
            np.ndarray: The values of the coalitions.

        Note:
            This method has to be implemented in the inheriting class.
        """
        raise NotImplementedError("The value function has to be implemented in inherited classes.")

    def precompute(self, coalitions: Optional[np.ndarray] = None) -> None:
        """Precompute the game values for all or a given set of coalitions.

        The pre-computation iterates over the powerset of all coalitions or a given set of
        coalitions and stores the values of the coalitions in a dictionary.

        Args:
            coalitions: The set of coalitions to precompute. If None, the powerset of all
                coalitions will be used.

        Examples:
            >>> from shapiq.games import DummyGame
            >>> game = DummyGame(4, interaction=(1, 2))
            >>> game.precomputed, game.n_values_stored
            False, 0
            >>> game.precompute()
            >>> game.precomputed, game.n_values_stored
            True, 16
            >>> # precompute only a subset of coalitions
            >>> game = DummyGame(4, interaction=(1, 2))
            >>> coals = np.asarray([[True, False, False, False], [False, True, True, False]])
            >>> game.precompute(coalitions=coals)
            >>> game.precomputed, game.n_values_stored
            True, 2
            >>> # store values
            >>> game.save_values("dummy_game.npz")
            >>> # load values in other game
            >>> new_game = DummyGame(4, interaction=(1, 2))
            >>> new_game.load_values("dummy_game.npz")

        Note:
            The pre-computation can be slow for a large number of players since the powerset of
            all coalitions is evaluated. If the number of players is greater than 16 and no
            coalitions are given, a warning is raised to inform the user about the potential
            slow computation.
        """
        # if more than 16 players and no coalitions are given, warn the user
        if self.n_players > 16 and coalitions is None:
            warnings.warn(
                "The number of players is greater than 16. Precomputing all coalitions might "
                "take a long time. Consider providing a subset of coalitions to precompute. "
                "Note that 2 ** n_players coalitions will be evaluated for the pre-computation."
            )
        if coalitions is None:
            coalitions = list(powerset(range(self.n_players)))  # might be getting slow
            coalitions_array = transform_coalitions_to_array(coalitions, self.n_players)
            coalitions_dict = {coal: i for i, coal in enumerate(coalitions)}
        else:
            coalitions_array = coalitions
            coalitions_tuple = transform_array_to_coalitions(coalitions=coalitions_array)
            coalitions_dict = {coal: i for i, coal in enumerate(coalitions_tuple)}

        # run the game for all coalitions
        game_values: np.ndarray = self(coalitions_array)  # call the game with the coalitions

        # update the storage with the new coalitions and values
        self.value_storage = game_values.astype(float)
        self.coalition_lookup = coalitions_dict
        self.precompute_flag = True

    def save_values(self, path: str) -> None:
        """Saves the game values to the given path.

        Args:
            path: The path to save the game.
        """
        # check if path ends with .npz
        if not path.endswith(".npz"):
            path += ".npz"

        if not self.precomputed:
            warnings.warn(
                UserWarning("The game has not been precomputed yet. Saving the game may be slow.")
            )
            self.precompute()

        # transform the values_storage to float16 for compression
        self.value_storage = self.value_storage.astype(np.float16)

        # cast the coalitions_in_storage to bool
        coalitions_in_storage = transform_coalitions_to_array(
            coalitions=self.coalition_lookup, n_players=self.n_players
        ).astype(bool)

        # save the data
        np.savez_compressed(
            path,
            values=self.value_storage,
            coalitions=coalitions_in_storage,
            n_players=self.n_players,
        )

    def load_values(self, path: str, precomputed: bool = False) -> None:
        """Loads the game values from the given path.

        Args:
            path: The path to load the game values from.
            precomputed: Whether the game should be set to precomputed after loading the values no
                matter how many values are loaded. This can be useful if a game is loaded for a
                subset of all coalitions and only this subset will be used. Defaults to `False`.
        """
        # check if path ends with .npz
        if not path.endswith(".npz"):
            path += ".npz"

        data = np.load(path)
        n_players = data["n_players"]
        if n_players != self.n_players:
            raise ValueError(
                f"The number of players in the game ({self.n_players}) does not match the number "
                f"of players in the saved game ({n_players})."
            )
        self.value_storage = data["values"]
        self.coalition_lookup = transform_array_to_coalitions(data["coalitions"])
        self.precompute_flag = precomputed

    def save(self, path: str) -> None:
        """Saves and serializes the game object to the given path.

        Args:
            path: The path to save the game.
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "Game":
        """Load the game from a given path.

        Args:
            path: The path to load the game from.
        """
        with open(path, "rb") as f:
            game = pickle.load(f)
        return game

    def __repr__(self) -> str:
        """Return a string representation of the game."""
        return f"{self.__class__.__name__}({self.n_players} players, normalize={self.normalize})"

    def __str__(self) -> str:
        """Return a string representation of the game."""
        return self.__repr__()
