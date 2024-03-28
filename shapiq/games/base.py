"""This module contains the base class for all games in the shapiq package."""

import pickle
import warnings
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from shapiq.utils import powerset, transform_coalitions_to_array


class Game(ABC):
    """Base class for all games in the shapiq package.

    This class implements some common methods and attributes that all games should have.

    Note:
        This class is an abstract base class and should not be instantiated directly. All games
        should inherit from this class and implement the abstract methods.
    """

    @abstractmethod
    def __init__(self, n_players: int, *args, **kwargs):
        self.value_storage: Optional[np.ndarray] = None
        self.coalitions_in_storage: Optional[np.ndarray] = None

        # define some handy variables describing the game
        self.n_players: int = n_players

        # define some handy coalition variables
        self.empty_coalition = np.zeros(n_players, dtype=bool)
        self.grand_coalition = np.ones(n_players, dtype=bool)

        # get the empty value of the game
        self.empty_value: float = float(self(self.empty_coalition)[0])

    @abstractmethod
    def __call__(self, coalitions: np.ndarray) -> np.ndarray:
        """Calls the game with the given coalitions and returns the values of the coalitions.

        Args:
            coalitions: The coalitions to evaluate.

        Returns:
            np.ndarray: The values of the coalitions.

        Note:
            This method should be implemented by the inheriting class.
        """
        raise NotImplementedError

    def save_values(self, path: str) -> None:
        """Saves the game values to the given path.

        Args:
            path: The path to save the game.
        """
        # check if path ends with .npz
        if not path.endswith(".npz"):
            path += ".npz"

        if self.value_storage is None:
            warnings.warn(
                UserWarning("The game has not been precomputed yet. Saving the game may be slow.")
            )
            self.precompute()

        # transform the values_storage to float16 for compression
        self.value_storage = self.value_storage.astype(np.float16)

        # cast the coalitions_in_storage to bool
        self.coalitions_in_storage = self.coalitions_in_storage.astype(bool)

        # save the data
        np.savez_compressed(
            path,
            values=self.value_storage,
            coalitions=self.coalitions_in_storage,
            n_players=self.n_players,
            empty_value=self.empty_value,
        )

    def load_values(self, path: str) -> None:
        """Loads the game values from the given path.

        Args:
            path: The path to load the game values from.
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
        self.empty_value = data["empty_value"]
        self.value_storage = data["values"]
        self.coalitions_in_storage = data["coalitions"]

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

    def precompute(self, coalitions: Optional[np.ndarray] = None) -> None:
        """Precompute the game values for all or a given set of coalitions.

        The pre-computation iterates over the powerset of all coalitions or a given set of
        coalitions and stores the values of the coalitions in a dictionary.

        Args:
            coalitions: The set of coalitions to precompute. If None, the powerset of all
                coalitions will be used.
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
        game_values: np.ndarray = self(coalitions_array)  # call the game with the coalitions
        self.value_storage = game_values
        self.coalitions_in_storage = coalitions_array

    def __repr__(self) -> str:
        precomputed = "precomputed" if self.value_storage is not None else "not precomputed"
        return f"{self.__class__.__name__}({self.n_players} players, {precomputed})"

    def __str__(self) -> str:
        return self.__repr__()
