"""Base Game class for games and benchmarks."""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np
from tqdm.auto import tqdm

from shapiq.utils import (
    powerset,
    raise_deprecation_warning,
    transform_array_to_coalitions,
    transform_coalitions_to_array,
)

if TYPE_CHECKING:
    from shapiq.interaction_values import InteractionValues
    from shapiq.typing import CoalitionMatrix, GameValues, JSONType, MetadataBlock

    class GameJSONMetadata(TypedDict):
        """Metadata for the game loaded from JSON."""

        n_players: int
        game_name: str
        normalization_value: float
        precompute_flag: bool
        precomputed: bool
        player_names: list[str] | None

    class GameJSON(MetadataBlock, TypedDict):
        """Types denoting the Fields loaded from JSON."""

        metadata: GameJSONMetadata
        data: dict[str, float]


class Game:
    """Base class for games/benchmarks/imputers in the ``shapiq`` package.

    This class implements some common methods and attributes that all games should have.

    Properties:
        n_values_stored: The number of values stored in the game.
        precomputed: Indication whether the game has been precomputed.
        normalize: Indication whether the game values are normalized.
        game_name: The name of the game.

    Attributes:
        _precompute_flag: A flag to manually override the precomputed check. If set to ``True``, the
            game is considered precomputed and only uses the lookup.
        value_storage: The storage for the game values without normalization applied.
        coalition_lookup: A lookup dictionary mapping from coalitions to indices in the
            ``value_storage``.
        n_players: The number of players in the game.
        normalization_value: The value to normalize and center the game values with.
        empty_coalition: The empty coalition of the game.
        grand_coalition: The grand coalition of the game.
        verbose: Whether to show a progress bar for the evaluation.

    Note:
        This class is a base class and all games should inherit from this class and implement the
            `value_function` methods. Usually, this Game class is only directly used when dealing
            with precomputed / stored games.

    Examples:
        >>> from shapiq.games import Game
        >>> from shapiq.games.benchmark.synthetic import DummyGame
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
        >>> game.precomputed, game.n_values_stored
        True, 2
        >>> # you can also load a game of any class with the Game class
        >>> new_game = Game(path_to_values="dummy_game.npz")
        >>> new_game.precomputed, new_game.n_values_stored
        True, 2
        >>> # save and load the game
        >>> game.save("game.pkl")
        >>> new_game = DummyGame.load("game.pkl")
        >>> new_game.precomputed, new_game.n_values_stored
        True, 2

    """

    n_players: int
    """The number of players in the game."""

    game_id: str
    """A unique identifier for the game, based on its class name and hash."""

    normalization_value: float
    """The value which is used to normalize (center) the game values such that the value for the
    empty coalition is zero. If this is zero, the game values are not normalized."""

    value_storage: GameValues
    """The storage for the game values without normalization applied."""

    def __init__(
        self,
        n_players: int | None = None,
        *,
        normalize: bool = True,
        normalization_value: float | None = None,
        path_to_values: Path | str | None = None,
        verbose: bool = False,
        player_names: list[str] | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Initialize the Game class.

        Args:
            n_players: The number of players in the game.

            normalize: Whether the game values should be normalized / centered. Defaults to
                ``True``. If ``True``, the game values are normalized such that the value for the
                empty coalition is zero. If ``False``, the game values are not normalized and the
                value for the empty coalition is not guaranteed to be zero. This is useful for
                algorithms that require the game values to be centered.

            normalization_value: The value to normalize and center the game values with such that the
                value for the empty coalition is zero. Defaults to ``None``.  If ``normalization`` is set
                to ``False`` this value is not required. Otherwise, the value is needed to normalize and
                center the game. If no value is provided, the game raises a warning.

            path_to_values: The path to load the game values from. If the path is provided, the game
                values are loaded from the given path. Defaults to ``None``.

            verbose: Whether to show a progress bar for the evaluation. Defaults to ``False``. Note
                that this only has an effect if the game is not precomputed and may slow down the
                evaluation.

            player_names: An optional list of player names. If provided, the coalitions can be
                provided as strings instead of integers.

            kwargs: Additional keyword arguments (not used).

        """
        # manual flag for choosing precomputed values even if not all values might be stored
        self._precompute_flag: bool = False  # flag to manually override the precomputed check

        # define storage variables
        self.value_storage: GameValues = np.zeros(0, dtype=float)
        self.coalition_lookup: dict[tuple[int, ...], int] = {}
        self.n_players = n_players

        if path_to_values is not None:
            msg = (
                "Initializing a Game with `path_to_values` is deprecated and will be removed in a"
                " future version. Use `Game.load` or `Game().load_values()` instead."
            )
            raise_deprecation_warning(message=msg, deprecated_in="1.3.1", removed_in="1.4.0")

        if n_players is None and path_to_values is None:
            msg = "The number of players has to be provided if game is not loaded from values."
            raise ValueError(msg)

        # setup normalization of the game
        self.normalization_value = 0.0
        if normalize and path_to_values is None:
            self.normalization_value = normalization_value
            if normalization_value is None:
                # this is desired behavior, as in some cases normalization is set by the subclasses
                # after init of the base Game class. For example, in the imputer classes.
                warnings.warn(
                    RuntimeWarning(
                        "Normalization value is set to `None`. No normalization value was provided"
                        " at initialization. Make sure to set the normalization value before"
                        " calling the game.",
                    ),
                    stacklevel=2,
                )

        game_id = str(hash(self))[:8]
        self.game_id = f"{self.get_game_name()}_{game_id}"
        if path_to_values is not None:
            self.load_values(path_to_values, precomputed=True)
            self.game_id = str(path_to_values).split(os.path.sep)[-1].split(".")[0]
            # if game should not be normalized, reset normalization value to 0
            if not normalize and self.normalization_value != 0:
                self.normalization_value = 0.0

        # define some handy coalition variables
        self.empty_coalition = np.zeros(self.n_players, dtype=bool)
        self.grand_coalition = np.ones(self.n_players, dtype=bool)
        self._empty_coalition_value_property = None
        self._grand_coalition_value_property = None

        # define player_names
        self.player_name_lookup: dict[str, int] = (
            {name: i for i, name in enumerate(player_names)} if player_names is not None else None
        )

        self.verbose = verbose

    @property
    def n_values_stored(self) -> int:
        """The number of values stored in the game."""
        return len(self.coalition_lookup)

    @property
    def precomputed(self) -> bool:
        """Indication whether the game has been precomputed."""
        return self.n_values_stored >= 2**self.n_players or self._precompute_flag

    @property
    def normalize(self) -> bool:
        """Indication whether the game values are getting normalized."""
        return self.normalization_value != 0

    @property
    def is_normalized(self) -> bool:
        """Checks if the game is normalized/centered."""
        return self(self.empty_coalition) == 0

    def to_json_file(
        self,
        path: Path,
        *,
        desc: str | None = None,
        created_from: object | None = None,
        **kwargs: JSONType,
    ) -> None:
        """Saves the game as a JSON file.

        Args:
            path: Path to the JSON file. If the path does not end with ``.json``, it will be
                automatically appended.
            desc: A description of the game. Defaults to ``None``.
            created_from: An object from which the game was created. Defaults to ``None``.
            **kwargs: Additional keyword arguments to pass to the metadata block.

        """
        from shapiq.utils.saving import lookup_and_values_to_dict, make_file_metadata, save_json

        meta_data = make_file_metadata(
            object_to_store=self,
            data_type="game",
            desc=desc,
            created_from=created_from,
            parameters=kwargs,
        )
        data = {
            **meta_data,
            "metadata": {
                "n_players": self.n_players,
                "game_name": self.game_name,
                "normalization_value": self.normalization_value,
                "precompute_flag": self._precompute_flag,
                "precomputed": self.precomputed,
                "player_names": list(self.player_name_lookup.keys())
                if self.player_name_lookup
                else None,
            },
            "data": lookup_and_values_to_dict(self.coalition_lookup, self.value_storage),
        }
        save_json(data, path)

    @classmethod
    def from_json_file(cls, path: Path | str, *, normalize: bool = True) -> Game:
        """Loads the game from a JSON file.

        Args:
            path: Path to the JSON file.
            normalize: Whether to normalize the game values. Defaults to ``True``.

        Returns:
            Game: The loaded game object.
        """
        from shapiq.utils.saving import dict_to_lookup_and_values

        with Path(path).open("r") as file:
            data: GameJSON = json.load(file)

        metadata = data["metadata"]
        n_players = metadata["n_players"]
        normalization_value = metadata["normalization_value"]
        player_names = metadata["player_names"]

        game = Game(
            n_players=n_players,
            normalization_value=normalization_value,
            normalize=normalize,
            player_names=player_names,
        )
        game._precompute_flag = metadata["precompute_flag"]
        game.coalition_lookup, game.value_storage = dict_to_lookup_and_values(data["data"])
        return game

    def _check_coalitions(
        self,
        coalitions: (
            CoalitionMatrix
            | list[tuple[int, ...]]
            | list[tuple[str, ...]]
            | tuple[int, ...]
            | tuple[str, ...]
        ),
    ) -> CoalitionMatrix:
        """Validates the coalitions and convert them to one-hot encoding.

        Check if the coalitions are in the correct format and convert them to one-hot encoding.
        The format may either be a numpy array containg the coalitions in one-hot encoding or a
        list of tuples with integers or strings.

        Args:
            coalitions: The coalitions to convert to one-hot encoding.

        Returns:
            np.ndarray: The coalitions in the correct format

        Raises:
            TypeError: If the coalitions are not in the correct format.

        Examples:
            >>> coalitions = np.asarray([[1, 0, 0, 0], [0, 1, 1, 0]])
            >>> coalitions = [(0, 1), (1, 2)]
            >>> coalitions = [()]
            >>> coalitions = [(0, 1), (1, 2), (0, 1, 2)]
            if player_name_lookup is not None:
            >>> coalitions = [("Alice", "Bob"), ("Bob", "Charlie")]
            Wrong format:
            >>> coalitions = [1, 0, 0, 0]
            >>> coalitions = [(1, "Alice")]
            >>> coalitions = np.array([1, -1, 2])

        """
        error_message = (
            "List may only contain tuples of integers or strings. The tuples are not allowed to "
            "have heterogeneous types. See the docs for correct format of coalitions. If strings "
            "are used, the player_name_lookup has to be provided during initialization."
        )
        # check for array input and do validation
        if isinstance(coalitions, np.ndarray):
            if len(coalitions) == 0:  # check that coalition is contained in array
                msg = "The array of coalitions is empty."
                raise TypeError(msg)
            if coalitions.ndim == 1:  # check if single coalition is correctly given
                if len(coalitions) < self.n_players or len(coalitions) > self.n_players:
                    msg = (
                        "The array of coalitions is not correctly formatted."
                        f"It should have a length of {self.n_players}"
                    )
                    raise TypeError(msg)
                coalitions = coalitions.reshape((1, self.n_players))
            if coalitions.shape[1] != self.n_players:  # check if players match
                msg = (
                    f"Number of players in the coalitions ({coalitions.shape[1]}) does not match "
                    f"the number of players in the game ({self.n_players})."
                )
                raise TypeError(msg)
            # check that values of numpy array are either 0 or 1
            if not np.all(np.logical_or(coalitions == 0, coalitions == 1)):
                msg = "The values in the array of coalitions are not binary."
                raise TypeError(msg)
            return coalitions
        # try for list of tuples
        if isinstance(coalitions, tuple):
            coalitions = [coalitions]
        try:
            # convert list of tuples to one-hot encoding
            return transform_coalitions_to_array(coalitions, self.n_players)
        except (IndexError, TypeError):
            pass
        # assuming str input
        if self.player_name_lookup is None:
            msg = "Player names are not provided. Cannot convert string to integer."
            raise ValueError(msg)
        try:
            coalitions_from_str = []
            for coalition in coalitions:
                coal_indices = sorted([self.player_name_lookup[player] for player in coalition])
                coalitions_from_str.append(tuple(coal_indices))
            return transform_coalitions_to_array(coalitions_from_str, self.n_players)
        except Exception as error:
            raise TypeError(error_message) from error

    def __call__(
        self,
        coalitions: (
            CoalitionMatrix
            | list[tuple[int, ...]]
            | list[tuple[str, ...]]
            | tuple[int, ...]
            | tuple[str, ...]
        ),
        *,
        verbose: bool = False,
    ) -> GameValues:
        """Calls the game with the given coalitions.

        Calls the game's value function with the given coalitions and returns the output of the
        value function. The call also checks if the coalitions are in the correct format and
        converts if necessary. If the game is precomputed, the values are looked up in internal
        storage without calling the value function.

        Args:
            coalitions: The coalitions to evaluate as a one-hot matrix or a list of tuples.

            verbose: Whether to show a progress bar for the evaluation. Defaults to ``False``.

        Returns:
            The values of the coalitions.

        """
        coalitions = self._check_coalitions(coalitions)
        verbose = verbose or self.verbose
        if not self.precomputed and not verbose:
            values = self.value_function(coalitions)
        elif not self.precomputed and verbose:
            values = np.zeros(coalitions.shape[0], dtype=float)
            for i, coalition in enumerate(
                tqdm(coalitions, desc="Evaluating game", unit=" coalition"),
            ):
                values[i] = self.value_function(coalition.reshape((1, self.n_players)))[0]
        else:
            values = self._lookup_coalitions(coalitions)
        return values - self.normalization_value

    def _lookup_coalitions(self, coalitions: CoalitionMatrix) -> GameValues:
        """Lookup the values of the coalitions in the storage."""
        values = np.zeros(coalitions.shape[0], dtype=float)
        for i, coalition in enumerate(coalitions):
            # convert one-hot vector to tuple
            coalition_tuple = tuple(np.where(coalition)[0])
            try:
                values[i] = self.value_storage[self.coalition_lookup[coalition_tuple]]
            except KeyError as error:
                msg = (
                    f"The coalition {coalition_tuple} is not stored in the game. "
                    f"Are all values pre-computed?"
                )
                raise KeyError(msg) from error
        return values

    def value_function(self, coalitions: CoalitionMatrix) -> GameValues:
        """Returns the value of the coalitions.

        The value function of the game, which models the behavior of the game. The value function
        is the core of the game and should be implemented in the inheriting class. A value function
        should return the worth of a coalition of players.

        Args:
            coalitions: The coalitions to evaluate.

        Returns:
            np.ndarray: The values of the coalitions.

        Note:
            This method has to be implemented in the inheriting class.

        """
        msg = "The value function has to be implemented in inherited classes."
        raise NotImplementedError(msg)

    def precompute(self, coalitions: CoalitionMatrix | None = None) -> None:
        """Precompute the game values for all or a given set of coalitions.

        The pre-computation iterates over the powerset of all coalitions or a given set of
        coalitions and stores the values of the coalitions in a dictionary.

        Args:
            coalitions: The set of coalitions to precompute. If None, the powerset of all
                coalitions will be used.

        Examples:
            >>> from shapiq.games.benchmark import DummyGame
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
                "Note that 2 ** n_players coalitions will be evaluated for the pre-computation.",
                stacklevel=2,
            )
        if coalitions is None:
            all_coalitions = list(powerset(range(self.n_players)))  # might be getting slow
            coalitions = transform_coalitions_to_array(all_coalitions, self.n_players)
            coalitions_dict = {coal: i for i, coal in enumerate(all_coalitions)}
        else:
            coalitions_tuple = transform_array_to_coalitions(coalitions=coalitions)
            coalitions_dict = {coal: i for i, coal in enumerate(coalitions_tuple)}

        # run the game for all coalitions (no normalization)
        norm_value, self.normalization_value = self.normalization_value, 0
        game_values = self(coalitions)  # call the game with the coalitions
        self.normalization_value = norm_value

        # update the storage with the new coalitions and values
        self.value_storage = game_values.astype(float)
        self.coalition_lookup = coalitions_dict
        self._precompute_flag = True

    def compute(
        self, coalitions: CoalitionMatrix
    ) -> tuple[np.ndarray, dict[tuple[int, ...], int], float]:
        """Compute the game values for all or a given set of coalitions.

        Args:
            coalitions: The coalitions to evaluate.

        Returns:
            A tuple containing:
            - The computed game values in the same order of the coalitions.
            - A lookup dictionary mapping from coalitions to the indices in the array.
            - The normalization value used to center/normalize the game values.

        Note:
            This method does not change the state of the game and does not normalize the values.

        Examples:
            >>> from shapiq.games.benchmark import DummyGame
            >>> game = DummyGame(4, interaction=(1, 2))
            >>> game.compute(np.array([[0, 1, 0, 0], [0, 1, 1, 0]], dtype=bool))
            (array([0.25, 1.5]), {(1): 0, (1, 2): 1.5}, 0.0)

        """
        game_values = self.value_function(self._check_coalitions(coalitions))

        return game_values, self.coalition_lookup, self.normalization_value

    def save_values(
        self, path: Path | str, *, as_npz: bool = False, **kwargs: dict[str, JSONType]
    ) -> None:
        """Saves the game values to the given path.

        Args:
            path: The path to save the game.
            as_npz: Whether to save the game as a numpy array (``True``) or as a JSON file.
                Defaults to ``False`` (saves as JSON file).
            **kwargs: Additional keyword arguments to pass to :meth:`~Game.to_json_file`.

        """
        # make sure path is a Path object
        path = Path(path)

        if not self.precomputed:
            warnings.warn(
                UserWarning("The game has not been precomputed yet. Saving the game may be slow."),
                stacklevel=2,
            )
            self.precompute()

        if as_npz or path.name.endswith(".npz"):
            path = path.with_suffix(".npz") if not path.name.endswith(".npz") else path  # add .npz
            coalitions_in_storage = transform_coalitions_to_array(
                coalitions=self.coalition_lookup,
                n_players=self.n_players,
            ).astype(bool)
            np.savez_compressed(
                path,
                values=self.value_storage,
                coalitions=coalitions_in_storage,
                n_players=self.n_players,
                normalization_value=self.normalization_value,
            )
        else:
            # store as JSON file
            self.to_json_file(path, **kwargs)

    def load_values(self, path: Path | str, *, precomputed: bool = False) -> None:
        """Loads the game values from the given path.

        Args:
            path: The path to load the game values from.
            precomputed: Whether the game should be set to precomputed after loading the values no
                matter how many values are loaded. This can be useful if a game is loaded for a
                subset of all coalitions and only this subset will be used. Defaults to ``False``.

        """
        path = Path(path)
        if path.name.endswith(".npz"):
            self._load_npz_values(path)
        elif path.name.endswith(".json"):
            self._load_json_values(path)
        else:
            msg = "The path to the game values must either be a .npz or .json file."
            raise ValueError(msg)
        self._precompute_flag = precomputed

    def _load_json_values(self, path: Path) -> None:
        """Loads game values from a JSON file."""
        game = self.from_json_file(path)
        self._validate_and_set_players_from_save(game.n_players)
        self.value_storage = game.value_storage
        self.coalition_lookup = game.coalition_lookup
        self.normalization_value = game.normalization_value

    def _validate_and_set_players_from_save(self, n_players: int) -> None:
        """Validates and sets the number of players from the saved game."""
        if self.n_players is not None and n_players != self.n_players:
            msg = (
                f"The number of players in the game ({self.n_players}) does not match the number "
                f"of players in the saved game ({n_players})."
            )
            raise ValueError(msg)
        self.n_players = int(n_players)

    def _load_npz_values(self, path: Path) -> None:
        """Load game values from a npz archive file."""
        data = np.load(path)
        self._validate_and_set_players_from_save(data["n_players"])
        self.value_storage = data["values"]
        coalition_lookup: list[tuple] = transform_array_to_coalitions(data["coalitions"])
        self.coalition_lookup = {coal: i for i, coal in enumerate(coalition_lookup)}
        self.normalization_value = float(data["normalization_value"])

    def save(self, path: Path | str, **kwargs: dict[str, JSONType]) -> None:
        """Saves and serializes the game object to the given path.

        Args:
            path: The path to save the game. If the path does not end with ``.json``, it will be
                automatically appended.
            **kwargs: Additional keyword arguments to pass to :meth:`~Game.to_json_file`.

        """
        self.to_json_file(Path(path), **kwargs)

    @classmethod
    def load(cls, path: Path | str, *, normalize: bool = True) -> Game:
        """Load the game from a given path.

        Args:
            path: The path to load the game from.
            normalize: Weather the game values should be normalized or not after loading.

        """
        return cls.from_json_file(path, normalize=normalize)

    def __repr__(self) -> str:
        """Return a string representation of the game."""
        return (
            f"{self.__class__.__name__}({self.n_players} players, normalize={self.normalize}"
            f", normalization_value={self.normalization_value}, precomputed={self.precomputed})"
        )

    def __str__(self) -> str:
        """Return a string representation of the game."""
        return self.__repr__()

    def exact_values(self, index: str, order: int) -> InteractionValues:
        """Uses the ExactComputer to compute the exact interaction values.

        Args:
            index: The index to compute the interaction values for. Choose from ``{"SII", "k-SII"}``.
            order: The maximum order of the interaction values.

        Returns:
            InteractionValues: The exact interaction values.

        """
        from shapiq.game_theory.exact import ExactComputer

        # raise warning if the game is not precomputed and n_players > 16
        if not self.precomputed and self.n_players > 16:
            warnings.warn(
                "The game is not precomputed and the number of players is greater than 16. "
                "Computing the exact interaction values via brute force may take a long time.",
                stacklevel=2,
            )

        exact_computer = ExactComputer(self.n_players, game=self)
        return exact_computer(index=index, order=order)

    @property
    def game_name(self) -> str:
        """Return the name of the game and the first class in the inheritance hierarchy."""
        return self.get_game_name()

    @classmethod
    def get_game_name(cls) -> str:
        """Return the name of the game and the first class in the inheritance hierarchy."""
        class_names = [cls.__name__]
        parent = cls.__base__
        while parent:
            parent_name = parent.__name__
            class_names.append(parent_name)
            if parent_name == Game.__name__:
                break
            parent = parent.__base__
        return "_".join(class_names)

    @property
    def empty_coalition_value(self) -> float:
        """Return the value of the empty coalition."""
        if self._empty_coalition_value_property is None:
            self._empty_coalition_value_property = float(self(self.empty_coalition))
        return self._empty_coalition_value_property

    @property
    def grand_coalition_value(self) -> float:
        """Return the value of the grand coalition."""
        if self._grand_coalition_value_property is None:
            self._grand_coalition_value_property = float(self(self.grand_coalition))
        return self._grand_coalition_value_property

    def __getitem__(self, item: tuple[int, ...]) -> float:
        """Return the value of the given coalition. Only retrieves precomputed/store values.

        Args:
            item: The coalition to evaluate.

        Returns:
            The value of the coalition

        Raises:
            KeyError: If the coalition is not stored in the game.

        """
        try:
            return float(self.value_storage[self.coalition_lookup[tuple(sorted(item))]])
        except (KeyError, IndexError) as error:
            msg = f"The coalition {item} is not stored in the game. Is it precomputed?"
            raise KeyError(msg) from error
