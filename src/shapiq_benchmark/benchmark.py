"""A benchmark is a game plus a computer of its exact values."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from shapiq import InteractionValues
from shapiq_benchmark.datasets import get_data_dir

from .computers import DEFAULT_MAX_PLAYERS, default_computer

if TYPE_CHECKING:
    from shapiq import Game

    from .computers import Computer
    from .setups import Setup

__all__ = ["Benchmark"]


class Benchmark:
    """A game together with the computer of its ground truth.

    ``Benchmark(game)`` works for any game, your own included, and computes its exact values on
    demand. :meth:`from_setup` builds the game of a :class:`~shapiq_benchmark.setups.Setup` and
    caches its exact values locally in ``<data dir>/ground_truth/<setup name>/<setup key>/`` (see
    :func:`shapiq_benchmark.datasets.get_data_dir`), keyed by the computer, index, and order. The
    setup identifies the game; a new model belongs under a new name. Delete the directory, or
    pass ``cache=False``, to recompute.

    Examples:
        >>> from shapiq_benchmark.setups import PathDependentTreeSetup
        >>> setup = PathDependentTreeSetup(dataset="xor", model="random_forest")
        >>> benchmark = Benchmark.from_setup(setup)  # uses the path-dependent tree computer
        >>> ground_truth = benchmark.exact_values(index="k-SII", order=2)
    """

    def __init__(
        self,
        game: Game,
        computer: Computer | None = None,
        *,
        max_players: int = DEFAULT_MAX_PLAYERS,
    ) -> None:
        """Create a benchmark of a game. Its exact values are not cached.

        Args:
            game: The game.
            computer: The ground-truth computer, bound to ``game``. Defaults to
                :func:`default_computer` of the game.
            max_players: The player cap of brute force when no computer is given.

        Raises:
            ValueError: If the computer is bound to another game.
        """
        self.game = game
        self.computer = (
            computer if computer is not None else default_computer(game, max_players=max_players)
        )
        if self.computer.game is not game:
            msg = "The computer must be bound to the benchmark's game."
            raise ValueError(msg)
        self.setup: Setup | None = None
        self.cache = False

    @classmethod
    def from_setup(
        cls,
        setup: Setup,
        computer: type[Computer] | None = None,
        *,
        cache: bool = True,
        max_players: int = DEFAULT_MAX_PLAYERS,
    ) -> Benchmark:
        """Build the game of a setup and benchmark it, caching its exact values.

        Args:
            setup: The setup.
            computer: The class of the ground-truth computer. Defaults to
                :func:`default_computer` of the game.
            cache: Whether to cache the exact values under the setup's key. Defaults to ``True``.
            max_players: The player cap of brute force when no computer is given.

        Returns:
            The benchmark.
        """
        game = setup.build()
        benchmark = cls(
            game,
            computer(game) if computer is not None else None,
            max_players=max_players,
        )
        benchmark.setup = setup
        benchmark.cache = cache
        return benchmark

    @property
    def key(self) -> str | None:
        """The key of the setup, or ``None`` for a benchmark of a game without setup."""
        return self.setup.key if self.setup is not None else None

    def _cache_dir(self) -> Path | None:
        if not self.cache or self.setup is None:
            return None
        return get_data_dir() / "ground_truth" / self.setup.name / self.setup.key

    def supports(self, index: str, order: int) -> bool:
        """Return whether the computer supports ``index`` up to ``order``."""
        return self.computer.supports(index, order)

    def exact_values(self, index: str, order: int) -> InteractionValues:
        """Return the exact interaction values of order 1 to ``order`` (cached if possible).

        Args:
            index: The interaction index.
            order: The highest interaction order.

        Returns:
            The exact values (see :meth:`Computer.exact_values`).
        """
        directory = self._cache_dir()
        path = (
            None if directory is None else directory / f"{self.computer.name}_{index}_{order}.json"
        )
        if path is not None and path.exists():
            return InteractionValues.from_json_file(path)
        values = self.computer.exact_values(index, order)
        if directory is not None and path is not None and self.setup is not None:
            directory.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=directory) as tmp:
                tmp_path = Path(tmp) / path.name
                values.to_json_file(tmp_path)
                tmp_path.replace(path)
                setup_path = Path(tmp) / "setup.json"  # what the key stands for, for humans
                setup_path.write_text(json.dumps(self.setup.to_dict(), indent=2, sort_keys=True))
                setup_path.replace(directory / "setup.json")
        return values

    def __repr__(self) -> str:
        """Return a short description of the benchmark."""
        setup = f", setup={self.setup.name}, key={self.key}" if self.setup is not None else ""
        return (
            f"Benchmark(game={type(self.game).__name__}(n_players={self.game.n_players}), "
            f"computer={self.computer.name}{setup})"
        )
