"""A benchmark is a game plus a computer of its exact values."""

from __future__ import annotations

import hashlib
import json
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING

from shapiq import InteractionValues
from shapiq_games.datasets import get_data_dir

from .computers import DEFAULT_MAX_PLAYERS, default_computer

if TYPE_CHECKING:
    from shapiq import Game

    from .computers import Computer

__all__ = ["Benchmark", "environment_key"]

_VERSIONED_PACKAGES = (
    "shapiq",
    "numpy",
    "scikit-learn",
    "xgboost",
    "lightgbm",
    "catboost",
    "tabpfn",
    "torch",
    "transformers",
)


def environment_key() -> str:
    """Return a short hash of the installed versions of shapiq and the model libraries.

    Exact values are computed by shapiq from models fitted by these libraries, so cached ground
    truth is only reused within the same versions.
    """
    versions = {}
    for package in _VERSIONED_PACKAGES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    digest = hashlib.sha256(json.dumps(versions, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()[:12]


class Benchmark:
    """A game together with the computer of its ground truth.

    Exact values of games configured with ``from_config`` (games with a ``fingerprint``) are cached
    locally in ``<data dir>/ground_truth/<fingerprint>/<environment>/`` (see
    :func:`shapiq_games.datasets.get_data_dir` and :func:`environment_key`), keyed by the
    computer, index, and order. Upgrading shapiq or a model library therefore never reuses
    ground truth computed with other versions.

    Examples:
        >>> from shapiq_games import PathDependentTreeGame
        >>> game = PathDependentTreeGame.from_config(dataset="xor", model="random_forest")
        >>> benchmark = Benchmark(game)  # uses the path-dependent tree computer
        >>> ground_truth = benchmark.exact_values(index="k-SII", order=2)
    """

    def __init__(
        self,
        game: Game,
        computer: Computer | None = None,
        *,
        cache: bool = True,
        max_players: int = DEFAULT_MAX_PLAYERS,
    ) -> None:
        """Create the benchmark.

        Args:
            game: The game.
            computer: The ground-truth computer. Defaults to :func:`default_computer` of the game.
            cache: Whether to cache exact values of fingerprinted games. Defaults to ``True``.
            max_players: The player cap of brute force when no computer is given.
        """
        self.game = game
        self.computer = (
            computer if computer is not None else default_computer(game, max_players=max_players)
        )
        if self.computer.game is not game:
            msg = "The computer must be bound to the benchmark's game."
            raise ValueError(msg)
        self.cache = cache

    @property
    def fingerprint(self) -> str | None:
        """The fingerprint of the game, or ``None`` if it was not built from a configuration."""
        return getattr(self.game, "fingerprint", None)

    def _cache_path(self, index: str, order: int) -> Path | None:
        if not self.cache or self.fingerprint is None:
            return None
        return (
            get_data_dir()
            / "ground_truth"
            / self.fingerprint
            / environment_key()
            / f"{self.computer.name}_{index}_{order}.json"
        )

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
        path = self._cache_path(index, order)
        if path is not None and path.exists():
            return InteractionValues.from_json_file(path)
        values = self.computer.exact_values(index, order)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=path.parent) as tmp:
                tmp_path = Path(tmp) / path.name
                values.to_json_file(tmp_path)
                tmp_path.replace(path)
        return values

    def __repr__(self) -> str:
        """Return a short description of the benchmark."""
        return (
            f"Benchmark(game={type(self.game).__name__}(n_players={self.game.n_players}), "
            f"computer={self.computer.name}, fingerprint={self.fingerprint})"
        )
