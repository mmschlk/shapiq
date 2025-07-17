"""Collects all deprecated behaviour tests."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

from .features import register_deprecated


@register_deprecated(name="Game(path_to_values=...)", deprecated_in="1.3.1", removed_in="1.4.0")
def deprecated_game_init_with_path(request: pytest.FixtureRequest) -> None:
    from shapiq.games.base import Game

    tmp_path = request.getfixturevalue("tmp_path")
    game = request.getfixturevalue("cooking_game_pre_computed")
    path = tmp_path / "dummy_game.json"
    game.save(path)
    Game(path_to_values=path)


@register_deprecated(
    name="InteractionValues.save(..., as_pickle=True)", deprecated_in="1.3.1", removed_in="1.4.0"
)
def save_interaction_values_as_pickle(request: pytest.FixtureRequest) -> None:
    """Tests that old methods work but also warn with deprecation."""
    import pathlib

    tmp_path = request.getfixturevalue("tmp_path")
    iv = request.getfixturevalue("iv_7_all")

    path = tmp_path / pathlib.Path("test_interaction_values")
    iv.save(path, as_pickle=True)


@register_deprecated(
    name="InteractionValues.save(..., as_npz=True)", deprecated_in="1.3.1", removed_in="1.4.0"
)
def save_interaction_values_as_npz(request: pytest.FixtureRequest) -> None:
    """Tests that old methods work but also warn with deprecation."""
    import pathlib

    tmp_path = request.getfixturevalue("tmp_path")
    iv = request.getfixturevalue("iv_7_all")

    path = tmp_path / pathlib.Path("test_interaction_values")
    iv.save(path, as_npz=True)


@register_deprecated(
    name="InteractionValues.load() from non-json file", deprecated_in="1.3.1", removed_in="1.4.0"
)
def load_interaction_values_from_non_json(request: pytest.FixtureRequest) -> None:
    """Tests that old methods work but also warn with deprecation."""
    import pathlib

    from shapiq.interaction_values import InteractionValues

    tmp_path = request.getfixturevalue("tmp_path")
    iv: InteractionValues = request.getfixturevalue("iv_7_all")
    path = tmp_path / pathlib.Path("test_interaction_values.npz")

    # supress the warning for saving as .npz
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        iv.save(path, as_npz=True)

    InteractionValues.load(path)
