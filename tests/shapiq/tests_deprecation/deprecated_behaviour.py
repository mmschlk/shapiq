"""Collects all deprecated behavior tests.

Test functions in this module are registered as deprecated features using the `register_deprecated`
decorator from `features.py`. Each test function should be decorated with `register_deprecated`,
which takes the name of the deprecated feature, the version in which it was deprecated, and the
version in which it will be removed as arguments.

Example:
    >>> from .features import register_deprecated

    >>> @register_deprecated(name="Game(path_to_values=...)", deprecated_in="1.3.1", removed_in="1.4.0")
    >>> def deprecated_game_init_with_path(request: pytest.FixtureRequest) -> None:
    >>> from shapiq.game import Game
    >>>
    >>> tmp_path = request.getfixturevalue("tmp_path")
    >>> game = request.getfixturevalue("cooking_game_pre_computed")
    >>> path = tmp_path / "dummy_game.json"
    >>> game.save(path)
    >>> Game(path_to_values=path)

"""

from __future__ import annotations
