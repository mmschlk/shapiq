"""Pre-compute the values for the games and store them in a file."""

import multiprocessing as mp
import os
from typing import Optional, Union

from tqdm.auto import tqdm

from ..base import Game

__all__ = [
    "pre_compute_and_store",
    "pre_compute_and_store_from_list",
    "SHAPIQ_DATA_DIR",
    "get_game_files",
]


SHAPIQ_DATA_DIR = os.path.join(os.path.dirname(__file__), "precomputed")


def get_game_files(game: Union[Game, Game.__class__, str], n_players: int) -> list[str]:
    """Get the files for the given game and number of players.

    Args:
        game: The game to get the files for or the game name (as provided by `game.game_name` or
            `GameClass.get_game_name()`).
        n_players: The number of players in the game. If not provided, all files are returned.

    Returns:
        The list of files for the given game and number of players.
    """
    game_name = game
    if not isinstance(game, str):
        game_name = game.get_game_name()
    save_dir = os.path.join(SHAPIQ_DATA_DIR, game_name, str(n_players))
    try:
        return os.listdir(save_dir)
    except FileNotFoundError:
        return []


def pre_compute_and_store(
    game: Game, save_dir: Optional[str] = None, game_id: Optional[str] = None
) -> str:
    """Pre-compute the values for the given game and store them in a file.

    Args:
        game: The game to pre-compute the values for.
        save_dir: The path to the directory where the values are stored. If not provided, the
            directory is determined at random.
        game_id: A identifier of the game. If not provided, the ID is determined at random.

    Returns:
        The path to the file where the values are stored.
    """

    if save_dir is None:
        # this file path
        save_dir = os.path.dirname(__file__)
        save_dir = os.path.join(save_dir, "precomputed", game.game_name, str(game.n_players))
    else:  # check if n_players is in the path
        if str(game.n_players) not in save_dir:
            save_dir = os.path.join(save_dir, str(game.n_players))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if game_id is None:
        game_id = str(hash(game))[:8]

    save_path = os.path.join(save_dir, game_id)

    game.precompute()
    game.save_values(path=save_path)
    return save_path


def pre_compute_and_store_from_list(
    games: list[Game],
    save_dir: Optional[str] = None,
    game_ids: Optional[list[str]] = None,
    n_jobs: int = 1,
) -> list[str]:
    """Pre-compute the values for the games stored in the given file.

    Args:
        games: The games to pre-compute the values for.
        save_dir: The path to the directory where the values are stored. If not provided, the
            directory is determined at random.
        game_ids: The IDs of the games. If not provided, the IDs are determined at random.
        n_jobs: The number of parallel jobs to run.

    Returns:
        The paths to the files where the values are stored.
    """

    if game_ids is None:
        game_ids = [None] * len(games)

    if n_jobs == 1:
        return [
            pre_compute_and_store(game, save_dir, game_id) for game, game_id in zip(games, game_ids)
        ]

    with mp.Pool(n_jobs) as pool:
        results = list(
            tqdm(
                pool.starmap(
                    pre_compute_and_store,
                    [(game, save_dir, game_id) for game, game_id in zip(games, game_ids)],
                ),
                total=len(games),
            )
        )

    return results
