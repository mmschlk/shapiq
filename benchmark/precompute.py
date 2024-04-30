import multiprocessing as mp
import os
from typing import Optional

from tqdm.asyncio import tqdm

from shapiq.games import Game


def pre_compute_and_store(
    game: Game, save_dir: Optional[str] = None, game_id: Optional[str] = None
) -> str:
    """Pre-compute the values for the given game and store them in a file.

    Args:
        game: The game to pre-compute the values for.
        save_dir: The path to the directory where the values are stored. If not provided, the
            directory is determined at random.
        game_id: The ID of the game. If not provided, the ID is determined at random.

    Returns:
        The path to the file where the values are stored.
    """

    if save_dir is None:
        save_dir = os.path.join(os.getcwd(), game.game_name, str(game.n_players))
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
