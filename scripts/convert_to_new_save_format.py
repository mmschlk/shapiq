"""This script converts old save files to the new JSON save format.

It finds all `.npz` and `.pkl` files in the given directory and converts them to `.json`,
either as Game or InteractionValues objects, depending on the command-line arguments.
"""

import argparse
import pathlib
import sys

import numpy as np

from shapiq import Game, InteractionValues


def str2bool(v: str) -> bool:
    """Convert a string to a boolean value."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    msg = "Boolean value expected, got: " + str(v)
    raise argparse.ArgumentTypeError(msg)


def get_all_files(directory: pathlib.Path) -> list[pathlib.Path]:
    """Recursively get all .npz and .pkl files in the directory."""
    return [f for f in directory.rglob("*") if f.is_file() and f.suffix in {".npz", ".pkl"}]


def _check_game_equality(game1: Game, game2: Game) -> bool:
    """Check if two games are equal."""
    return (
        game1.n_players == game2.n_players
        and game1.normalization_value == game2.normalization_value
        and game1.coalition_lookup == game2.coalition_lookup
        and np.allclose(game1.value_storage, game2.value_storage)
        and game1.grand_coalition_value == game2.grand_coalition_value
        and game1.empty_coalition_value == game2.empty_coalition_value
        and game1.n_values_stored == game2.n_values_stored
    )


def convert_game(path: pathlib.Path, *, delete: bool) -> bool:
    """Try converting a Game file. Return True if successful."""
    try:
        game = Game(path_to_values=path)
        new_path = path.with_suffix(".json")
        game.save_values(new_path)
        reloaded = Game.load(new_path)
        if not _check_game_equality(game, reloaded):
            new_path.unlink()
            msg = "Equality check failed after conversion."
            raise ValueError(msg)
        if delete:
            path.unlink()
        print(f"[✓] Game converted: {path} -> {new_path}")
        return True
    except Exception:
        return False


def convert_iv(path: pathlib.Path, *, delete: bool) -> bool:
    """Try converting an InteractionValues file. Return True if successful."""
    try:
        iv = InteractionValues.load(path)
        new_path = path.with_suffix(".json")
        iv.save(new_path)
        reloaded = InteractionValues.load(new_path)
        if reloaded != iv:
            new_path.unlink()
            msg = "Equality check failed after conversion."
            raise ValueError(msg)
        if delete:
            path.unlink()
        print(f"[✓] InteractionValues converted: {path} -> {new_path}")
        return True
    except Exception:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert old save files to new JSON format.")
    parser.add_argument(
        "--directory", type=str, default=".", help="Directory to search (default: current)."
    )
    parser.add_argument(
        "--delete_old_files",
        type=str2bool,
        default=False,
        help="Delete original files (default: False).",
    )
    parser.add_argument(
        "--convert_iv",
        type=str2bool,
        default=True,
        help="Convert InteractionValues (default: True).",
    )
    parser.add_argument(
        "--convert_game", type=str2bool, default=True, help="Convert Game files (default: True)."
    )
    args = parser.parse_args()

    directory = pathlib.Path(args.directory)
    if not directory.is_dir():
        msg = f"The path '{directory}' is not a valid directory."
        raise ValueError(msg)

    print(f"Searching in: {directory}")
    print(
        f"Convert IV: {args.convert_iv} | Convert Game: {args.convert_game} | Delete old: {args.delete_old_files}"
    )

    files = get_all_files(directory)
    if not files:
        print("No matching files found.")
        sys.exit(0)

    for file in files:
        converted = False

        if args.convert_iv:
            converted = convert_iv(file, delete=args.delete_old_files)

        if not converted and args.convert_game:
            converted = convert_game(file, delete=args.delete_old_files)

        if not converted:
            try:
                Game(path_to_values=file)
            except Exception:
                try:
                    InteractionValues.load(file)
                except Exception:
                    print(f"[ ] Skipped (not convertible): {file}")
                    continue
