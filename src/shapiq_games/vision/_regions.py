"""The players of the image games: regions of an image, as the player of every pixel.

Two grids live here, on purpose with different splits: :func:`grid_regions` divides pixels
(``floor(i * rows / height)``, so uneven rows alternate), while :func:`token_players` divides a
transformer's token grid like the paper scripts (``np.array_split``, so the first rows get one
more token). Merging them would move the DINOv2 and CLIP regions.
"""

from __future__ import annotations

import numpy as np

__all__ = ["check_regions", "grid_regions", "pixel_regions", "token_players"]


def grid_regions(height: int, width: int, rows: int, cols: int) -> np.ndarray:
    """Split an image into a ``rows x cols`` grid of regions.

    Args:
        height: The image height in pixels.
        width: The image width in pixels.
        rows: The number of grid rows.
        cols: The number of grid columns.

    Returns:
        The region of every pixel, of shape ``(height, width)``, numbered row by row from ``0``.

    Examples:
        >>> grid_regions(4, 6, rows=2, cols=3)
        array([[0, 0, 1, 1, 2, 2],
               [0, 0, 1, 1, 2, 2],
               [3, 3, 4, 4, 5, 5],
               [3, 3, 4, 4, 5, 5]])
    """
    row = np.minimum(np.arange(height) * rows // height, rows - 1)
    col = np.minimum(np.arange(width) * cols // width, cols - 1)
    return row[:, None] * cols + col[None, :]


def check_regions(regions: np.ndarray, image: np.ndarray) -> np.ndarray:
    """Return own regions as integers after checking that they label every pixel of ``image``.

    Raises:
        ValueError: If the regions do not have the image's shape, or do not use every label
            ``0, ..., n - 1``.
    """
    regions = np.asarray(regions)
    if regions.shape != image.shape[:2]:
        msg = f"regions must have the image's shape {image.shape[:2]}, got {regions.shape}."
        raise ValueError(msg)
    labels = np.unique(regions)
    if not np.issubdtype(regions.dtype, np.integer) or not np.array_equal(
        labels, np.arange(labels.shape[0])
    ):
        msg = "regions must label every pixel with a player 0, ..., n - 1, using every label."
        raise ValueError(msg)
    return regions.astype(int)


def token_players(side: int, rows: int, cols: int) -> np.ndarray:
    """Group a ``side x side`` token grid into ``rows x cols`` near-equal rectangular players.

    Args:
        side: The number of tokens per side of the grid.
        rows: The number of player rows.
        cols: The number of player columns.

    Returns:
        The player of every token, of shape ``(side, side)``, numbered row by row. Uneven splits
        give the first rows and columns one token more (``np.array_split``).

    Examples:
        >>> token_players(5, 2, 2)
        array([[0, 0, 0, 1, 1],
               [0, 0, 0, 1, 1],
               [0, 0, 0, 1, 1],
               [2, 2, 2, 3, 3],
               [2, 2, 2, 3, 3]])
    """
    players = np.empty((side, side), dtype=int)
    row_groups = np.array_split(np.arange(side), rows)
    col_groups = np.array_split(np.arange(side), cols)
    for row, row_tokens in enumerate(row_groups):
        for col, col_tokens in enumerate(col_groups):
            players[np.ix_(row_tokens, col_tokens)] = row * cols + col
    return players


def pixel_regions(players: np.ndarray, patch_size: int) -> np.ndarray:
    """Spread the players of a token grid over the pixels, each token covering a square patch.

    Args:
        players: The player of every token, of shape ``(side, side)``.
        patch_size: The side length of a patch in pixels.

    Returns:
        The player of every pixel, of shape ``(side * patch_size, side * patch_size)``.
    """
    return np.repeat(np.repeat(players, patch_size, axis=0), patch_size, axis=1)
