"""Removing players from a vision transformer by dropping their patch tokens.

A vision transformer reads an image as a sequence of patch tokens (plus a class token). Dropping
the tokens of absent players from the sequence, while the present tokens keep their position
embeddings, removes those regions without inventing replacement pixels. The players are
rectangular blocks of the token grid.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ._batching import pad_batch

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["TokenDropper", "pixel_regions", "token_players"]


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


class TokenDropper:
    """Encode the class token and the patch tokens of each coalition's players.

    Coalitions with the same number of kept tokens are encoded together. Every forward pass is
    padded to ``batch_size`` rows, so a coalition's output does not depend on the coalitions
    evaluated with it (torch picks its kernels by the shape of the batch).

    Args:
        torch: The ``torch`` module.
        cls_token: The class token, of shape ``(1, 1, dim)``.
        patch_tokens: The patch tokens with their position embeddings, of shape
            ``(n_tokens, dim)``.
        token_player: The player of every patch token, of shape ``(n_tokens,)``.
        encode: Maps a batch of token sequences ``(batch, 1 + kept, dim)`` to outputs
            ``(batch, out_dim)``.
        batch_size: The number of sequences per forward pass.
    """

    def __init__(
        self,
        torch: Any,  # noqa: ANN401
        cls_token: Any,  # noqa: ANN401
        patch_tokens: Any,  # noqa: ANN401
        token_player: np.ndarray,
        encode: Callable[[Any], Any],
        batch_size: int,
    ) -> None:
        """Store the tokens and the encoder."""
        self._torch = torch
        self._cls_token = cls_token
        self._patch_tokens = patch_tokens
        self.token_player = np.asarray(token_player).reshape(-1)
        self._encode = encode
        self.batch_size = batch_size

    def __call__(self, coalitions: np.ndarray) -> np.ndarray:
        """Return the encoder output of every coalition, of shape ``(n_coalitions, out_dim)``."""
        torch = self._torch
        kept = np.asarray(coalitions, dtype=bool)[:, self.token_player]
        counts = kept.sum(axis=1)
        outputs: np.ndarray | None = None
        for count in np.unique(counts):
            rows = np.flatnonzero(counts == count)
            # the kept token indices of each coalition, in token order
            indices = np.nonzero(kept[rows])[1].reshape(rows.shape[0], int(count))
            for start in range(0, rows.shape[0], self.batch_size):
                chunk = indices[start : start + self.batch_size]
                index = torch.as_tensor(pad_batch(chunk, self.batch_size))
                tokens = self._patch_tokens[index.to(self._patch_tokens.device)]
                cls = self._cls_token.expand(index.shape[0], -1, -1)
                with torch.no_grad():
                    encoded = self._encode(torch.cat((cls, tokens), dim=1))
                result = encoded[: chunk.shape[0]].float().cpu().numpy().astype(float)
                if outputs is None:
                    outputs = np.empty((kept.shape[0], result.shape[1]))
                outputs[rows[start : start + chunk.shape[0]]] = result
        if outputs is None:
            msg = "Expected at least one coalition."
            raise ValueError(msg)
        return outputs
