"""Removing players from a vision transformer in token space.

A vision transformer reads an image as a sequence of patch tokens, after a class token in most
models (SigLIP has none: it pools all tokens by attention). The tokens of absent players are either dropped from the sequence (``"remove"``), while the present tokens
keep their position embeddings, or masked (``"mask"``): their content is replaced by the model's
mask token, and they keep their position. Neither invents replacement pixels. The players are
rectangular blocks of the token grid.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ._batching import pad_batch

if TYPE_CHECKING:
    from collections.abc import Callable

    from shapiq.typing import CoalitionMatrix
    from shapiq_games.typing import MaskStrategy

__all__ = ["TokenDropper", "TokenMasker", "pixel_regions", "token_players", "token_remover"]


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
    """Encode the prefix tokens and the patch tokens of each coalition's players.

    Coalitions with the same number of kept tokens are encoded together. Every forward pass is
    padded to ``batch_size`` rows, so a coalition's output does not depend on the coalitions
    evaluated with it (torch picks its kernels by the shape of the batch).

    Args:
        torch: The ``torch`` module.
        prefix_tokens: The tokens before the patch tokens (the class token), of shape
            ``(1, n_prefix, dim)``; ``n_prefix`` may be zero.
        patch_tokens: The patch tokens with their position embeddings, of shape
            ``(n_tokens, dim)``.
        token_player: The player of every patch token, of shape ``(n_tokens,)``.
        encode: Maps a batch of token sequences ``(batch, n_prefix + kept, dim)`` to outputs
            ``(batch, out_dim)``.
        batch_size: The number of sequences per forward pass.
    """

    def __init__(
        self,
        torch: Any,  # noqa: ANN401
        prefix_tokens: Any,  # noqa: ANN401
        patch_tokens: Any,  # noqa: ANN401
        token_player: np.ndarray,
        encode: Callable[[Any], Any],
        batch_size: int,
    ) -> None:
        """Store the tokens and the encoder."""
        self._torch = torch
        self._prefix_tokens = prefix_tokens
        self._patch_tokens = patch_tokens
        self.token_player = np.asarray(token_player).reshape(-1)
        self._encode = encode
        self.batch_size = batch_size

    def __call__(self, coalitions: CoalitionMatrix) -> np.ndarray:
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
                prefix = self._prefix_tokens.expand(index.shape[0], -1, -1)
                with torch.no_grad():
                    encoded = self._encode(torch.cat((prefix, tokens), dim=1))
                result = encoded[: chunk.shape[0]].float().cpu().numpy().astype(float)
                if outputs is None:
                    outputs = np.empty((kept.shape[0], result.shape[1]))
                outputs[rows[start : start + chunk.shape[0]]] = result
        if outputs is None:
            msg = "Expected at least one coalition."
            raise ValueError(msg)
        return outputs


class TokenMasker:
    """Encode each coalition's token sequence with the absent players' patch tokens masked.

    A masked token is the model's mask token (or zeros) plus the token's position embedding, so
    every sequence keeps all tokens and their positions. Every forward pass is padded to
    ``batch_size`` rows, so a coalition's output does not depend on the coalitions evaluated
    with it.

    Args:
        torch: The ``torch`` module.
        prefix_tokens: The tokens before the patch tokens (the class token), of shape
            ``(1, n_prefix, dim)``; ``n_prefix`` may be zero.
        patch_tokens: The patch tokens with their position embeddings, of shape
            ``(n_tokens, dim)``.
        masked_tokens: The masked patch tokens (mask token plus position embedding), of shape
            ``(n_tokens, dim)``.
        token_player: The player of every patch token, of shape ``(n_tokens,)``.
        encode: Maps a batch of token sequences ``(batch, n_prefix + n_tokens, dim)`` to outputs
            ``(batch, out_dim)``.
        batch_size: The number of sequences per forward pass.
    """

    def __init__(
        self,
        torch: Any,  # noqa: ANN401
        prefix_tokens: Any,  # noqa: ANN401
        patch_tokens: Any,  # noqa: ANN401
        masked_tokens: Any,  # noqa: ANN401
        token_player: np.ndarray,
        encode: Callable[[Any], Any],
        batch_size: int,
    ) -> None:
        """Store the tokens and the encoder."""
        self._torch = torch
        self._prefix_tokens = prefix_tokens
        self._patch_tokens = patch_tokens
        self._masked_tokens = masked_tokens
        self.token_player = np.asarray(token_player).reshape(-1)
        self._encode = encode
        self.batch_size = batch_size

    def __call__(self, coalitions: CoalitionMatrix) -> np.ndarray:
        """Return the encoder output of every coalition, of shape ``(n_coalitions, out_dim)``."""
        torch = self._torch
        kept = np.asarray(coalitions, dtype=bool)[:, self.token_player]
        outputs = []
        for start in range(0, kept.shape[0], self.batch_size):
            chunk = kept[start : start + self.batch_size]
            mask = torch.as_tensor(pad_batch(chunk, self.batch_size))
            mask = mask.to(self._patch_tokens.device)[..., None]
            tokens = torch.where(mask, self._patch_tokens, self._masked_tokens)
            prefix = self._prefix_tokens.expand(mask.shape[0], -1, -1)
            with torch.no_grad():
                encoded = self._encode(torch.cat((prefix, tokens), dim=1))
            outputs.append(encoded[: chunk.shape[0]].float().cpu().numpy().astype(float))
        if not outputs:
            msg = "Expected at least one coalition."
            raise ValueError(msg)
        return np.concatenate(outputs)


def token_remover(
    mask_strategy: MaskStrategy,
    torch: Any,  # noqa: ANN401
    embeddings: Any,  # noqa: ANN401
    masked_embeddings: Any,  # noqa: ANN401
    token_player: np.ndarray,
    encode: Callable[[Any], Any],
    batch_size: int,
    *,
    n_prefix: int = 1,
) -> TokenDropper | TokenMasker:
    """Return the token remover of a strategy for an embedded image.

    Args:
        mask_strategy: ``"mask"`` (mask the absent tokens) or ``"remove"`` (drop them).
        torch: The ``torch`` module.
        embeddings: The prefix tokens and the patch tokens with their position embeddings, of
            shape ``(1, n_prefix + n_tokens, dim)``.
        masked_embeddings: The same with every patch token masked, of the same shape (only used
            for ``"mask"``).
        token_player: The player of every patch token, of shape ``(n_tokens,)``.
        encode: Maps a batch of token sequences to outputs ``(batch, out_dim)``.
        batch_size: The number of sequences per forward pass.
        n_prefix: The number of tokens before the patch tokens, which are never removed.
            Defaults to ``1`` (the class token); SigLIP has ``0``, so its empty coalition is an
            empty sequence.

    Returns:
        A callable mapping a coalition matrix to the encoder outputs.

    Raises:
        ValueError: If ``mask_strategy`` is unknown.
    """
    prefix, patch_tokens = embeddings[:, :n_prefix], embeddings[0, n_prefix:]
    if mask_strategy == "remove":
        return TokenDropper(torch, prefix, patch_tokens, token_player, encode, batch_size)
    if mask_strategy == "mask":
        masked = masked_embeddings[0, n_prefix:]
        return TokenMasker(torch, prefix, patch_tokens, masked, token_player, encode, batch_size)
    msg = f"mask_strategy must be 'mask' or 'remove', got {mask_strategy!r}."
    raise ValueError(msg)
