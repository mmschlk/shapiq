"""Fixed-size batches for the torch models of the image games."""

from __future__ import annotations

import numpy as np

__all__ = ["pad_batch"]


def pad_batch(rows: np.ndarray, batch_size: int) -> np.ndarray:
    """Pad a batch to ``batch_size`` rows by repeating its last row.

    A torch forward pass can pick different kernels for different batch sizes, which changes the
    outputs in the last digits. Padding every pass to one size makes the value of a coalition
    independent of how many coalitions are evaluated together.

    Args:
        rows: The batch, with at least one row.
        batch_size: The size of every forward pass.

    Returns:
        The batch with at least ``batch_size`` rows.
    """
    missing = batch_size - rows.shape[0]
    if missing <= 0:
        return rows
    return np.concatenate([rows, np.repeat(rows[-1:], missing, axis=0)])
