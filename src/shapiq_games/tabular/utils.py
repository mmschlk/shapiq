"""Utility functions for shapiq-games."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def get_x_explain(
    x: NDArray | int | None,
    data: NDArray,
    random_state: int | None = None,
) -> NDArray:
    """Returns a single data point to explain given the input.

    Args:
        x: The data point to explain. Can be an index of the background data or a 1d matrix of shape
            (n_features).
        data: The data set to select the data point from. Should be a 2d matrix of shape
            (n_samples, n_features).
        random_state: An optional random state for reproducibility. If `None`, a random state will
            be chosen internally.

    Returns:
        The data point to explain as a numpy array.

    """
    if x is None:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(data.shape[0])
        return data[idx]
    if isinstance(x, int):
        return data[x]
    return x
