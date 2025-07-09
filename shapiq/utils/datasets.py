"""This module contains utility functions for datasets and data."""

from __future__ import annotations

import numpy as np


def shuffle_data(
    x_data: np.ndarray,
    y_data: np.ndarray,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Shuffle the data.

    Args:
        x_data: The data features.
        y_data: The data labels.
        random_state: The random state to use for shuffling. Defaults to `None`.

    Returns:
        The shuffled data.

    """
    rng = np.random.default_rng(random_state)
    indices = np.arange(len(x_data))
    rng.shuffle(indices)
    return x_data[indices], y_data[indices]
