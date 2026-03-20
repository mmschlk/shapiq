"""Types used in tests and benchmarks."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

NumericArray = NDArray[np.number]


class TabularDataSet(NamedTuple):
    """A named tuple to hold the dataset and its target values."""

    x_train: NumericArray
    y_train: NumericArray
    x_test: NumericArray
    y_test: NumericArray
