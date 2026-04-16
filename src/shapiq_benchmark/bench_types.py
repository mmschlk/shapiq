"""Types for benchmark data loading."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

NumericArray = NDArray[np.number]


@dataclass(frozen=True)
class BenchmarkDataset:
    """Container for benchmark dataset splits and metadata."""

    x_train: NumericArray
    y_train: NumericArray
    x_test: NumericArray
    y_test: NumericArray
    data_type: str
    x_explain: NumericArray
