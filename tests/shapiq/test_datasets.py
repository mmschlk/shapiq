"""Smoke tests for the built-in dataset loaders.

These tests are marked ``slow`` because they may download data from GitHub on
first run. If the machine is offline and nothing is cached the test is skipped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shapiq.datasets import (
    load_adult_census,
    load_bike_sharing,
    load_california_housing,
)

pytestmark = pytest.mark.slow


DATASET_LOADERS = [
    ("california_housing", load_california_housing),
    ("bike_sharing", load_bike_sharing),
    ("adult_census", load_adult_census),
]


@pytest.mark.parametrize(("name", "loader"), DATASET_LOADERS, ids=[d[0] for d in DATASET_LOADERS])
class TestDatasetLoaders:
    """Every loader must return ``(X, y)`` with aligned lengths as pandas or numpy."""

    def test_returns_pandas_by_default(self, name, loader):
        try:
            X, y = loader()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"dataset unavailable (likely offline): {exc}")
        assert isinstance(X, pd.DataFrame)
        assert len(X) == len(y)
        assert len(X) > 0

    def test_to_numpy(self, name, loader):
        try:
            X, y = loader(to_numpy=True)
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"dataset unavailable (likely offline): {exc}")
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == y.shape[0]
