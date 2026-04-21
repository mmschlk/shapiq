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

# Shapes advertised in each loader's docstring. Regressions in preprocessing
# (extra rows dropped, columns added) break the user-visible contract.
EXPECTED_SHAPES: dict[str, tuple[int, int]] = {
    "california_housing": (20640, 8),
    "bike_sharing": (17379, 12),
    "adult_census": (45222, 14),
}

# The target column must be popped out of ``X`` by every loader.
TARGET_COLUMN_NAMES: dict[str, str] = {
    "california_housing": "MedHouseVal",
    "bike_sharing": "count",
    "adult_census": "class",
}


def _load_or_skip(loader, *, to_numpy: bool = False):
    try:
        return loader(to_numpy=to_numpy)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"dataset unavailable (likely offline): {exc}")


@pytest.mark.parametrize(("name", "loader"), DATASET_LOADERS, ids=[d[0] for d in DATASET_LOADERS])
class TestDatasetLoaders:
    """Every loader must return ``(X, y)`` with aligned lengths as pandas or numpy."""

    def test_returns_pandas_by_default(self, name, loader):
        X, y = _load_or_skip(loader)
        assert isinstance(X, pd.DataFrame)
        assert len(X) == len(y)
        assert len(X) > 0

    def test_to_numpy(self, name, loader):
        X, y = _load_or_skip(loader, to_numpy=True)
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == y.shape[0]

    def test_shape_matches_docstring(self, name, loader):
        """Guard the exact ``(rows, cols)`` shape each loader advertises.

        The docstring examples show ``X.shape`` for each dataset; if this
        fires, either the upstream CSV changed or the preprocessing pipeline
        drifted. Update both the docstring and this table together.
        """
        X, y = _load_or_skip(loader)
        assert X.shape == EXPECTED_SHAPES[name]
        assert y.shape == (EXPECTED_SHAPES[name][0],)

    def test_target_column_not_leaked_into_features(self, name, loader):
        """The ``y`` column must be removed from ``X`` by the loader."""
        X, _ = _load_or_skip(loader)
        assert TARGET_COLUMN_NAMES[name] not in X.columns

    def test_no_missing_values(self, name, loader):
        """Preprocessing must leave no NaNs in either ``X`` or ``y``."""
        X, y = _load_or_skip(loader)
        assert not X.isna().any().any()
        assert not pd.Series(y).isna().any()

    def test_numpy_and_pandas_agree(self, name, loader):
        """``to_numpy=True`` is equivalent to calling ``.to_numpy()`` on the pandas output.

        Catches divergence between the two return paths, e.g. if the numpy
        branch ever skipped a preprocessing step.
        """
        X_df, y_series = _load_or_skip(loader)
        X_np, y_np = _load_or_skip(loader, to_numpy=True)
        assert X_np.shape == X_df.shape
        assert y_np.shape == np.asarray(y_series).shape
        np.testing.assert_array_equal(X_np, X_df.to_numpy())
        np.testing.assert_array_equal(y_np, np.asarray(y_series))


def test_adult_census_labels_are_binary():
    """``y`` must be strictly ``{0, 1}`` for adult census.

    The loader re-encodes ``'>50K'`` → 1 and everything else → 0. A typo in
    the comparison string silently collapses every label to 0, which this
    test catches in one line.
    """
    _, y = _load_or_skip(load_adult_census)
    assert set(pd.unique(y)) == {0, 1}
