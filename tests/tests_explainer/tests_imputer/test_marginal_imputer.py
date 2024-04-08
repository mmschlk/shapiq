"""This test module contains all tests for the marginal imputer module of the shapiq package."""

import numpy as np

from shapiq.games.imputer import MarginalImputer


def test_marginal_imputer_init():
    """Test the initialization of the marginal imputer."""

    def model(x: np.ndarray) -> np.ndarray:
        return np.asarray([0.0])

    # get np data set of 10 rows and 3 columns of random numbers
    data = np.random.rand(10, 3)

    imputer = MarginalImputer(
        model=model,
        data=data,
        sample_replacements=True,
        sample_size=10,
        random_state=42,
    )
    assert imputer._sample_replacements
    assert imputer._sample_size == 10
    assert imputer._random_state == 42
    assert imputer._n_features == 3

    # test with x
    x = np.random.rand(1, 3)
    imputer = MarginalImputer(
        model=model,
        data=data,
        x=x,
        random_state=42,
        sample_replacements=False,
    )
    assert np.array_equal(imputer._x, x)
    assert imputer._n_features == 3
    assert imputer._random_state == 42
    assert not imputer._sample_replacements

    # check with categorical features and a wrong numerical feature
    data = np.asarray([["a", "b", 1], ["c", "d", 2], ["e", "f", 3]])
    categorical_features = [0]  # only first specified
    imputer = MarginalImputer(
        model=model,
        data=data,
        categorical_features=categorical_features,
        random_state=42,
        sample_replacements=False,
    )
    assert imputer._cat_features == [0]


def test_sample_replacements():
    def model(x: np.ndarray) -> np.ndarray:
        return np.asarray([0.0])

    # get np data set of 10 rows and 3 columns of random numbers
    data = np.random.rand(10, 3)

    imputer = MarginalImputer(
        model=model,
        data=data,
        sample_replacements=True,
        sample_size=10,
        random_state=42,
    )

    imputed_values = imputer(np.array([[True, False, True], [False, True, False]]))
    assert len(imputed_values) == 2
