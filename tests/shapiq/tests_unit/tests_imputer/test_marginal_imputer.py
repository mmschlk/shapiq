"""This test module contains all tests for the marginal imputer module of the shapiq package."""

from __future__ import annotations

import itertools
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pytest
from sklearn.tree import DecisionTreeRegressor

from shapiq import ExactComputer, TabularExplainer
from shapiq.imputer import MarginalImputer

if TYPE_CHECKING:
    from collections.abc import Callable


def test_marginal_imputer_init():
    """Test the initialization of the marginal imputer."""

    def model(x: np.ndarray) -> np.ndarray:
        return np.sum(x, axis=1)

    # get np data set of 10 rows and 3 columns of random numbers
    data = np.random.rand(10, 3)

    imputer = MarginalImputer(
        model=model,
        data=data,
        sample_size=10,
        random_state=42,
    )
    assert imputer.sample_size == 10
    assert imputer.random_state == 42
    assert imputer.n_features == 3

    # test with x and normalize
    x = np.random.rand(1, 3)
    imputer = MarginalImputer(
        model=model,
        data=data,
        x=x,
        random_state=42,
    )
    assert np.array_equal(imputer.x, x)
    assert imputer.n_features == 3
    assert imputer.random_state == 42

    # check with categorical features and a wrong numerical feature

    def model_cat(x: np.ndarray) -> np.ndarray:
        return np.zeros(x.shape[0])

    data = np.asarray([["a", "b", 1], ["c", "d", 2], ["e", "f", 3]])
    categorical_features = [0]  # only first specified
    imputer = MarginalImputer(
        model=model_cat,
        data=data,
        categorical_features=categorical_features,
        random_state=42,
    )
    assert imputer._cat_features == [0]


def test_marginal_imputer_value_function():
    """Test the value function of the marginal imputer."""

    def model(x: np.ndarray) -> np.ndarray:
        return np.sum(x, axis=1)

    # get np data set of 10 rows and 3 columns of random numbers
    data = np.random.rand(10, 3)

    imputer = MarginalImputer(
        model=model,
        data=data,
        x=np.ones((1, 3)),
        sample_size=8,
        random_state=42,
    )

    imputed_values = imputer(np.array([[True, False, True], [False, True, False]]))
    assert len(imputed_values) == 2

    # test with normalization
    imputer = MarginalImputer(
        model=model,
        data=data,
        x=np.ones((1, 3)),
        sample_size=8,
        random_state=42,
        normalize=True,
    )
    imputed_out = imputer(np.array([[False, False, False], [False, True, False]]))
    assert imputed_out[0] == 0.0
    assert imputed_out[1] != 0.0


def test_joint_marginal_distribution():
    """Test weather the marginal imputer correctly samples replacement values."""

    def model(x: np.ndarray) -> np.ndarray:
        return np.sum(x, axis=1)

    data = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
    data_as_tuples = [tuple(row) for row in data]
    data = np.array(data)
    x = np.array([1, 1, 1])

    imputer = MarginalImputer(
        model=model,
        data=data,
        x=x,
        sample_size=3,
        random_state=42,
        joint_marginal_distribution=False,
    )
    replacement_data_independent = imputer._sample_replacement_data(3)

    imputer = MarginalImputer(
        model=model,
        data=data,
        x=x,
        sample_size=3,
        random_state=42,
        joint_marginal_distribution=True,
    )
    replacement_data_joint = imputer._sample_replacement_data(3)
    for i in range(3):
        assert tuple(replacement_data_joint[i]) in data_as_tuples
        # the below only works because of the random seed (might break in future)
        assert tuple(replacement_data_independent[i]) not in data_as_tuples


def _sum_model(x: np.ndarray) -> np.ndarray:
    return np.sum(x, axis=1)


ALL_COALITIONS = np.array(list(itertools.product([False, True], repeat=3)))


def test_sample_size_is_an_upper_bound():
    """Test that a background smaller than ``sample_size`` is used completely without a warning."""
    data = np.random.default_rng(0).normal(size=(10, 3))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        imputer = MarginalImputer(model=_sum_model, data=data, sample_size=100, random_state=42)
    assert imputer.sample_size == 10
    assert np.array_equal(imputer._sampled_replacement_data, data)


@pytest.mark.parametrize("joint_marginal_distribution", [True, False])
def test_sample_size_none_uses_all_rows(*, joint_marginal_distribution: bool):
    """Test that ``sample_size=None`` uses the complete background data."""
    data, model = _null_player_setup()
    kwargs = {
        "model": model,
        "data": data,
        "x": data[0],
        "random_state": 0,
        "normalize": False,
        "joint_marginal_distribution": joint_marginal_distribution,
    }
    imputer_all = MarginalImputer(sample_size=None, **kwargs)
    imputer_n = MarginalImputer(sample_size=len(data), **kwargs)
    assert imputer_all.sample_size == len(data)
    assert np.array_equal(imputer_all(ALL_COALITIONS), imputer_n(ALL_COALITIONS))
    if joint_marginal_distribution:  # the baseline is the mean prediction over all rows
        assert imputer_all.empty_prediction == np.mean(model(data))


def test_init_background_recomputes_sample_size():
    """Test that a smaller earlier background does not cap the rows used of a later one."""
    data = np.random.default_rng(0).normal(size=(500, 3))
    imputer = MarginalImputer(model=_sum_model, data=data[:50], random_state=0)
    assert imputer.sample_size == 50
    imputer.init_background(data)
    assert imputer.sample_size == 100  # the default upper bound
    assert imputer._sampled_replacement_data.shape == (100, 3)

    imputer = MarginalImputer(model=_sum_model, data=data[:50], sample_size=None, random_state=0)
    imputer.init_background(data)
    assert imputer.sample_size == 500


@pytest.mark.parametrize("sample_size", [0, -1])
def test_invalid_sample_size(sample_size: int):
    """Test that a sample size smaller than one raises an error."""
    with pytest.raises(ValueError, match="positive integer or None"):
        MarginalImputer(model=_sum_model, data=np.zeros((5, 3)), sample_size=sample_size)


def _null_player_setup() -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """Returns background data and a model that never uses feature 0 (a null player)."""
    rng = np.random.default_rng(0)
    data = rng.normal(size=(2000, 3))
    tree = DecisionTreeRegressor(max_depth=6, random_state=0)
    tree.fit(data[:, 1:], data[:, 1] + data[:, 2])

    def model(x: np.ndarray) -> np.ndarray:
        return tree.predict(x[:, 1:])

    return data, model


@pytest.mark.parametrize("joint_marginal_distribution", [True, False])
def test_null_player_gets_zero_value(*, joint_marginal_distribution: bool):
    """Regression test: v(S + {0}) == v(S) for all S (including S = {}) and SV of 0 is zero."""
    data, model = _null_player_setup()
    imputer = MarginalImputer(
        model=model,
        data=data,
        x=data[0],
        random_state=0,
        normalize=False,
        joint_marginal_distribution=joint_marginal_distribution,
    )

    coalitions_without_0 = np.array(
        [[False, False, False], [False, True, False], [False, False, True], [False, True, True]]
    )
    coalitions_with_0 = coalitions_without_0.copy()
    coalitions_with_0[:, 0] = True
    values_without_0 = imputer(coalitions_without_0)
    values_with_0 = imputer(coalitions_with_0)
    assert values_without_0[0] == pytest.approx(values_with_0[0], abs=1e-12)  # S = {}
    assert np.allclose(values_without_0, values_with_0, rtol=0, atol=1e-12)

    sv = ExactComputer(game=imputer, n_players=3)(index="SV", order=1)
    assert sv[(0,)] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("joint_marginal_distribution", [True, False])
def test_efficiency(*, joint_marginal_distribution: bool):
    """Test that the Shapley values sum up to v(N) - v({})."""
    data, model = _null_player_setup()
    imputer = MarginalImputer(
        model=model,
        data=data,
        x=data[0],
        random_state=0,
        normalize=False,
        joint_marginal_distribution=joint_marginal_distribution,
    )
    v_empty, v_grand = imputer(np.array([[False, False, False], [True, True, True]]))
    sv = ExactComputer(game=imputer, n_players=3)(index="SV", order=1)
    sum_sv = sum(sv[(i,)] for i in range(3))
    assert sum_sv == pytest.approx(v_grand - v_empty, abs=1e-12)


def test_normalization_consistent_with_empty_value():
    """Test that empty prediction, normalization value, and baseline value are equal to v({})."""
    data, model = _null_player_setup()
    x = data[0]
    empty = np.zeros((1, 3), dtype=bool)
    v_empty = MarginalImputer(model=model, data=data, x=x, random_state=0, normalize=False)(empty)

    imputer = MarginalImputer(model=model, data=data, x=x, random_state=0, normalize=True)
    assert imputer.empty_prediction == v_empty[0]
    assert imputer.normalization_value == v_empty[0]
    assert imputer(empty)[0] == 0.0

    explainer = TabularExplainer(
        model=model,
        data=data,
        imputer="marginal",
        index="SV",
        max_order=1,
        random_state=0,
        sample_size=100,
    )
    iv = explainer.explain(x, budget=2**3)
    assert iv.baseline_value == v_empty[0]
    sum_sv = sum(iv[(i,)] for i in range(3))
    assert sum_sv + iv.baseline_value == pytest.approx(model(x.reshape(1, -1))[0])


@pytest.mark.parametrize("joint_marginal_distribution", [True, False])
@pytest.mark.parametrize("random_state", [None, 42])
def test_deterministic_within_instance(
    random_state: int | None, *, joint_marginal_distribution: bool
):
    """Test that repeated and permuted evaluations of coalitions return identical values."""
    data, model = _null_player_setup()
    imputer = MarginalImputer(
        model=model,
        data=data,
        x=data[0],
        random_state=random_state,
        joint_marginal_distribution=joint_marginal_distribution,
    )
    coalitions = np.array(
        [
            [False, False, False],
            [True, False, False],
            [False, True, False],
            [False, True, True],
            [True, True, True],
        ]
    )
    values = imputer(coalitions)
    assert np.array_equal(imputer(coalitions), values)  # repeated call
    permutation = np.random.default_rng(1).permutation(len(coalitions))
    assert np.array_equal(imputer(coalitions[permutation]), values[permutation])  # permuted batch
    for coalition, value in zip(coalitions, values, strict=True):  # single coalitions
        assert imputer(coalition.reshape(1, -1))[0] == value


@pytest.mark.parametrize("joint_marginal_distribution", [True, False])
def test_deterministic_across_instances(*, joint_marginal_distribution: bool):
    """Test that imputers with the same integer random state return identical values."""
    data, model = _null_player_setup()
    coalitions = np.array([[False, False, False], [False, True, False], [True, True, False]])

    def get_imputer(random_state: int | None) -> MarginalImputer:
        return MarginalImputer(
            model=model,
            data=data,
            x=data[0],
            random_state=random_state,
            joint_marginal_distribution=joint_marginal_distribution,
        )

    imputer_1, imputer_2 = get_imputer(7), get_imputer(7)
    assert imputer_1.empty_prediction == imputer_2.empty_prediction
    assert np.array_equal(imputer_1(coalitions), imputer_2(coalitions))

    # setting the same random state afterwards also leads to identical values
    imputer_1, imputer_2 = get_imputer(None), get_imputer(None)
    imputer_1.set_random_state(7)
    imputer_2.set_random_state(7)
    assert imputer_1.empty_prediction == imputer_2.empty_prediction
    assert np.array_equal(imputer_1(coalitions), imputer_2(coalitions))


@pytest.mark.parametrize("joint_marginal_distribution", [True, False])
def test_value_function_matches_reference(*, joint_marginal_distribution: bool):
    """Test the vectorized value function against a direct per-coalition computation."""
    data, model = _null_player_setup()
    imputer = MarginalImputer(
        model=model,
        data=data,
        x=data[0],
        random_state=0,
        normalize=False,
        joint_marginal_distribution=joint_marginal_distribution,
    )
    expected = []
    for coalition in ALL_COALITIONS:
        imputed = imputer._sampled_replacement_data.copy()
        imputed[:, coalition] = data[0][coalition]
        expected.append(np.mean(model(imputed)))
    assert np.allclose(imputer(ALL_COALITIONS), expected, rtol=0, atol=1e-12)


def test_value_function_chunks_model_calls(monkeypatch: pytest.MonkeyPatch):
    """Test that splitting the coalitions over several model calls does not change the values."""
    data, model = _null_player_setup()
    call_sizes = []

    def counting_model(x: np.ndarray) -> np.ndarray:
        call_sizes.append(x.shape[0])
        return model(x)

    imputer = MarginalImputer(model=counting_model, data=data, x=data[0], random_state=0)
    expected = imputer(ALL_COALITIONS)
    n_samples = imputer.sample_size
    # allow three coalitions per model call
    monkeypatch.setattr(
        "shapiq.imputer.marginal_imputer._MAX_ELEMENTS_PER_PREDICTION", 3 * n_samples * 3
    )
    call_sizes.clear()
    assert np.array_equal(imputer(ALL_COALITIONS), expected)
    assert call_sizes == [3 * n_samples, 3 * n_samples, 2 * n_samples]
