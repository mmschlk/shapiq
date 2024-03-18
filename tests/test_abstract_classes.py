"""This test module contains all tests regarding the base approximator class."""
import numpy as np
import pytest

from shapiq.approximator._base import Approximator
from shapiq.explainer.imputer._base import Imputer
from shapiq.explainer._base import Explainer


def concreter(abclass):
    """Class decorator to create a concrete class from an abstract class.
    Directly taken from https://stackoverflow.com/a/37574495."""

    class concreteCls(abclass):
        pass

    concreteCls.__abstractmethods__ = frozenset()
    return type("DummyConcrete" + abclass.__name__, (concreteCls,), {})


def test_approximator():
    approx = concreter(Approximator)(n=7, max_order=2, index="SII", top_order=False)
    assert approx.n == 7
    assert approx.max_order == 2
    assert approx.index == "SII"
    assert approx.top_order is False

    with pytest.raises(NotImplementedError):
        approx.approximate(budget=100, game=lambda x: x)

    wrong_index = "something"
    with pytest.raises(ValueError):
        approx = concreter(Approximator)(n=7, max_order=2, index=wrong_index, top_order=False)


def test_imputer():
    model = lambda x: x
    background_data = np.asarray([[1, 2, 3], [4, 5, 6]])
    imputer = concreter(Imputer)(model, background_data)
    assert imputer._model == model
    assert np.all(imputer._background_data == background_data)
    assert imputer._n_features == 3
    assert imputer._cat_features == []
    assert imputer._random_state is None
    assert imputer._rng is not None

    with pytest.raises(NotImplementedError):
        imputer(np.array([[True, False, True]]))


def test_explainer():
    explainer = concreter(Explainer)()
    with pytest.raises(NotImplementedError):
        explainer.explain(np.array([[1, 2, 3]]))
