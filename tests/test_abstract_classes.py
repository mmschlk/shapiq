"""This test module contains all tests regarding the base approximator class."""

import numpy as np
import pytest

from shapiq.approximator._base import Approximator
from shapiq.explainer._base import Explainer
from shapiq.games.base import Game
from shapiq.games.imputer.base import Imputer


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
    def model(x):
        return x

    data = np.asarray([[1, 2, 3], [4, 5, 6]])
    imputer = concreter(Imputer)(model, data)
    assert imputer.model == model
    assert np.all(imputer.data == data)
    assert imputer._n_features == 3
    assert imputer._cat_features == []
    assert imputer._random_state is None
    assert imputer._rng is not None

    with pytest.raises(NotImplementedError):
        imputer(np.array([[True, False, True]]))


def test_explainer():
    with pytest.raises(TypeError):
        Explainer()


def test_game():
    n = 6
    game = concreter(Game)(n_players=n)
    with pytest.raises(NotImplementedError):
        game(np.array([[True for _ in range(n)]]))
