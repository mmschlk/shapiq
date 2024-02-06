"""This test module contains all tests regarding the base approximator class."""
import pytest

from shapiq.approximator._base import Approximator


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
