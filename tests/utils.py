"""This module contains utility functions for testing purposes."""


def get_concrete_class(abclass):
    """Class decorator to create a concrete class from an abstract class.

    The function is used to test abstract classes and their methods.
    Directly taken from https://stackoverflow.com/a/37574495.

    Args:
        abclass: The abstract class to create a concrete class from.

    Returns:
        The concrete class.
    """

    class concreteCls(abclass):
        pass

    concreteCls.__abstractmethods__ = frozenset()
    return type("DummyConcrete" + abclass.__name__, (concreteCls,), {})
