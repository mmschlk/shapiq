"""This module contains all tests for the TreeExplainer class of the shapiq package."""
import pytest

from shapiq.explainer import TreeExplainer


def test_init():
    with pytest.raises(NotImplementedError):
        explainer = TreeExplainer()
