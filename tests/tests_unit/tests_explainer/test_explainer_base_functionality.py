"""This module tests the base functionality of the explainer class."""

from __future__ import annotations

import pytest

from shapiq import Explainer


def test_explainer():
    """Tests if the attributes and properties of explainers are set correctly."""
    with pytest.raises(TypeError):
        Explainer()
