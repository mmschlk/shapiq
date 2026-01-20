from __future__ import annotations

from typing import get_args

import pytest

from shapiq.explainer.custom_types import ExplainerIndices
from shapiq.explainer.nn._util import assert_valid_index_and_order


def test_valid_index_and_order():
    assert_valid_index_and_order("SV", 1)
    assert_valid_index_and_order("k-SII", 1)


def test_invalid_index_and_order():
    for index in get_args(ExplainerIndices):
        if index in ["SV", "k-SII"]:
            continue
        with pytest.raises(ValueError, match="Explainer index .* is invalid"):
            assert_valid_index_and_order(index, 1)

    expected_exception = pytest.raises(ValueError, match="Explanation order .* is invalid")
    with expected_exception:
        assert_valid_index_and_order("SV", 0)
    with expected_exception:
        assert_valid_index_and_order("SV", 2)
