from __future__ import annotations

import logging
from typing import get_args

import pytest

from shapiq.explainer.custom_types import ExplainerIndices
from shapiq.explainer.nn._util import assert_valid_index_and_order, warn_ignored_parameters


def test_warn_ignored_parameters_emits_warning(caplog):
    local_vars = {
        "foo": 123,
        "bar": None,
    }
    ignored = ["foo", "bar"]

    class_name = "MyClass"
    with caplog.at_level(logging.WARNING):
        warn_ignored_parameters(
            local_vars=local_vars,
            ignored_parameter_names=ignored,
            class_name=class_name,
        )

    # Assert exactly one warning was emitted
    assert len(caplog.records) == 1

    record = caplog.records[0]
    assert record.levelno == logging.WARNING
    assert "ignored" in record.message
    assert class_name in record.message


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
