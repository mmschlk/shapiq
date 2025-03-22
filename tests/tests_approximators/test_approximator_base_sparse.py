"""This test module contains all tests regarding the base sparse approximator."""

import pytest
import numpy as np

from shapiq.approximator.sparse import Sparse
from shapiq.games.benchmark import DummyGame
from shapiq.interaction_values import InteractionValues


@pytest.mark.parametrize(
    "n, max_order, index, top_order, transform_type, decoder_type",
    [
        (7, 2, "FBII", False, "fourier", "soft"),
        (7, 2, "FBII", False, "FouRier", "Soft"),
        (7, 2, "FBII", False, "fourier", "Hard"),
        (7, 2, "wrong_index", False, "fourier", "soft"),  # Should raise ValueError
        (7, 2, "wrong_index", False, "fourier", "wrong_dec_type"),  # Should raise ValueError
        (7, 2, "FBII", True, "mobius", None),
        (7, 2, "FBII", True, "Mobius", 'abc'),
        (7, 2, "FBII", False, "invalid_type", None),  # Should raise ValueError
    ],
)

def test_initialization(n, max_order, index, top_order, transform_type, decoder_type):
    """Tests the initialization of the Sparse approximator."""

    if index == "wrong_index":
        with pytest.raises(ValueError):
            _ = Sparse(n,
                       max_order,
                       index=index,
                       top_order=top_order,
                       transform_type=transform_type,
                       decoder_type=decoder_type,
                       )
        return

    if transform_type == "invalid_type":
        with pytest.raises(ValueError):
            _ = Sparse(n,
                       max_order,
                       index=index,
                       top_order=top_order,
                       transform_type=transform_type,
                       decoder_type=decoder_type,
                       )
        return

    if decoder_type == "wrong_dec_type" or (transform_type.lower() == "mobius" and decoder_type is not None):
        with pytest.raises(ValueError):
            _ = Sparse(n,
                       max_order,
                       index=index,
                       top_order=top_order,
                       transform_type=transform_type,
                       decoder_type=decoder_type,
                       )
        return

    approximator = Sparse(
        n,
        max_order,
        index=index,
        top_order=top_order,
        transform_type=transform_type,
        decoder_type=decoder_type,
    )

    assert approximator.n == n
    assert approximator.max_order == max_order
    assert approximator.top_order is top_order
    assert approximator.min_order == (max_order if top_order else 0)
    assert approximator.index == index
    assert approximator.transform_type == transform_type.lower()
    if decoder_type is not None or transform_type.lower() == "fourier":
        channel_method = approximator.decoder_args['reconstruct_method_channel']
        if channel_method is None or decoder_type.lower() == 'soft':
            assert channel_method == 'identity-siso'
        else:
            assert channel_method == 'identity'
