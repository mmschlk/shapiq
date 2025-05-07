"""This module contains all tests for the network plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy as sp

from shapiq.interaction_values import InteractionValues
from shapiq.plot import network_plot


def test_network_plot():
    """Tests whether the network plot can be created."""
    n_players = 5
    n_values = n_players + int(sp.special.binom(n_players, 2))
    iv = InteractionValues(
        values=np.random.rand(n_values),
        index="k-SII",
        n_players=n_players,
        min_order=1,
        max_order=2,
        baseline_value=0.0,
    )
    fig, axes = network_plot(interaction_values=iv)
    assert fig is not None
    assert axes is not None
    plt.close(fig)

    # value error if neither first_order_values nor interaction_values are given
    with pytest.raises(ValueError):
        network_plot()

    # test show=True
    output = network_plot(interaction_values=iv, show=True)
    assert output is None
    plt.close("all")

    assert True
