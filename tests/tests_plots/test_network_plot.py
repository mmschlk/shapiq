"""This module contains all tests for the network plots."""
import numpy as np
import matplotlib.pyplot as plt
import pytest
from PIL import Image
from scipy.special import binom

from shapiq.plot import network_plot
from interaction_values import InteractionValues


def test_network_plot():
    """Tests whether the network plot can be created."""

    first_order_values = np.asarray([0.1, -0.2, 0.3, 0.4, 0.5, 0.6])
    second_order_values = np.random.rand(6, 6) - 0.5

    fig, axes = network_plot(
        first_order_values=first_order_values,
        second_order_values=second_order_values,
    )
    assert fig is not None
    assert axes is not None
    plt.close(fig)

    fig, axes = network_plot(
        first_order_values=first_order_values[0:4],
        second_order_values=second_order_values[0:4, 0:4],
        feature_names=["a", "b", "c", "d"],
    )
    assert fig is not None
    assert axes is not None
    plt.close(fig)

    # test with InteractionValues object
    n_players = 5
    n_values = n_players + int(binom(n_players, 2))
    iv = InteractionValues(
        values=np.random.rand(n_values),
        index="k-SII",
        n_players=n_players,
        min_order=1,
        max_order=2,
    )
    fig, axes = network_plot(interaction_values=iv)
    assert fig is not None
    assert axes is not None
    plt.close(fig)

    # value error if neither first_order_values nor interaction_values are given
    with pytest.raises(ValueError):
        network_plot()

    assert True


def test_network_plot_with_image_or_text():
    first_order_values = np.asarray([0.1, -0.2, 0.3, 0.4, 0.5, 0.6])
    second_order_values = np.random.rand(6, 6) - 0.5
    n_features = len(first_order_values)

    # create dummyimage
    image = np.random.rand(100, 100, 3)
    image = (image * 255).astype(np.uint8)
    image = Image.fromarray(image)

    feature_image_patches: dict[int, Image.Image] = {}
    feature_image_patches_size: dict[int, float] = {}
    for feature_idx in range(n_features):
        feature_image_patches[feature_idx] = image
        feature_image_patches_size[feature_idx] = 0.1

    fig, axes = network_plot(
        first_order_values=first_order_values,
        second_order_values=second_order_values,
        center_image=image,
        feature_image_patches=feature_image_patches,
    )
    assert fig is not None
    assert axes is not None
    plt.close(fig)

    fig, axes = network_plot(
        first_order_values=first_order_values,
        second_order_values=second_order_values,
        center_image=image,
        feature_image_patches=feature_image_patches,
        feature_image_patches_size=feature_image_patches_size,
    )
    assert fig is not None
    assert axes is not None
    plt.close(fig)

    # with text
    fig, axes = network_plot(
        first_order_values=first_order_values,
        second_order_values=second_order_values,
        center_text="center text",
    )
    assert fig is not None
    assert axes is not None
    plt.close(fig)
    assert True
