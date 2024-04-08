"""This test module contains the tests for all image classifier games."""

import os

import numpy as np
import pytest

from shapiq.games._vit_setup import ViTModel
from shapiq.games.image_classifier import ImageClassifierGame


def test_vit_model_class(test_image_and_path):
    """Tests the creation of the ViTModel class."""
    test_image, _ = test_image_and_path
    vit_model = ViTModel(n_patches=16, input_image=test_image, verbose=False)
    assert vit_model.n_patches == 16
    assert float(vit_model(coalitions=np.ones(16))) >= 0.9195  # check that call works
    assert float(vit_model(coalitions=np.zeros(16))) == vit_model.empty_value

    vit_model = ViTModel(n_patches=9, input_image=test_image, verbose=True)
    assert vit_model.n_patches == 9

    with pytest.raises(ValueError):
        ViTModel(n_patches=10, input_image=test_image)


def test_image_classifier_game_vit(test_image_and_path):
    """Tests the ImageClassifierGame with the ViT models."""
    test_image, path_from_test_root = test_image_and_path
    game = ImageClassifierGame(
        x_explain=path_from_test_root, model="vit_9_patches", normalize=True, verbose=True
    )
    assert game.n_players == 9
    assert game.normalization_value == game.model_function.empty_value
    assert game.normalize  # should be True as empty value is around 0.005 and not 0

    test_coalitions_to_precompute = np.array(
        [
            np.zeros(9),
            np.ones(9),
            np.array([0, 1, 0, 1, 0, 1, 0, 1, 0]),
            np.array([1, 0, 1, 0, 1, 0, 1, 0, 1]),
        ],
        dtype=bool,
    )

    game.precompute(test_coalitions_to_precompute)
    assert game.n_values_stored == 4

    # check that the values are stored and loaded correctly
    game.save_values("test_values.npz")
    assert os.path.exists("test_values.npz")

    # load
    new_game = ImageClassifierGame(path_to_values="test_values.npz")
    assert new_game.n_values_stored == 4
    assert np.allclose(game.value_storage, new_game.value_storage)

    # cleanup
    os.remove("test_values.npz")
    assert not os.path.exists("test_values.npz")

    # create vit with 16 patches
    game_16 = ImageClassifierGame(
        x_explain=path_from_test_root, model="vit_16_patches", normalize=True, verbose=False
    )
    assert game_16.n_players == 16
    assert game_16.normalization_value == game.normalization_value  # should be the same as 9 patch

    # wrong model
    with pytest.raises(ValueError):
        _ = ImageClassifierGame(x_explain=path_from_test_root, model="wrong_model")

    # no image path
    with pytest.raises(ValueError):
        _ = ImageClassifierGame(model="vit_9_patches")
