"""This test module contains all tests regarding the LocalExplanation game."""

import os

import numpy as np
import pytest

from shapiq.games.base import Game
from shapiq.games.benchmark import (
    LocalExplanation,
    AdultCensusLocalXAI,
    CaliforniaHousingLocalXAI,
    BikeSharingLocalXAI,
    SentimentAnalysisLocalXAI,
    ImageClassifierLocalXAI,
)
from shapiq.games.benchmark._setup._vit_setup import ViTModel
from shapiq.games.benchmark._setup._resnet_setup import ResNetModel


def test_basic_function(background_reg_dataset, dt_reg_model):
    """Tests the base LocalExplanation game with a small regression dataset."""
    x_data, y_data = background_reg_dataset
    x = x_data[0].copy()  # get first instance

    model = dt_reg_model

    # init game
    game = LocalExplanation(data=x_data, model=model.predict, x=x)
    game.precompute()

    # test save and load
    game.save("test_game.pkl")
    assert os.path.exists("test_game.pkl")

    # load new game
    new_game = LocalExplanation.load("test_game.pkl")
    assert new_game.n_values_stored == game.n_values_stored
    assert new_game.n_players == game.n_players
    assert new_game.normalize == game.normalize
    assert new_game.precomputed == game.precomputed

    # compare output on same input
    test_coalition = new_game.empty_coalition
    test_coalition[0] = True
    assert new_game(test_coalition) == game(test_coalition)

    # clean up
    os.remove("test_game.pkl")
    assert not os.path.exists("test_game.pkl")

    # init game with integer
    game = LocalExplanation(x=0, data=x_data, model=model.predict)
    # check if the x_explain is valid
    assert np.all(game.x == x_data[0])

    # test game with no instance
    game = LocalExplanation(x=None, data=x_data, model=model.predict)
    assert game.x is not None


@pytest.mark.slow
@pytest.mark.parametrize("model", ["sklearn_rf", "invalid"])
def test_basic_function(model):
    """Tests the AdultCensus LocalExplanation game."""

    game_n_players = 14

    if model == "invalid":
        with pytest.raises(ValueError):
            _ = AdultCensusLocalXAI(model_name=model, x=0)
        return

    x_explain_id = 1
    game = AdultCensusLocalXAI(x=x_explain_id, model_name=model)
    assert game.n_players == game_n_players

    # test full prediction output against underlying model
    full_pred = float(game(np.ones(game_n_players, dtype=bool)))
    assert full_pred + game.normalization_value == 0.28  # for x_explain_id=1 it should be 0.28

    test_coalitions_precompute = np.array(
        [
            np.zeros(game_n_players),
            np.ones(game_n_players),
            np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0]),
            np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0]),
        ],
        dtype=bool,
    )

    game.precompute(coalitions=test_coalitions_precompute)
    assert game.n_players == game_n_players
    assert len(game.feature_names) == game_n_players
    assert game.n_values_stored == 4
    assert game.precomputed

    # test save and load values
    path = f"test_values_bike.npz"
    game.save_values(path)
    assert os.path.exists(path)

    # test init from values file
    new_game = Game(path_to_values=path, normalize=True)
    out = new_game(test_coalitions_precompute)
    assert new_game.n_values_stored == game.n_values_stored
    assert new_game.n_players == game.n_players
    assert new_game.normalize == game.normalize
    assert new_game.normalization_value == game.normalization_value
    assert np.allclose(new_game.value_storage, game.value_storage)
    assert np.allclose(out, game.value_storage - game.normalization_value)

    # clean up
    os.remove(path)
    assert not os.path.exists(path)

    # value error for wrong class
    with pytest.raises(ValueError):
        _ = AdultCensusLocalXAI(x=x_explain_id, class_to_explain=2)


@pytest.mark.slow
@pytest.mark.parametrize("model", ["torch_nn", "sklearn_gbt", "invalid"])
def test_basic_function(model):
    """Tests the CaliforniaHousing game with a small regression dataset."""
    game_n_players = 8

    if model == "invalid":
        with pytest.raises(ValueError):
            _ = CaliforniaHousingLocalXAI(model=model, x=0)
        return

    x_id = 0
    if model == "torch_nn":  # test here the auto select
        x_id = None

    test_coalitions_precompute = np.array(
        [
            np.zeros(game_n_players),
            np.ones(game_n_players),
            np.array([0, 1, 0, 1, 0, 1, 0, 1]),
            np.array([1, 0, 1, 0, 1, 0, 1, 0]),
        ],
        dtype=bool,
    )

    game = CaliforniaHousingLocalXAI(x=x_id, model=model)
    game.precompute(coalitions=test_coalitions_precompute)
    assert game.n_players == game_n_players
    assert len(game.feature_names) == game_n_players
    assert game.n_values_stored == len(test_coalitions_precompute)
    assert game.precomputed

    # test save and load values
    path = f"california_local_xai_{model}_id_{x_id}.npz"
    game.save_values(path)

    assert os.path.exists(path)

    # test init from values file
    new_game = Game(path_to_values=path, normalize=True)
    out = new_game(test_coalitions_precompute)
    assert new_game.n_values_stored == game.n_values_stored
    assert new_game.n_players == game.n_players
    assert new_game.normalize == game.normalize
    assert np.allclose(new_game.value_storage, game.value_storage)
    assert np.allclose(out, game.value_storage - game.normalization_value)

    # clean up
    os.remove(path)
    assert not os.path.exists(path)


@pytest.mark.parametrize("model", ["xgboost", "invalid"])
def test_basic_function(model):
    """Tests the BikeSharing game."""

    game_n_players = 12

    if model == "invalid":
        with pytest.raises(ValueError):
            _ = BikeSharingLocalXAI(model=model, x=0)
        return

    x_explain_id = 0
    game = BikeSharingLocalXAI(x=x_explain_id, model=model)
    assert game.n_players == game_n_players

    test_coalitions_precompute = np.array(
        [
            np.zeros(game_n_players),
            np.ones(game_n_players),
            np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
            np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]),
        ],
        dtype=bool,
    )

    game.precompute(coalitions=test_coalitions_precompute)
    assert game.n_players == game_n_players
    assert len(game.feature_names) == game_n_players
    assert game.n_values_stored == len(test_coalitions_precompute)
    assert game.precomputed

    # test save and load values
    path = f"test_values_bike.npz"
    game.save_values(path)
    assert os.path.exists(path)

    # test init from values file
    new_game = Game(path_to_values=path, normalize=True)
    out = new_game(test_coalitions_precompute)
    assert new_game.n_values_stored == game.n_values_stored
    assert new_game.n_players == game.n_players
    assert new_game.normalize == game.normalize
    assert np.allclose(new_game.value_storage, game.value_storage)
    assert np.allclose(out, game.value_storage - game.normalization_value)

    # clean up
    os.remove(path)
    assert not os.path.exists(path)


@pytest.mark.slow
@pytest.mark.parametrize("mask_strategy", ["remove", "mask"])
def test_basic_function(mask_strategy):
    """Tests the SentimentClassificationGame with a small input text."""
    input_text = "this is a six word sentence"
    n_players = 6
    game = SentimentAnalysisLocalXAI(
        input_text=input_text, normalize=True, mask_strategy=mask_strategy
    )

    assert game.n_players == n_players
    assert game.original_input_text == input_text
    assert game.original_model_output == game._full_output

    assert game.normalization_value == game._empty_output
    assert game.normalize  # should be normalized

    # test value function
    test_coalition = np.array([[0, 0, 0, 0, 0, 0]], dtype=bool)
    assert game.value_function(test_coalition) == game._empty_output
    assert game(test_coalition) == game._empty_output - game.normalization_value

    test_coalition = np.array([[1, 1, 1, 1, 1, 1]], dtype=bool)
    assert game.value_function(test_coalition) == game._full_output
    assert game(test_coalition) == game._full_output - game.normalization_value

    test_coalition = np.array([[1, 0, 1, 0, 1, 0]], dtype=bool)
    assert game.value_function(test_coalition) == game(test_coalition) + game.normalization_value

    # test ValueError with wrong param
    with pytest.raises(ValueError):
        _ = SentimentAnalysisLocalXAI(
            input_text=input_text, normalize=True, mask_strategy="undefined"
        )


@pytest.mark.slow
def test_resnet_model_class(test_image_and_path):
    """Tests the creation of the ResNet Model class."""
    test_image, _ = test_image_and_path
    resnet_model = ResNetModel(n_superpixels=14, input_image=test_image, verbose=True, batch_size=2)
    assert resnet_model.n_superpixels == 14

    test_coalitions = np.asarray(
        [
            [False for _ in range(14)],
            [True for _ in range(14)],
            [False for _ in range(14)],
        ],
        dtype=bool,
    )

    output = resnet_model(test_coalitions)
    assert len(output) == 3
    assert output[0] == pytest.approx(resnet_model.empty_value, abs=1e-3)
    assert output[1] == pytest.approx(0.2925054, abs=1e-3)


@pytest.mark.slow
def test_image_classifier_game_resnet(test_image_and_path):
    """Tests the ImageClassifierGame with the ResNet models."""
    # TODO: maybe remove this test and check all of it in the ImageClassifierGame test
    test_image, path_from_test_root = test_image_and_path
    game = ImageClassifierLocalXAI(
        model_name="resnet_18", verbose=True, x_explain_path=path_from_test_root
    )
    assert game.n_players == 14
    assert game.normalization_value == game.model_function.empty_value
    assert game.normalize  # should be True as empty value is around 0.005 and not 0
    grand_coal_output = game(game.grand_coalition)
    assert grand_coal_output == pytest.approx(0.2925054 - game.normalization_value, abs=1e-3)

    game_small = ImageClassifierLocalXAI(
        model_name="resnet_18",
        verbose=False,
        x_explain_path=path_from_test_root,
        n_superpixel_resnet=5,
    )
    assert game_small.n_players == 5
    assert game_small(game_small.grand_coalition) == grand_coal_output


@pytest.mark.slow
def test_vit_model_class(test_image_and_path):
    """Tests the creation of the ViTModel class."""
    # TODO: maybe remove this test and check all of it in the ImageClassifierGame test
    test_image, _ = test_image_and_path
    vit_model = ViTModel(n_patches=16, input_image=test_image, verbose=False)
    assert vit_model.n_patches == 16
    assert float(vit_model(coalitions=np.ones(16))) >= 0.9195  # check that call works
    assert float(vit_model(coalitions=np.zeros(16))) == vit_model.empty_value

    vit_model = ViTModel(n_patches=9, input_image=test_image, verbose=True)
    assert vit_model.n_patches == 9

    with pytest.raises(ValueError):
        ViTModel(n_patches=10, input_image=test_image)


@pytest.mark.slow
def test_image_classifier_game_vit(test_image_and_path):
    """Tests the ImageClassifierGame with the ViT models."""
    test_image, path_from_test_root = test_image_and_path
    game = ImageClassifierLocalXAI(
        x_explain_path=path_from_test_root, model_name="vit_9_patches", normalize=True, verbose=True
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
    new_game = Game(path_to_values="test_values.npz")
    assert new_game.n_values_stored == 4
    assert np.allclose(game.value_storage, new_game.value_storage)

    # cleanup
    os.remove("test_values.npz")
    assert not os.path.exists("test_values.npz")

    # create vit with 16 patches
    game_16 = ImageClassifierLocalXAI(
        x_explain_path=path_from_test_root,
        model_name="vit_16_patches",
        normalize=True,
        verbose=False,
    )
    assert game_16.n_players == 16
    assert game_16.normalization_value == game.normalization_value  # should be the same as 9 patch

    # wrong model
    with pytest.raises(ValueError):
        _ = ImageClassifierLocalXAI(x_explain_path=path_from_test_root, model_name="wrong_model")

    # no image path
    with pytest.raises(ValueError):
        _ = ImageClassifierLocalXAI(model_name="vit_9_patches")
