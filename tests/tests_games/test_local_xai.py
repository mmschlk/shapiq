"""This test module contains all tests regarding the LocalExplanation game."""

import os

import numpy as np
import pytest

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
    x_id = 0
    game = LocalExplanation(x=x_id, data=x_data, model=model.predict)
    assert np.all(game.x == x_data[x_id])

    # test game with no instance
    game = LocalExplanation(x=None, data=x_data, model=model.predict)
    assert game.x is not None


@pytest.mark.parametrize(
    "model", ["decision_tree", "random_forest", "gradient_boosting", "invalid"]
)
def test_adult_census(model):
    """Tests the AdultCensus LocalExplanation game."""

    game_n_players = 14
    x_explain_id = 1
    game_name = "AdultCensus_LocalExplanation_Game"

    if model == "invalid":
        with pytest.raises(ValueError):
            _ = AdultCensusLocalXAI(model_name=model, x=0)
        return

    game = AdultCensusLocalXAI(x=x_explain_id, model_name=model)
    assert game.n_players == game_n_players
    assert game.game_name == game_name

    test_coalitions_precompute = np.array(
        [
            np.zeros(game_n_players),
            np.ones(game_n_players),
            np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0]),
            np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0]),
        ],
        dtype=bool,
    )

    values = game(test_coalitions_precompute)
    assert values.shape == (4,)

    # value error for wrong class
    with pytest.raises(ValueError):
        _ = AdultCensusLocalXAI(x=x_explain_id, class_to_explain=2)


@pytest.mark.parametrize(
    "model", ["neural_network", "decision_tree", "random_forest", "gradient_boosting", "invalid"]
)
def test_california_housing(model):
    """Tests the CaliforniaHousing game local XAI."""

    game_n_players = 8
    x_id = 0
    game_name = "CaliforniaHousing_LocalExplanation_Game"

    if model == "invalid":
        with pytest.raises(ValueError):
            _ = CaliforniaHousingLocalXAI(model_name=model, x=0)
        return

    game = CaliforniaHousingLocalXAI(x=x_id, model_name=model)
    assert game.n_players == game_n_players
    assert game.game_name == game_name

    test_coalitions = np.array(
        [
            np.zeros(game_n_players),
            np.ones(game_n_players),
            np.array([0, 1, 0, 1, 0, 1, 0, 1]),
            np.array([1, 0, 1, 0, 1, 0, 1, 0]),
        ],
        dtype=bool,
    )

    values = game(test_coalitions)
    assert values.shape == (4,)


@pytest.mark.parametrize(
    "model", ["decision_tree", "random_forest", "gradient_boosting", "invalid"]
)
def test_bike_sharing(model):
    """Tests the BikeSharing local XAI game."""

    game_n_players = 12
    x_explain_id = 0
    game_name = "BikeSharing_LocalExplanation_Game"

    if model == "invalid":
        with pytest.raises(ValueError):
            _ = BikeSharingLocalXAI(model_name=model, x=x_explain_id)
        return

    game = BikeSharingLocalXAI(x=x_explain_id, model_name=model)
    assert game.n_players == game_n_players
    assert game.game_name == game_name

    test_coalitions_precompute = np.array(
        [
            np.zeros(game_n_players),
            np.ones(game_n_players),
            np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]),
            np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]),
        ],
        dtype=bool,
    )

    values = game(test_coalitions_precompute)
    assert values.shape == (4,)


@pytest.mark.parametrize("mask_strategy", ["remove", "mask"])
def test_sentiment_classifier(mask_strategy):
    """Tests the SentimentClassificationGame with a small input text."""
    input_text = "this is a six word sentence"
    n_players = 6
    game = SentimentAnalysisLocalXAI(
        input_text=input_text, normalize=True, mask_strategy=mask_strategy
    )

    assert game.n_players == n_players
    assert game.game_name == "SentimentAnalysis_Game"
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
    assert output[1] != 0.0


@pytest.mark.slow
def test_image_classifier_game_resnet(test_image_and_path):
    """Tests the ImageClassifierGame with the ResNet models."""
    test_image, path_from_test_root = test_image_and_path
    game = ImageClassifierLocalXAI(
        model_name="resnet_18", verbose=True, x_explain_path=path_from_test_root
    )
    assert game.n_players == 14
    assert game.game_name == "ImageClassifier_Game"
    assert game.normalization_value == game.model_function.empty_value
    assert game.normalize  # should be True as empty value is around 0.005 and not 0
    grand_coal_output = game(game.grand_coalition)

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

    values = game(test_coalitions_to_precompute)
    assert values.shape == (4,)

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
