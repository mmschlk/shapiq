"""This module contains the tests for the `ClusterExplanation` class."""

import os

import numpy as np
import pytest

from shapiq.games.base import Game
from shapiq.games.benchmark import (
    ClusterExplanation,
    AdultCensusClusterExplanation,
    BikeSharingClusterExplanation,
    CaliforniaHousingClusterExplanation,
)


@pytest.mark.parametrize(
    "cluster_method, score_method",
    [
        ("kmeans", "calinski_harabasz_score"),
        ("kmeans", "silhouette_score"),
        ("agglomerative", "calinski_harabasz_score"),
        ("agglomerative", "silhouette_score"),
    ],
)
def test_base_class(cluster_method, score_method):
    """This function tests the setup and logic of the game."""

    n_players = 4

    # create synthetic data
    data = np.random.rand(100, n_players)
    cluster_params = {"n_clusters": 3}

    # setup game
    game = ClusterExplanation(
        data=data,
        cluster_method=cluster_method,
        score_method=score_method,
        cluster_params=cluster_params,
        random_state=42,
        normalize=True,
    )
    assert game.n_players == n_players

    # test the game
    test_coalitions = [[False, False, False, False], [True, False, False, False]]
    test_coalitions = np.array(test_coalitions, dtype=bool)
    test_values = game(test_coalitions)
    assert test_values.shape == (2,)
    assert test_values[0] == 0.0
    assert test_values[1] != 0.0

    # test pre-compute
    game.precompute(test_coalitions)
    assert game.n_values_stored == 2

    # test save and load
    game.save("test_game.pkl")
    assert os.path.exists("test_game.pkl")
    loaded_game = Game.load("test_game.pkl")
    assert loaded_game.n_players == n_players
    assert loaded_game.n_values_stored == 2
    os.remove("test_game.pkl")
    assert not os.path.exists("test_game.pkl")

    # test save and load values
    game.save_values("test_values.npz")
    assert os.path.exists("test_values.npz")
    loaded_values = Game(path_to_values="test_values.npz")
    assert loaded_values.n_players == n_players
    assert loaded_values.n_values_stored == 2
    os.remove("test_values.npz")
    assert not os.path.exists("test_values.npz")

    # test with wrong cluster method
    with pytest.raises(ValueError):
        _ = ClusterExplanation(
            data=data,
            cluster_method="wrong",
            score_method=score_method,
            cluster_params=cluster_params,
            random_state=42,
            normalize=True,
        )

    # test with wrong score method
    with pytest.raises(ValueError):
        _ = ClusterExplanation(
            data=data,
            cluster_method=cluster_method,
            score_method="wrong",
            cluster_params=cluster_params,
            random_state=42,
            normalize=True,
        )

    # test setup without any cluster params
    game = ClusterExplanation(
        data=data,
        cluster_method=cluster_method,
        score_method=score_method,
        random_state=42,
        normalize=True,
    )
    assert game.n_players == n_players


@pytest.mark.parametrize(
    "cluster_method, score_method",
    [
        ("kmeans", "calinski_harabasz_score"),
        ("agglomerative", "silhouette_score"),
    ],
)
def test_california(cluster_method, score_method):
    """Tests the california housing Dataset Valuation Benchmark game."""
    n_players = 8
    # setup game
    game = CaliforniaHousingClusterExplanation(
        cluster_method=cluster_method, score_method=score_method
    )
    assert game.n_players == n_players
    assert game.game_name == "CaliforniaHousing_ClusterExplanation_Game"
    # no run tests here since it takes too long


@pytest.mark.parametrize(
    "cluster_method, score_method",
    [
        ("kmeans", "calinski_harabasz_score"),
        ("agglomerative", "silhouette_score"),
    ],
)
def test_bike(cluster_method, score_method):
    """Tests the bike sharing Dataset Valuation Benchmark game."""
    n_players = 12
    # setup game
    game = BikeSharingClusterExplanation(cluster_method=cluster_method, score_method=score_method)
    assert game.n_players == n_players
    assert game.game_name == "BikeSharing_ClusterExplanation_Game"
    # no run tests here since it takes too long


@pytest.mark.parametrize(
    "cluster_method, score_method",
    [
        ("kmeans", "calinski_harabasz_score"),
        ("agglomerative", "silhouette_score"),
    ],
)
def test_adult_census(cluster_method, score_method):
    """Tests the adult census Dataset Valuation Benchmark game."""
    n_players = 14
    # setup game
    game = AdultCensusClusterExplanation(cluster_method=cluster_method, score_method=score_method)
    assert game.n_players == n_players
    assert game.game_name == "AdultCensus_ClusterExplanation_Game"
    # no run tests here since it takes too long
