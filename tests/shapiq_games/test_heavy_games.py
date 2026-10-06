"""Opt-in tests of games that download pretrained models or data from third-party hosts.

These tests need network access to Hugging Face, download.pytorch.org, OpenML, or the UCI
repository, and some need optional packages. They are skipped unless the environment variable
``SHAPIQ_RUN_HEAVY_TESTS=1`` is set.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from shapiq_games.datasets import load_dataset
from tests.shapiq_games.helpers import is_installed

pytestmark = pytest.mark.skipif(
    os.environ.get("SHAPIQ_RUN_HEAVY_TESTS") != "1",
    reason="set SHAPIQ_RUN_HEAVY_TESTS=1 to run tests that download models and data",
)


def _assert_deterministic(game) -> None:
    coalitions = np.random.default_rng(0).random((4, game.n_players)) < 0.5
    np.testing.assert_allclose(game(coalitions), game(coalitions[::-1])[::-1], rtol=1e-6)


@pytest.mark.skipif(not is_installed("transformers"), reason="transformers is not installed")
def test_vision_transformer_game() -> None:
    from shapiq_games import ImageClassifier

    game = ImageClassifier.from_config(image=0, model="vit_9_patches")
    assert game.n_players == 9
    assert 0.0 <= game(game.grand_coalition)[0] + game.normalization_value <= 1.0
    _assert_deterministic(game)


@pytest.mark.skipif(not is_installed("torchvision"), reason="torchvision is not installed")
def test_resnet_game() -> None:
    from shapiq_games import ImageClassifier

    game = ImageClassifier.from_config(image=1, model="resnet_18", n_superpixels=8)
    assert game.n_players == 8
    _assert_deterministic(game)


@pytest.mark.skipif(not is_installed("transformers"), reason="transformers is not installed")
def test_sentiment_game() -> None:
    from shapiq_games import SentimentAnalysis

    game = SentimentAnalysis.from_config(input_text="This movie was surprisingly good.")
    assert -1.0 <= game.original_model_output <= 1.0
    _assert_deterministic(game)


@pytest.mark.skipif(not is_installed("tabpfn"), reason="tabpfn is not installed")
def test_tabpfn_recontextualization_game() -> None:
    from shapiq_games import LocalExplanation

    game = LocalExplanation.from_config(
        dataset="xor", model="tabpfn", imputer="tabpfn", n_background=50
    )
    _assert_deterministic(game)


@pytest.mark.skipif(not is_installed("tabpfn"), reason="tabpfn is not installed")
def test_confounding_game_with_tabpfn() -> None:
    from shapiq_games import GlobalConfoundingXAI

    game = GlobalConfoundingXAI.from_config(n=100)
    assert game.fingerprint is not None
    _assert_deterministic(game)


@pytest.mark.skipif(not is_installed("openml"), reason="openml is not installed")
def test_tabarena_dataset_download() -> None:
    dataset = load_dataset("tabarena_blood_transfusion")
    assert dataset.task == "classification"
    assert dataset.n_samples == 748


@pytest.mark.parametrize("name", ["wine_quality", "forest_fires"])
def test_uci_dataset_download(name: str) -> None:
    dataset = load_dataset(name)
    assert dataset.task == "regression"
    assert dataset.n_samples > 0
