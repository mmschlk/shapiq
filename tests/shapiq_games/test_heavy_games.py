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

    game = ImageClassifier.from_config(index=0, size="160px", model="vit_9_patches")
    assert game.n_players == 9
    assert 0.0 <= game(game.grand_coalition)[0] + game.normalization_value <= 1.0
    _assert_deterministic(game)


@pytest.mark.skipif(not is_installed("torchvision"), reason="torchvision is not installed")
def test_resnet_game() -> None:
    from shapiq_games import ImageClassifier

    game = ImageClassifier.from_config(
        index=1, size="160px", model="resnet_18", n_superpixels=8, class_index="label"
    )
    assert game.class_index == 0  # the first Imagenette class, tench, is ImageNet class 0
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


def test_imagenette_download() -> None:
    from shapiq_games.datasets import load_imagenette

    images = load_imagenette(split="val", size="160px")
    assert len(images) == 3925
    assert sorted(set(images.labels.tolist())) == [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]
    assert min(images[0].shape[:2]) == 160


_PREVIOUS_FILES = (
    "https://raw.githubusercontent.com/mmschlk/shapiq/5ff37e6ce6ccefe4ca8ef938d16e61da3d779242/"
    "src/shapiq_games/datasets/data/"
)


@pytest.mark.skipif(
    not (is_installed("openml") and is_installed("ucimlrepo")),
    reason="openml and ucimlrepo are not installed",
)
@pytest.mark.parametrize(
    "name",
    [
        "adult_census",
        "amazon",
        "annealing",
        "arrhythmia",
        "bike_sharing",
        "bioresponse",
        "california_housing",
        "hepatitis",
        "ionosphere",
        "leukemia",
        "microresponse",
        "mushroom",
        "nursery",
        "soybean",
        "thyroid",
        "zoo",
    ],
)
def test_upstream_table_matches_the_previously_bundled_file(name: str) -> None:
    """The original source still serves the table the loaders were written for."""
    import pandas as pd

    from shapiq_games.datasets import _tabular

    previous_name = {"bike_sharing": "bike"}.get(name, name)
    previous = pd.read_csv(f"{_PREVIOUS_FILES}{previous_name}.csv", low_memory=False)
    table = _tabular._read_table(name)
    assert list(table.columns) == list(previous.columns)
    pd.testing.assert_frame_equal(table, previous, check_dtype=False, check_exact=False, rtol=1e-12)
