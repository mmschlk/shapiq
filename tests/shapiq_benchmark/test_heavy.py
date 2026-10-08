"""Opt-in tests of setups and datasets that download pretrained models or data from third-party hosts.

These tests need network access to Hugging Face, download.pytorch.org, OpenML, or the UCI
repository, and some need optional packages. They are skipped unless the environment variable
``SHAPIQ_RUN_HEAVY_TESTS=1`` is set.
"""

from __future__ import annotations

import dataclasses
import os

import numpy as np
import pytest

from shapiq_benchmark.datasets import load_dataset
from shapiq_benchmark.setups import (
    GlobalConfoundingSetup,
    ImageClassifierSetup,
    ImageTextSimilaritySetup,
    SentimentAnalysisSetup,
    TabularLocalExplanationSetup,
)
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
    game = ImageClassifierSetup(index=0, size="160px", model="vit_9_patches").build()
    assert game.n_players == 9
    assert 0.0 <= game(game.grand_coalition)[0] + game.normalization_value <= 1.0
    _assert_deterministic(game)


@pytest.mark.skipif(not is_installed("torchvision"), reason="torchvision is not installed")
def test_resnet_game() -> None:
    setup = ImageClassifierSetup(
        index=1, size="160px", model="resnet_18", n_superpixels=8, class_index="label"
    )
    game = setup.build()
    assert game.class_index == 0  # the first Imagenette class, tench, is ImageNet class 0
    assert game.n_players == 8
    _assert_deterministic(game)


@pytest.mark.skipif(not is_installed("transformers"), reason="transformers is not installed")
def test_dinov2_token_drop_game() -> None:
    game = ImageClassifierSetup(index=388, model="dinov2_20_patches", batch_size=4).build()
    assert game.n_players == 20
    assert game.image.shape == (224, 224, 3)
    assert game.class_index == 217  # the image's true class, English springer
    _assert_deterministic(game)


@pytest.mark.skipif(not is_installed("transformers"), reason="transformers is not installed")
def test_clip_image_text_game() -> None:
    game = ImageTextSimilaritySetup(index=388, grid=(5, 4), batch_size=4).build()
    assert game.n_players == 20
    assert "spaniel" in game.label  # CLIP's zero-shot label of an English springer
    assert 0.2 < game.original_model_output < 0.4
    _assert_deterministic(game)
    labelled = ImageTextSimilaritySetup(index=388, text="label", model="clip_vit_b32").build()
    assert labelled.text == "a photo of a English springer."


@pytest.mark.skipif(not is_installed("transformers"), reason="transformers is not installed")
@pytest.mark.parametrize("removal", [{"mask_strategy": "remove"}, {"fill": "blur"}])
def test_vision_transformer_removal_strategies(removal: dict) -> None:
    """Every way of removing players keeps the full image's prediction and is batch-independent."""
    default = ImageClassifierSetup(index=388, size="160px", model="vit_9_patches", normalize=False)
    game = dataclasses.replace(default, **removal).build()
    expected = default.build()(np.ones((1, game.n_players), dtype=bool))[0]
    assert game(game.grand_coalition)[0] == pytest.approx(expected, abs=1e-6)
    _assert_deterministic(game)


@pytest.mark.skipif(not is_installed("transformers"), reason="transformers is not installed")
@pytest.mark.parametrize("removal", [{"mask_strategy": "mask"}, {"fill": "gray"}])
def test_clip_removal_strategies(removal: dict) -> None:
    default = ImageTextSimilaritySetup(index=388, model="clip_vit_b32", text="label")
    game = dataclasses.replace(default, **removal).build()
    assert game.original_model_output == pytest.approx(default.build().original_model_output)
    _assert_deterministic(game)


@pytest.mark.skipif(not is_installed("tabpfn"), reason="tabpfn is not installed")
def test_tabpfn_missing_value_game() -> None:
    """TabPFN v2 (no license token needed) reads absent features masked as +inf."""
    setup = TabularLocalExplanationSetup(
        dataset="california_housing",
        model="tabpfn",
        imputer="baseline",
        baseline="missing",
        n_train=500,  # tabpfn refuses more than 1,000 rows on a CPU by default
        model_params={"n_estimators": 1},
    )
    game = setup.build()
    assert game.n_players == 8
    assert game(game.grand_coalition)[0] != 0.0
    _assert_deterministic(game)


@pytest.mark.skipif(not is_installed("transformers"), reason="transformers is not installed")
def test_sentiment_game() -> None:
    game = SentimentAnalysisSetup(input_text="This movie was surprisingly good.").build()
    assert -1.0 <= game.original_model_output <= 1.0
    _assert_deterministic(game)


@pytest.mark.skipif(not is_installed("tabpfn"), reason="tabpfn is not installed")
def test_tabpfn_recontextualization_game() -> None:
    setup = TabularLocalExplanationSetup(
        dataset="xor", model="tabpfn", imputer="tabpfn", n_background=50
    )
    game = setup.build()
    _assert_deterministic(game)


@pytest.mark.skipif(not is_installed("tabpfn"), reason="tabpfn is not installed")
def test_confounding_game_with_tabpfn() -> None:
    game = GlobalConfoundingSetup(n=100).build()  # the default regressor is TabPFN v2
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
    from shapiq_benchmark.datasets import load_imagenette

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

    from shapiq_benchmark.datasets import _tabular

    previous_name = {"bike_sharing": "bike"}.get(name, name)
    previous = pd.read_csv(f"{_PREVIOUS_FILES}{previous_name}.csv", low_memory=False)
    table = _tabular._read_table(name)
    assert list(table.columns) == list(previous.columns)
    pd.testing.assert_frame_equal(table, previous, check_dtype=False, check_exact=False, rtol=1e-12)
