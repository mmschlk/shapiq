"""The setups: every game but the synthetic ones builds from names, reproducibly, with a stable key."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pytest

import shapiq_games as sg
from shapiq_benchmark import setups
from shapiq_benchmark.setups import (
    SETUPS,
    ClusterExplanationSetup,
    DatasetValuationSetup,
    DataValuationSetup,
    EnsembleSelectionSetup,
    FeatureSelectionSetup,
    GlobalConfoundingSetup,
    GlobalExplanationSetup,
    ImageClassifierSetup,
    InterventionalTreeSetup,
    KNNSetup,
    LocalConfoundingSetup,
    LocalExplanationSetup,
    PathDependentTreeSetup,
    ProductKernelSetup,
    RandomForestEnsembleSelectionSetup,
    SentimentAnalysisSetup,
    Setup,
    ThresholdNNSetup,
    UncertaintyExplanationSetup,
    UnsupervisedDataSetup,
    WeightedKNNSetup,
    setup_from_dict,
)

if TYPE_CHECKING:
    from shapiq import Game

_SYNTHETIC_GAMES = {"DummyGame", "RandomTableGame", "SOUM", "UnanimityGame"}

# setups that build offline (registered synthetic datasets and scikit-learn's breast cancer)
OFFLINE: dict[str, tuple[Setup, type[Game]]] = {
    "path_dependent_tree": (
        PathDependentTreeSetup(dataset="xor", model="random_forest"),
        sg.PathDependentTreeGame,
    ),
    "interventional_tree": (
        InterventionalTreeSetup(dataset="breast_cancer", n_reference=20, normalize=True),
        sg.InterventionalTreeGame,
    ),
    "knn": (KNNSetup(dataset="xor", n_train=8), sg.KNNGame),
    "weighted_knn": (
        WeightedKNNSetup(dataset="breast_cancer", n_train=8, n_bits=3),
        sg.WeightedKNNGame,
    ),
    "threshold_nn": (
        ThresholdNNSetup(dataset="xor", n_train=8, model_params={"radius": 1.0}),
        sg.ThresholdNNGame,
    ),
    "product_kernel": (
        ProductKernelSetup(dataset="breast_cancer", n_train=100, normalize=True),
        sg.ProductKernelGame,
    ),
    "local_explanation": (
        LocalExplanationSetup(dataset="condind", imputer="baseline"),
        sg.LocalExplanation,
    ),
    "global_explanation": (
        GlobalExplanationSetup(dataset="xor", model="decision_tree"),
        sg.GlobalExplanation,
    ),
    "feature_selection": (
        FeatureSelectionSetup(dataset="breast_cancer", n_train=100),
        sg.FeatureSelection,
    ),
    "data_valuation": (
        DataValuationSetup(dataset="xor", n_players=8, empty_value=0.5),
        sg.DataValuation,
    ),
    "dataset_valuation": (
        DatasetValuationSetup(
            dataset="independentlinear60",
            dataset_params={"n_samples": 300},
            n_players=5,
            player_sizes="increasing",
        ),
        sg.DatasetValuation,
    ),
    "ensemble_selection": (
        EnsembleSelectionSetup(dataset="breast_cancer", members=("linear", "decision_tree")),
        sg.EnsembleSelection,
    ),
    "random_forest_ensemble_selection": (
        RandomForestEnsembleSelectionSetup(dataset="xor", n_members=5),
        sg.RandomForestEnsembleSelection,
    ),
    "uncertainty_explanation": (
        UncertaintyExplanationSetup(dataset="breast_cancer", n_background=20),
        sg.UncertaintyExplanation,
    ),
    "cluster_explanation": (
        ClusterExplanationSetup(dataset="group", n_samples=200),
        sg.ClusterExplanation,
    ),
    "unsupervised_data": (UnsupervisedDataSetup(dataset="xor"), sg.UnsupervisedData),
    "global_confounding": (
        GlobalConfoundingSetup(n=200, regressor="linear"),
        sg.GlobalConfoundingXAI,
    ),
    "local_confounding": (
        LocalConfoundingSetup(n=200, unit=2, mode="sq", regressor="linear"),
        sg.LocalConfoundingXAI,
    ),
}


def test_every_game_but_the_synthetic_ones_has_one_setup() -> None:
    games = {name for name in sg.__all__ if name not in _SYNTHETIC_GAMES}
    built = {game.__name__ for _, game in OFFLINE.values()} | {
        "ImageClassifier",  # tested below with a stand-in model
        "SentimentAnalysis",
    }
    assert built == games
    assert len(SETUPS) == len(games)
    assert set(OFFLINE) | {"image_classifier", "sentiment_analysis"} == set(SETUPS)


@pytest.mark.parametrize("name", sorted(OFFLINE))
def test_setups_build_their_game_reproducibly(name: str) -> None:
    setup, game_class = OFFLINE[name]
    game, again = setup.build(), setup.build()
    assert type(game) is game_class
    rng = np.random.default_rng(0)
    coalitions = rng.random((6, game.n_players)) < 0.5
    np.testing.assert_allclose(game(coalitions), again(coalitions), rtol=1e-12, atol=1e-12)
    assert setup_from_dict(setup.to_dict()) == setup
    assert setup_from_dict(setup.to_dict()).key == setup.key


def test_the_key_identifies_the_game() -> None:
    first = KNNSetup(dataset="xor", n_train=8, random_state=0)
    assert len(first.key) == 16
    assert first.key == KNNSetup(dataset="xor", n_train=8, random_state=0).key
    assert first.key != KNNSetup(dataset="xor", n_train=8, random_state=1).key
    # numpy scalars are stored as Python numbers
    numpy_setup = KNNSetup(dataset="xor", n_train=8, x=np.int64(0), random_state=np.int64(0))
    assert numpy_setup.key == first.key
    assert type(numpy_setup.to_dict()["random_state"]) is int
    # a different setup with equal field values is a different game
    assert first.key != WeightedKNNSetup(dataset="xor", n_train=8, random_state=0).key
    # an int in a float field is the same game as the float
    assert (
        DataValuationSetup(dataset="xor", empty_value=0).key
        == DataValuationSetup(dataset="xor").key
    )


def test_a_new_recipe_version_is_a_new_key(monkeypatch: pytest.MonkeyPatch) -> None:
    first = KNNSetup(dataset="xor", n_train=8).key
    monkeypatch.setattr(KNNSetup, "version", 2)
    assert KNNSetup(dataset="xor", n_train=8).key != first


def test_setups_are_frozen_hashable_and_round_trip() -> None:
    """Fields are stored read-only in their JSON form, so the key cannot change under the cache."""
    params = {"hidden_layer_sizes": (8, 8)}
    setup = LocalExplanationSetup(dataset="xor", model="mlp", model_params=params)
    params["hidden_layer_sizes"] = (2,)  # the caller's dict is not the setup's
    assert setup.model_params == {"hidden_layer_sizes": (8, 8)}
    with pytest.raises(TypeError, match="read-only"):
        setup.model_params["alpha"] = 1.0  # type: ignore[index]
    again = setup_from_dict(setup.to_dict())
    assert again == setup
    assert hash(again) == hash(setup)
    assert {setup: 1}[again] == 1
    assert pickle.loads(pickle.dumps(setup)) == setup  # noqa: S301
    assert EnsembleSelectionSetup(dataset="xor", members=["linear"]).members == ("linear",)


def test_unregistered_subclasses_cannot_share_their_parents_cache() -> None:
    @dataclass(frozen=True, kw_only=True)
    class ExplainAnotherPoint(PathDependentTreeSetup):
        def build(self) -> Game:
            return PathDependentTreeSetup(dataset=self.dataset, x=1).build()

    with pytest.raises(TypeError, match="not a registered setup"):
        ExplainAnotherPoint(dataset="xor")


@pytest.mark.parametrize(
    ("make", "error", "match"),
    [
        (lambda: KNNSetup(dataset="xor", dataset_params={"noise": object()}), TypeError, "JSON"),
        (lambda: KNNSetup(dataset="xor", model_params={"weights": {0: 1}}), TypeError, "strings"),
        (lambda: KNNSetup(dataset="xor", x=1.5), TypeError, "KNNSetup.x"),
        (lambda: KNNSetup(dataset="nope"), ValueError, "Unknown dataset 'nope'"),
        (lambda: KNNSetup(dataset="independentlinear60"), ValueError, "classification"),
        (lambda: UncertaintyExplanationSetup(dataset="independentlinear60"), ValueError, "class"),
        (lambda: LocalExplanationSetup(dataset="xor", model="nope"), ValueError, "Unknown model"),
        (lambda: LocalExplanationSetup(dataset="xor", imputer="foo"), ValueError, "one of"),
        (lambda: LocalExplanationSetup(dataset="xor", preset="tuned"), ValueError, "No tuned"),
        (lambda: LocalExplanationSetup(dataset="xor", imputer="tabpfn"), ValueError, "'tabpfn'"),
        (lambda: PathDependentTreeSetup(dataset="xor", model="svm"), ValueError, "one of"),
        (lambda: ProductKernelSetup(dataset="xor", model="linear"), ValueError, "one of"),
        (lambda: EnsembleSelectionSetup(dataset="xor", members=("nope",)), ValueError, "member"),
        (lambda: GlobalConfoundingSetup(regressor="nope"), ValueError, "Unknown regressor"),
        (lambda: ImageClassifierSetup(class_index="labels"), ValueError, "one of"),
        (lambda: setup_from_dict({"setup": "nope"}), ValueError, "Unknown setup 'nope'"),
    ],
)
def test_setups_are_checked_when_created(make, error: type[Exception], match: str) -> None:  # noqa: ANN001
    with pytest.raises(error, match=match):
        make()


def test_setup_names_are_unique() -> None:
    with pytest.raises(ValueError, match="already registered"):

        @dataclass(frozen=True, kw_only=True)
        class Duplicate(KNNSetup, name="knn"):
            pass


def test_background_size_is_the_imputer_sample_size() -> None:
    game = LocalExplanationSetup(dataset="breast_cancer", model="decision_tree", n_background=150)
    assert game.build().imputer.sample_size == 150
    uncertainty = UncertaintyExplanationSetup(dataset="breast_cancer", n_background=150).build()
    assert uncertainty.imputer.sample_size == 150


def test_image_classifier_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    """The image comes from Imagenette; device and batch size do not change the key."""
    calls: list[dict] = []

    class FakeImages(list):
        labels = np.array([7, 3])

    class FakeImageClassifier:
        def __init__(self, image: np.ndarray, model: str, **kwargs: object) -> None:
            calls.append({"image": image, "model": model, **kwargs})

    monkeypatch.setattr(setups._vision, "load_imagenette", lambda **_: FakeImages([1, 2]))
    monkeypatch.setattr(setups._vision, "ImageClassifier", FakeImageClassifier)
    setup = ImageClassifierSetup(index=1, model="resnet_18", class_index="label", revision="v1")
    setup.build()
    assert calls[0]["image"] == 2
    assert calls[0]["class_index"] == 3  # the label of image 1
    assert calls[0]["revision"] == "v1"
    assert ImageClassifierSetup(index=1, device="cuda", batch_size=64).key == (
        ImageClassifierSetup(index=1).key
    )
    assert setup.key != ImageClassifierSetup(index=1, model="resnet_18", class_index="label").key
    assert setup_from_dict(setup.to_dict()) == setup


def test_sentiment_analysis_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict] = []

    class FakeSentimentAnalysis:
        def __init__(self, input_text: str, **kwargs: object) -> None:
            calls.append({"input_text": input_text, **kwargs})

    monkeypatch.setattr(setups._language, "SentimentAnalysis", FakeSentimentAnalysis)
    setup = SentimentAnalysisSetup(input_text="a fine film", mask_strategy="remove", device="cpu")
    setup.build()
    assert calls == [
        {
            "input_text": "a fine film",
            "mask_strategy": "remove",
            "revision": None,
            "device": "cpu",
            "normalize": True,
        }
    ]
    assert setup.key == SentimentAnalysisSetup(input_text="a fine film", mask_strategy="remove").key
