"""Semantic tests for the machine learning game families."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from shapiq_games import (
    ClusterExplanation,
    DatasetValuation,
    DataValuation,
    EnsembleSelection,
    FeatureSelection,
    GlobalConfoundingXAI,
    GlobalExplanation,
    ImageClassifier,
    LocalExplanation,
    RandomForestEnsembleSelection,
    SentimentAnalysis,
    UncertaintyExplanation,
    UnsupervisedData,
)
from shapiq_games.unsupervised import total_correlation
from tests.shapiq_games.helpers import (
    FakeSentimentPipeline,
    mean_brightness_classifier,
    skip_if_no_skimage,
)


@pytest.fixture(scope="module")
def data() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 4))
    y_class = (x[:, 0] + 0.5 * x[:, 1] > 0).astype(int)
    y_reg = x[:, 0] - 2 * x[:, 2]
    return {
        "x_train": x[:300],
        "x_test": x[300:],
        "yc_train": y_class[:300],
        "yc_test": y_class[300:],
        "yr_train": y_reg[:300],
        "yr_test": y_reg[300:],
    }


def test_local_explanation_full_coalition_is_the_prediction(data) -> None:
    model = DecisionTreeRegressor(max_depth=4, random_state=0).fit(
        data["x_train"], data["yr_train"]
    )
    game = LocalExplanation(model, data["x_train"], x=data["x_test"][0], normalize=False)
    assert game.class_index is None
    assert game(game.grand_coalition)[0] == pytest.approx(model.predict(data["x_test"][:1])[0])
    assert game(game.empty_coalition)[0] == pytest.approx(game.empty_prediction_value)


def test_local_explanation_classifier_and_callable(data) -> None:
    model = DecisionTreeClassifier(max_depth=3, random_state=0).fit(
        data["x_train"], data["yc_train"]
    )
    game = LocalExplanation(model, data["x_train"], x=5, normalize=False)
    assert game.class_index == 1
    assert game(game.grand_coalition)[0] == pytest.approx(
        model.predict_proba(data["x_train"][5:6])[0, 1]
    )
    game = LocalExplanation(lambda z: z.sum(axis=1), data["x_train"], x=0, normalize=False)
    assert game.class_index is None
    assert game(game.grand_coalition)[0] == pytest.approx(data["x_train"][0].sum())
    with pytest.raises(ValueError, match="Unknown imputer"):
        LocalExplanation(model, data["x_train"], imputer="nearest")  # type: ignore[arg-type]
    with pytest.raises(IndexError, match="out of range"):
        LocalExplanation(model, data["x_train"], x=10_000)


def test_global_explanation_explains_loss_reduction(data) -> None:
    model = DecisionTreeRegressor(max_depth=4, random_state=0).fit(
        data["x_train"], data["yr_train"]
    )
    game = GlobalExplanation(model, data["x_test"], n_samples=50, normalize=False)
    assert game(game.grand_coalition)[0] == pytest.approx(0.0)  # no loss with all features
    assert game(game.empty_coalition)[0] < 0.0
    # the model never splits on feature 3, so it explains nothing on its own
    assert 3 not in model.tree_.feature
    single = game(np.eye(4, dtype=bool)) - game(game.empty_coalition)[0]
    assert single[3] == pytest.approx(0.0, abs=1e-12)
    assert single[0] > 0 and single[2] > 0


def test_feature_selection_empty_and_full_coalition(data) -> None:
    model = DecisionTreeClassifier(max_depth=3, random_state=0)
    game = FeatureSelection(
        model, data["x_train"], data["yc_train"], data["x_test"], data["yc_test"],
        task="classification", normalize=False,
    )  # fmt: skip
    majority = max(data["yc_train"].mean(), 1 - data["yc_train"].mean())
    majority_label = int(data["yc_train"].mean() > 0.5)
    assert game(game.empty_coalition)[0] == pytest.approx(
        np.mean(data["yc_test"] == majority_label)
    )
    assert 0.0 < majority < 1.0
    fitted = DecisionTreeClassifier(max_depth=3, random_state=0).fit(
        data["x_train"], data["yc_train"]
    )
    assert game(game.grand_coalition)[0] == pytest.approx(
        fitted.score(data["x_test"], data["yc_test"])
    )


def test_valuation_accuracy_uses_labels_not_column_indices(data) -> None:
    """A training subset with one class predicts that class (regression test of the old games)."""
    model = DecisionTreeClassifier(random_state=0)
    game = DataValuation(
        model, data["x_train"][:10], data["yc_train"][:10], data["x_test"], data["yc_test"],
        task="classification", normalize=False,
    )  # fmt: skip
    ones = np.flatnonzero(data["yc_train"][:10] == 1)
    coalition = np.zeros((1, 10), dtype=bool)
    coalition[0, ones] = True
    assert game(coalition)[0] == pytest.approx(np.mean(data["yc_test"] == 1))
    assert game(game.empty_coalition)[0] == 0.0


def test_dataset_valuation_groups(data) -> None:
    model = DecisionTreeRegressor(random_state=0)
    game = DatasetValuation(
        model, data["x_train"], data["yr_train"], data["x_test"], data["yr_test"],
        task="regression", n_players=4, player_sizes="increasing", random_state=0,
    )  # fmt: skip
    sizes = [group.size for group in game.groups]
    assert sum(sizes) == 300
    assert sizes == sorted(sizes)
    assert np.unique(np.concatenate(game.groups)).size == 300  # disjoint and complete
    explicit = DatasetValuation(
        model, data["x_train"], data["yr_train"], data["x_test"], data["yr_test"],
        task="regression", groups=[np.arange(100), np.arange(100, 300)],
    )  # fmt: skip
    assert explicit.n_players == 2


def test_ensemble_selection_single_members_and_votes(data) -> None:
    members = [
        DecisionTreeClassifier(max_depth=depth, random_state=0).fit(
            data["x_train"], data["yc_train"]
        )
        for depth in (1, 2, 3)
    ]
    game = EnsembleSelection(
        members, data["x_test"], data["yc_test"], task="classification", normalize=False
    )
    values = game(np.eye(3, dtype=bool))
    for member, value in zip(members, values, strict=True):
        assert value == pytest.approx(member.score(data["x_test"], data["yc_test"]))
    assert game.player_name_lookup["0_DecisionTreeClassifier"] == 0
    regression = EnsembleSelection(
        [LinearRegression().fit(data["x_train"], data["yr_train"])] * 2,
        data["x_test"], data["yr_test"], task="regression", normalize=False,
    )  # fmt: skip
    assert regression(regression.grand_coalition)[0] == pytest.approx(1.0)
    with pytest.raises(TypeError, match="random forest"):
        RandomForestEnsembleSelection.from_forest(
            LinearRegression(), data["x_test"], data["yr_test"], task="regression"
        )


def test_uncertainty_decomposition(data) -> None:
    forest = RandomForestClassifier(n_estimators=5, random_state=0).fit(
        data["x_train"], data["yc_train"]
    )
    values = {
        kind: UncertaintyExplanation(
            forest, data["x_train"], x=1, uncertainty=kind, normalize=False
        )
        for kind in ("total", "aleatoric", "epistemic")
    }
    full = {kind: game(game.grand_coalition)[0] for kind, game in values.items()}
    assert full["total"] == pytest.approx(full["aleatoric"] + full["epistemic"])
    with pytest.raises(TypeError, match="RandomForestClassifier"):
        UncertaintyExplanation(
            DecisionTreeClassifier().fit(data["x_train"], data["yc_train"]), data["x_train"]
        )


def test_clustering_and_unsupervised_games(data) -> None:
    game = ClusterExplanation(data["x_train"], n_clusters=2, normalize=False)
    assert game(game.empty_coalition)[0] == 0.0
    assert game(game.grand_coalition)[0] > 0.0
    with pytest.raises(ValueError, match="method"):
        ClusterExplanation(data["x_train"], method="dbscan")  # type: ignore[arg-type]

    duplicated = np.column_stack(
        [data["x_train"][:, 0], data["x_train"][:, 0], data["x_train"][:, 1]]
    )
    game = UnsupervisedData(duplicated, n_bins=5)
    np.testing.assert_allclose(game(np.eye(3, dtype=bool)), 0.0)  # single features
    assert (
        game(np.array([[1, 1, 0]], dtype=bool))[0] > 0.0
    )  # duplicated features share all information
    discrete = np.array([[0, 0], [1, 1], [0, 0], [1, 1]])
    assert total_correlation(discrete) == pytest.approx(np.log(2))


def test_confounding_game_with_a_linear_regressor() -> None:
    game = GlobalConfoundingXAI.from_config(n=300, regressor=LinearRegression)
    assert game.fingerprint is None  # custom regressors are not fingerprinted
    # adjusting for every covariate reproduces the reference effect
    assert game(game.grand_coalition)[0] == pytest.approx(0.0, abs=1e-10)
    naive = game.Y[game.A == 1].mean() - game.Y[game.A == 0].mean()
    assert game(game.empty_coalition)[0] == pytest.approx(game.tau_hat.mean() - naive)


@skip_if_no_skimage
def test_image_classifier_superpixels_cover_every_player() -> None:
    image = np.random.default_rng(0).integers(0, 255, (60, 60, 3), dtype=np.uint8)
    game = ImageClassifier(
        image, model=mean_brightness_classifier, n_superpixels=9, normalize=False
    )
    assert game.superpixels is not None
    assert set(np.unique(game.superpixels)) == set(range(1, game.n_players + 1))
    expected = mean_brightness_classifier(image[None])[0, game.class_index]
    assert game(game.grand_coalition)[0] == pytest.approx(expected)
    grayscale = ImageClassifier(image[..., 0], model=mean_brightness_classifier, n_superpixels=4)
    assert grayscale.image.shape == (60, 60, 3)
    assert game.model_commit is None
    with pytest.raises(ValueError, match="vision transformer"):
        ImageClassifier(image, model=mean_brightness_classifier, revision="main")


def test_image_classifier_forwards_and_records_the_vit_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The revision reaches the Hugging Face loader and the loaded commit enters the config."""
    calls: list[dict] = []

    class FakeViT:
        def __init__(self, image: np.ndarray, n_players: int, **kwargs: object) -> None:
            calls.append(kwargs)
            self.n_players = n_players
            self.class_index = 3
            self.model_commit = f"commit-of-{kwargs['revision']}"

        def __call__(self, coalitions: np.ndarray) -> np.ndarray:
            return coalitions.mean(axis=1)

    import shapiq_games.vision.image_classifier as module

    monkeypatch.setattr(module, "ViTPatchModel", FakeViT)
    monkeypatch.setattr(module, "load_example_image", lambda _: np.zeros((8, 8, 3), np.uint8))
    game = ImageClassifier.from_config(image=0, model="vit_16_patches", revision="v1")
    other = ImageClassifier.from_config(image=0, model="vit_16_patches", revision="v2")
    assert calls[0]["revision"] == "v1"
    assert game.model_commit == "commit-of-v1"
    assert game.config is not None
    assert game.config["model_commit"] == "commit-of-v1"
    assert game.fingerprint != other.fingerprint


def test_sentiment_analysis_with_a_fake_pipeline() -> None:
    pipeline = FakeSentimentPipeline()
    game = SentimentAnalysis("good good bad plot", classifier=pipeline, normalize=False)
    assert game.n_players == 4
    assert game.original_model_output == pytest.approx(0.6)
    # masking the two 'good' tokens leaves one 'bad': a negative score
    assert game(np.array([[0, 0, 1, 1]], dtype=bool))[0] == pytest.approx(-0.6)
    removed = SentimentAnalysis(
        "good bad", classifier=pipeline, mask_strategy="remove", normalize=False
    )
    assert removed(np.array([[1, 0]], dtype=bool))[0] == pytest.approx(0.6)
    with pytest.raises(ValueError, match="mask_strategy"):
        SentimentAnalysis("good", classifier=pipeline, mask_strategy="drop")  # type: ignore[arg-type]
    assert game.model_commit is None  # a pipeline without a Hugging Face model


def test_sentiment_analysis_with_other_labels_and_tokenizers() -> None:
    labelled = FakeSentimentPipeline(labels=("LABEL_1", "LABEL_0"))
    game = SentimentAnalysis(
        "good good bad", classifier=labelled, positive_label="LABEL_1", normalize=False
    )
    assert game.original_model_output == pytest.approx(0.6)
    # special tokens are left out of the players, whatever the tokenizer adds
    assert game.n_players == 3

    no_mask = FakeSentimentPipeline()
    no_mask.tokenizer.mask_token_id = None  # type: ignore[assignment]
    with pytest.raises(ValueError, match="no mask token"):
        SentimentAnalysis("good", classifier=no_mask)
    removed = SentimentAnalysis("good bad", classifier=no_mask, mask_strategy="remove")
    assert removed.n_players == 2
