"""Media replicates change explanation inputs while retaining shared model clusters."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from shapiq_benchmark.families import _dataset, feature_subset
from shapiq_benchmark.media import make_extra


@pytest.mark.parametrize(
    ("dataset", "n_players"),
    [
        (dataset, count)
        for dataset, counts in (
            ("wine", (11, 12, 13)),
            ("breast_cancer", (11, 12, 16, 20)),
            ("digits", (11, 12, 16, 20, None)),
            ("iris", (None,)),
        )
        for count in counts
    ],
)
def test_tabpfn_feature_selection_uses_actual_training_rows(
    monkeypatch: pytest.MonkeyPatch, dataset: str, n_players: int | None
) -> None:
    """All matrix instances admit every singleton, retaining original columns and legacy recipes."""

    def imputer(
        model: object, x_train: np.ndarray, *args: object, **kwargs: object
    ) -> SimpleNamespace:
        return SimpleNamespace(x_train=x_train, n_players=x_train.shape[1], fit=lambda point: None)

    monkeypatch.setitem(
        sys.modules, "tabpfn", SimpleNamespace(TabPFNClassifier=lambda **kwargs: None)
    )
    monkeypatch.setattr("shapiq.imputer.tabpfn_imputer.TabPFNImputer", imputer)
    for seed in range(4):
        game, metadata = make_extra(
            "tabpfn", dataset=dataset, n_players=n_players, instance_seed=seed
        )
        x, _, train, _, names = _dataset(dataset, seed)
        features = metadata["feature_indices"]
        assert game.n_players == (n_players or x.shape[1])
        assert metadata["train_indices"] == train[:64].tolist()
        assert metadata["feature_names"] == [str(names[i]) for i in features]
        np.testing.assert_array_equal(game.x_train, x[train[:64]][:, features])
        if n_players is not None:
            assert np.all(np.ptp(game.x_train, axis=0) > 0)
        if dataset == "digits" and n_players is not None:
            assert metadata["parameters"]["feature_rule"] == (
                "seeded subset of columns nonconstant on the 64 TabPFN training rows"
            )
        else:
            np.testing.assert_array_equal(features, feature_subset(x, n_players, seed))
            assert metadata["parameters"] == {"n_estimators": 1, "device": "cpu", "class_index": 1}


def test_text_instances_use_distinct_inputs_and_shared_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A different seed must select another text rather than repeat the same prediction game."""
    inputs = []

    def factory(text: str, **kwargs: object) -> SimpleNamespace:
        inputs.append(text)
        config = SimpleNamespace(_commit_hash="test-revision")
        return SimpleNamespace(
            n_players=8, _classifier=SimpleNamespace(model=SimpleNamespace(config=config))
        )

    monkeypatch.setattr(
        "shapiq_games.benchmark.local_xai.benchmark_language.SentimentAnalysis", factory
    )
    metadata = [make_extra("text", instance_seed=seed)[1] for seed in range(4)]
    assert len(set(inputs)) == 4
    assert len({m["cluster_id"] for m in metadata}) == 1
    assert len({m["input_id"] for m in metadata}) == 4
    assert {m["instance_seed"] for m in metadata} == set(range(4))


def test_image_instances_use_distinct_shipped_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Image replicas select distinct files and preserve the verified null-player removal."""
    inputs = []

    class Image:
        n_players = 9

        def __init__(self, *, x_explain_path: str, **kwargs: object) -> None:
            inputs.append(x_explain_path)
            self.model_function = SimpleNamespace(
                superpixels=np.arange(1, 9).reshape(2, 4), batch_size=0
            )

        def __call__(self, rows: np.ndarray) -> np.ndarray:
            return rows[:, :8].sum(axis=1)

    monkeypatch.setattr("shapiq_games.benchmark.local_xai.benchmark_image.ImageClassifier", Image)
    games = [make_extra("image", instance_seed=seed) for seed in range(4)]
    assert len(set(inputs)) == 4
    assert len({metadata["data_sha256"] for _, metadata in games}) == 4
    assert len({metadata["cluster_id"] for _, metadata in games}) == 1
    assert all(
        game.n_players == 8 and game.game.model_function.batch_size == 1 for game, _ in games
    )


@pytest.mark.parametrize("n_players", [11, 12])
def test_larger_images_keep_only_nonempty_segments(
    monkeypatch: pytest.MonkeyPatch, n_players: int
) -> None:
    """Requested player counts refer to active segments, never the legacy null slot."""

    class Image:
        def __init__(self, *, n_superpixel_resnet: int, **kwargs: object) -> None:
            self.n_players = n_superpixel_resnet
            self.model_function = SimpleNamespace(
                superpixels=np.arange(1, self.n_players), batch_size=0
            )

        def __call__(self, rows: np.ndarray) -> np.ndarray:
            return rows[:, :-1].sum(axis=1)

    monkeypatch.setattr("shapiq_games.benchmark.local_xai.benchmark_image.ImageClassifier", Image)
    for seed in range(4):
        game, metadata = make_extra("image", instance_seed=seed, n_players=n_players)
        assert game.n_players == n_players
        assert metadata["parameters"]["requested_superpixels"] == n_players + 1
        np.testing.assert_array_equal(game(np.eye(n_players, dtype=bool)), np.ones(n_players))


@pytest.mark.parametrize("name", ["causal_global", "causal_local"])
@pytest.mark.parametrize("n_players", [11, 12])
def test_larger_causal_games_use_requested_covariates(
    monkeypatch: pytest.MonkeyPatch, name: str, n_players: int
) -> None:
    """The actual SCM dimensions and local inputs agree with the declared players."""

    def factory(*, n: int, d: int, seed: int, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            X=np.random.default_rng(seed).normal(size=(n, d)),
            A=np.arange(n) % 2,
            Y=np.arange(n, dtype=float),
            tau_hat=np.ones(n),
            n_players=d,
        )

    def local(*, X: np.ndarray, x_i: np.ndarray, **kwargs: object) -> SimpleNamespace:
        assert x_i.shape == (X.shape[1],)
        return SimpleNamespace(n_players=X.shape[1])

    monkeypatch.setattr("shapiq_games.benchmark.causal_xai.benchmark.CurthVDS", factory)
    monkeypatch.setattr("shapiq_games.benchmark.causal_xai.base.LocalConfoundingXAI", local)
    for seed in range(4):
        game, metadata = make_extra(name, instance_seed=seed, n_players=n_players)
        assert game.n_players == n_players
        assert metadata["parameters"]["d"] == n_players
        assert metadata["parameters"]["seed"] == seed


def test_requested_text_players_must_match_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unexpected tokenizer counts fail preparation rather than adding dummy players."""
    config = SimpleNamespace(_commit_hash="test-revision")
    monkeypatch.setattr(
        "shapiq_games.benchmark.local_xai.benchmark_language.SentimentAnalysis",
        lambda *args, **kwargs: SimpleNamespace(
            n_players=10, _classifier=SimpleNamespace(model=SimpleNamespace(config=config))
        ),
    )
    with pytest.raises(ValueError, match="Requested 11 active players"):
        make_extra("text", n_players=11)
