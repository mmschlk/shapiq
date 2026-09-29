"""Media replicates change explanation inputs while retaining shared model clusters."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pytest

from shapiq_benchmark.media import make_extra


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
