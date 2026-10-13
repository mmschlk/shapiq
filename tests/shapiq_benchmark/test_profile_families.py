"""Compact profiles retrain with signed targets and incomplete coalition classes."""

from __future__ import annotations

import numpy as np
import pytest

from shapiq_benchmark import campaign_models as models
from shapiq_benchmark.families import make_family


@pytest.mark.parametrize("profile", ["linear", "random_forest"])
@pytest.mark.parametrize("classification", [False, True])
def test_coalition_retraining_on_actual_target_scale(monkeypatch, profile, classification):
    rng = np.random.default_rng(41)
    x = rng.normal(size=(120, 4))
    y = np.digitize(x[:, 0], [-0.5, 0.5]) if classification else -100 + 4 * x[:, 0]
    task = "classification" if classification else "regression"
    monkeypatch.setitem(
        models.DATASETS,
        "fixture",
        {"task": task, "source": "fixture", "n_features": 4, "n_classes": 3},
    )
    monkeypatch.setattr(models, "load_raw_dataset", lambda _: (x, y, [f"f{i}" for i in range(4)]))
    for family in ("feature_selection", "dataset_valuation"):
        game, metadata = make_family(family, dataset="fixture", n_players=4, model_profile=profile)
        coalitions = ((np.arange(16)[:, None] >> np.arange(4)) & 1).astype(bool)
        scores = game(coalitions)
        assert np.isfinite(scores).all()
        assert scores[0] == 0
        np.testing.assert_allclose(scores, game(coalitions[::-1])[::-1], atol=1e-12)
        assert metadata["output_scale"] == ("accuracy" if classification else "negative MSE")
        if family == "dataset_valuation" and not classification:
            assert np.all(scores[1:] <= 0)
