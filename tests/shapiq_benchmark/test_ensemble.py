"""Signed combinations, dataset isolation and error geometry on known vectors."""

from __future__ import annotations

import numpy as np
import pytest

from shapiq_benchmark.ensemble import aligned_block, fit_panel


def panels() -> list[dict]:
    """Two biased estimates whose signed combination recovers every reference."""
    result = []
    for dataset in range(6):
        for seed in range(2):
            y = np.array([1.0, 2.0 + dataset, -1.0 - seed])
            error = np.array([2.0 + seed, -1.0, dataset + 1.0])
            result.append(
                {
                    "id": f"{dataset}/{seed}",
                    "dataset": str(dataset),
                    "truth": y,
                    "estimates": np.column_stack((y + error, y + 2 * error)),
                    "weight": 1.0,
                }
            )
    return result


def test_signed_recovery_and_dataset_folds() -> None:
    result = fit_panel(["a", "b"], panels())
    np.testing.assert_allclose(result["coefficients"], [2, -1], atol=1e-4)
    assert result["train"]["nmse"] < 1e-8
    assert result["held_out"]["nmse"] < 1e-8
    assert result["held_out"]["r2"] == pytest.approx(1)
    assert result["equal_weights"]["nmse"] > 0.1
    held_out = [dataset for fold in result["folds"] for dataset in fold["held_out_datasets"]]
    assert sorted(held_out) == [str(i) for i in range(6)]
    assert result["pca"]["explained_variance"][0] == pytest.approx(1)


def test_alignment_keeps_extra_estimates_and_rejects_missing_methods() -> None:
    game = {
        "id": "g",
        "n_players": 3,
        "order": 1,
        "metadata": {"dataset": "d"},
        "truth": {"coordinates": [[], [0]], "values": [99, 1]},
    }
    rows = [
        {
            "game_id": "g",
            "budget": 8,
            "seed": 0,
            "method": method,
            "status": "ok",
            "estimate": {"coordinates": [[], [0], [1]], "values": [55, 1, extra]},
        }
        for method, extra in [("b", -2), ("a", 3)]
    ]
    block = aligned_block(game, rows, ["a", "b"], 1)
    assert block["truth"] == [1, 0, 0]
    assert block["estimates"] == [[1, 1], [3, -2], [0, 0]]
    assert block["dataset"] == "d"
    with pytest.raises(ValueError, match="complete"):
        aligned_block(game, rows[:1], ["a", "b"], 1)


def test_scaling_and_weighted_nmse() -> None:
    blocks = panels()
    for i, block in enumerate(blocks):
        block["weight"] = i + 1
    result = fit_panel(["a", "b"], blocks)
    expected = sum(
        block["weight"]
        * np.sum((block["estimates"].mean(axis=1) - block["truth"]) ** 2)
        / np.sum(block["truth"] ** 2)
        for block in blocks
    ) / sum(block["weight"] for block in blocks)
    assert result["equal_weights"]["nmse"] == pytest.approx(expected)
    for i, block in enumerate(blocks):
        block["truth"] *= (i + 1) * 10
        block["estimates"] *= (i + 1) * 10
    scaled = fit_panel(["a", "b"], blocks)
    np.testing.assert_allclose(scaled["coefficients"], result["coefficients"], atol=1e-12)
    assert scaled["equal_weights"]["nmse"] == pytest.approx(expected)


def test_identical_estimators_have_identical_pca_positions() -> None:
    blocks = panels()
    for block in blocks:
        block["estimates"][:, 1] = block["estimates"][:, 0]
    result = fit_panel(["a", "b"], blocks)
    assert result["rank"] == 1
    assert result["coefficients"][0] == pytest.approx(result["coefficients"][1])
    assert result["pca"]["explained_variance"] == [0.0, 0.0]
    assert all(point["x"] == point["y"] == 0 for point in result["pca"]["points"])


def test_large_finite_weights_do_not_overflow() -> None:
    blocks = panels()
    expected = fit_panel(["a", "b"], blocks)
    for block in blocks:
        block["weight"] = 1e308
    actual = fit_panel(["a", "b"], blocks)
    assert actual["equal_weights"] == expected["equal_weights"]
    np.testing.assert_allclose(actual["coefficients"], expected["coefficients"])
    with pytest.raises(ValueError, match="folds"):
        fit_panel(["a", "b"], blocks, folds=2.5)


@pytest.mark.parametrize("problem", ["duplicate", "zero", "missing", "one_dataset"])
def test_reject_invalid_panels(problem: str) -> None:
    blocks = panels()
    if problem == "duplicate":
        blocks.append(blocks[0])
    elif problem == "zero":
        blocks[0]["truth"][:] = 0
    elif problem == "missing":
        blocks[0]["estimates"][0, 0] = np.nan
    else:
        for block in blocks:
            block["dataset"] = "same"
    with pytest.raises(ValueError):
        fit_panel(["a", "b"], blocks)


def test_ridge_matches_augmented_least_squares_with_variable_block_sizes() -> None:
    blocks = panels()
    for i, block in enumerate(blocks):
        block["weight"] = i + 1
        block["truth"] = block["truth"][: 1 + i % 3]
        block["estimates"] = block["estimates"][: 1 + i % 3]
    penalty = 0.03
    result = fit_panel(["a", "b"], blocks, penalties=[penalty])
    total = sum(block["weight"] for block in blocks)
    x, y = [], []
    for block in blocks:
        scale = np.sqrt(block["weight"] / total) / np.linalg.norm(block["truth"])
        x.append(block["estimates"] * scale)
        y.append(block["truth"] * scale)
    expected = np.linalg.lstsq(
        np.vstack([*x, np.sqrt(penalty) * np.eye(2)]),
        np.concatenate([*y, np.zeros(2)]),
        rcond=None,
    )[0]
    np.testing.assert_allclose(result["coefficients"], expected, atol=1e-12)


def test_near_duplicate_columns_do_not_create_huge_cancelling_weights() -> None:
    blocks = panels()
    for block in blocks:
        y = block["truth"]
        error = block["estimates"][:, 0] - y
        block["estimates"] = np.column_stack((y + error, y + (1 + 1e-7) * error))
    result = fit_panel(["a", "b"], blocks)
    assert max(abs(a) for a in result["coefficients"]) < 1
    assert np.isfinite(result["held_out"]["nmse"])


def test_outer_heldout_truth_cannot_change_its_fit_or_penalty() -> None:
    blocks = panels()
    result = fit_panel(["a", "b"], blocks)
    first = result["folds"][0]
    for block in blocks:
        if block["dataset"] in first["held_out_datasets"]:
            block["truth"] *= -100
    altered = fit_panel(["a", "b"], blocks)["folds"][0]
    assert altered["penalty"] == first["penalty"]
    np.testing.assert_array_equal(altered["coefficients"], first["coefficients"])


@pytest.mark.parametrize("penalties", [[], [0], [-1], [np.nan], [np.inf]])
def test_reject_invalid_ridge_penalties(penalties: list) -> None:
    with pytest.raises(ValueError, match="penalties"):
        fit_panel(["a", "b"], panels(), penalties=penalties)


def test_reject_outer_folds_with_only_one_training_dataset() -> None:
    with pytest.raises(ValueError, match="two training datasets"):
        fit_panel(["a", "b"], panels()[:6], folds=2)
