"""Exact large-table mathematics and authenticated payoff checkpoints."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest

from shapiq.game_theory import ExactComputer
from shapiq_benchmark.materialize import exact_table_truth, prepare_families, prepare_family_chunk
from shapiq_benchmark.runner import table_game

if TYPE_CHECKING:
    from pathlib import Path

TARGETS = [
    {"index": index, "order": 1 if index == "SV" else 2}
    for index in ("SV", "SII", "k-SII", "STII", "FSII", "FBII")
]


@pytest.mark.parametrize("n", [3, 5, 8])
def test_exact_table_coefficients_match_existing_solver(n: int) -> None:
    """All six definitions and FBII's fitted intercept agree on arbitrary real payoffs."""
    values = np.random.default_rng(n).normal(size=2**n) + 3
    results = exact_table_truth(values, n, TARGETS)
    reference = ExactComputer(table_game(values, n), n_players=n)
    for target in TARGETS:
        key = target["index"], target["order"]
        expected = reference(*key)
        actual = results[key]
        # The legacy FSII regression imposes efficiency with finite big-M weights.
        tolerance = 1e-8 if target["index"] == "FSII" else 1e-12
        np.testing.assert_allclose(
            actual["values"],
            [expected[tuple(c)] for c in actual["coordinates"]],
            atol=tolerance,
            rtol=tolerance,
        )
        assert actual["baseline"] == pytest.approx(expected.baseline_value, abs=tolerance)


def test_twenty_player_exact_interactions_have_analytic_values() -> None:
    """A million-row game combines a three-player unanimity term and an additive player."""
    masks = np.arange(2**20)
    values = 3.0 + 2 * ((masks & 7) == 7) + 5 * ((masks & (1 << 19)) != 0)
    results = exact_table_truth(values, 20, TARGETS)
    singleton = {
        "SV": 2 / 3,
        "SII": 2 / 3,
        "k-SII": -1 / 3,
        "STII": 0,
        "FSII": -1 / 3,
        "FBII": -0.5,
    }
    pair = {"SII": 1, "k-SII": 1, "STII": 2 / 3, "FSII": 1, "FBII": 1}
    for (index, _), result in results.items():
        for coordinate, value in zip(result["coordinates"], result["values"], strict=True):
            expected = 0
            if len(coordinate) == 1:
                expected = 5 if coordinate == [19] else singleton[index] if coordinate[0] < 3 else 0
            elif max(coordinate) < 3:
                expected = pair[index]
            assert value == pytest.approx(expected, abs=1e-12)
        assert result["baseline"] == pytest.approx(3.25 if index == "FBII" else 3)


def test_parity_zero_and_tiny_real_shapley_signal_are_preserved() -> None:
    """No coefficient threshold hides a real small signal or creates false zero-energy noise."""
    masks = np.arange(2**20, dtype=np.uint32)
    parity = np.where(np.bitwise_count(masks) % 2, -1.0, 1.0)
    targets = [{"index": "SV", "order": 1}]
    zero = exact_table_truth(parity, 20, targets)["SV", 1]
    assert zero["energy"] == 0
    for epsilon in (1e-12, 1e-8):
        values = parity + epsilon * (masks & 1)
        expected = ((-1.0 + epsilon) + (1.0 + epsilon)) / 2
        result = exact_table_truth(values, 20, targets)["SV", 1]
        assert result["values"][0] == pytest.approx(expected, rel=1e-6, abs=1e-17)
        np.testing.assert_allclose(result["values"][1:], 0, atol=1e-17)
        assert result["energy"] > 0


def test_cached_chunks_resume_validate_and_combine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Paid oracle work is reused only for the same source, recipe and complete payload."""
    from shapiq_benchmark import materialize, runner

    calls = []

    class Game:
        n_players = 13

        def __call__(self, rows: np.ndarray) -> np.ndarray:
            calls.append(len(rows))
            return rows @ np.arange(1, 14, dtype=float)

    monkeypatch.setattr(materialize, "make_family", lambda *args, **kwargs: (Game(), {}))
    monkeypatch.setattr(runner, "provenance", lambda: {"source_sha256": "fixed-source"})
    spec = {"id": "chunked", "family": "dummy", "n_players": 13}
    first = prepare_family_chunk(spec, 0, 0, tmp_path)
    assert calls == [4096]
    assert prepare_family_chunk(spec, 0, 0, tmp_path) == first and calls == [4096]
    with pytest.raises(ValueError, match="does not match"):
        prepare_family_chunk({**spec, "dataset": "changed"}, 0, 0, tmp_path)
    games, coverage = prepare_families([spec], TARGETS, tmp_path, instance_seed=0)
    assert coverage[0]["status"] == "measured" and len(games) == 6
    assert calls.count(4096) == 2  # only the missing second chunk was evaluated
    with np.load(tmp_path / games[0]["artifact"]) as artifact:
        costs = artifact["evaluation_seconds"]
        batches = json.loads(str(artifact["evaluation_batches"]))
        assert costs.shape == artifact["values"].shape == (8192,)
        assert np.isfinite(costs).all() and np.all(costs >= 0)
        assert sum(costs) == pytest.approx(sum(batch["seconds"] for batch in batches))
    assert games[0]["metadata"]["evaluation_timing"]["protocol"] == materialize.COST_PROTOCOL
    assert games[0]["metadata"]["signal_ratio"] > 0
    with np.load(first) as saved:
        arrays = dict(saved)
    manifest = json.loads(str(arrays["manifest"]))
    for field, invalid in (
        ("start", 1),
        ("stop", 4095),
        ("seconds", -1),
        ("seconds", float("inf")),
        ("seconds", manifest["batch"]["seconds"] * 2 + 1),
    ):
        damaged = json.loads(str(arrays["manifest"]))
        damaged["batch"][field] = invalid
        np.savez_compressed(first, **{**arrays, "manifest": json.dumps(damaged)})
        with pytest.raises(ValueError, match="does not match"):
            prepare_family_chunk(spec, 0, 0, tmp_path)
    np.savez_compressed(first, **arrays)
    monkeypatch.setattr(
        runner, "provenance", lambda: {"source_sha256": "fixed-source", "numpy": "changed"}
    )
    with pytest.raises(ValueError, match="does not match"):
        prepare_family_chunk(spec, 0, 0, tmp_path)
    monkeypatch.setattr(runner, "provenance", lambda: {"source_sha256": "fixed-source"})
    arrays["values"][0] += 1
    np.savez_compressed(first, **arrays)
    with pytest.raises(ValueError, match="does not match"):
        prepare_family_chunk(spec, 0, 0, tmp_path)


def test_signal_rms_uses_full_coordinate_space_for_sparse_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Omitted zero coordinates must not increase a game's score-eligibility signal."""
    from shapiq_benchmark import materialize

    class Game:
        n_players = 4

        def __call__(self, rows: np.ndarray) -> np.ndarray:
            return rows[:, 0].astype(float)

    monkeypatch.setattr(materialize, "make_family", lambda *args, **kwargs: (Game(), {}))
    monkeypatch.setattr(
        materialize,
        "truth_dict",
        lambda truth: {"coordinates": [[0]], "values": [1.0], "baseline": 0.0, "energy": 1.0},
    )
    games, _ = prepare_families(["dummy"], [{"index": "k-SII", "order": 2}], tmp_path)
    assert games[0]["metadata"]["signal_ratio"] == pytest.approx(np.sqrt(1 / 10) / 0.5)


def test_structured_fsii_validation_avoids_finite_endpoint_reference_floor() -> None:
    """Analytic cubic utility has null players; finite-penalty regression invents tiny terms."""
    from shapiq import InteractionValues
    from shapiq_benchmark.games import validate_truth

    def oracle(coalitions: np.ndarray) -> np.ndarray:
        return 7.0 + 100 * np.all(coalitions[:, :3], axis=1)

    truth = InteractionValues(
        values={
            (0,): -100 / 6,
            (1,): -100 / 6,
            (2,): -100 / 6,
            (0, 1): 50.0,
            (0, 2): 50.0,
            (1, 2): 50.0,
        },
        index="FSII",
        min_order=1,
        max_order=2,
        n_players=8,
        estimated=False,
        baseline_value=7.0,
    )
    legacy = ExactComputer(oracle, n_players=8)("FSII", order=2)
    coordinates = (truth.dict_values.keys() | legacy.dict_values.keys()) - {()}
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            [truth[key] for key in coordinates],
            [legacy[key] for key in coordinates],
            rtol=1e-8,
            atol=1e-10,
        )
    assert validate_truth(oracle, truth, exhaustive=True) < 1e-12
    # Keep efficiency intact: the coefficient cross-check must still detect genuine errors.
    truth[(0,)] += 0.01
    truth[(1,)] -= 0.01
    with pytest.raises(AssertionError):
        validate_truth(oracle, truth, exhaustive=True)
