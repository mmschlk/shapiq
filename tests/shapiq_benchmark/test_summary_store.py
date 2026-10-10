"""Disk summaries preserve complete-panel arithmetic without reading full raw rows."""

from __future__ import annotations

import copy
import json
import math
from typing import TYPE_CHECKING

import pytest

from shapiq_benchmark.record_store import RecordStore
from shapiq_benchmark.summary import iter_summaries, summarize
from tests.shapiq_benchmark.test_summary import fixture_data

if TYPE_CHECKING:
    from pathlib import Path


def mixed_panel() -> dict:
    """Exercise target, model/dataset, controls, partial and zero-truth projections."""
    games: list[dict] = [
        {
            "family": "local" if i < 8 else "global",
            "stratum": str(i // 4),
            "n_players": 11 + i % 2,
            "index": "SV" if i < 8 else "SII",
            "order": 1 if i < 8 else 2,
            "metadata": {
                "instance_seed": i % 4,
                "dataset": "a" if i < 8 else "b",
                "model_profile": "rf" if i < 4 else "xgb",
                "game_quality": {"role": "control" if i == 11 else "core"},
                "order_scores": {
                    "1": {"score_eligible": True},
                    "2": {"score_eligible": i != 10},
                },
            },
        }
        for i in range(12)
    ]
    data = fixture_data(
        games, {"KernelSHAP": [1.0] * 12, "SVARM": [2.0] * 12, "OddSHAP": [3.0] * 12}
    )
    data["snapshot_id"] = "exact-panel"
    data["suite"].update(relative_budgets=[0.5, 1, 2], min_signal_ratio=1e-6)
    data["suite"]["budgets_by_game"] = {
        game["id"]: [math.ceil(r * game["n_players"]) for r in [0.5, 1, 2]]
        for game in data["games"]
    }
    records = []
    for position, source_row in enumerate(data["records"]):
        for budget in data["suite"]["budgets_by_game"][source_row["game_id"]]:
            if source_row["method"] == "OddSHAP" and position % 5 == 0:
                continue
            row = {
                **source_row,
                "budget": budget,
                "nmse": (position % 7) / (budget + 1),
                "worker": {"irrelevant_large_diagnostics": "x" * 4096},
                "order_scores": {"1": {"nmse": position / 11}, "2": {"nmse": position / 17}},
            }
            if row["method"] == "OddSHAP" and position % 3 == 0:
                row.update(status="failed", nmse=None)
            elif row["method"] == "SVARM" and position % 7 == 0:
                row.update(status="unsupported", nmse=None)
            records.append(row)
    data["records"] = records
    data["suite"]["budgets"] = sorted({row["budget"] for row in records})
    # Unknown methods still contribute global row-derived zero flags, exactly as
    # in the legacy summary; the complete game's metadata handles pending cases.
    records.append({**records[0], "method": "unknown", "zero_truth_energy": "truthy"})
    data["games"][1]["metadata"]["zero_truth_energy"] = True
    data["games"][2]["metadata"]["score_eligible"] = False
    return data


@pytest.mark.parametrize("degree", [None, 1, 2])
@pytest.mark.parametrize("controls", [False, True])
def test_disk_presets_match_legacy_with_all_filters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, degree: int | None, *, controls: bool
) -> None:
    """Every score, weight, ID, CI, Elo pool and historical value remains exact."""
    data = mixed_panel()
    before = copy.deepcopy(data)
    options: dict = {"bootstrap_draws": 4, "score_order": degree, "include_controls": controls}
    expected = summarize(data, **options)
    with RecordStore(tmp_path / "scores.sqlite") as store:
        store.extend(reversed(data["records"]))

        def forbidden(*_args, **_kwargs):
            message = "full row materialization forbidden"
            raise AssertionError(message)

        monkeypatch.setattr(RecordStore, "select", forbidden)
        monkeypatch.setattr(RecordStore, "__iter__", forbidden)
        actual = list(iter_summaries({**data, "records": store}, **options))
        assert json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)
        assert summarize({**data, "records": store}, **options) == actual
    assert data == before


def test_default_200_draw_intervals_and_history_match(tmp_path: Path) -> None:
    """Default uncertainty is computed on the full synchronized panel, not batch averages."""
    data = fixture_data(
        [{"family": "a", "stratum": "a", "metadata": {"cluster_id": str(i)}} for i in range(4)],
        {"KernelSHAP": [0.1, 2, 0.3, 4], "SVARM": [1, 0.2, 3, 0.4]},
    )
    expected = summarize(data)
    assert expected[0]["uncertainty"]["draws"] == 200
    assert expected[0]["common_panel"]["cells"] == 8
    with RecordStore(tmp_path / "scores.sqlite") as store:
        store.extend(data["records"])
        actual = list(iter_summaries({**data, "records": store}))
    assert json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)


def test_measurement_projection_exact_order_missing_and_interleaving(tmp_path: Path) -> None:
    """JSON projection preserves signed zero/null and never joins another partition."""
    rows = [
        {"game_id": "a", "method": "x", "budget": 1, "seed": 0, "status": "ok", "nmse": -0.0},
        {"game_id": "b", "method": "x", "budget": 2, "seed": 1, "status": "failed", "nmse": None},
        {"game_id": "a", "method": "y", "budget": 1, "seed": 0, "status": "ok", "nmse": 1},
    ]
    cells = [("b", 2, 1), ("a", 1, 0), ("missing", 1, 0)]
    with RecordStore(tmp_path / "scores.sqlite") as store:
        store.extend(rows)
        other = store.fork()
        other.extend([{**rows[0], "nmse": 100}])
        first = store.measurements(cells, ["x", "y"])
        second = store.measurements(cells[::-1], ["y", "never"])
        method, values = next(first)
        assert method == "x" and values[2] is None
        assert values[0] is not None and values[1] is not None
        assert values[0]["nmse"] is None
        assert json.dumps(values[1]["nmse"]) == "-0.0"
        assert next(second)[0] == "y"
        first.close()
        assert list(second) == [("never", [None, None, None])]
        other_value = next(other.measurements(cells, ["x"]))[1][1]
        assert other_value is not None and other_value["nmse"] == 100


def test_legacy_duplicate_guard_includes_hidden_rows() -> None:
    """Filtering controls/order must not hide duplicate input keys."""
    data = mixed_panel()
    hidden = next(row for row in data["records"] if row["game_id"] == "11")
    data["records"].append(hidden.copy())
    with pytest.raises(ValueError, match="duplicate result cells"):
        summarize(data, score_order=1)
