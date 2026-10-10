"""Focused weights are fixed by qualified applications, never method outcomes."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from shapiq_benchmark import prepare
from shapiq_benchmark.report import METADATA_FIELDS
from shapiq_benchmark.summary import bootstrap, comparisons, weights_for


def games():
    """Unbalanced recipes, constructions and budgets must still balance applications."""
    return [
        {
            "id": f"{recipe}-{seed}",
            "family": "same",
            "stratum": "same",
            "metadata": {
                "dataset": "shared",
                "instance_seed": seed,
                "focused_design": {"application": app, "subtype": subtype, "recipe": recipe},
            },
        }
        for app, subtype, recipe in [
            ("local", "", "l1"),
            ("local", "", "l2"),
            ("features", "", "f1"),
            ("data", "groups", "d1"),
            ("data", "neighbors", "d2"),
            ("data", "neighbors", "d3"),
        ]
        for seed in range(4)
    ]


@pytest.mark.parametrize("neighbors", [True, False])
def test_weights_browser_bootstrap_and_elo(neighbors):
    """No neighbor interaction reference redistributes within data, not across applications."""
    panel = [
        g for g in games() if neighbors or g["metadata"]["focused_design"]["subtype"] != "neighbors"
    ]
    budgets = {g["id"]: [12, 24] if g["id"].startswith("l") else [12] for g in panel}
    cells, weights = weights_for(panel, budgets, [0])
    lookup = {g["id"]: g for g in panel}
    masses = {}
    for (name, _, _), mass in zip(cells, weights, strict=True):
        masses[name] = masses.get(name, 0) + mass
    for app in ("local", "data", "features"):
        assert sum(
            masses[g["id"]] for g in panel if g["metadata"]["focused_design"]["application"] == app
        ) == pytest.approx(1 / 3)
    assert sum(
        masses[g["id"]] for g in panel if g["metadata"]["focused_design"]["subtype"] == "groups"
    ) == pytest.approx(1 / 6 if neighbors else 1 / 3)

    # Scores vary only by application: every paired bootstrap draw must retain the same mean.
    values = np.array(
        [
            [
                {"local": 1, "data": 4, "features": 7}[
                    lookup[cell[0]]["metadata"]["focused_design"]["application"]
                ]
                for cell in cells
            ]
        ]
    )
    intervals, status = bootstrap(
        panel, budgets, [0], cells, values, ["a"], draws=20, include_elo=False
    )
    assert status["available"]
    assert intervals["a"]["mean"] == pytest.approx([4, 4])
    # Pairwise Elo overlap keeps original mass even when a competitor misses an application.
    other = values[0].astype(float) + 1
    other[[lookup[c[0]]["metadata"]["focused_design"]["application"] == "data" for c in cells]] = (
        np.nan
    )
    matches, _ = comparisons(np.array([values[0], other]), weights, ["a", "b"])
    assert matches[0]["observed_weight"] == pytest.approx(2 / 3)

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required to check browser weighting")
    query = Path(__file__).resolve().parents[2] / "benchmark/site/query.js"
    output = subprocess.run(
        [
            node,
            "-e",
            "require(process.argv[1]); console.log(JSON.stringify(Object.fromEntries(BenchmarkQuery.gameWeights(JSON.parse(process.argv[2])))));",
            str(query),
            json.dumps(panel),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(output.stdout) == pytest.approx(masses)


def test_snapshot_attaches_exportable_focused_identity(tmp_path, monkeypatch):
    """Both native and table target IDs receive recipe identity before snapshot hashing."""
    monkeypatch.setattr(prepare, "provenance", dict)
    (tmp_path / "game.npz").write_bytes(b"fixture")
    recipe = {"id": "local-01", "application": "local", "subtype": ""}
    suite = {"focused_design": {"recipes": [recipe]}}
    panel = [
        {"id": name, "artifact": "game.npz", "n_players": 12}
        for name in ("local-01-i0-sv", "local-01-sv-i1")
    ]
    snapshot = prepare.write_snapshot(suite, panel, tmp_path)
    expected = {"application": "local", "subtype": "", "recipe": "local-01"}
    assert all(g["metadata"]["focused_design"] == expected for g in snapshot["games"])
    assert "focused_design" in METADATA_FIELDS
    with pytest.raises(ValueError, match="missing focused recipe"):
        prepare.write_snapshot(suite, [{**panel[0], "id": "unknown"}], tmp_path)


def test_mixed_weighting_fails_closed():
    panel = games()
    panel[0]["metadata"].pop("focused_design")
    with pytest.raises(ValueError, match="mix focused and legacy"):
        weights_for(panel, [12], [0])
