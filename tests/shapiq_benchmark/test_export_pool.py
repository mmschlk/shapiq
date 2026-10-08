"""Small actual references exercise terminal export without campaign journals."""

from __future__ import annotations

import copy
import importlib.metadata
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from shapiq import ExactComputer
from shapiq_benchmark import runner
from shapiq_benchmark.duplicates import remove_aliases
from shapiq_benchmark.exact import exact_table_truth
from shapiq_benchmark.games import signal_metadata, truth_dict
from shapiq_benchmark.results_io import Checkpoint
from shapiq_benchmark.summary import weights_for
from tests.shapiq_benchmark import test_collect_pool as fixtures
from tests.shapiq_benchmark.test_collect_pool import collector, lines, save

campaign = fixtures.campaign
sys.modules["collect_pool"] = collector
spec = importlib.util.spec_from_file_location(
    "export_pool_tested", Path(__file__).resolve().parents[2] / "benchmark/export_pool.py"
)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


def table_snapshot(root, *, additive=False):
    n = 4
    bits = ((np.arange(2**n)[:, None] >> np.arange(n)) & 1).astype(bool)
    values = bits.sum(axis=1).astype(float)
    if not additive:
        values += 2 * bits[:, 0] * bits[:, 1] + 3 * bits[:, 1] * bits[:, 2] * bits[:, 3]
    np.savez(root / "values.npz", values=values, evaluation_seconds=np.zeros(len(values)))
    exact = ExactComputer(runner.table_game(values, n), n_players=n)
    games = []
    for index, order in [
        ("SV", 1),
        *((index, 2) for index in ("SII", "k-SII", "STII", "FSII", "FBII")),
    ]:
        truth = truth_dict(exact(index, order=order))
        scale = float(np.std(values, dtype=np.longdouble))
        ratio = (
            math.sqrt(truth["energy"] / sum(math.comb(n, k) for k in range(1, order + 1))) / scale
        )
        games.append(
            {
                "id": index,
                "n_players": n,
                "index": index,
                "order": order,
                "oracle": "table",
                "artifact": "values.npz",
                "truth": truth,
                "metadata": {
                    "truth_method": "exhaustive frozen table",
                    "truth_queries": 2**n,
                    "payoff_std": scale,
                    "signal_ratio": ratio,
                    "score_eligible": True,
                },
            }
        )
    return {"games": games, "suite": {"min_signal_ratio": 1e-6}}, values


def test_all_table_targets_keep_truth_and_fbii_baseline(tmp_path):
    snapshot, _ = table_snapshot(tmp_path)
    original = copy.deepcopy(snapshot)
    checks, fingerprints = exporter.check_snapshot(snapshot, tmp_path)
    assert len(checks) == len(fingerprints) == 6 and snapshot == original
    assert snapshot["games"][-1]["truth"]["baseline"] != 0
    assert next(c for c in checks if c["game_id"] == "FSII")["finite_endpoint_fsii"]
    assert next(c for c in checks if c["game_id"] == "k-SII")["implicit_zero_coordinates"] > 0


@pytest.mark.parametrize("mutation", ["coefficient", "energy", "eligibility", "cost"])
def test_invalid_reference_or_eligibility_is_rejected(tmp_path, mutation):
    snapshot, values = table_snapshot(tmp_path)
    game = snapshot["games"][0]
    if mutation == "coefficient":
        game["truth"]["values"][0] += 1
        game["truth"]["energy"] = sum(v * v for v in game["truth"]["values"])
    elif mutation == "energy":
        game["truth"]["energy"] += 1
    elif mutation == "eligibility":
        game["metadata"]["score_eligible"] = False
    else:
        np.savez(tmp_path / "values.npz", values=values, evaluation_seconds=-np.ones(len(values)))
    with pytest.raises(ValueError):
        exporter.check_snapshot(snapshot, tmp_path)


def test_fsii_floor_cannot_create_eligible_order(tmp_path):
    snapshot, values = table_snapshot(tmp_path, additive=True)
    game = next(g for g in snapshot["games"] if g["index"] == "FSII")
    pair = next(i for i, c in enumerate(game["truth"]["coordinates"]) if len(c) == 2)
    game["truth"]["values"][pair] = 1e-3
    game["truth"]["energy"] = sum(v * v for v in game["truth"]["values"])
    reference = exact_table_truth(values, 4, [{"index": "FSII", "order": 2}])["FSII", 2]
    with pytest.raises(ValueError, match="changes order eligibility"):
        exporter.table_reference(game, reference, values)


def test_entirely_weak_fsii_replays_frozen_solver():
    values = np.full(16, 7.0)
    truth = ExactComputer(runner.table_game(values, 4), n_players=4)("FSII", order=2)
    game = {
        "id": "weak",
        "index": "FSII",
        "order": 2,
        "n_players": 4,
        "truth": truth_dict(truth),
        "metadata": {"payoff_std": 0.0},
    }
    reference = exact_table_truth(values, 4, [{"index": "FSII", "order": 2}])["FSII", 2]
    check = exporter.table_reference(game, reference, values)
    assert check["weak_fsii_frozen_solver_replay"]
    game["truth"]["values"][0] += 1e-5
    game["truth"]["energy"] = sum(v * v for v in game["truth"]["values"])
    with pytest.raises(ValueError, match="frozen solver replay"):
        exporter.table_reference(game, reference, values)


@pytest.mark.parametrize("kind", ["knn", "tnn"])
def test_native_neighbors_reconstruct_actual_saved_utility(tmp_path, kind):
    from shapiq.explainer.nn import KNNExplainer, ThresholdNNExplainer
    from shapiq_benchmark.games import knn_game, tnn_game, validate_truth

    x, y, point = np.arange(16).reshape(8, 2).astype(float), np.arange(8) % 2, np.array([3.2, 4.2])
    if kind == "knn":
        model, oracle = knn_game(x, y, point, point_label=1)
        computer = KNNExplainer
    else:
        model, oracle = tnn_game(x, y, point, {"radius": 4.0, "n_jobs": 1}, 1)
        computer = ThresholdNNExplainer
    truth = computer(model, class_index=oracle.class_index).explain(point)
    if kind == "tnn":
        truth.baseline_value = truth[()]
    error = validate_truth(oracle, truth, exhaustive=True)
    np.savez(
        tmp_path / "native.npz",
        x_train=x,
        y_train=y,
        point=point,
        **({"sortperm": oracle.sortperm} if kind == "knn" else {}),
    )
    metadata = {
        "small_validation_players": 8,
        "small_validation_max_error": error,
        "model_parameters": model.get_params(),
        "point_label": 1,
        "class_index": oracle.class_index,
        "sklearn_version": importlib.metadata.version("scikit-learn"),
        **signal_metadata(truth, 1.0),
        "score_eligible": True,
    }
    game = {
        "id": kind,
        "index": "SV",
        "order": 1,
        "n_players": 8,
        "oracle": kind,
        "artifact": "native.npz",
        "truth": truth_dict(truth),
        "metadata": metadata,
    }
    snapshot = {"games": [game], "suite": {"min_signal_ratio": 1e-6}}
    checks, fingerprints = exporter.check_snapshot(snapshot, tmp_path)
    assert not fingerprints and checks[0]["exhaustive_actual_game"]
    assert checks[0]["order_eligibility"]["1"]["signal_reference"] == "full_target_rms"


def test_duplicate_ownership_is_stable_and_preserves_weight_branches():
    first = {
        "id": "a",
        "metadata": {
            "instance_seed": 0,
            "focused_design": {"application": "local", "subtype": "tree", "recipe": "rf"},
        },
    }
    second = copy.deepcopy(first)
    second["id"], second["metadata"]["instance_seed"] = "b", 1
    assert exporter.aliases_for([second, first], {"a": "same", "b": "same"}) == {"b": "a"}
    second["metadata"]["focused_design"]["recipe"] = "different"
    assert exporter.aliases_for([first, second], {"a": "same", "b": "same"}) == {}


@pytest.mark.parametrize("dimension", ["application", "subtype", "recipe", "role"])
def test_equal_payoffs_preserve_each_declared_branch(dimension):
    games = []
    for name, branch, seed in [("a", 0, 0), ("b", 0, 1), ("c", 1, 0), ("d", 1, 1)]:
        design = {"application": "local", "subtype": "generic", "recipe": "rf"}
        if branch and dimension != "role":
            design[dimension] = "another"
        games.append(
            {
                "id": name,
                "n_players": 8,
                "metadata": {
                    "instance_seed": seed,
                    "focused_design": design,
                    "game_quality": {
                        "role": "control" if branch and dimension == "role" else "core"
                    },
                },
            }
        )
    original = copy.deepcopy(games)
    aliases = exporter.aliases_for(list(reversed(games)), dict.fromkeys("abcd", "equal"))
    assert aliases == {"b": "a", "d": "c"}
    assert games == original
    # Canonical ownership cannot favor a successful later seed over a failed first seed.
    data = {
        "games": games,
        "records": [
            {"game_id": g["id"], "status": "failed" if g["id"] in "ac" else "ok"} for g in games
        ],
        "suite": {},
    }
    remove_aliases(data, aliases)
    assert [g["id"] for g in data["games"]] == ["a", "c"]
    assert all(row["status"] == "failed" for row in data["records"])
    cells, weights = weights_for(data["games"], [8], [0])
    assert dict(zip(cells, weights, strict=True)) == {("a", 8, 0): 0.5, ("c", 8, 0): 0.5}


def exportable_campaign(campaign):
    config, directory, raw, envelope, intent = campaign
    snapshot = json.loads((directory / "prepared/snapshot.json").read_text())
    artifact = directory / "prepared/payoffs.npz"
    values = np.array([i.bit_count() for i in range(16)], dtype=float)
    np.savez(artifact, values=values, evaluation_seconds=np.zeros(16))
    game = snapshot["games"][0]
    game["metadata"].update(
        truth_method="exhaustive frozen table",
        truth_queries=16,
        signal_ratio=1.0,
        score_eligible=True,
    )
    snapshot["coverage"] = [
        {
            "family": "recipe",
            "status": "measured",
            "reason": "distinctive qualification evidence",
            "game_ids": [game["id"]],
        }
    ]
    snapshot["artifacts"]["payoffs.npz"] = collector.digest(artifact)
    snapshot["snapshot_id"] = runner.identity(
        {k: v for k, v in snapshot.items() if k != "snapshot_id"}
    )
    pin = save(directory / "prepared/snapshot.json", snapshot)
    prepared = json.loads((directory / "prepared.json").read_text())
    prepared.update(snapshot_id=snapshot["snapshot_id"], snapshot_sha256=pin["sha256"])
    save(directory / "prepared.json", prepared)
    raw.update(
        snapshot_id=snapshot["snapshot_id"], games=snapshot["games"], coverage=snapshot["coverage"]
    )
    intent["expected_snapshot_id"] = snapshot["snapshot_id"]
    lines(directory / "intents.jsonl", [intent])
    Checkpoint(directory / "results", directory / "prepared", raw)
    return config, directory, raw


@pytest.mark.parametrize("interrupted", [False, True])
def test_full_export_accepts_explicit_failure_and_preserves_original_run(
    campaign, tmp_path, monkeypatch, interrupted
):
    config, directory, raw = exportable_campaign(campaign)
    monkeypatch.setattr(exporter, "terminal_accounting", collector.terminal_accounting)
    if interrupted:
        for name in (
            "responses.jsonl",
            "results/results.json",
            "results/records.jsonl",
            "outcome.json",
        ):
            (directory / name).unlink()
    path = tmp_path / "config.json"
    save(path, config)
    output, audit_path = tmp_path / "public", tmp_path / "audit.json"
    audit = exporter.export(path, output, tmp_path / "export.sqlite", audit_path)
    assert audit["outcome_complete"] and audit["score_complete"] is not interrupted
    assert audit["public_rows"] == 2 and not audit["publication_ready"]
    assert (output / "about.md").is_file()
    manifest = json.loads((output / "data.json").read_text())
    assert manifest["layout"] == "partitioned-v1" and manifest["record_count"] == 2
    if not interrupted:
        assert runner.identity(raw) in (output / manifest["assets"]["runs"][0]["file"]).read_text()
    assert json.loads(audit_path.read_text())["report_id"] == manifest["snapshot_id"]


def test_export_refuses_never_attempted_supported_cells(campaign, tmp_path):
    config, directory, _ = exportable_campaign(campaign)
    for name in (
        "intents.jsonl",
        "responses.jsonl",
        "results/results.json",
        "results/records.jsonl",
        "outcome.json",
    ):
        (directory / name).unlink()
    path = tmp_path / "config.json"
    save(path, config)
    with pytest.raises(ValueError, match="never-attempted"):
        exporter.export(
            path, tmp_path / "public", tmp_path / "export.sqlite", tmp_path / "audit.json"
        )
    assert not (tmp_path / "public").exists()


def test_output_path_aliases_are_rejected_before_inputs(tmp_path):
    with pytest.raises(ValueError, match="new separate"):
        exporter.export(
            tmp_path / "missing-config",
            tmp_path / "public",
            tmp_path / "unused/../public",
            tmp_path / "audit.json",
        )


def seal_closeout(config, tmp_path):
    closed = {
        "reason": "admitted_allocations_ended",
        "pool_directory": config["pool_directory"],
        "suite_sha256": config["suite"]["sha256"],
        "source_sha256": config["source"]["sha256"],
        "admissions": config["jobs"],
        "slurm_accounting": collector.terminal_accounting(config["jobs"]),
    }
    config["closeout"] = save(tmp_path / "closeout.json", closed)
    return closed


@pytest.mark.parametrize("missing", ["evaluation", "preparation"])
def test_authenticated_closeout_publishes_exact_missing_design(
    campaign, tmp_path, monkeypatch, missing
):
    from tests.shapiq_benchmark.test_partitioned import decoded

    config, directory, _ = exportable_campaign(campaign)
    monkeypatch.setattr(exporter, "terminal_accounting", collector.terminal_accounting)
    if missing == "evaluation":
        for name in (
            "intents.jsonl",
            "responses.jsonl",
            "results/results.json",
            "results/records.jsonl",
            "outcome.json",
        ):
            (directory / name).unlink()
    else:
        suite = json.loads(Path(config["suite"]["path"]).read_text())
        suite["game_seeds"] = [0, 1]
        config["suite"] = save(Path(config["suite"]["path"]), suite)
        tasks = json.loads(Path(config["inventory"]["path"]).read_text())
        tasks.append({**tasks[0], "case": 1, "seed": 1})
        config["inventory"] = save(Path(config["inventory"]["path"]), tasks)
        for path in (
            directory / "claim.json",
            Path(config["pool_directory"]) / "allocations/1/started.json",
        ):
            value = json.loads(path.read_text())
            value["suite_sha256"] = config["suite"]["sha256"]
            save(path, value)
    seal_closeout(config, tmp_path)
    path = tmp_path / "config.json"
    save(path, config)
    output = tmp_path / "public"
    audit = exporter.export(path, output, tmp_path / "export.sqlite", tmp_path / "audit.json")
    assert audit["accounting_complete"] and not audit["outcome_complete"]
    manifest = json.loads((output / "data.json").read_text())
    summary = manifest["campaign_coverage"]
    assert summary["closed"] and summary["closure_reason"] == "admitted_allocations_ended"
    assert (
        summary["planned_supported_cells"]
        == summary["responses"]
        + summary["unanswered_intents"]
        + summary["never_attempted_supported"]
        + summary["not_reached_supported"]
    )
    details = {
        r["id"]: r["value"] for r in decoded(output, manifest, "details") if r["type"] == "report"
    }
    assert details["preparation_coverage"][0]["reason"] == "distinctive qualification evidence"
    assert len(details["coverage"]) == summary["intended_instances"]
    if missing == "evaluation":
        assert summary["never_attempted_supported"] == 1 and manifest["record_count"] == 1
        assert manifest["catalog"]["planned_cells"] - manifest["record_count"] == 1
    else:
        assert summary["unprepared_instances"] == summary["not_reached_supported"] == 1
        assert manifest["game_count"] == 1 and manifest["record_count"] == 2
        assert details["coverage"][1]["preparation_status"] == "unclaimed"


@pytest.mark.parametrize(
    "fault", ["reason", "admissions", "slurm_accounting", "hash", "intent_tail"]
)
def test_closeout_cannot_override_integrity_or_claim_budget_exhaustion(campaign, tmp_path, fault):
    config, directory, _ = exportable_campaign(campaign)
    closed = seal_closeout(config, tmp_path)
    if fault == "reason":
        closed["reason"] = "budget_exhausted"
    elif fault in {"admissions", "slurm_accounting"}:
        closed[fault] = {}
    elif fault == "hash":
        config["closeout"]["sha256"] = "0" * 64
    else:
        with (directory / "intents.jsonl").open("ab") as stream:
            stream.write(b'{"partial":')
    if fault in {"reason", "admissions", "slurm_accounting"}:
        config["closeout"] = save(tmp_path / "closeout.json", closed)
    path = tmp_path / "config.json"
    save(path, config)
    with pytest.raises(ValueError):
        exporter.export(
            path, tmp_path / "public", tmp_path / "export.sqlite", tmp_path / "audit.json"
        )
    assert not (tmp_path / "public").exists()
