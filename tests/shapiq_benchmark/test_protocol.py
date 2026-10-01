"""The phased protocol cannot silently broaden a run or change its experiment controls."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from shapiq_benchmark.datasets import DATASETS
from shapiq_benchmark.protocol import (
    BUDGET_MULTIPLIERS,
    CONSTRUCTIONS,
    DATASET_PHASES,
    FIRST_DATASETS,
    NEXT_DATASETS,
    build_phase,
    main,
    phase_candidates,
    protocol_manifest,
)


@pytest.fixture
def base_suite() -> dict:
    """Read the existing estimator/target catalog rather than a second hand-maintained list."""
    path = Path(__file__).resolve().parents[2] / "benchmark/suites/all-families.json"
    return json.loads(path.read_text())


def test_phase_two_is_exactly_the_agreed_64_games(base_suite: dict) -> None:
    """Every dataset/model/construction has four instances; phase planning never mutates its base."""
    original = copy.deepcopy(base_suite)
    suite = build_phase(2, base_suite)
    assert base_suite == original
    assert suite == build_phase(2, base_suite)
    assert len(suite["families"]) * len(suite["game_seeds"]) == 64
    assert len({row["id"] for row in suite["families"]}) == 16
    expected = {
        (dataset, model, family, 12)
        for dataset in FIRST_DATASETS
        for model in ("random_forest", "xgboost")
        for family in ("local_baseline", "local_marginal")
    }
    assert {
        (row["dataset"], row["model_profile"], row["family"], row["n_players"])
        for row in suite["families"]
    } == expected
    assert all(row["model_profile"] in row["id"] for row in suite["families"])
    assert suite["games"] == []
    assert suite["phase_plan"]["counts"] == {"selected": 16, "planned": 0, "excluded": 0}
    for key in ("methods", "targets", "relative_budgets", "game_seeds", "seeds"):
        assert suite[key] == base_suite[key]
    assert suite["relative_budgets"] == list(BUDGET_MULTIPLIERS)


@pytest.mark.parametrize("phase", [3, 4, 5, 6, 7])
def test_unimplemented_phases_cannot_be_scheduled_as_if_ready(phase: int, base_suite: dict) -> None:
    """Future inventory is honest about pending model/output qualification."""
    suite = build_phase(phase, base_suite)
    assert not suite["families"] and not suite["games"]
    assert suite["phase_plan"]["counts"]["planned"] > 0
    assert suite["phase_plan"]["counts"]["selected"] == 0
    candidates = suite["phase_plan"]["candidates"]
    assert len({row["id"] for row in candidates}) == len(candidates)
    assert all(row["n_players"] >= 11 for row in candidates)
    if phase != 7:
        assert all(row["n_players"] <= 20 for row in candidates)
        initial = {row["id"] for row in phase_candidates(2)}
        assert initial <= {row["id"] for row in candidates}
    else:
        assert all(row["n_players"] > 20 and row["reference"] == "structured" for row in candidates)


def test_full_dataset_inventory_matches_registered_loader_expansion() -> None:
    """All 63 loader identities survive the phased rollout; historical Wine is untouched."""
    assert len(DATASET_PHASES) == 63
    assert set(DATASET_PHASES) == set(DATASETS) - {"wine"}
    assert sum(name.startswith("tabarena_") for name in DATASET_PHASES) == 51
    assert {name for name, phase in DATASET_PHASES.items() if phase == 2} == set(FIRST_DATASETS)
    assert {name for name, phase in DATASET_PHASES.items() if phase == 3} == set(NEXT_DATASETS)
    assert DATASETS["wine"]["task"] == "classification"
    assert DATASETS["wine_quality"]["task"] == "regression"
    assert {row["dataset"] for row in phase_candidates(6)} - {None} == set(DATASET_PHASES)


def test_filter_understands_player_units_and_unsupported_task_pairs() -> None:
    """Feature limits must not exclude row players, or admit invented padded columns."""
    rows = {
        (row["family"], row["dataset"], row["model_profile"], row["n_players"]): row
        for row in phase_candidates(6)
    }
    assert rows["local_baseline", "iris", "random_forest", 11]["status"] == "excluded"
    assert rows["data_valuation", "iris", "random_forest", 11]["status"] == "planned"
    assert rows["knn", "wine_quality", "knn", 11]["reason"] == "Requires class labels."
    assert rows["product_kernel", "digits", "rbf_svm", 11]["status"] == "excluded"
    assert rows["local_gaussian", "mushroom", "random_forest", 11]["status"] == "excluded"
    assert rows["tabpfn", "wine_quality", "tabpfn", 11]["status"] == "planned"
    assert "mlp" not in CONSTRUCTIONS["interventional_tree"]["models"]
    assert CONSTRUCTIONS["product_kernel"]["models"] == ["rbf_svm"]
    assert rows["image", None, "vit", 16]["status"] == "planned"


def test_manifest_explains_models_sources_budgets_and_actual_run_distinction() -> None:
    """Public protocol metadata contains parameter/source provenance without claiming completion."""
    manifest = protocol_manifest(2)
    models = {row["id"]: row for row in manifest["models"]}
    assert models["random_forest"]["parameters"]["n_estimators"] == 100
    assert models["xgboost"]["parameters"]["max_depth"] == 8
    assert models["mlp"]["status"] == "planned"
    assert models["gaussian_process"]["status"] == "planned"
    assert manifest["training"]["fit_rows_max"] == 5000
    assert "ceil" in manifest["budget_rule"]
    assert manifest["minimum_players"] == 11
    assert manifest["maximum_enumerated_players"] == 20
    assert all(row["source_url"].startswith("https://") for row in manifest["constructions"])
    assert all(row["source"] for row in manifest["datasets"])
    manifest["models"][0]["parameters"]["n_estimators"] = 1
    assert protocol_manifest(2)["models"][0]["parameters"]["n_estimators"] == 100


@pytest.mark.parametrize("phase", [0, 1, 8, True, "2"])
def test_invalid_phase_rejected(phase: object) -> None:
    """A typo must not silently select a different experiment cohort."""
    with pytest.raises(ValueError, match="Phase must"):
        protocol_manifest(phase)


@pytest.mark.parametrize(
    "key,value",
    [("relative_budgets", [2, 4, 64]), ("seeds", [0, 1]), ("game_seeds", [0]), ("min_players", 8)],
)
def test_protocol_drift_rejected(base_suite: dict, key: str, value: object) -> None:
    """Explicit protocol controls cannot drift with a legacy suite or accidental edit."""
    base_suite[key] = value
    with pytest.raises(ValueError, match="Base suite disagrees"):
        build_phase(2, base_suite)


def test_cli_uses_explicit_base_and_never_overwrites_it(
    base_suite: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The generated frozen manifest is separate from the existing suite file."""
    base = tmp_path / "base.json"
    base.write_text(json.dumps(base_suite))
    output = tmp_path / "phase2.json"
    monkeypatch.setattr(
        "sys.argv", ["protocol", "--phase", "2", "--base", str(base), "--output", str(output)]
    )
    main()
    assert json.loads(output.read_text()) == build_phase(2, base_suite)
    monkeypatch.setattr(
        "sys.argv", ["protocol", "--phase", "2", "--base", str(base), "--output", str(base)]
    )
    with pytest.raises(SystemExit):
        main()
