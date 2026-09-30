"""Matrix selection respects player meaning and qualifies no unseen large-game semantics."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from shapiq_benchmark.families import DATASETS
from shapiq_benchmark.matrix import REASON_LABELS, expand_matrix, main


@pytest.fixture
def matrix_inputs() -> tuple[dict, dict]:
    directory = Path(__file__).resolve().parents[2] / "benchmark/suites"
    config = json.loads((directory / "matrix.json").read_text())
    base = json.loads((directory / config["base_suite"]).read_text())
    return config, base


def test_selected_matrix_keeps_real_dimensions_and_base_protocol(matrix_inputs: tuple) -> None:
    """All generated tables have feasible native dimensions and unchanged experiment controls."""
    config, base = matrix_inputs
    original = copy.deepcopy(base)
    suite = expand_matrix(config, base)
    assert base == original
    assert suite == expand_matrix(config, base)
    for key in (
        "methods",
        "relative_budgets",
        "seeds",
        "game_seeds",
        "targets",
        "games",
        "min_players",
    ):
        assert suite[key] == base[key]
    selected = [
        row for row in suite["matrix_coverage"]["candidates"] if row["status"] == "selected"
    ]
    assert len(selected) == 196
    assert len({(row["recipe"], row["dataset"], row["n_players"]) for row in selected}) == 196
    for row in selected:
        assert row["n_players"] in (11, 12)
        if row["player_unit"] == "feature":
            assert row["n_players"] <= DATASETS[row["dataset"]]["n_features"]
    assert suite["matrix_coverage"]["counts"]["game_definitions"] == 5084
    assert all(row["reason"] in REASON_LABELS for row in suite["matrix_coverage"]["candidates"])
    assert "base_suite" not in suite["matrix_definition"]


def test_feature_width_is_not_training_row_count_and_large_adapters_stay_distinct(
    matrix_inputs: tuple,
) -> None:
    """Small-feature datasets still admit row-player games; a different exact oracle is not substituted."""
    config, base = matrix_inputs
    suite = expand_matrix(config, base)
    rows = {
        (row["recipe"], row["dataset"], row["n_players"]): row
        for row in suite["matrix_coverage"]["candidates"]
    }
    assert rows["local_baseline", "iris", 11]["reason"] == "insufficient_features"
    assert rows["data_valuation", "iris", 11]["status"] == "selected"
    assert ("data_valuation", "iris", 4) not in rows
    assert rows["local_baseline", "iris", 4]["reason"] == "below_minimum"
    assert rows["interventional_tree", "breast_cancer", 30]["reason"] == "unqualified_large_adapter"
    assert rows["product_kernel", "digits", 64]["status"] == "excluded"
    assert rows["product_kernel", "digits", 64]["reason"] == "requires_binary_target"
    assert rows["local_gaussian", "bike_sharing", 11]["reason"] == "binary_calendar_features"
    assert rows["tabpfn", "diabetes", 11]["reason"] == "requires_class_labels"
    structured = suite["matrix_coverage"]["structured"]
    assert any(
        row["oracle"] == "tree" and row["dataset"] == "breast_cancer" and row["n_players"] == 30
        for row in structured
    )
    assert any(
        row["oracle"] == "product_kernel" and row["dataset"] == "digits" and row["n_players"] == 64
        for row in structured
    )
    assert all(
        (row["index"], row["order"]) == ("SV", 1)
        for row in structured
        if row["oracle"] in ("knn", "product_kernel")
    )
    assert all(row["recipe"].startswith("structured_") for row in structured)


def test_cli_resolves_base_relative_to_config(
    matrix_inputs: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A command launched elsewhere still expands the authenticated config's base suite."""
    config, base = matrix_inputs
    (tmp_path / config["base_suite"]).write_text(json.dumps(base))
    config_path = tmp_path / "matrix.json"
    config_path.write_text(json.dumps(config))
    output = tmp_path / "output/suite.json"
    monkeypatch.setattr(
        "sys.argv", ["matrix", "--config", str(config_path), "--output", str(output)]
    )
    monkeypatch.chdir(tmp_path.parent)
    main()
    assert json.loads(output.read_text()) == expand_matrix(config, base)
    monkeypatch.setattr(
        "sys.argv", ["matrix", "--config", str(config_path), "--output", str(config_path)]
    )
    with pytest.raises(SystemExit):
        main()


@pytest.mark.parametrize(
    "change",
    [
        {"unexpected": "/private/path"},
        {"datasets": ["wine", "wine"]},
        {"datasets": ["unknown"]},
        {"families": ["image"]},
        {"player_counts": [11, True]},
        {"player_counts": [0]},
        {"include_native_width": "yes"},
    ],
)
def test_invalid_matrix_config_is_rejected(matrix_inputs: tuple, change: dict) -> None:
    """Public inventory accepts only the declared registry and simple typed configuration."""
    config, base = matrix_inputs
    with pytest.raises(ValueError, match="Matrix"):
        expand_matrix({**config, **change}, base)


def test_unqualified_structured_target_is_not_relabelled(matrix_inputs: tuple) -> None:
    """An SV-only adapter cannot silently appear as qualified interaction ground truth."""
    config, base = matrix_inputs
    game = next(game for game in base["games"] if game["oracle"] == "knn")
    game.update(index="k-SII", order=2)
    with pytest.raises(ValueError, match="no qualified dataset/target adapter"):
        expand_matrix(config, base)


def test_structured_counts_match_actual_adapter_semantics(matrix_inputs: tuple) -> None:
    """KNN defaults to 128 rows; tree/kernel adapters cannot silently select feature subsets."""
    config, base = matrix_inputs
    knn = next(game for game in base["games"] if game["oracle"] == "knn")
    del knn["n_players"]
    suite = expand_matrix(config, base)
    row = next(row for row in suite["matrix_coverage"]["structured"] if row["id"] == knn["id"])
    assert row["n_players"] == 128
    tree = next(game for game in base["games"] if game["oracle"] == "tree")
    tree["n_players"] = 11
    with pytest.raises(ValueError, match="require the native feature width"):
        expand_matrix(config, base)
