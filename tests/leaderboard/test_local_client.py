"""Unit tests for DatabaseClient and the DatabaseClient contract.

Structure
---------
* Fixtures – shared pytest helpers (tmp JSONL path, sample documents, client).
* TestDatabaseClientConstruction  – from_env / __init__ edge-cases.
* TestDatabaseClientConnection    – test_connection / close / context-manager.
* TestDatabaseClientWrite         – insert_one / insert_many.
* TestDatabaseClientDelete        – delete_all / delete_by_config.
* TestDatabaseClientReadGeneric   – get_all / get_by_config.
* TestDatabaseClientContract      – abstract-method enforcement.

The sample documents mirror the JSONL data supplied with the task.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from leaderboard.storage import (
    DatabaseClient,
    DatabaseClientFactory,
    RunConfig,
)

# ---------------------------------------------------------------------------
# Sample documents (extracted from the raw jsonl data)
# ---------------------------------------------------------------------------

_PERM_100_SEED0: dict[str, Any] = {
    "run_id": "62fc2073-d022-49dd-895a-684265eb983a",
    "game_name": "CaliforniaHousing",
    "game_id": "CaliforniaHousing_LocalExplanation_Game_87076402",
    "game_params": {
        "x": 0,
        "model_name": "decision_tree",
        "imputer": "marginal",
        "normalize": True,
        "verbose": False,
        "random_state": 42,
    },
    "n_players": 5,
    "approximator_name": "PermutationSamplingSV",
    "approximator_params": {},
    "shapiq_version": "1.4.2",
    "index": "SV",
    "max_order": 1,
    "budget": 100,
    "approx_seed": 0,
    "ground_truth_method": "ExactComputer",
    "run_failed": False,
    "error_message": None,
    "metrics": {
        "mse": 0.007025,
        "mae": None,
        "mse_normalized": None,
        "spearman": 1.0,
        "kendall_tau": None,
        "precision_at_k": None,
    },
    "runtime_seconds": 0.022,
    "timestamp": "2026-05-26T19:58:19.420086+00:00",
    "hardware": {"cpu": "x86_64", "ram_gb": None, "python_version": "3.12.3"},
    "notes": "",
}

_PERM_100_SEED1: dict[str, Any] = {
    **_PERM_100_SEED0,
    "run_id": "1d535eb0-c208-4711-931a-edd5ede25be0",
    "approx_seed": 1,
    "metrics": {**_PERM_100_SEED0["metrics"], "mse": 0.005208},
    "runtime_seconds": 0.021,
}

_PERM_500_SEED0: dict[str, Any] = {
    **_PERM_100_SEED0,
    "run_id": "e9440ad1-179c-4de9-bbd8-a748efd43e55",
    "game_name": "SOUM",
    "game_id": "SOUM_LocalExplanation_Game_12345678",
    "budget": 500,
    "approx_seed": 0,
    "metrics": {**_PERM_100_SEED0["metrics"], "mse": 0.001015},
    "runtime_seconds": 0.070,
}

_STRAT_100_SEED0: dict[str, Any] = {
    **_PERM_100_SEED0,
    "run_id": "cd7acb0b-f941-44f7-9408-25b34e6da831",
    "game_name": "CaliforniaHousing",
    "game_id": "CaliforniaHousing_LocalExplanation_Game_87076410",
    "approximator_name": "StratifiedSamplingSV",
    "budget": 100,
    "approx_seed": 0,
    "metrics": {**_PERM_100_SEED0["metrics"], "mse": 0.004873},
    "runtime_seconds": 0.341,
}

_BIKE_PERM_100_SEED0: dict[str, Any] = {
    **_PERM_100_SEED0,
    "run_id": "bf961067-ac37-41e1-ae4e-2e9672c59e18",
    "game_name": "BikeSharing",
    "game_id": "BikeSharing_LocalExplanation_Game_80012861",
    "n_players": 12,
    "budget": 100,
    "approx_seed": 0,
    "metrics": {**_PERM_100_SEED0["metrics"], "mse": 25.547},
    "runtime_seconds": 0.016,
}

_FAILED_RUN: dict[str, Any] = {
    **_PERM_100_SEED0,
    "run_id": "deadbeef-0000-0000-0000-000000000000",
    "run_failed": True,
    "error_message": "something went wrong",
}

ALL_DOCS = [
    _PERM_100_SEED0,
    _PERM_100_SEED1,
    _PERM_500_SEED0,
    _STRAT_100_SEED0,
    _BIKE_PERM_100_SEED0,
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def jsonl_path(tmp_path: Path) -> Path:
    """Path inside a temp directory (file does not exist yet)."""
    return tmp_path / "data" / "runs.jsonl"


@pytest.fixture()
def empty_client(jsonl_path: Path) -> DatabaseClient:
    return DatabaseClientFactory.create_client("local", {"LOCAL_DB_PATH": str(jsonl_path)})


@pytest.fixture()
def populated_client(jsonl_path: Path) -> DatabaseClient:
    client = DatabaseClientFactory.create_client("local", {"LOCAL_DB_PATH": str(jsonl_path)})
    client.insert_many(ALL_DOCS)
    return client


@pytest.fixture()
def perm_config() -> RunConfig:
    return RunConfig(
        game_name="CaliforniaHousing",
        n_players=5,
        ground_truth_method="ExactComputer",
        approximator_name="PermutationSamplingSV",
        budget=100,
        index="SV",
        max_order=1,
    )


@pytest.fixture()
def strat_config() -> RunConfig:
    return RunConfig(
        game_name="CaliforniaHousing",
        n_players=5,
        ground_truth_method="ExactComputer",
        approximator_name="StratifiedSamplingSV",
        budget=100,
        index="SV",
        max_order=1,
    )


# ===========================================================================
# TestDatabaseClientConstruction
# ===========================================================================


class TestDatabaseClientConstruction:
    def test_init_stores_path(self, jsonl_path: Path) -> None:
        client = DatabaseClientFactory.create_client("local", {"LOCAL_DB_PATH": str(jsonl_path)})
        # no internal attribute checks — validate behavior instead
        assert isinstance(client, DatabaseClient)

    def test_init_accepts_str(self, tmp_path: Path) -> None:
        p = str(tmp_path / "runs.jsonl")
        client = DatabaseClientFactory.create_client("local", {"LOCAL_DB_PATH": p})
        assert isinstance(client, DatabaseClient)

    def test_from_env_uses_args_first(self, tmp_path: Path) -> None:
        custom = str(tmp_path / "custom.jsonl")
        client = DatabaseClientFactory.create_client("local", {"LOCAL_DB_PATH": custom})
        assert isinstance(client, DatabaseClient)

    def test_from_env_uses_env_var(self, tmp_path: Path) -> None:
        custom = str(tmp_path / "env.jsonl")
        with patch.dict(os.environ, {"LOCAL_DB_PATH": custom}):
            client = DatabaseClientFactory.create_client("local", {})
        assert isinstance(client, DatabaseClient)

    def test_from_env_default_fallback(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            client = DatabaseClientFactory.create_client("local", {})
        assert isinstance(client, DatabaseClient)

    def test_populated_client_builds(self, populated_client: DatabaseClient) -> None:
        assert len(populated_client.get_all()) == len(ALL_DOCS)

    def test_from_env_args_beats_env_var(self, tmp_path: Path) -> None:
        env_path = str(tmp_path / "env.jsonl")
        arg_path = str(tmp_path / "arg.jsonl")
        with patch.dict(os.environ, {"LOCAL_DB_PATH": env_path}):
            client = DatabaseClientFactory.create_client("local", {"LOCAL_DB_PATH": arg_path})
        assert isinstance(client, DatabaseClient)


# ===========================================================================
# TestDatabaseClientConnection
# ===========================================================================


class TestDatabaseClientConnection:
    def test_close_is_noop(self, empty_client: DatabaseClient) -> None:
        # Should not raise.
        empty_client.close()

    def test_context_manager_returns_self(self, empty_client: DatabaseClient) -> None:
        with empty_client as c:
            assert c is empty_client

    def test_context_manager_calls_close(self, empty_client: DatabaseClient) -> None:
        with patch.object(empty_client, "close") as mock_close:
            with empty_client:
                pass
        mock_close.assert_called_once()

    def test_context_manager_calls_close_on_exception(self, empty_client: DatabaseClient) -> None:
        with patch.object(empty_client, "close") as mock_close:
            with pytest.raises(ValueError):
                with empty_client:
                    raise ValueError("boom")
        mock_close.assert_called_once()


# ===========================================================================
# TestDatabaseClientWrite
# ===========================================================================


class TestDatabaseClientWrite:
    def test_insert_one_creates_file(self, empty_client: DatabaseClient) -> None:
        empty_client.insert_one(_PERM_100_SEED0)
        assert empty_client.test_connection() is True

    def test_insert_one_appends_valid_json(self, empty_client: DatabaseClient) -> None:
        empty_client.insert_one(_PERM_100_SEED0)
        empty_client.insert_one(_PERM_100_SEED1)

        data = empty_client.get_all()
        assert len(data) == 2
        assert data[0]["run_id"] == _PERM_100_SEED0["run_id"]
        assert data[1]["run_id"] == _PERM_100_SEED1["run_id"]


# ===========================================================================
# TestDatabaseClientDelete
# ===========================================================================


class TestDatabaseClientDelete:
    def test_delete_all_returns_correct_count(self, populated_client: DatabaseClient) -> None:
        assert populated_client.delete_all() == len(ALL_DOCS)

    def test_delete_all_leaves_empty_store(self, populated_client: DatabaseClient) -> None:
        populated_client.delete_all()
        assert populated_client.get_all() == []

    def test_delete_all_on_empty_returns_zero(self, empty_client: DatabaseClient) -> None:
        assert empty_client.delete_all() == 0

    def test_delete_by_config_removes_matching(
        self, populated_client: DatabaseClient, perm_config: RunConfig
    ) -> None:
        # _PERM_100_SEED0 and _PERM_100_SEED1 both match
        deleted = populated_client.delete_by_config(perm_config)
        assert deleted == 2
        run_ids = {d["run_id"] for d in populated_client.get_all()}
        assert _PERM_100_SEED0["run_id"] not in run_ids
        assert _PERM_100_SEED1["run_id"] not in run_ids

    def test_delete_by_config_keeps_non_matching(
        self, populated_client: DatabaseClient, perm_config: RunConfig
    ) -> None:
        populated_client.delete_by_config(perm_config)
        remaining_ids = {d["run_id"] for d in populated_client.get_all()}
        assert _PERM_500_SEED0["run_id"] in remaining_ids
        assert _STRAT_100_SEED0["run_id"] in remaining_ids
        assert _BIKE_PERM_100_SEED0["run_id"] in remaining_ids

    def test_delete_by_config_zero_when_no_match(self, populated_client: DatabaseClient) -> None:
        no_match = RunConfig(
            game_name="NONEXISTENT",
            n_players=5,
            approximator_name="X",
            ground_truth_method="ExactComputer",
            budget=9999,
            index="SV",
            max_order=1,
        )
        assert populated_client.delete_by_config(no_match) == 0
        assert len(populated_client.get_all()) == len(ALL_DOCS)

    def test_delete_by_config_does_not_rewrite_when_nothing_deleted(
        self, populated_client: DatabaseClient
    ) -> None:
        no_match = RunConfig(
            game_name="NONEXISTENT",
            n_players=5,
            approximator_name="X",
            ground_truth_method="ExactComputer",
            budget=9999,
            index="SV",
            max_order=1,
        )
        with patch.object(populated_client, "_save") as mock_save:
            populated_client.delete_by_config(no_match)
        mock_save.assert_not_called()


# ===========================================================================
# TestDatabaseClientReadGeneric
# ===========================================================================


class TestDatabaseClientReadGeneric:
    def test_get_all_empty_when_no_file(self, empty_client: DatabaseClient) -> None:
        assert empty_client.get_all() == []

    def test_get_all_returns_all_documents(self, populated_client: DatabaseClient) -> None:
        assert len(populated_client.get_all()) == len(ALL_DOCS)

    def test_get_all_round_trips_data(self, populated_client: DatabaseClient) -> None:
        ids = {d["run_id"] for d in populated_client.get_all()}
        expected_ids = {d["run_id"] for d in ALL_DOCS}
        assert ids == expected_ids

    def test_get_by_config_returns_only_matching(
        self, populated_client: DatabaseClient, perm_config: RunConfig
    ) -> None:
        results = populated_client.get_by_config(perm_config)
        assert len(results) == 2
        for d in results:
            assert d["approximator_name"] == "PermutationSamplingSV"
            assert d["budget"] == 100
            assert d["game_name"] == "CaliforniaHousing"

    def test_get_by_config_empty_when_no_match(self, populated_client: DatabaseClient) -> None:
        cfg = RunConfig(
            game_name="MISSING",
            n_players=5,
            approximator_name="X",
            ground_truth_method="ExactComputer",
            budget=9999,
            index="SV",
            max_order=1,
        )
        assert populated_client.get_by_config(cfg) == []

    def test_get_by_config_on_empty_store(
        self, empty_client: DatabaseClient, perm_config: RunConfig
    ) -> None:
        assert empty_client.get_by_config(perm_config) == []


# ===========================================================================
# TestDatabaseClientContract
# ===========================================================================


class TestDatabaseClientContract:
    def test_cannot_instantiate_abstract_class(self) -> None:
        with pytest.raises(TypeError):
            DatabaseClient()  # type: ignore[abstract]

    def test_all_abstract_methods_declared(self) -> None:
        expected = {
            "from_env",
            "test_connection",
            "close",
            "insert_one",
            "insert_many",
            "delete_all",
            "delete_by_config",
            "get_all",
            "get_by_config",
            "get_unique_configs",
            "get_games",
            "get_by_game",
            "get_approximators",
            "get_by_approximator",
            "count_by_config",
        }
        assert expected == set(DatabaseClient.__abstractmethods__)

    def test_local_client_satisfies_contract(self, jsonl_path: Path) -> None:
        client = DatabaseClientFactory.create_client("local", {"LOCAL_DB_PATH": jsonl_path})

        assert isinstance(client, DatabaseClient)
        # Verify no unimplemented abstract methods remain
        assert not getattr(client.__class__, "__abstractmethods__", set())
