"""Unit tests for LocalClient and the DatabaseClient contract.

Structure
---------
* Fixtures – shared pytest helpers (tmp JSONL path, sample documents, client).
* TestLocalClientConstruction  – from_env / __init__ edge-cases.
* TestLocalClientConnection    – test_connection / close / context-manager.
* TestLocalClientWrite         – insert_one / insert_many.
* TestLocalClientDelete        – delete_all / delete_by_config.
* TestLocalClientReadGeneric   – get_all / get_by_config.
* TestLocalClientReadDomain    – get_unique_configs / get_games / get_by_game /
                                  get_approximators / get_by_approximator /
                                  count_by_config.
* TestLocalClientLoadDataframe – load_dataframe integration with _process.
* TestDatabaseClientContract   – abstract-method enforcement.
* TestHelperFunctions          – _json_default / _matches_config unit tests.

The sample documents mirror the JSONL data supplied with the task.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from leaderboard.storage import (
    LocalClient,
    _json_default,
    _matches_config,
    DatabaseClient,
    RunConfig,
)

# ---------------------------------------------------------------------------
# Sample documents (representative subset from the supplied JSONL)
# ---------------------------------------------------------------------------

_PERM_100_SEED0: dict[str, Any] = {
    "run_id": "62fc2073-d022-49dd-895a-684265eb983a",
    "game_name": "CaliforniaHousing",
    "game_id": "CaliforniaHousing_LocalExplanation_Game_87076402",
    "game_params": {
        "x": 0, "model_name": "decision_tree", "imputer": "marginal",
        "normalize": True, "verbose": False, "random_state": 42,
    },
    "n_players": 8,
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
        "mse": 0.007025, "mae": None, "mse_normalized": None,
        "spearman": 1.0, "kendall_tau": None, "precision_at_k": None,
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
    "game_id": "CaliforniaHousing_LocalExplanation_Game_87076400",
    "budget": 500,
    "approx_seed": 0,
    "metrics": {**_PERM_100_SEED0["metrics"], "mse": 0.001015},
    "runtime_seconds": 0.070,
}

_STRAT_100_SEED0: dict[str, Any] = {
    **_PERM_100_SEED0,
    "run_id": "cd7acb0b-f941-44f7-9408-25b34e6da831",
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
def empty_client(jsonl_path: Path) -> LocalClient:
    return LocalClient(path=jsonl_path)


@pytest.fixture()
def populated_client(jsonl_path: Path) -> LocalClient:
    client = LocalClient(path=jsonl_path)
    client.insert_many(ALL_DOCS)
    return client


@pytest.fixture()
def perm_config() -> RunConfig:
    return RunConfig(
        game_name="CaliforniaHousing",
        approximator_name="PermutationSamplingSV",
        budget=100,
        index="SV",
        max_order=1,
    )


@pytest.fixture()
def strat_config() -> RunConfig:
    return RunConfig(
        game_name="CaliforniaHousing",
        approximator_name="StratifiedSamplingSV",
        budget=100,
        index="SV",
        max_order=1,
    )


# ===========================================================================
# TestLocalClientConstruction
# ===========================================================================


class TestLocalClientConstruction:
    def test_init_stores_path(self, jsonl_path: Path) -> None:
        client = LocalClient(path=jsonl_path)
        assert client._path == jsonl_path

    def test_init_accepts_str(self, tmp_path: Path) -> None:
        p = str(tmp_path / "runs.jsonl")
        client = LocalClient(path=p)
        assert client._path == Path(p)

    def test_from_env_uses_args_first(self, tmp_path: Path) -> None:
        custom = str(tmp_path / "custom.jsonl")
        client = LocalClient.from_env({"LOCAL_DB_PATH": custom})
        assert client._path == Path(custom)

    def test_from_env_uses_env_var(self, tmp_path: Path) -> None:
        custom = str(tmp_path / "env.jsonl")
        with patch.dict(os.environ, {"LOCAL_DB_PATH": custom}):
            client = LocalClient.from_env({})
        assert client._path == Path(custom)

    def test_from_env_default_fallback(self) -> None:
        env = {k: v for k, v in os.environ.items() if k != "LOCAL_DB_PATH"}
        with patch.dict(os.environ, env, clear=True):
            client = LocalClient.from_env({})
        assert client._path == Path("data/runs.jsonl")

    def test_from_env_args_beats_env_var(self, tmp_path: Path) -> None:
        env_path = str(tmp_path / "env.jsonl")
        arg_path = str(tmp_path / "arg.jsonl")
        with patch.dict(os.environ, {"LOCAL_DB_PATH": env_path}):
            client = LocalClient.from_env({"LOCAL_DB_PATH": arg_path})
        assert client._path == Path(arg_path)


# ===========================================================================
# TestLocalClientConnection
# ===========================================================================


class TestLocalClientConnection:
    def test_test_connection_always_true(self, empty_client: LocalClient) -> None:
        assert empty_client.test_connection() is True

    def test_close_is_noop(self, empty_client: LocalClient) -> None:
        # Should not raise.
        empty_client.close()

    def test_context_manager_returns_self(self, empty_client: LocalClient) -> None:
        with empty_client as c:
            assert c is empty_client

    def test_context_manager_calls_close(self, empty_client: LocalClient) -> None:
        with patch.object(empty_client, "close") as mock_close:
            with empty_client:
                pass
        mock_close.assert_called_once()

    def test_context_manager_calls_close_on_exception(
        self, empty_client: LocalClient
    ) -> None:
        with patch.object(empty_client, "close") as mock_close:
            with pytest.raises(ValueError):
                with empty_client:
                    raise ValueError("boom")
        mock_close.assert_called_once()


# ===========================================================================
# TestLocalClientWrite
# ===========================================================================


class TestLocalClientWrite:
    def test_insert_one_creates_file(self, empty_client: LocalClient) -> None:
        empty_client.insert_one(_PERM_100_SEED0)
        assert empty_client._path.exists()

    def test_insert_one_creates_parent_dirs(self, tmp_path: Path) -> None:
        deep = tmp_path / "a" / "b" / "c" / "runs.jsonl"
        client = LocalClient(path=deep)
        client.insert_one(_PERM_100_SEED0)
        assert deep.exists()

    def test_insert_one_appends_valid_json(self, empty_client: LocalClient) -> None:
        empty_client.insert_one(_PERM_100_SEED0)
        empty_client.insert_one(_PERM_100_SEED1)
        lines = empty_client._path.read_text().splitlines()
        assert len(lines) == 2
        parsed = [json.loads(l) for l in lines]
        assert parsed[0]["run_id"] == _PERM_100_SEED0["run_id"]
        assert parsed[1]["run_id"] == _PERM_100_SEED1["run_id"]

    def test_insert_many_noop_on_empty(self, empty_client: LocalClient) -> None:
        empty_client.insert_many([])
        # File should not be created for an empty list
        assert not empty_client._path.exists()

    def test_insert_many_writes_all(self, empty_client: LocalClient) -> None:
        docs = [_PERM_100_SEED0, _PERM_100_SEED1, _PERM_500_SEED0]
        empty_client.insert_many(docs)
        lines = empty_client._path.read_text().splitlines()
        assert len(lines) == 3

    def test_insert_many_appends_to_existing(self, empty_client: LocalClient) -> None:
        empty_client.insert_one(_PERM_100_SEED0)
        empty_client.insert_many([_PERM_100_SEED1, _PERM_500_SEED0])
        assert len(empty_client.get_all()) == 3

    def test_insert_one_serialises_numpy_integer(
        self, empty_client: LocalClient
    ) -> None:
        import numpy as np
        doc = {**_PERM_100_SEED0, "numpy_int": np.int64(42)}
        empty_client.insert_one(doc)
        loaded = empty_client.get_all()[0]
        assert loaded["numpy_int"] == 42

    def test_insert_one_serialises_numpy_float(
        self, empty_client: LocalClient
    ) -> None:
        import numpy as np
        doc = {**_PERM_100_SEED0, "numpy_float": np.float32(3.14)}
        empty_client.insert_one(doc)
        loaded = empty_client.get_all()[0]
        assert abs(loaded["numpy_float"] - 3.14) < 1e-4

    def test_insert_one_serialises_numpy_array(
        self, empty_client: LocalClient
    ) -> None:
        import numpy as np
        doc = {**_PERM_100_SEED0, "arr": np.array([1, 2, 3])}
        empty_client.insert_one(doc)
        loaded = empty_client.get_all()[0]
        assert loaded["arr"] == [1, 2, 3]


# ===========================================================================
# TestLocalClientDelete
# ===========================================================================


class TestLocalClientDelete:
    def test_delete_all_returns_correct_count(
        self, populated_client: LocalClient
    ) -> None:
        assert populated_client.delete_all() == len(ALL_DOCS)

    def test_delete_all_leaves_empty_store(
        self, populated_client: LocalClient
    ) -> None:
        populated_client.delete_all()
        assert populated_client.get_all() == []

    def test_delete_all_on_empty_returns_zero(
        self, empty_client: LocalClient
    ) -> None:
        assert empty_client.delete_all() == 0

    def test_delete_by_config_removes_matching(
        self, populated_client: LocalClient, perm_config: RunConfig
    ) -> None:
        # _PERM_100_SEED0 and _PERM_100_SEED1 both match
        deleted = populated_client.delete_by_config(perm_config)
        assert deleted == 2
        run_ids = {d["run_id"] for d in populated_client.get_all()}
        assert _PERM_100_SEED0["run_id"] not in run_ids
        assert _PERM_100_SEED1["run_id"] not in run_ids

    def test_delete_by_config_keeps_non_matching(
        self, populated_client: LocalClient, perm_config: RunConfig
    ) -> None:
        populated_client.delete_by_config(perm_config)
        remaining_ids = {d["run_id"] for d in populated_client.get_all()}
        assert _PERM_500_SEED0["run_id"] in remaining_ids
        assert _STRAT_100_SEED0["run_id"] in remaining_ids
        assert _BIKE_PERM_100_SEED0["run_id"] in remaining_ids

    def test_delete_by_config_zero_when_no_match(
        self, populated_client: LocalClient
    ) -> None:
        no_match = RunConfig(
            game_name="NONEXISTENT", approximator_name="X",
            budget=9999, index="SV", max_order=1,
        )
        assert populated_client.delete_by_config(no_match) == 0
        assert len(populated_client.get_all()) == len(ALL_DOCS)

    def test_delete_by_config_does_not_rewrite_when_nothing_deleted(
        self, populated_client: LocalClient
    ) -> None:
        no_match = RunConfig(
            game_name="NONEXISTENT", approximator_name="X",
            budget=1, index="SV", max_order=1,
        )
        with patch.object(populated_client, "_save") as mock_save:
            populated_client.delete_by_config(no_match)
        mock_save.assert_not_called()


# ===========================================================================
# TestLocalClientReadGeneric
# ===========================================================================


class TestLocalClientReadGeneric:
    def test_get_all_empty_when_no_file(self, empty_client: LocalClient) -> None:
        assert empty_client.get_all() == []

    def test_get_all_returns_all_documents(
        self, populated_client: LocalClient
    ) -> None:
        assert len(populated_client.get_all()) == len(ALL_DOCS)

    def test_get_all_round_trips_data(self, populated_client: LocalClient) -> None:
        ids = {d["run_id"] for d in populated_client.get_all()}
        expected_ids = {d["run_id"] for d in ALL_DOCS}
        assert ids == expected_ids

    def test_get_by_config_returns_only_matching(
        self, populated_client: LocalClient, perm_config: RunConfig
    ) -> None:
        results = populated_client.get_by_config(perm_config)
        assert len(results) == 2
        for d in results:
            assert d["approximator_name"] == "PermutationSamplingSV"
            assert d["budget"] == 100
            assert d["game_name"] == "CaliforniaHousing"

    def test_get_by_config_empty_when_no_match(
        self, populated_client: LocalClient
    ) -> None:
        cfg = RunConfig(
            game_name="MISSING", approximator_name="X",
            budget=9999, index="SV", max_order=1,
        )
        assert populated_client.get_by_config(cfg) == []

    def test_get_by_config_on_empty_store(
        self, empty_client: LocalClient, perm_config: RunConfig
    ) -> None:
        assert empty_client.get_by_config(perm_config) == []


# ===========================================================================
# TestLocalClientReadDomain
# ===========================================================================


class TestLocalClientReadDomain:
    def test_get_games_returns_sorted_distinct(
        self, populated_client: LocalClient
    ) -> None:
        games = populated_client.get_games()
        assert games == sorted(set(games))
        assert set(games) == {"CaliforniaHousing", "BikeSharing"}

    def test_get_games_empty_store(self, empty_client: LocalClient) -> None:
        assert empty_client.get_games() == []

    def test_get_by_game_returns_only_that_game(
        self, populated_client: LocalClient
    ) -> None:
        results = populated_client.get_by_game("BikeSharing")
        assert len(results) == 1
        assert all(d["game_name"] == "BikeSharing" for d in results)

    def test_get_by_game_missing_game(self, populated_client: LocalClient) -> None:
        assert populated_client.get_by_game("NONEXISTENT") == []

    def test_get_approximators_returns_sorted_distinct(
        self, populated_client: LocalClient
    ) -> None:
        approx = populated_client.get_approximators()
        assert approx == sorted(set(approx))
        assert set(approx) == {"PermutationSamplingSV", "StratifiedSamplingSV"}

    def test_get_approximators_empty_store(self, empty_client: LocalClient) -> None:
        assert empty_client.get_approximators() == []

    def test_get_by_approximator_filters_correctly(
        self, populated_client: LocalClient
    ) -> None:
        results = populated_client.get_by_approximator("StratifiedSamplingSV")
        assert len(results) == 1
        assert all(d["approximator_name"] == "StratifiedSamplingSV" for d in results)

    def test_get_by_approximator_missing(
        self, populated_client: LocalClient
    ) -> None:
        assert populated_client.get_by_approximator("NONEXISTENT") == []

    def test_get_unique_configs_no_duplicates(
        self, populated_client: LocalClient
    ) -> None:
        configs = populated_client.get_unique_configs()
        keys = [tuple(sorted(c.to_dict().items())) for c in configs]
        assert len(keys) == len(set(keys))

    def test_get_unique_configs_count(self, populated_client: LocalClient) -> None:
        # 4 distinct (game, approx, budget, index, max_order) combos:
        #   CalHousing/Perm/100, CalHousing/Perm/500, CalHousing/Strat/100,
        #   BikeSharing/Perm/100
        configs = populated_client.get_unique_configs()
        keys = {tuple(sorted(c.to_dict().items())) for c in configs}
        assert len(keys) == 4

    def test_count_by_config_correct(
        self, populated_client: LocalClient, perm_config: RunConfig
    ) -> None:
        assert populated_client.count_by_config(perm_config) == 2

    def test_count_by_config_zero_for_missing(
        self, populated_client: LocalClient
    ) -> None:
        cfg = RunConfig(
            game_name="MISSING", approximator_name="X",
            budget=1, index="SV", max_order=1,
        )
        assert populated_client.count_by_config(cfg) == 0

    def test_count_by_config_empty_store(
        self, empty_client: LocalClient, perm_config: RunConfig
    ) -> None:
        assert empty_client.count_by_config(perm_config) == 0


# ===========================================================================
# TestLocalClientLoadDataframe
# ===========================================================================


class TestLocalClientLoadDataframe:
    def test_load_dataframe_returns_dataframe(
        self, populated_client: LocalClient
    ) -> None:
        import pandas as pd
        df = populated_client.load_dataframe()
        assert isinstance(df, pd.DataFrame)

    def test_load_dataframe_empty_store(self, empty_client: LocalClient) -> None:
        import pandas as pd
        df = empty_client.load_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert df.empty

    def test_load_dataframe_excludes_failed_runs(
        self, jsonl_path: Path
    ) -> None:
        client = LocalClient(path=jsonl_path)
        client.insert_many([_PERM_100_SEED0, _FAILED_RUN])
        df = client.load_dataframe()
        assert len(df) == 1
        assert df.iloc[0]["game_name"] == "CaliforniaHousing"

    def test_load_dataframe_has_expected_columns(
        self, populated_client: LocalClient
    ) -> None:
        df = populated_client.load_dataframe()
        for col in ("game_name", "approximator_name", "budget", "mse"):
            assert col in df.columns

    def test_load_dataframe_row_count_excludes_failures(
        self, jsonl_path: Path
    ) -> None:
        client = LocalClient(path=jsonl_path)
        client.insert_many(ALL_DOCS + [_FAILED_RUN])
        df = client.load_dataframe()
        assert len(df) == len(ALL_DOCS)


# ===========================================================================
# TestHelperFunctions
# ===========================================================================


class TestHelperFunctions:
    def test_json_default_numpy_integer(self) -> None:
        import numpy as np
        result = _json_default(np.int32(7))
        assert result == 7
        assert isinstance(result, int)

    def test_json_default_numpy_floating(self) -> None:
        import numpy as np
        result = _json_default(np.float64(3.14))
        assert isinstance(result, float)
        assert abs(result - 3.14) < 1e-9

    def test_json_default_numpy_array(self) -> None:
        import numpy as np
        result = _json_default(np.array([1, 2, 3]))
        assert result == [1, 2, 3]

    def test_json_default_fallback_to_str(self) -> None:
        class _Weird:
            def __str__(self) -> str:
                return "weird"
        assert _json_default(_Weird()) == "weird"

    def test_matches_config_true(self) -> None:
        cfg = RunConfig(game_name="CaliforniaHousing", approximator_name="X")
        doc = {"game_name": "CaliforniaHousing", "approximator_name": "X", "extra": 1}
        assert _matches_config(doc, cfg) is True

    def test_matches_config_false_wrong_value(self) -> None:
        cfg = RunConfig(game_name="CaliforniaHousing", approximator_name="X")
        doc = {"game_name": "CaliforniaHousing", "approximator_name": "Y"}
        assert _matches_config(doc, cfg) is False

    def test_matches_config_false_missing_key(self) -> None:
        cfg = RunConfig(game_name="CaliforniaHousing", approximator_name="X")
        doc = {"game_name": "CaliforniaHousing"}
        assert _matches_config(doc, cfg) is False

    def test_matches_config_empty_config(self) -> None:
        """An empty RunConfig matches any document."""
        cfg = RunConfig()
        doc = {"game_name": "CaliforniaHousing", "budget": 100}
        assert _matches_config(doc, cfg) is True


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
        client = LocalClient(path=jsonl_path)
        assert isinstance(client, DatabaseClient)
        # Verify no unimplemented abstract methods remain
        assert not getattr(client.__class__, "__abstractmethods__", set())
