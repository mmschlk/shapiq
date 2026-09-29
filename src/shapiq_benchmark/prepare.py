"""Freeze a small real-data explanation game and its exhaustive ground truth."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from shapiq.datasets import load_california_housing
from shapiq.game_theory import ExactComputer
from shapiq_benchmark.games import prepare_structured
from shapiq_benchmark.runner import digest, identity, provenance, table_game, validate_suite


def prepare(suite_path: Path, output: Path) -> dict:
    """Train once, enumerate 256 coalitions, and store non-executable artifacts."""
    suite = json.loads(suite_path.read_text())
    validate_suite(suite)
    game_seeds = suite.get("game_seeds")
    if game_seeds is not None and (
        not isinstance(game_seeds, list)
        or not game_seeds
        or any(type(seed) is not int or seed < 0 for seed in game_seeds)
        or len(set(game_seeds)) != len(game_seeds)
    ):
        message = "game_seeds must be a nonempty list of unique nonnegative integers."
        raise ValueError(message)
    specs = suite.get("games", [])
    if game_seeds is not None:
        specs = [
            {
                **spec,
                "id": f"{spec['id']}-i{seed}",
                "instance_seed": seed,
                "basecase_id": spec.get(
                    "basecase_id",
                    (
                        f"{spec.get('dataset', 'breast_cancer')}-{spec['oracle']}-"
                        f"{spec.get('n_players', 'native')}"
                    ),
                ),
            }
            for spec in specs
            for seed in game_seeds
        ]
    if "families" in suite:
        from shapiq_benchmark.materialize import prepare_families

        games, coverage = [], []
        for seed in game_seeds if game_seeds is not None else [None]:
            instances, entries = prepare_families(
                suite["families"], suite["targets"], output, instance_seed=seed
            )
            games.extend(instances)
            coverage.extend(entries)
        if specs:
            structured = prepare_structured(specs, output)
            games.extend(structured)
            coverage.extend(
                {
                    "family": g["id"],
                    "status": "measured",
                    "source": "src/shapiq_benchmark/games.py",
                    "game_ids": [g["id"]],
                }
                for g in structured
            )
        return write_snapshot(suite, games, output, coverage=coverage)
    if "games" in suite:
        games = prepare_structured(specs, output)
        return write_snapshot(suite, games, output)
    if game_seeds is not None:
        message = "Use family or structured recipes for multiple game constructions."
        raise ValueError(message)
    if suite["game"] != "california_tree":
        message = "The pilot supports only california_tree."
        raise ValueError(message)
    features, target = load_california_housing()
    features = pd.DataFrame(features)
    x, y = np.asarray(features), np.asarray(target)
    train, test = train_test_split(np.arange(len(x)), test_size=0.2, random_state=0)
    model = DecisionTreeRegressor(max_depth=5, min_samples_leaf=20, random_state=0)
    model.fit(x[train], y[train])
    background_indices = np.random.default_rng(0).choice(train, size=32, replace=False)
    background, point = x[background_indices], x[test[0]]
    n = x.shape[1]
    coalitions = ((np.arange(2**n)[:, None] >> np.arange(n)) & 1).astype(bool)
    values = np.array(
        [model.predict(np.where(coalition, point, background)).mean() for coalition in coalitions]
    )
    truth = ExactComputer(table_game(values, n), n_players=n)("SV", order=1)
    output.mkdir(parents=True, exist_ok=True)
    artifact = output / "california_tree.npz"
    np.savez_compressed(
        artifact,
        values=values,
        point=point,
        background=background,
        train_indices=train,
        test_indices=test,
        background_indices=background_indices,
        children_left=model.tree_.children_left,
        children_right=model.tree_.children_right,
        feature=model.tree_.feature,
        threshold=model.tree_.threshold,
        tree_value=model.tree_.value,
    )
    data_hash = hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest()
    game = {
        "id": "california-tree-point0-sv",
        "family": "local_explanation",
        "stratum": "california_tree_8",
        "n_players": n,
        "index": "SV",
        "order": 1,
        "artifact": artifact.name,
        "oracle": "table",
        "truth": {
            "coordinates": [[i] for i in range(n)],
            "values": [float(truth[(i,)]) for i in range(n)],
            "baseline": float(values[0]),
            "energy": float(sum(truth[(i,)] ** 2 for i in range(n))),
        },
        "metadata": {
            "dataset": "California Housing",
            "data_sha256": data_hash,
            "data_source": "shapiq.datasets.load_california_housing",
            "features": list(features.columns),
            "model": "DecisionTreeRegressor",
            "model_parameters": model.get_params(),
            "test_r2": model.score(x[test], y[test]),
            "point_row": int(test[0]),
            "background_size": 32,
            "semantics": "mean prediction over fixed joint background rows",
            "truth_method": "exhaustive enumeration",
            "truth_queries": 2**n,
        },
    }
    return write_snapshot(suite, [game], output)


def write_snapshot(
    suite: dict, games: list[dict], output: Path, *, coverage: list | None = None
) -> dict:
    """Write one content-addressed manifest for either preparation route."""
    if suite.get("relative_budgets"):
        suite = {
            **suite,
            "budgets_by_game": {
                game["id"]: sorted(
                    {math.ceil(ratio * game["n_players"]) for ratio in suite["relative_budgets"]}
                )
                for game in games
            },
        }
        suite["budgets"] = sorted(
            {budget for grid in suite["budgets_by_game"].values() for budget in grid}
        )
    if not games:
        message = "No family qualified; inspect the local preparation diagnostics."
        raise ValueError(message)
    snapshot: dict = {
        "schema_version": 1,
        "suite": suite,
        "provenance": provenance(),
        "artifacts": {game["artifact"]: digest(output / game["artifact"]) for game in games},
        "games": games,
    }
    if coverage is not None:
        snapshot["coverage"] = coverage
    snapshot["snapshot_id"] = identity(snapshot)
    serialized = json.dumps(snapshot, indent=2, allow_nan=False) + "\n"
    (output / "snapshot.json").write_text(serialized)
    return json.loads(serialized)


def main() -> None:
    """Prepare the command-line pilot snapshot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.suite, args.output)


if __name__ == "__main__":
    main()
