"""Check one terminal comprehensive pool and write its public partitioned report.

Run under the pool's frozen scientific runtime in an accounted allocation. This
creates a local candidate and one consolidated audit; it does not publish it.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

from collect_pool import Inputs, collect, digest, require, terminal_accounting


def coefficients(game: dict) -> dict:
    """Validate the saved sparse reference and its recorded nonempty energy."""
    truth = game["truth"]
    keys = [tuple(key) for key in truth["coordinates"]]
    require(
        len(keys) == len(set(keys)) == len(truth["values"])
        and all(
            1 <= len(key) <= game["order"]
            and tuple(sorted(set(key))) == key
            and all(type(i) is int and 0 <= i < game["n_players"] for i in key)
            for key in keys
        ),
        "Invalid reference coordinates",
    )
    require(
        all(math.isfinite(v) for v in [*truth["values"], truth["baseline"], truth["energy"]]),
        "Nonfinite reference",
    )
    result = dict(zip(keys, truth["values"], strict=True))
    require(
        math.isclose(
            math.fsum(v * v for v in result.values()), truth["energy"], rel_tol=1e-12, abs_tol=1e-25
        ),
        "Reference energy differs",
    )
    return result


def close(actual: float, expected: float, message: str) -> None:
    """Use the existing preparation audit's scalar comparison tolerance."""
    require(
        math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-25),
        message,
    )


def table_reference(game: dict, expected: dict, values: object) -> dict:
    """Compare saved coefficients without replacing finite-endpoint FSII references."""
    import numpy as np

    from shapiq_benchmark.order_metrics import order_scores

    actual = coefficients(game)
    reference = dict(zip(map(tuple, expected["coordinates"]), expected["values"], strict=True))
    require(actual.keys() <= reference.keys(), f"Foreign reference coordinates: {game['id']}")
    # ExactComputer's k-SII aggregation omits exact zero coefficients. Preserve
    # that sparse representation and compare its implicit zeros with the table.
    differences = [actual.get(key, 0.0) - value for key, value in reference.items()]
    squared = math.fsum(value * value for value in differences)
    maximum = max(map(abs, differences), default=0.0)
    baseline = abs(game["truth"]["baseline"] - expected["baseline"])
    tolerance = 1e-10 * max(1.0, float(np.max(np.abs(values))))
    require(
        all(math.isfinite(v) for v in (squared, maximum, baseline)),
        "Nonfinite reference comparison",
    )
    require(baseline <= tolerance, "Reference baseline differs")
    saved_orders = order_scores(actual, actual, game)
    floor = game["n_players"] <= 12 and game["index"] == "FSII"
    weak_replay = False
    if game["n_players"] > 12:
        require(
            actual.keys() == reference.keys() and squared == baseline == 0,
            "Direct reference replay differs",
        )
    elif not floor:
        require(maximum <= tolerance, "Independent reference discrepancy")
    else:
        # Same policy as the focused numerical review: the total discrepancy
        # must be below 1e-12 of every eligible order's energy. Also check weak
        # orders explicitly, rather than letting the solver floor create signal.
        reference_orders = order_scores(reference, reference, game)
        require(
            all(
                saved["score_eligible"] == reference_orders[degree]["score_eligible"]
                for degree, saved in saved_orders.items()
            ),
            "FSII finite-endpoint floor changes order eligibility",
        )
        for saved in saved_orders.values():
            if saved["score_eligible"]:
                require(
                    squared / saved["truth_energy"] < 1e-12,
                    "FSII finite-endpoint floor approaches eligible order signal",
                )
        if not any(saved["score_eligible"] for saved in saved_orders.values()):
            from shapiq import ExactComputer
            from shapiq_benchmark.runner import table_game

            replay = ExactComputer(
                table_game(values, game["n_players"]), n_players=game["n_players"]
            )("FSII", order=game["order"])
            replay_values = {key: value for key, value in replay.dict_values.items() if key}
            require(
                actual == replay_values and game["truth"]["baseline"] == replay.baseline_value,
                "Entirely weak FSII reference does not match frozen solver replay",
            )
            weak_replay = True
    return {
        "comparison": "direct-formula deterministic replay"
        if game["n_players"] > 12
        else "direct-formula cross-check of ExactComputer",
        "maximum_absolute_difference": maximum,
        "squared_difference": squared,
        "baseline_difference": baseline,
        "finite_endpoint_fsii": floor,
        "weak_fsii_frozen_solver_replay": weak_replay,
        "implicit_zero_coordinates": len(reference.keys() - actual.keys()),
        "order_eligibility": saved_orders,
    }


def check_snapshot(snapshot: dict, root: Path) -> tuple[list[dict], dict]:
    """Check frozen table references and the recorded native qualification separately."""
    import numpy as np

    from shapiq import InteractionValues
    from shapiq_benchmark.duplicates import payoff_fingerprint
    from shapiq_benchmark.exact import exact_table_truth
    from shapiq_benchmark.games import load_game, validate_truth
    from shapiq_benchmark.order_metrics import order_scores

    by_artifact = defaultdict(list)
    for game in snapshot["games"]:
        by_artifact[game["artifact"]].append(game)
    checks, fingerprints = [], {}
    for artifact, games in by_artifact.items():
        table = games[0].get("oracle", "table") == "table"
        require(
            all((g.get("oracle", "table") == "table") == table for g in games),
            "Artifact mixes table and native oracles",
        )
        if table:
            n = games[0]["n_players"]
            require(all(g["n_players"] == n for g in games), "Artifact player counts differ")
            with np.load(root / artifact, allow_pickle=False) as archive:
                values, costs = archive["values"], archive["evaluation_seconds"]
                require(
                    values.shape == costs.shape == (2**n,)
                    and np.isfinite(values).all()
                    and np.isfinite(costs).all()
                    and (costs >= 0).all(),
                    "Invalid payoff/cost table",
                )
                scale = float(np.std(values, dtype=np.longdouble))
                references = exact_table_truth(
                    values, n, [{"index": g["index"], "order": g["order"]} for g in games]
                )
            for game in games:
                metadata = game["metadata"]
                require(
                    metadata["truth_method"] == "exhaustive frozen table"
                    and metadata["truth_queries"] == 2**n,
                    "Incomplete table qualification",
                )
                close(metadata["payoff_std"], scale, "Payoff deviation differs")
                checked = table_reference(game, references[game["index"], game["order"]], values)
                fingerprints[game["id"]] = payoff_fingerprint(game, root)
                checks.append({"game_id": game["id"], **checked})
        for game in games:
            saved = coefficients(game)
            metadata = game["metadata"]
            dimension = sum(math.comb(game["n_players"], k) for k in range(1, game["order"] + 1))
            if not table:
                require(
                    game["oracle"] in {"tree", "pathdependent_tree", "knn", "tnn"},
                    "Unreviewed native oracle",
                )
                if game["oracle"] in {"tree", "pathdependent_tree"}:
                    require(
                        metadata.get("structured_format") == "numeric-model-v1",
                        "Native trees must reconstruct numeric arrays without refitting",
                    )
                require(
                    metadata["small_validation_players"] == 8
                    and math.isfinite(metadata["small_validation_max_error"])
                    and metadata["small_validation_max_error"] >= 0,
                    "Missing native qualification",
                )
                bound = metadata["payoff_range_upper_bound"]
                require(math.isfinite(bound) and bound >= 0, "Invalid native payoff range bound")
                scale = bound / 2
                truth = InteractionValues(
                    saved,
                    index=game["index"],
                    max_order=game["order"],
                    n_players=game["n_players"],
                    min_order=0,
                    baseline_value=game["truth"]["baseline"],
                )
                error = validate_truth(
                    load_game(game, root), truth, exhaustive=game["n_players"] <= 8
                )
                checks.append(
                    {
                        "game_id": game["id"],
                        "comparison": "native saved qualification plus endpoint/efficiency/determinism replay",
                        "exhaustive_actual_game": game["n_players"] <= 8,
                        "actual_small_game_maximum_error": error,
                        "recorded_small_game_maximum_error": metadata["small_validation_max_error"],
                        "order_eligibility": order_scores(saved, saved, game),
                    }
                )
            ratio = math.sqrt(game["truth"]["energy"] / dimension) / scale if scale else 0.0
            close(metadata["signal_ratio"], ratio, "Signal ratio differs")
            require(
                metadata["score_eligible"] == (ratio >= snapshot["suite"]["min_signal_ratio"]),
                "Saved signal eligibility differs",
            )
    return checks, fingerprints


def aliases_for(games: list[dict], fingerprints: dict) -> dict:
    """Deduplicate within a weighting branch; never silently remove another branch."""
    groups = defaultdict(list)
    for game in games:
        fingerprint = fingerprints.get(game["id"])
        if fingerprint is not None:
            groups[fingerprint].append(game)
    aliases = {}
    for group in groups.values():
        if len(group) < 2:
            continue
        branches = {
            (
                *[
                    game["metadata"]["focused_design"][key]
                    for key in ("application", "subtype", "recipe")
                ],
                game["metadata"].get("game_quality", {}).get("role", "unqualified"),
            )
            for game in group
        }
        require(
            len(branches) == 1,
            "Identical payoffs cross weighting branches; explicit reconciliation required",
        )
        canonical = min(group, key=lambda g: (g["metadata"]["instance_seed"], g["id"]))["id"]
        aliases.update({g["id"]: canonical for g in group if g["id"] != canonical})
    return aliases


def export(config_path: Path, output: Path, database: Path, audit_path: Path) -> dict:
    """Create a new local report and one audit after complete outcome coverage."""
    from shapiq_benchmark.duplicates import remove_aliases
    from shapiq_benchmark.partitioned import write_partitioned_report
    from shapiq_benchmark.record_store import RecordStore
    from shapiq_benchmark.runner import identity, load_snapshot

    paths = [p.resolve() for p in (output, database, audit_path)]
    require(
        len(set(paths)) == 3
        and not any(p.exists() for p in paths)
        and not any(p.is_symlink() for p in (output, database, audit_path))
        and all(paths[0] not in p.parents for p in paths[1:]),
        "Use new separate output, database and audit paths",
    )
    inputs = Inputs()
    config = inputs.read(config_path)
    require(
        "comprehensive_design" in inputs.pinned(config["suite"]), "Need the comprehensive cohort"
    )
    for path in (Path(__file__), Path(__file__).with_name("collect_pool.py")):
        inputs.pin(path)
    assets = Path(__file__).with_name("site")
    names = inputs.read(assets / "assets.json")
    require(
        len(names) == len(set(names)) and all(Path(name).name == name for name in names),
        "Invalid public asset manifest",
    )
    for name in names:
        inputs.pin(assets / name)
    with RecordStore(database) as records:
        data, audit = collect(config, records)
        require(audit["outcome_complete"], "Unclaimed or never-attempted cells remain")
        require(data["games"], "No qualified measured games to export")
        for path, sha in audit["input_hashes"].items():
            inputs.pin(path, sha)
        inputs.absent.update(audit["absent_files"])
        checks, fingerprints = [], {}
        pool = Path(config["pool_directory"])
        for case in audit["cases"]:
            path = pool / "tasks" / f"case-{case['case']:06d}" / "prepared/snapshot.json"
            if str(path.absolute()) not in audit["input_hashes"]:
                continue
            snapshot, root = load_snapshot(path)
            checked, found = check_snapshot(snapshot, root)
            checks.extend(checked)
            fingerprints.update(found)
        require(
            {r["game_id"] for r in checks} == {g["id"] for g in data["games"]}
            and len(checks) == len(data["games"]),
            "Reference inventory differs",
        )
        aliases = aliases_for(data["games"], fingerprints)
        remove_aliases(data, aliases)
        collection_id = data["snapshot_id"]
        data["composition"] = {
            "kind": "terminal-comprehensive-pool",
            "collection_id": collection_id,
            "reference_check_sha256": identity({"checks": checks}),
            "aliases": data["duplicate_games"],
            "intended_instances": audit["intended_instances"],
        }
        data["snapshot_id"] = identity(data["composition"])
        inputs.stable()
        write_partitioned_report(data, output)
        for name in names:
            shutil.copyfile(assets / name, output / name)
        shutil.copyfile(output / "data.json", output / "about.json")
        files = {
            str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*")) if p.is_file()
        }
        size = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
        require(size <= 950 * 1024 * 1024, "Report exceeds the Pages payload allowance")
        require(
            terminal_accounting(config["jobs"]) == audit["slurm_accounting"],
            "Allocation accounting changed",
        )
        inputs.stable()
        audit.update(
            status="EXPORTED_NOT_PUBLISHED",
            publication_ready=False,
            scope="Terminal comprehensive outcome collection, reference checks and public packaging; browser verification and publication remain separate.",
            report_id=data["snapshot_id"],
            collection_id=collection_id,
            reference_checks=checks,
            duplicate_games=data["duplicate_games"],
            native_duplicate_uniqueness="not established by table fingerprints",
            input_hashes=inputs.hashes,
            absent_files=sorted(inputs.absent),
            output_files=files,
            output_bytes=size,
            public_targets=len(data["games"]),
            public_rows=len(records),
        )
    with audit_path.open("x") as stream:
        json.dump(audit, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return audit


def main() -> None:
    """Expose the single consolidated terminal collection/export command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    result = export(args.config, args.output, args.database, args.audit)
    print(  # noqa: T201 -- command-line result
        json.dumps(
            {key: result[key] for key in ("status", "report_id", "public_rows", "output_bytes")}
        )
    )


if __name__ == "__main__":
    main()
