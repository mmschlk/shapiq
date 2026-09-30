"""Freeze bounded representatives of shipped families into exact coalition tables."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import time
from itertools import combinations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np

from shapiq.game_theory import ExactComputer
from shapiq_benchmark.families import FAMILY_CATALOG, MAX_ENUMERATION_PLAYERS, make_family
from shapiq_benchmark.games import truth_dict
from shapiq_benchmark.media import EXTRA_CATALOG, make_extra

CATALOG = {**FAMILY_CATALOG, **EXTRA_CATALOG}
CHUNK_SIZE = 4096
CHUNK_PROTOCOL = "ascending-4096-fresh-seeded-recipe-v1"
COST_PROTOCOL = "batch-amortized-wall-seconds-v1"


def _cost_batch(start: int, stop: int, seconds: float) -> dict:
    """Record measured batch work without local host names or paths."""
    from shapiq_benchmark.execution import THREAD_VARIABLES, hardware

    observed = hardware()
    return {
        "start": start,
        "stop": stop,
        "seconds": seconds,
        "cpu_model": observed["cpu_model"],
        "affinity": observed["affinity"],
        "threads": {name: os.environ.get(name) for name in THREAD_VARIABLES},
    }


def _chunk_identity(spec: dict, seed: int, start: int, source: dict) -> dict:
    software = {
        key: source.get(key)
        for key in ("python", "numpy", "scikit-learn", "shapiq", "installed_packages")
    }
    return {
        "spec": spec,
        "instance_seed": seed,
        "start": start,
        "stop": min(start + CHUNK_SIZE, 2 ** spec["n_players"]),
        "protocol": CHUNK_PROTOCOL,
        "source_sha256": source["source_sha256"],
        "software_sha256": hashlib.sha256(
            json.dumps(software, sort_keys=True).encode()
        ).hexdigest(),
    }


def _read_chunk(path: Path, expected: dict) -> tuple:
    """Reject incomplete, corrupt, or incompatible checkpointed payoff work."""
    with np.load(path, allow_pickle=False) as saved:
        values, costs = saved["values"], saved["evaluation_seconds"]
        manifest = json.loads(str(saved["manifest"]))
    count = expected["stop"] - expected["start"]
    batch = manifest["batch"]
    if (
        manifest["identity"] != expected
        or values.shape != (count,)
        or costs.shape != (count,)
        or not np.isfinite(values).all()
        or not np.isfinite(costs).all()
        or np.any(costs < 0)
        or batch["start"] != expected["start"]
        or batch["stop"] != expected["stop"]
        or not math.isfinite(batch["seconds"])
        or batch["seconds"] < 0
        or not math.isclose(batch["seconds"], float(np.sum(costs)), rel_tol=1e-12, abs_tol=0)
        or hashlib.sha256(values.tobytes() + costs.tobytes()).hexdigest()
        != manifest["payload_sha256"]
    ):
        message = "Cached payoff chunk is incomplete or does not match its source/recipe/protocol."
        raise ValueError(message)
    return values, costs, manifest


def prepare_family_chunk(spec: dict, instance_seed: int, start: int, output: Path) -> Path:
    """Checkpoint one independently reproducible large-table chunk; safe to rerun.

    Every chunk reconstructs the same fitted game with the same seed. Stochastic
    imputers therefore reuse their initial random stream across chunks: this is
    an explicitly frozen realization, not independent Monte Carlo replicates.
    """
    from shapiq_benchmark.runner import provenance

    n = spec.get("n_players")
    if (
        type(n) is not int
        or not 12 < n <= MAX_ENUMERATION_PLAYERS
        or type(instance_seed) is not int
        or instance_seed < 0
        or type(start) is not int
        or not 0 <= start < 2**n
        or start % CHUNK_SIZE
        or not re.fullmatch(r"[a-zA-Z0-9_-]+", spec.get("id", ""))
    ):
        message = "Chunks require an explicit 13-20 player recipe and an aligned bitmask start."
        raise ValueError(message)
    expected = _chunk_identity(spec, instance_seed, start, provenance())
    path = output / ".chunks" / f"{spec['id']}-i{instance_seed}-{start}.npz"
    if path.exists():
        _read_chunk(path, expected)
        return path
    random.seed(instance_seed)
    np.random.seed(instance_seed % 2**32)  # noqa: NPY002 -- optional backend global RNGs
    factory = make_extra if spec["family"] in EXTRA_CATALOG else make_family
    game, metadata = factory(
        spec["family"],
        instance_seed=instance_seed,
        **{key: spec[key] for key in ("dataset", "n_players") if key in spec},
    )
    if game.n_players != n:
        message = "Constructed player count does not match the chunk recipe."
        raise ValueError(message)
    stop = expected["stop"]
    coalitions = ((np.arange(start, stop)[:, None] >> np.arange(n)) & 1).astype(bool)
    started = time.perf_counter()
    values = np.asarray(game(coalitions), dtype=float)
    elapsed = time.perf_counter() - started
    if values.shape != (stop - start,) or not np.isfinite(values).all():
        message = "The game returned invalid payoff chunk values."
        raise ValueError(message)
    costs = np.full(len(values), elapsed / len(values))
    manifest = {
        "identity": expected,
        "metadata": metadata,
        "batch": _cost_batch(start, stop, elapsed),
        "payload_sha256": hashlib.sha256(values.tobytes() + costs.tobytes()).hexdigest(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            values=values,
            evaluation_seconds=costs,
            manifest=json.dumps(manifest, allow_nan=False),
        )
    temporary.replace(path)
    return path


def exact_table_truth(values: np.ndarray, n: int, targets: list[dict]) -> dict:
    """Combine an exhaustive bitmask table into the six supported low-order indices.

    Direct first/second differences avoid high-order Möbius cancellation and the
    library FII solver's square diagonal matrix. Weights are the library's
    discrete-derivative formulas; at order two, faithful and k-SII singleton
    coefficients are their first-order values minus half the incident pairs.
    Storage is O(2**n); pair differences are reused for every interaction index.
    """
    supported = {("SV", 1), *((index, 2) for index in ("SII", "k-SII", "STII", "FSII", "FBII"))}
    if any((target["index"], target["order"]) not in supported for target in targets):
        message = "Large exhaustive tables support SV and the five order-two interaction targets."
        raise ValueError(message)
    if values.shape != (2**n,) or not np.isfinite(values).all():
        message = "Exact truth requires one finite payoff per coalition."
        raise ValueError(message)
    values = np.asarray(values, dtype=np.longdouble)
    baseline = float(values[0])
    sizes = np.zeros(2 ** (n - 1), dtype=np.uint8)
    for bit in range(n - 1):
        step = 1 << bit
        sizes[step : 2 * step] = sizes[:step] + 1
    weights = np.array([np.longdouble(1) / (n * math.comb(n - 1, size)) for size in range(n)])
    rest = np.arange(2 ** (n - 1), dtype=np.uint32)
    shapley, banzhaf = {}, {}
    for player in range(n):
        bit = 1 << player
        absent = (rest & (bit - 1)) | ((rest >> player) << (player + 1))
        delta = values[absent | bit] - values[absent]
        # Complementary coalitions have equal weights. Pair before summation to
        # preserve cancellation, including exactly zero SV for even parity games.
        symmetric = (delta + delta[::-1]) / 2
        shapley[player] = np.sum(symmetric * weights[sizes])
        banzhaf[player] = np.mean(symmetric)
    pairs = {index: {} for index in ("SII", "STII", "FSII", "FBII")}
    if any(target["order"] == 2 for target in targets):
        rest = np.arange(2 ** (n - 2), dtype=np.uint32)
        pair_sizes = sizes[: len(rest)]
        sii = np.array(
            [np.longdouble(1) / ((n - 1) * math.comb(n - 2, size)) for size in range(n - 1)]
        )
        fsii = np.array(
            [sii[size] * 6 * (size + 1) * (n - size - 1) / (n * (n + 1)) for size in range(n - 1)]
        )
        stii = np.array([np.longdouble(2) / (n * math.comb(n - 1, size)) for size in range(n - 1)])
        for left, right in combinations(range(n), 2):
            a, b = 1 << left, 1 << right
            absent = (rest & (a - 1)) | ((rest >> left) << (left + 1))
            absent = (absent & (b - 1)) | ((absent >> right) << (right + 1))
            delta = (
                values[absent | a | b] - values[absent | a] - values[absent | b] + values[absent]
            )
            symmetric = (delta + delta[::-1]) / 2
            pairs["SII"][left, right] = np.sum(symmetric * sii[pair_sizes])
            pairs["FSII"][left, right] = np.sum(symmetric * fsii[pair_sizes])
            pairs["STII"][left, right] = np.sum(delta * stii[pair_sizes])
            pairs["FBII"][left, right] = np.mean(symmetric)
    results = {}
    for target in targets:
        index, order = target["index"], target["order"]
        selected = pairs["SII" if index == "k-SII" else index] if order == 2 else {}
        singles = banzhaf if index == "FBII" else shapley
        coefficients = {(player,): value for player, value in singles.items()}
        if index == "STII":
            coefficients = {(player,): values[1 << player] - values[0] for player in range(n)}
        if index in ("k-SII", "FSII", "FBII"):
            for (left, right), value in selected.items():
                coefficients[left,] -= value / 2
                coefficients[right,] -= value / 2
        coefficients.update(selected)
        output_baseline = baseline
        if index == "FBII":
            output_baseline = float(
                np.mean(values) - sum(banzhaf.values()) / 2 + sum(selected.values()) / 4
            )
        numbers = [float(value) for value in coefficients.values()]
        results[index, order] = {
            "coordinates": [list(players) for players in coefficients],
            "values": numbers,
            "baseline": output_baseline,
            "energy": math.fsum(value**2 for value in numbers),
        }
    return results


def prepare_families(
    names: list[str | dict], targets: list[dict], output: Path, *, instance_seed: int | None = None
) -> tuple[list, list]:
    """Qualify each recipe, preserving unavailable families in the coverage catalog.

    Sampled games are explicitly frozen in canonical bitmask order. Their exact
    coefficients describe that saved realization, not an expected stochastic game.
    """
    from shapiq_benchmark.runner import provenance, table_game

    specs = [{"id": entry, "family": entry} if isinstance(entry, str) else entry for entry in names]
    if not specs or any(
        not isinstance(spec, dict)
        or set(spec) - {"id", "family", "dataset", "n_players"}
        or not isinstance(spec.get("family"), str)
        or spec.get("family") not in CATALOG
        or not isinstance(spec.get("id"), str)
        or not re.fullmatch(r"[a-zA-Z0-9_-]+", spec["id"])
        or ("dataset" in spec and not isinstance(spec["dataset"], str))
        or (
            "n_players" in spec
            and (
                type(spec["n_players"]) is not int
                or not 1 <= spec["n_players"] <= MAX_ENUMERATION_PLAYERS
            )
        )
        for spec in specs
    ):
        message = (
            "Families must select catalog names or explicit id/family/dataset/n_players specs."
        )
        raise ValueError(message)
    if len({spec["id"] for spec in specs}) != len(specs):
        message = "Family recipe IDs must be unique."
        raise ValueError(message)
    pairs = [(target.get("index"), target.get("order")) for target in targets]
    if (
        not pairs
        or len(set(pairs)) != len(pairs)
        or any(
            index not in ExactComputer.valid_indices
            or type(order) is not int
            or not 1 <= order <= 12
            or ((index in ("SV", "BV")) != (order == 1))
            for index, order in pairs
        )
    ):
        message = "Targets must be unique supported index/order pairs; values use order one."
        raise ValueError(message)
    output.mkdir(parents=True, exist_ok=True)
    games, coverage = [], []
    for spec in specs:
        name, case_id = spec["family"], spec["id"]
        instance_id = case_id if instance_seed is None else f"{case_id}-i{instance_seed}"
        entry = {"family": instance_id, **CATALOG[name], "status": "unavailable"}
        try:
            factory = make_extra if name in EXTRA_CATALOG else make_family
            options = {key: spec[key] for key in ("dataset", "n_players") if key in spec}
            if instance_seed is not None:
                options["instance_seed"] = instance_seed
            game, metadata = factory(name, **options)
            n = game.n_players
            if not 1 <= n <= MAX_ENUMERATION_PLAYERS:
                message = f"Exhaustive family preparation is limited to {MAX_ENUMERATION_PLAYERS} players."
                raise ValueError(message)  # noqa: TRY301 -- preserve per-family coverage failures
            values = np.empty(2**n, dtype=float)
            costs = np.empty_like(values)
            batches = []
            if n <= 12:
                coalitions = ((np.arange(2**n)[:, None] >> np.arange(n)) & 1).astype(bool)
                started = time.perf_counter()
                batch = np.asarray(game(coalitions), dtype=float)
                elapsed = time.perf_counter() - started
                if batch.shape != values.shape:
                    message = "The family produced incorrectly shaped batch values."
                    raise ValueError(message)  # noqa: TRY301
                values[:] = batch
                costs[:] = elapsed / len(values)
                batches.append(_cost_batch(0, len(values), elapsed))
            else:
                source = provenance()
                chunk_spec = {**spec, "n_players": n}
                for start in range(0, len(values), CHUNK_SIZE):
                    path = prepare_family_chunk(chunk_spec, instance_seed or 0, start, output)
                    expected = _chunk_identity(chunk_spec, instance_seed or 0, start, source)
                    batch, batch_costs, manifest = _read_chunk(path, expected)
                    if json.dumps(metadata, sort_keys=True) != json.dumps(
                        manifest["metadata"], sort_keys=True
                    ):
                        message = "Chunk construction metadata differs from the same seeded game."
                        raise ValueError(message)  # noqa: TRY301
                    values[start : expected["stop"]] = batch
                    costs[start : expected["stop"]] = batch_costs
                    batches.append(manifest["batch"])
            if values.shape != (2**n,) or not np.all(np.isfinite(values)):
                message = "The family produced nonfinite or incorrectly shaped values."
                raise ValueError(message)  # noqa: TRY301 -- preserve per-family coverage failures
            if not metadata.get("stochastic_frozen", False):
                positions = np.random.default_rng(0).choice(len(values), 8)
                probe = ((positions[:, None] >> np.arange(n)) & 1).astype(bool)
                np.testing.assert_allclose(values[positions], game(probe), rtol=1e-8, atol=1e-10)
                np.testing.assert_allclose(
                    game(probe), game(probe[::-1])[::-1], rtol=1e-8, atol=1e-10
                )
                np.testing.assert_allclose(
                    game(probe),
                    np.concatenate([game(row[None]) for row in probe]),
                    rtol=1e-8,
                    atol=1e-10,
                )
            artifact = output / f"{instance_id}.npz"
            np.savez_compressed(
                artifact,
                values=values,
                evaluation_seconds=costs,
                evaluation_batches=json.dumps(batches, allow_nan=False),
            )
            profiles = {
                json.dumps({key: batch[key] for key in ("cpu_model", "threads")}, sort_keys=True)
                for batch in batches
            }
            metadata["evaluation_timing"] = {
                "protocol": COST_PROTOCOL,
                "profiles": [json.loads(profile) for profile in sorted(profiles)],
            }
            metadata["oracle_cost_protocol"] = COST_PROTOCOL
            exact = ExactComputer(table_game(values, n), n_players=n) if n <= 12 else None
            large_truth = exact_table_truth(values, n, targets) if n > 12 else {}
            payoff_std = float(np.std(values, dtype=np.longdouble))
            qualified = []
            for target in targets:
                index, order = target["index"], target["order"]
                encoded = (
                    truth_dict(exact(index, order=order)) if exact else large_truth[index, order]
                )
                if not np.isfinite(encoded["energy"]) or not np.all(np.isfinite(encoded["values"])):
                    message = "Ground truth is nonfinite."
                    raise ValueError(message)  # noqa: TRY301 -- preserve per-family coverage failures
                slug = re.sub(r"[^a-zA-Z0-9_-]", "-", index).lower()
                qualified.append(
                    {
                        "id": f"{instance_id}-{slug}-{order}",
                        "family": metadata.get("application_family", name),
                        "stratum": f"{case_id}_{metadata.get('dataset', 'fixed')}_{n}",
                        "n_players": n,
                        "index": index,
                        "order": order,
                        "oracle": "table",
                        "artifact": artifact.name,
                        "truth": encoded,
                        "metadata": {
                            **metadata,
                            "case_id": case_id,
                            "game_kind": name,
                            "instance_seed": instance_seed or 0,
                            "cluster_id": metadata.get("cluster_id", instance_id),
                            "truth_method": "exhaustive frozen table",
                            "truth_queries": 2**n,
                            "payoff_std": payoff_std,
                            "signal_ratio": (
                                math.sqrt(
                                    encoded["energy"]
                                    / sum(math.comb(n, degree) for degree in range(1, order + 1))
                                )
                                / payoff_std
                                if payoff_std
                                else 0.0
                            ),
                            "materialization": (
                                "ascending bitmask, player zero least significant, one full batch"
                                if n <= 12
                                else CHUNK_PROTOCOL
                            ),
                            **(
                                {
                                    "truth_algorithm": "vectorized exact first/second discrete derivatives"
                                }
                                if n > 12
                                else {}
                            ),
                        },
                    }
                )
            games.extend(qualified)
            entry.update(status="measured", n_players=n, game_ids=[g["id"] for g in qualified])
        except Exception as error:  # noqa: BLE001 -- one broken family must not erase the inventory
            entry.update(reason=f"Preparation failed: {type(error).__name__}")
            # Full diagnostics remain local; public metadata excludes raw exception text.
            (output / f"{instance_id}-error.txt").write_text(f"{type(error).__name__}: {error}\n")
        coverage.append(entry)
    return games, coverage
