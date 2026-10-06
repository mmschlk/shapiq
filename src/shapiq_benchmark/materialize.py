"""Freeze bounded representatives of shipped families into exact coalition tables."""

from __future__ import annotations

import json
import math
import random
import re
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np

from shapiq.game_theory import ExactComputer

# Keep the existing materialize imports available to scripts and local candidates.
from shapiq_benchmark.exact import exact_table_truth
from shapiq_benchmark.families import FAMILY_CATALOG, MAX_ENUMERATION_PLAYERS, make_family
from shapiq_benchmark.games import truth_dict
from shapiq_benchmark.media import EXTRA_CATALOG, make_extra, preparation_backend
from shapiq_benchmark.payoff_cache import (
    CHUNK_PROTOCOL,
    CHUNK_SIZE,
    COST_PROTOCOL,
    chunk_identity as _base_chunk_identity,
    cost_batch as _cost_batch,
    read_chunk as _read_chunk,
    write_chunk,
)
from shapiq_benchmark.quality import (
    QUALITY_PROTOCOL,
    clustering_diagnostics,
    imputation_stability,
    payoff_diagnostics,
)
from shapiq_benchmark.spectrum import fourier_spectrum

CATALOG = {**FAMILY_CATALOG, **EXTRA_CATALOG}


def _chunk_identity(spec: dict, seed: int, start: int, source: dict) -> dict:
    """Bind CUDA chunks to the actual device and precision before reading any payoff."""
    identity = _base_chunk_identity(spec, seed, start, source)
    if spec.get("device") == "cuda":
        identity["preparation_hardware"] = preparation_backend("cuda")
    return identity


def _synchronize(spec: dict) -> None:
    """Include completed CUDA work in the measured oracle cost."""
    if spec.get("device") == "cuda":
        import torch

        torch.cuda.synchronize()


def _timing_batch(start: int, stop: int, elapsed: float, metadata: dict) -> dict:
    """Keep accelerator details alongside CPU details in cached timing records."""
    batch = _cost_batch(start, stop, elapsed)
    if "preparation_hardware" in metadata:
        batch["preparation_hardware"] = metadata["preparation_hardware"]
    return batch


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
        **{
            key: spec[key]
            for key in (
                "dataset",
                "n_players",
                "model_profile",
                "device",
                "quality_protocol",
                "input_features",
                "feature_rule",
            )
            if key in spec and (key != "quality_protocol" or factory is make_family)
        },
        **({"model_cache": str(output.parent / ".models")} if "model_profile" in spec else {}),
    )
    if game.n_players != n:
        message = "Constructed player count does not match the chunk recipe."
        raise ValueError(message)
    stop = expected["stop"]
    coalitions = ((np.arange(start, stop)[:, None] >> np.arange(n)) & 1).astype(bool)
    _synchronize(spec)
    started = time.perf_counter()
    values = np.asarray(game(coalitions), dtype=float)
    _synchronize(spec)
    elapsed = time.perf_counter() - started
    if values.shape != (stop - start,) or not np.isfinite(values).all():
        message = "The game returned invalid payoff chunk values."
        raise ValueError(message)
    costs = np.full(len(values), elapsed / len(values))
    manifest = {
        "identity": expected,
        "metadata": metadata,
        "batch": _timing_batch(start, stop, elapsed, metadata),
    }
    write_chunk(path, values, costs, manifest)
    return path


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
        or set(spec)
        - {
            "id",
            "family",
            "dataset",
            "n_players",
            "model_profile",
            "device",
            "quality_protocol",
            "input_features",
            "feature_rule",
        }
        or not isinstance(spec.get("family"), str)
        or spec.get("family") not in CATALOG
        or not isinstance(spec.get("id"), str)
        or not re.fullmatch(r"[a-zA-Z0-9_-]+", spec["id"])
        or ("dataset" in spec and not isinstance(spec["dataset"], str))
        or ("model_profile" in spec and not isinstance(spec["model_profile"], str))
        or spec.get("quality_protocol") not in (None, QUALITY_PROTOCOL)
        or (
            "input_features" in spec
            and (
                spec["family"] not in {"data_valuation", "dataset_valuation"}
                or "model_profile" not in spec
                or type(spec["input_features"]) is not int
                or spec["input_features"] < 1
            )
        )
        or (
            "feature_rule" in spec
            and (
                spec["family"] not in {"feature_selection", "data_valuation", "dataset_valuation"}
                or "model_profile" not in spec
                or spec["feature_rule"] not in {"all", "nested"}
            )
        )
        or (
            "device" in spec
            and (
                (spec["family"] != "tabpfn" and spec.get("model_profile") != "tabpfn_prediction")
                or spec["device"] not in ("cpu", "cuda")
            )
        )
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
            options = {
                key: spec[key]
                for key in (
                    "dataset",
                    "n_players",
                    "model_profile",
                    "device",
                    "quality_protocol",
                    "input_features",
                    "feature_rule",
                )
                if key in spec and (key != "quality_protocol" or factory is make_family)
            }
            if "model_profile" in spec:
                options["model_cache"] = str(output.parent / ".models")
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
                _synchronize(spec)
                started = time.perf_counter()
                batch = np.asarray(game(coalitions), dtype=float)
                _synchronize(spec)
                elapsed = time.perf_counter() - started
                if batch.shape != values.shape:
                    message = "The family produced incorrectly shaped batch values."
                    raise ValueError(message)  # noqa: TRY301
                values[:] = batch
                costs[:] = elapsed / len(values)
                batches.append(_timing_batch(0, len(values), elapsed, metadata))
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
                repeated = {
                    "batch": np.asarray(game(probe)),
                    "repeat": np.asarray(game(probe)),
                    "reversed": np.asarray(game(probe[::-1]))[::-1],
                    "singletons": np.concatenate([game(row[None]) for row in probe]),
                }
                tolerance = {"rtol": 1e-8, "atol": 1e-10}
                if metadata.get("model_profile") == "tabpfn_prediction":
                    # Float32 inference depends slightly on batch shape. Qualify a
                    # bounded numerical realization; exact truth uses saved payoffs.
                    bound = 64 * np.finfo(np.float32).eps * max(1.0, float(np.max(np.abs(values))))
                    tolerance = {"rtol": 0.0, "atol": bound}
                    metadata["oracle_validation"] = {
                        "protocol": "float32 frozen batch realization",
                        "absolute_tolerance": float(bound),
                        "observed_differences": {
                            name: float(np.max(np.abs(values[positions] - v)))
                            for name, v in repeated.items()
                        },
                        "reference": "Exact coefficients of the saved canonical payoff table",
                    }
                for observed in repeated.values():
                    if observed.shape != (len(positions),) or not np.isfinite(observed).all():
                        message = "The oracle returned invalid qualification probes."
                        raise ValueError(message)  # noqa: TRY301 -- local diagnostic retained
                    np.testing.assert_allclose(values[positions], observed, **tolerance)
                    if metadata.get("model_profile") != "tabpfn_prediction":
                        np.testing.assert_allclose(repeated["batch"], observed, **tolerance)
            artifact = output / f"{instance_id}.npz"
            np.savez_compressed(
                artifact,
                values=values,
                evaluation_seconds=costs,
                evaluation_batches=json.dumps(batches, allow_nan=False),
            )
            profiles = {
                json.dumps(
                    {
                        key: batch[key]
                        for key in ("cpu_model", "threads", "preparation_hardware")
                        if key in batch
                    },
                    sort_keys=True,
                )
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
            metadata["fourier_spectrum"] = fourier_spectrum(values, n)
            if spec.get("quality_protocol") == QUALITY_PROTOCOL:
                metadata["quality_protocol"] = QUALITY_PROTOCOL
                metadata["game_quality"] = payoff_diagnostics(values, n)
                clustering_diagnostics(values, metadata, metadata["game_quality"])
                if metadata.get("synthetic"):
                    metadata["game_quality"]["control_reasons"].append("synthetic_control")
                if metadata.get("stochastic_frozen"):
                    stability = (
                        imputation_stability(game, seed=instance_seed or 0)
                        if name in {"local_gaussian", "local_copula", "local_conditional"}
                        else {"status": "unqualified", "reason": "No matched stability pilot"}
                    )
                    metadata["imputation_stability"] = stability
                    if stability["status"] != "stable":
                        metadata["game_quality"]["control_reasons"].append(
                            "unqualified_stochastic_payoffs"
                        )
                if metadata["game_quality"]["control_reasons"]:
                    metadata["game_quality"]["role"] = "control"
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
