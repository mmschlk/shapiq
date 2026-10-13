"""Freeze bounded representatives of shipped families into exact coalition tables."""

from __future__ import annotations

import json
import math
import re
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np

from shapiq.game_theory import ExactComputer
from shapiq_benchmark.families import FAMILY_CATALOG, MAX_ENUMERATION_PLAYERS, make_family
from shapiq_benchmark.games import truth_dict
from shapiq_benchmark.payoff_cache import (
    CHUNK_PROTOCOL,
    COST_PROTOCOL,
    cost_batch as _cost_batch,
)
from shapiq_benchmark.quality import (
    QUALITY_PROTOCOL,
    payoff_diagnostics,
)
from shapiq_benchmark.spectrum import fourier_spectrum

CATALOG = FAMILY_CATALOG


def _timing_batch(start: int, stop: int, elapsed: float, metadata: dict) -> dict:
    """Keep accelerator details alongside CPU details in cached timing records."""
    batch = _cost_batch(start, stop, elapsed)
    if "preparation_hardware" in metadata:
        batch["preparation_hardware"] = metadata["preparation_hardware"]
    return batch


def prepare_families(
    names: list[str | dict], targets: list[dict], output: Path, *, instance_seed: int | None = None
) -> tuple[list, list]:
    """Qualify each recipe, preserving unavailable families in the coverage catalog.

    Sampled games are explicitly frozen in canonical bitmask order. Their exact
    coefficients describe that saved realization, not an expected stochastic game.
    """
    from shapiq_benchmark.campaign_runner import table_game

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
                spec["family"]
                not in {
                    "feature_selection",
                    "data_valuation",
                    "dataset_valuation",
                    "local_baseline",
                }
                or "model_profile" not in spec
                or spec["feature_rule"] not in {"all", "nested"}
            )
        )
        or (spec.get("device", "cpu") != "cpu")
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
            factory = make_family
            options: dict = {
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
                if key in spec
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
            coalitions = ((np.arange(2**n)[:, None] >> np.arange(n)) & 1).astype(bool)
            started = time.perf_counter()
            batch = np.asarray(game(coalitions), dtype=float)
            elapsed = time.perf_counter() - started
            if batch.shape != values.shape:
                message = "The family produced incorrectly shaped batch values."
                raise ValueError(message)  # noqa: TRY301
            values[:] = batch
            costs[:] = elapsed / len(values)
            batches.append(_timing_batch(0, len(values), elapsed, metadata))
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
                for observed in repeated.values():
                    if observed.shape != (len(positions),) or not np.isfinite(observed).all():
                        message = "The oracle returned invalid qualification probes."
                        raise ValueError(message)  # noqa: TRY301 -- local diagnostic retained
                    np.testing.assert_allclose(values[positions], observed, rtol=1e-8, atol=1e-10)
                    np.testing.assert_allclose(repeated["batch"], observed, rtol=1e-8, atol=1e-10)
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
            exact = ExactComputer(table_game(values, n), n_players=n)
            payoff_std = float(np.std(values, dtype=np.longdouble))
            metadata["fourier_spectrum"] = fourier_spectrum(values, n)
            if spec.get("quality_protocol") == QUALITY_PROTOCOL:
                metadata["quality_protocol"] = QUALITY_PROTOCOL
                metadata["game_quality"] = payoff_diagnostics(values, n)
                if metadata.get("synthetic"):
                    metadata["game_quality"]["control_reasons"].append("synthetic_control")
                if metadata["game_quality"]["control_reasons"]:
                    metadata["game_quality"]["role"] = "control"
            qualified = []
            for target in targets:
                index, order = target["index"], target["order"]
                encoded = truth_dict(exact(index, order=order))
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
