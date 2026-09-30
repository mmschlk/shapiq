"""Freeze bounded representatives of shipped families into exact coalition tables."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np

from shapiq.game_theory import ExactComputer
from shapiq_benchmark.families import FAMILY_CATALOG, make_family
from shapiq_benchmark.games import truth_dict
from shapiq_benchmark.media import EXTRA_CATALOG, make_extra

CATALOG = {**FAMILY_CATALOG, **EXTRA_CATALOG}


def prepare_families(
    names: list[str | dict], targets: list[dict], output: Path, *, instance_seed: int | None = None
) -> tuple[list, list]:
    """Qualify each recipe, preserving unavailable families in the coverage catalog.

    Sampled games are explicitly frozen in canonical bitmask order. Their exact
    coefficients describe that saved realization, not an expected stochastic game.
    """
    from shapiq_benchmark.runner import table_game

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
            and (type(spec["n_players"]) is not int or not 1 <= spec["n_players"] <= 12)
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
            if not 1 <= n <= 12:
                message = "Exhaustive family preparation is limited to twelve players."
                raise ValueError(message)  # noqa: TRY301 -- preserve per-family coverage failures
            coalitions = ((np.arange(2**n)[:, None] >> np.arange(n)) & 1).astype(bool)
            # A fixed full-batch ordering is part of the sampled-game definition.
            values = np.asarray(game(coalitions), dtype=float)
            if values.shape != (2**n,) or not np.all(np.isfinite(values)):
                message = "The family produced nonfinite or incorrectly shaped values."
                raise ValueError(message)  # noqa: TRY301 -- preserve per-family coverage failures
            if not metadata.get("stochastic_frozen", False):
                positions = np.random.default_rng(0).choice(len(coalitions), 8)
                probe = coalitions[positions]
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
            np.savez_compressed(artifact, values=values)
            exact = ExactComputer(table_game(values, n), n_players=n)
            qualified = []
            for target in targets:
                index, order = target["index"], target["order"]
                truth = exact(index, order=order)
                encoded = truth_dict(truth)
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
                            "materialization": "ascending bitmask, player zero least significant, one full batch",
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
