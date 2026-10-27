"""Explain how SigLIP 2 matches the FIxLIP demo's elephant at every order: ``results/fixlip.json``.

The players are SigLIP 2's 8 x 8 patches of the photo and the 10 tokens of its caption ("a small
elephant standing next to a white bird."), and the value of a coalition is the cosine similarity of
the photo with only the coalition's patches and the caption with only its tokens: the game of the
FIxLIP demo, imported from its scripts (``../../fixlip/compute/explain.py``).

SigLIP 2 is a black box: unlike trees or graph neural networks, it has no structure that would give
its interactions exactly. ProxySHAP fits an XGBoost proxy to coalitions of the game, as many of every
size, and reads the proxy's interactions off its trees exactly. Without the residual adjustment, k-SII
of the proxy's depth is the full decomposition of the proxy: its trees split on at most that many
players, so no larger group has a value, and every lower order of k-SII follows from it.

Every order is scored by its R^2 on coalitions it was not fit to, every player present with
probability 1/2 (as for the molecule and the houses), against the model itself: the full
decomposition reproduces the proxy exactly, and the model as well as the proxy does.

Usage::

    python fixlip.py

The game's evaluations are kept in ``results/fixlip_coalitions.npz``: the model is evaluated once per
coalition (5,000 and 512 held out, about ten minutes on four CPU cores), and a run again only fits
the proxy.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import tomllib
from pathlib import Path

import numpy as np
import torch
from xgboost import XGBRegressor

import shapiq
from shapiq.game_theory.moebius_converter import MoebiusConverter

HERE = Path(__file__).resolve().parent
FIXLIP = HERE.parent.parent / "fixlip" / "compute"  # the FIxLIP demo's scripts and config
sys.path.insert(0, str(FIXLIP))
import explain as fixlip  # noqa: E402  (the FIxLIP demo's game)

PHOTO, MODEL = "elephant", "siglip2"
BUDGET = 5_000  # the coalitions the proxy is fit to
N_TEST = 512  # the coalitions every order is scored on
DEPTH = 5  # the proxy's depth: its highest order of interaction
PROXY = {"n_estimators": 1000, "max_depth": DEPTH, "learning_rate": 0.05, "reg_lambda": 5.0}


class CachedGame:
    """The game, with every evaluation kept on disk: the model sees every coalition once."""

    def __init__(self, game: shapiq.Game, path: Path) -> None:
        """Load the evaluations of earlier runs."""
        self.game, self.path = game, path
        self.values: dict[bytes, float] = {}
        if path.exists():
            stored = np.load(path)
            for coalition, value in zip(stored["coalitions"], stored["values"], strict=True):
                self.values[coalition.tobytes()] = float(value)

    def __call__(self, coalitions: np.ndarray) -> np.ndarray:
        """Return the game's (centered) values, evaluating the coalitions not seen before."""
        coalitions = np.asarray(coalitions, dtype=bool)
        missing = list({row.tobytes(): row for row in coalitions if row.tobytes() not in self.values}.values())
        for start in range(0, len(missing), 256):
            chunk = np.array(missing[start : start + 256])
            for row, value in zip(chunk, self.game(chunk), strict=True):
                self.values[row.tobytes()] = float(value)
            print(f"  {start + len(chunk)} of {len(missing)} coalitions evaluated", flush=True)
        if missing:
            keys = list(self.values)
            rows = np.array([np.frombuffer(key, dtype=bool) for key in keys])
            np.savez_compressed(self.path, coalitions=rows, values=np.array([self.values[k] for k in keys]))
        return np.array([self.values[row.tobytes()] for row in coalitions])


def build_game(results: Path) -> tuple[shapiq.Game, list[str]]:
    """Return the FIxLIP demo's game of SigLIP 2, the elephant and its caption, and the tokens."""
    config = tomllib.loads((FIXLIP / "config.toml").read_text(encoding="utf-8"))
    photo = next(p for p in config["photos"] if p["id"] == PHOTO)
    model = next(m for m in config["models"] if m["id"] == MODEL)
    name = f"{model['id']}_demo"  # a name starting with "siglip" runs as SigLIP (explain.py)
    fixlip.image_text.IMAGE_TEXT_MODEL_IDS[name] = model["checkpoint"]
    embedder = fixlip.image_text.ImageTextTokenModel(
        np.asarray(fixlip.load_photo(photo, results)),
        (model["grid"], model["grid"]),
        model=name,
        mask_strategy="mask",
        batch_size=32,
    )
    tokens = fixlip.caption_tokens(fixlip.Network(embedder), photo["caption"])
    game = fixlip.SimilarityGame(embedder, photo["caption"], model["grid"] ** 2, len(tokens))
    return game, [token["text"] for token in tokens]


def predictions(values: dict[tuple[int, ...], float], coalitions: np.ndarray) -> np.ndarray:
    """Return the sum of the values of every coalition's subsets: the explanation's prediction."""
    players = [interaction for interaction in values if interaction]
    members = np.zeros((len(players), coalitions.shape[1]))
    for row, interaction in enumerate(players):
        members[row, list(interaction)] = 1
    inside = coalitions.astype(float) @ members.T == members.sum(axis=1)
    return inside @ np.array([values[interaction] for interaction in players])


def main() -> None:
    """Evaluate the game, fit the proxy, and write k-SII of every order to ``results/fixlip.json``."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)

    game, tokens = build_game(args.results)
    n = game.n_players
    cached = CachedGame(game, args.results / "fixlip_coalitions.npz")

    # the full decomposition of the proxy: k-SII of its highest order, without adjustment
    start = time.time()
    approximator = shapiq.approximator.ProxySHAP(
        n=n,
        index="k-SII",
        max_order=DEPTH,
        proxy_model=XGBRegressor(**PROXY, random_state=0, n_jobs=4),
        adjustment=False,
        random_state=0,
    )
    full = approximator.approximate(budget=BUDGET, game=cached)
    print(f"proxy fit and read in {time.time() - start:.0f} s: {len(full.values)} values")
    moebius = shapiq.InteractionValues(
        values=full.values,
        index="Moebius",
        max_order=n,
        min_order=0,
        n_players=n,
        interaction_lookup=full.interaction_lookup,
        baseline_value=full.baseline_value,
    )
    converter = MoebiusConverter(moebius)

    test = np.random.default_rng(1).random((N_TEST, n)) < 0.5
    y = cached(test)
    orders = {}
    for order in range(1, DEPTH + 1):
        explanation = converter("k-SII", order)
        values = {
            key: float(explanation[key])
            for key in explanation.interaction_lookup
            if key and abs(explanation[key]) > 1e-9
        }
        residual = y - predictions(values, test)
        r2 = float(1 - np.sum(residual**2) / np.sum((y - y.mean()) ** 2))
        strongest = sorted(values.items(), key=lambda item: (len(item[0]), -abs(item[1])))
        orders[order] = {"r2": r2, "values": [[list(key), value] for key, value in strongest]}
        print(f"order {order}: {len(values)} values, R^2 {r2:.3f}", flush=True)

    # the lower orders from the full decomposition are ProxySHAP's own at that order
    pairs = shapiq.approximator.ProxySHAP(
        n=n,
        index="k-SII",
        max_order=2,
        proxy_model=XGBRegressor(**PROXY, random_state=0, n_jobs=4),
        adjustment=False,
        random_state=0,
    ).approximate(budget=BUDGET, game=cached)
    difference = max(abs(pairs[key] - converter("k-SII", 2)[key]) for key in pairs.interaction_lookup)
    print(f"largest difference to ProxySHAP of order 2: {difference:.1e}")

    result = {
        "grid": 8,
        "tokens": tokens,
        "largest_order": DEPTH,
        "budget": BUDGET,
        "similarity": float(game.empty_value + cached(np.ones((1, n), dtype=bool))[0]),
        "orders": orders,
    }
    (args.results / "fixlip.json").write_text(json.dumps(result))


if __name__ == "__main__":
    main()
