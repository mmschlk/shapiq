"""Explain how SigLIP 2 matches the FIxLIP demo's elephant at every order: ``results/fixlip.json``.

The players are SigLIP 2's 8 x 8 patches of the photo and the 10 tokens of its caption ("a small
elephant standing next to a white bird."), and the value of a coalition is the cosine similarity of
the photo with only the coalition's patches and the caption with only its tokens: the game of the
FIxLIP demo, imported from its scripts (``../../fixlip/compute/explain.py``).

SigLIP 2 is a black box: unlike trees or graph neural networks, it has no structure that would give
its interactions exactly. ProxySHAP fits an XGBoost proxy to coalitions of the game drawn from the
Shapley distribution (``faithfulness.py``), the distribution FSII is faithful to, and reads the
proxy's interactions off its trees exactly. Its sampler evaluates the empty and the full coalition
in any case (their weights are ignored), all coalitions of a size it would draw more often than the
size has coalitions (here the 74 of one player and the 74 of 73 players), and draws the others by
the weights of their sizes, each with its complement. Without the residual adjustment, FSII of the
proxy's depth is the full decomposition of the proxy: its trees split on at most that many players,
so no larger group has a value, and every lower order of FSII follows from it. The depth, 6, is the
best of 3 to 8 in a 5-fold cross-validation on the coalitions.

Every order is scored by its R^2 under the Shapley distribution (as for the molecule and the
houses), on 512 coalitions drawn from it independently of the proxy's (a coalition of 1 or 73
players is one the proxy has seen: it saw all of them), against the model itself. The game is
centered, and ProxySHAP gives the empty group the game's value of the empty coalition, 0, where the
proxy predicts -0.008 (the empty coalition shares its leaves with coalitions of one or two players,
whose values are about -0.007): the full decomposition reproduces the proxy up to this constant,
and fits the model less well than the proxy itself (R^2 = 0.967 against 0.984).

Usage::

    python fixlip.py

The game's evaluations are kept in ``results/fixlip_coalitions.npz``: the model is evaluated once
per coalition (10,000 for the proxy and 512 to score it, a run of about nine minutes on four CPU
cores), and a run again only fits the proxy.
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
from faithfulness import fsii, predictions, r2, sample, size_means, size_weights  # in this folder
from xgboost import XGBRegressor

import shapiq
from shapiq.game_theory.moebius_converter import MoebiusConverter

HERE = Path(__file__).resolve().parent
FIXLIP = HERE.parent.parent / "fixlip" / "compute"  # the FIxLIP demo's scripts and config
sys.path.insert(0, str(FIXLIP))
import explain as fixlip  # noqa: E402  (the FIxLIP demo's game)

PHOTO, MODEL = "elephant", "siglip2"
BUDGET = 10_000  # the coalitions the proxy is fit to
N_TEST = 512  # the coalitions every order is scored on, drawn independently of the proxy's
DEPTH = 6  # the proxy's depth, its highest order of interaction (the best in cross-validation)
PROXY = {"n_estimators": 1000, "max_depth": DEPTH, "learning_rate": 0.05, "reg_lambda": 5.0}


class CachedGame:
    """The game, with every evaluation kept on disk: the model sees every coalition once."""

    def __init__(self, game: shapiq.Game, path: Path) -> None:
        """Load the evaluations of earlier runs."""
        self.game, self.path = game, path
        self.values: dict[bytes, float] = {}
        self.asked: dict[bytes, float] = {}  # the coalitions of this run, with their values
        if path.exists():
            stored = np.load(path)
            for coalition, value in zip(stored["coalitions"], stored["values"], strict=True):
                self.values[coalition.tobytes()] = float(value)

    def __call__(self, coalitions: np.ndarray) -> np.ndarray:
        """Return the game's (centered) values, evaluating the coalitions not seen before."""
        coalitions = np.asarray(coalitions, dtype=bool)
        missing = list(
            {row.tobytes(): row for row in coalitions if row.tobytes() not in self.values}.values()
        )
        for start in range(0, len(missing), 256):
            chunk = np.array(missing[start : start + 256])
            for row, value in zip(chunk, self.game(chunk), strict=True):
                self.values[row.tobytes()] = float(value)
            print(f"  {start + len(chunk)} of {len(missing)} coalitions evaluated", flush=True)
        if missing:
            keys = list(self.values)
            rows = np.array([np.frombuffer(key, dtype=bool) for key in keys])
            np.savez_compressed(
                self.path, coalitions=rows, values=np.array([self.values[k] for k in keys])
            )
        for row in coalitions:
            self.asked[row.tobytes()] = self.values[row.tobytes()]
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


def main() -> None:
    """Evaluate the game, fit the proxy, and write FSII of every order: ``results/fixlip.json``."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)

    game, tokens = build_game(args.results)
    n = game.n_players
    cached = CachedGame(game, args.results / "fixlip_coalitions.npz")

    # the full decomposition of the proxy: FSII of its highest order, without adjustment
    start = time.time()
    approximator = shapiq.approximator.ProxySHAP(
        n=n,
        index="FSII",
        max_order=DEPTH,
        sampling_weights=size_weights(n),  # the Shapley distribution's sizes
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

    test = sample(n, N_TEST, seed=1)
    y = cached(test)
    # the mean value of every size, the yardstick of R^2 beyond size: from all coalitions of this run,
    # the proxy's and the held-out ones, all drawn alike within a size (the held-out ones alone are
    # too few for 73 sizes)
    asked = np.array([np.frombuffer(key, dtype=bool) for key in cached.asked])
    means = size_means(np.array(list(cached.asked.values())), asked.sum(axis=1))
    orders = {}
    for order in range(1, DEPTH + 1):
        values = fsii(converter, order)
        fit = r2(y, predictions(values, test), test.sum(axis=1), means=means)
        strongest = sorted(
            ((key, value) for key, value in values.items() if key),
            key=lambda item: (len(item[0]), -abs(item[1])),
        )
        orders[order] = {"r2": fit, "values": [[list(key), value] for key, value in strongest]}
        print(f"order {order}: {len(strongest)} values, R^2 {fit:.3f}", flush=True)

    # the lower orders from the full decomposition are ProxySHAP's own at that order
    pairs = shapiq.approximator.ProxySHAP(
        n=n,
        index="FSII",
        max_order=2,
        sampling_weights=size_weights(n),
        proxy_model=XGBRegressor(**PROXY, random_state=0, n_jobs=4),
        adjustment=False,
        random_state=0,
    ).approximate(budget=BUDGET, game=cached)
    difference = max(
        abs(pairs[key] - converter("FSII", 2)[key]) for key in pairs.interaction_lookup
    )
    print(f"largest difference to ProxySHAP of order 2: {difference:.1e}")

    result = {
        "index": "FSII",
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
