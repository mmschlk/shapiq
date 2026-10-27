"""The illustrative explanation of the FIxLIP demo's elephant at every order: ``results/fixlip.json``.

Unlike the molecule's, this explanation is not computed from a model: it illustrates what the
orders of an explanation of a vision-language model add, on the photo and caption of the FIxLIP
demo ("a small elephant standing next to a white bird."). The players are SigLIP 2's 8 x 8 patches
of the photo and the 10 tokens of the caption, as in the demo. A toy game is defined by its Möbius
transform: a few main effects, and interactions of patches and words that the demo's explanation
suggests, e.g. the egret's patch with "white" and "bird" (order 2 and 3), and "next to" with both
the elephant and the egret (order 4 and more). ``shapiq`` converts it into k-SII of every order, as
for the molecule.

The R^2 of the orders are illustrative too: they show how much more of a vision-language model's
output higher orders typically explain, and are not computed from the toy game.

Usage::

    python fixlip.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from shapiq import InteractionValues
from shapiq.game_theory.moebius_converter import MoebiusConverter

HERE = Path(__file__).resolve().parent
GRID = 8  # SigLIP 2 ViT-B/32 at 256 pixels: 8 x 8 patches, as in the FIxLIP demo
TOKENS = ["a", "small", "elephant", "standing", "next", "to", "a", "white", "bird", "."]
N_PLAYERS = GRID * GRID + len(TOKENS)  # the patches, then the tokens
A, SMALL, ELEPHANT, STANDING, NEXT, TO, A2, WHITE, BIRD, _ = range(GRID * GRID, N_PLAYERS)
HEAD, EGRET = 35, 40  # the elephant's head (row 4, column 3) and the egret (row 5, column 0)
BODY = [36, 38, 21, 20, 39]  # the patches of the elephant's body and ears
R2 = {1: 0.31, 2: 0.80, 3: 0.95, 4: 0.98, 8: 1.0}  # illustrative, by order (8: all orders)

# the toy game's Möbius transform: the interactions of patches and words, by order
MOEBIUS: dict[tuple[int, ...], float] = {
    # main effects: the elephant's patches and word, a little of the egret, and the grass
    (HEAD,): 0.020,
    **{
        (patch,): value
        for patch, value in zip(BODY, [0.006, 0.006, 0.005, 0.004, 0.004], strict=True)
    },
    (EGRET,): 0.004,
    (41,): 0.002,
    (ELEPHANT,): 0.030,
    (BIRD,): 0.006,
    (WHITE,): -0.004,
    (SMALL,): -0.002,
    (TO,): 0.003,
    (A2,): 0.002,
    (28,): -0.004,
    (56,): -0.003,
    (27,): -0.003,
    (60,): -0.002,
    # pairs: the words with the patches they describe
    (HEAD, ELEPHANT): 0.030,
    **{
        (patch, ELEPHANT): value
        for patch, value in zip(BODY, [0.008, 0.008, 0.006, 0.005, 0.005], strict=True)
    },
    (EGRET, BIRD): 0.030,
    (41, BIRD): 0.006,
    (32, BIRD): 0.004,
    (48, BIRD): 0.004,
    (EGRET, WHITE): 0.016,
    (27, SMALL): 0.004,
    (HEAD, SMALL): 0.004,
    (47, STANDING): 0.003,
    (EGRET, NEXT): 0.008,
    (EGRET, TO): 0.005,
    (HEAD, BIRD): -0.010,
    (EGRET, ELEPHANT): -0.008,
    (28, WHITE): -0.004,
    (36, BIRD): -0.004,
    # triples: "white bird" on the egret, "small elephant" on its head
    (EGRET, WHITE, BIRD): 0.022,
    (HEAD, SMALL, ELEPHANT): 0.012,
    (47, ELEPHANT, STANDING): 0.008,
    (EGRET, 41, BIRD): 0.008,
    (HEAD, 36, ELEPHANT): 0.010,
    (HEAD, WHITE, BIRD): -0.010,
    (EGRET, SMALL, ELEPHANT): -0.006,
    # higher orders: "next to" needs both animals, and the words that name them
    (HEAD, EGRET, NEXT, TO): 0.020,
    (HEAD, EGRET, ELEPHANT, BIRD): 0.016,
    (EGRET, A2, WHITE, BIRD): 0.010,
    (HEAD, A, SMALL, ELEPHANT): 0.008,
    (HEAD, EGRET, ELEPHANT, WHITE): -0.008,
    (HEAD, EGRET, ELEPHANT, NEXT, TO): 0.014,
    (HEAD, EGRET, NEXT, TO, BIRD): 0.012,
    (HEAD, EGRET, ELEPHANT, NEXT, TO, BIRD): 0.016,
    (HEAD, 36, EGRET, ELEPHANT, WHITE, BIRD): 0.008,
    (HEAD, EGRET, ELEPHANT, STANDING, NEXT, TO, BIRD): 0.010,
    (HEAD, EGRET, SMALL, ELEPHANT, NEXT, TO, WHITE, BIRD): 0.008,
}


def main() -> None:
    """Convert the toy game into k-SII of the orders of ``R2``, and write ``results/fixlip.json``."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)

    lookup = {(): 0}
    for interaction in MOEBIUS:
        lookup[tuple(sorted(interaction))] = len(lookup)
    moebius = InteractionValues(
        values=np.array([0.0, *MOEBIUS.values()]),
        index="Moebius",
        max_order=N_PLAYERS,
        min_order=0,
        n_players=N_PLAYERS,
        interaction_lookup=lookup,
        baseline_value=0.0,
    )
    converter = MoebiusConverter(moebius)
    orders = {}
    for order, r2 in R2.items():
        explanation = converter("k-SII", order)
        values = {
            interaction: float(explanation[interaction])
            for interaction in explanation.interaction_lookup
            if interaction and abs(explanation[interaction]) > 1e-9
        }
        strongest = sorted(values.items(), key=lambda item: (len(item[0]), -abs(item[1])))
        orders[order] = {
            "r2": r2,
            "values": [[list(interaction), value] for interaction, value in strongest],
        }
        print(f"order {order}: {len(values)} values", flush=True)

    result = {
        "grid": GRID,
        "tokens": TOKENS,
        "largest_order": max(len(interaction) for interaction in MOEBIUS),
        "orders": orders,
    }
    (args.results / "fixlip.json").write_text(json.dumps(result))


if __name__ == "__main__":
    main()
