"""The game of the talk's figures "lattice-sv" and "lattice-sii" (../assets/figure-lattice.js).

Four players of the giraffe of the FIxLIP demo, as the orders demo explains it: SigLIP 2 matches
the photo ``demos/fixlip/assets/giraffe-tree.jpg`` with its caption "a giraffe has its front legs
spread to bend down for a drink of water at a pond." (demos/orders/compute/fixlip.py, ``--photo
giraffe-tree``). Its players are the 64 patches of the photo (8 x 8, row by row) and the 19 tokens
of the caption. The four of the figures are

- patch 44, the giraffe's head as it drinks,
- patch 17, the trunk of the dead tree behind it,
- the token "giraffe" (player 65),
- the token "water" (player 78).

The game is the orders demo's full decomposition of SigLIP 2 (its order 10: the Möbius transform
of the ProxySHAP proxy, ``orders["10"]`` of ``results/fixlip_giraffe-tree.json``) with the other 79
players absent: the value of a coalition T of the four is the sum of the Möbius values of all
groups L within T, the proxy's (centered) cosine similarity of the photo with only T's patches and
the caption with only T's tokens. The results keep the 2,000 strongest groups of the order; 12 of
the 15 groups of the four are among them, and the other 3 ({17, 44}, {17, "water"} and
{17, 44, "water"}), weaker than every kept group (|m| < 0.0024), count as 0. The values are
rescaled to thousandths of the cosine similarity and rounded to whole numbers, so that every sum
the figures show adds up exactly; the figures compute the marginal contributions, the Shapley
values, the discrete derivatives and the Shapley interactions from these whole numbers.

No new evaluation of SigLIP 2 is needed. Run from this folder, with the orders demo's results of
the giraffe (by default where ``fixlip.py`` writes them):

    python lattice.py [--results ../../../demos/orders/compute/results]
    # rewrites the block LATTICE in ../assets/figure-lattice.js
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from pathlib import Path

HERE = Path(__file__).parent
PLAYERS = [44, 17, 65, 78]  # the giraffe's head, the dead tree, "giraffe", "water"
SCALE = 1000  # thousandths of the cosine similarity


def main() -> None:
    """Restrict the full decomposition to the four players and write the block LATTICE."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--results", type=Path, default=HERE / "../../../demos/orders/compute/results"
    )
    args = parser.parse_args()
    data = json.loads((args.results / "fixlip_giraffe-tree.json").read_text())
    moebius = {tuple(sorted(group)): value for group, value in data["orders"]["10"]["values"]}
    weakest = min(abs(value) for group, value in moebius.items() if len(group) > 1)

    values = []  # by bit mask: player PLAYERS[i] is bit i
    for mask in range(1 << len(PLAYERS)):
        members = sorted(p for i, p in enumerate(PLAYERS) if mask >> i & 1)
        groups = [g for r in range(1, len(members) + 1) for g in itertools.combinations(members, r)]
        missing = [g for g in groups if g not in moebius]
        values.append(round(SCALE * sum(moebius.get(g, 0.0) for g in groups)))
        if missing:
            print(f"{members}: {len(missing)} groups below |m| = {weakest:.4f} count as 0")

    block = {
        "players": [
            {"kind": "patch", "patch": 44, "name": "giraffe\u2019s head"},
            {"kind": "patch", "patch": 17, "name": "dead tree"},
            {"kind": "word", "word": data["tokens"][65 - 64]},
            {"kind": "word", "word": data["tokens"][78 - 64]},
        ],
        "grid": data["grid"],
        "values": values,
    }
    figure = HERE / "../assets/figure-lattice.js"
    text = figure.read_text()
    start, end = "  // LATTICE (compute/lattice.py)\n", "  // /LATTICE\n"
    line = (
        "  const LATTICE = " + json.dumps(block, ensure_ascii=False, separators=(",", ":")) + ";\n"
    )
    text, count = re.subn(
        re.escape(start) + ".*?" + re.escape(end),
        lambda _: start + line + end,
        text,
        flags=re.DOTALL,
    )
    if not count:
        msg = "no block LATTICE in figure-lattice.js"
        raise ValueError(msg)
    figure.write_text(text)
    print(values)


if __name__ == "__main__":
    main()
