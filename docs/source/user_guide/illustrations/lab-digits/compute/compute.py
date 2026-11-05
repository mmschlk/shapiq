"""Compute the data of the illustration "lab-digits" (chapter 2.1 of the user guide).

Definition 2.1 on an image model: a handwritten digit of scikit-learn's digits data (8 x 8 pixels,
ink from 0 to 16), a multi-layer perceptron that classifies it, and explanations of the probability
it gives the class it predicts. The features are the 16 patches of 2 x 2 pixels, a 4 x 4 grid,
numbered 1, ..., 16 row by row.

- the explanation game of the digit: the predicted probability with only the patches of a coalition
  kept and the others erased (their pixels set to 0, blank paper), for all 2^16 = 65,536
  coalitions, in batches;
- numbers: the Shapley values of the patches and their pairwise Shapley interactions (k-SII of
  order 2), with shapiq's ``ExactComputer``, and the Shapley values of the four rows of patches,
  each row one player (``partition_values``);
- curves: the probability as the ink of one patch (its four pixels), or of one row of patches, goes
  from 0 to 16 and the rest of the digit stays (its ICE curve); and for the strongest pairs, the
  part of the probability that neither patch explains alone as the ink of both varies (the pure
  interaction surface).

Probabilities are rounded to 4 decimals, the curves and surfaces to 3.

Run it from the repository's root with the docs' dependencies installed; it writes
``../assets/data.js``::

    python docs/source/user_guide/illustrations/lab-digits/compute/compute.py
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

import shapiq

SIDE = 8  # pixels per side of a digit
PATCH = 2  # pixels per side of a patch
GRID = SIDE // PATCH  # patches per side: 4, so 16 patches
INK = 16  # the darkest ink of a pixel in the data
DIGIT = 435  # the explained digit: its place in the held-out digits, a clear 7
HIDDEN = (128,)  # the hidden layer of the perceptron
ALPHA = 0.3  # its L2 penalty, which keeps its probabilities from saturating
TOP_PAIRS = 6  # the pairs with the strongest interactions, which get a surface
SURFACE_STEP = 2  # the ink of a surface's grid: 0, 2, ..., 16
BATCH = 8192  # images per call of the model
SEED = 0
OUT = Path(__file__).resolve().parents[1] / "assets" / "data.js"


def patch_pixels() -> np.ndarray:
    """A mask per patch of the pixels it covers: patch k is in row k // 4 and column k % 4."""
    masks = np.zeros((GRID * GRID, SIDE * SIDE), dtype=bool)
    for k in range(GRID * GRID):
        row, col = divmod(k, GRID)
        for dr, dc in itertools.product(range(PATCH), repeat=2):
            masks[k, (PATCH * row + dr) * SIDE + PATCH * col + dc] = True
    return masks


def main() -> None:
    """Train the model, explain the digit, and write the data."""
    digits = load_digits()
    x_train, x_test, y_train, y_test = train_test_split(
        digits.data, digits.target, test_size=0.25, random_state=SEED, stratify=digits.target
    )
    model = MLPClassifier(
        hidden_layer_sizes=HIDDEN, alpha=ALPHA, max_iter=2000, random_state=SEED
    ).fit(x_train / INK, y_train)
    accuracy = float(model.score(x_test / INK, y_test))

    digit = x_test[DIGIT]
    label = int(y_test[DIGIT])
    target = int(model.predict(digit[None] / INK)[0])  # the class it predicts: explained

    def predict(images: np.ndarray) -> np.ndarray:
        """The probability of the explained class for each image, in batches."""
        out = np.empty(len(images))
        for start in range(0, len(images), BATCH):
            batch = images[start : start + BATCH] / INK
            out[start : start + BATCH] = model.predict_proba(batch)[:, target]
        return out

    masks = patch_pixels()
    n = len(masks)

    def game(coalitions: np.ndarray) -> np.ndarray:
        """The probability with the patches of each coalition kept and the others erased."""
        kept = coalitions.astype(float) @ masks  # per coalition, 1 for the pixels it keeps
        return predict(kept * digit[None])

    computer = shapiq.ExactComputer(game=game, n_players=n)
    shapley = computer("SV", order=1)
    ksii = computer("k-SII", order=2)
    values, lookup = computer.game_values, computer.coalition_lookup

    def value(players: tuple[int, ...]) -> float:
        return float(values[lookup[tuple(sorted(players))]])

    prediction, empty = value(tuple(range(n))), value(())

    def partition_values(parts: list[tuple[int, ...]]) -> list[float]:
        """The Shapley values of the parts of a partition, in the game with each part one player."""
        m = len(parts)
        result = []
        for p, part in enumerate(parts):
            others = [q for q in range(m) if q != p]
            total = 0.0
            for size in range(m):
                weight = math.factorial(size) * math.factorial(m - size - 1) / math.factorial(m)
                for rest in itertools.combinations(others, size):
                    members = tuple(i for q in rest for i in parts[q])
                    total += weight * (value(members + part) - value(members))
            result.append(total)
        return result

    rows = [tuple(range(GRID * r, GRID * r + GRID)) for r in range(GRID)]
    row_values = partition_values(rows)

    def with_ink(changes: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        """Predict the digit with the pixels of each mask set to the (broadcast) inks."""
        shape = np.broadcast_shapes(*(ink.shape for _, ink in changes))
        images = np.repeat(digit[None], math.prod(shape), axis=0).astype(float)
        for mask, ink in changes:
            images[:, mask] = np.broadcast_to(ink, shape).reshape(-1, 1)
        return predict(images).reshape(shape)

    inks = np.arange(INK + 1, dtype=float)
    patch_curves = [with_ink([(masks[k], inks)]).tolist() for k in range(n)]
    row_curves = [with_ink([(masks[list(row)].any(axis=0), inks)]).tolist() for row in rows]

    pairs = {(i, j): float(ksii[(i, j)]) for i, j in itertools.combinations(range(n), 2)}
    strongest = sorted(pairs, key=lambda pair: -abs(pairs[pair]))[:TOP_PAIRS]
    grid = np.arange(0, INK + 1, SURFACE_STEP, dtype=float)
    surfaces = {}
    for i, j in strongest:
        both = with_ink([(masks[i], grid[:, None]), (masks[j], grid[None, :])])
        only_i = with_ink([(masks[i], grid)])[:, None]
        only_j = with_ink([(masks[j], grid)])[None, :]
        surfaces[f"{i}-{j}"] = (both - only_i - only_j + prediction).tolist()

    numbers = {
        "prediction": prediction,
        "empty": empty,
        "shapley": [float(shapley[(i,)]) for i in range(n)],
        "rows": row_values,
        "pairs": {f"{i}-{j}": v for (i, j), v in pairs.items()},
    }
    out = {
        "source": "scikit-learn's digits, a held-out 7; MLPClassifier on the 64 pixels",
        "accuracy": round(accuracy, 4),
        "train": len(x_train),
        "test": len(x_test),
        "digit": DIGIT,
        "label": label,
        "target": target,
        "pixels": digit.astype(int).tolist(),
        "groups": [list(row) for row in rows],  # their values: "rows"
        **_round(numbers, 4),
        "ink": inks.astype(int).tolist(),
        "surface_ink": grid.astype(int).tolist(),
        **_round({"curves": patch_curves, "row_curves": row_curves, "surfaces": surfaces}, 3),
    }
    text = json.dumps(out, separators=(",", ":"))
    OUT.write_text(
        "// The data of the illustration lab-digits (see illustration.js), written by\n"
        "// ../compute/compute.py. Values are probabilities of the predicted class.\n"
        f"window.SHAPIQ_LAB_DIGITS = {text};\n"
    )
    print(f"accuracy {accuracy:.4f} on {len(x_test)} held-out digits")
    print(f"digit {DIGIT}: label {label}, predicted {target} with probability {prediction:.4f}")
    print(f"all patches erased: {empty:.4f}; wrote {OUT} ({OUT.stat().st_size / 1000:.1f} kB)")
    print(
        "sum of Shapley values",
        round(sum(numbers["shapley"]), 6),
        "= prediction - erased",
        round(prediction - empty, 6),
    )
    print("sum of the rows' values", round(sum(row_values), 6))
    print("strongest pairs", [(i + 1, j + 1, round(pairs[(i, j)], 3)) for i, j in strongest])


def _round(value: object, digits: int) -> object:
    """Round floats to ``digits`` decimals, recursively, to keep the file small."""
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, list):
        return [_round(v, digits) for v in value]
    if isinstance(value, dict):
        return {k: _round(v, digits) for k, v in value.items()}
    return value


if __name__ == "__main__":
    main()
