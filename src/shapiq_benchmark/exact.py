"""Exact low-order interactions from fully enumerated coalition payoff tables."""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np


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
