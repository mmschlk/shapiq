"""Write the orders demo's data, ``../assets/data.js``, from the results of the other scripts.

Reads the explanations of the molecule (molecule.py), of the houses of California housing
(housing.py), of the same house by TabPFN (house_tabpfn.py), and of how SigLIP 2 matches the FIxLIP
demo's elephant and its caption (fixlip.py). Every order is FSII, which the data names as its
``index`` for the demo's labels. It keeps what the demo draws on its sheets, rounded: the orders 1
to 4 and the full decomposition (FSII of order n, the number of players, which equals the Möbius
transform; for the houses, of order 5, the trees' depth, which equals it already and is exact),
every main effect, and the 300 strongest interactions of each order (of the molecule, with the
centers of its groups where shapiq's SI graph plot places them); for the many houses, where each
is on the map of California, and the strongest features and groups of features with their value for
every house. For every example, also its problem in words: its input, and its model's
prediction. And the R^2 (beyond size, under the Shapley distribution) and the number of values of
every order up to the highest with an interaction, and of order n, for the plot of the fit by
order. It writes them as a script that sets ``window.SHAPIQ_ORDERS`` (a script loads from disk too,
where the browser would not fetch a JSON file).

Usage::

    python pack.py
    python pack.py --examples house houses

With ``--examples``, only those examples are packed anew, from their results; the others are kept
as the data file has them (but for their words and their picks, which are written anew), so that
one example can change without computing all of them again. The examples are the demo's tabs
(molecule, house, tabpfn, houses, fixlip) and the other photos of the last one (giraffe, sign).
"""

from __future__ import annotations

import argparse
import json
import re
import tomllib
from functools import partial
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Collection

HERE = Path(__file__).resolve().parent
ORDERS = (1, 2, 3, 4)  # the sheets, above the one of order n (the full decomposition)
# the interactions (of two or more players) drawn on a sheet, most of them weak: a haze around
# the strongest
STRONGEST = 300
# of the orders between, which show only while pointed at, small: their strongest interactions
BETWEEN_STRONGEST = 120
# R^2 to nine decimals: enough to tell the exact full decomposition (R^2 = 1) from an order that
# only nearly is (the house's order 4 misses by 3e-7)
R2_DIGITS = 9
COMPACTNESS = 1e10  # how hard a group's center is pulled to its atoms: si_graph_plot's default
ROWS = 12  # of the many houses, the features and groups of features a sheet keeps: its strongest
# the many houses' numbers as text, a character each (two for a coordinate): the printable ASCII
# characters from "#" on, without the backslash, 91 of them, which a script's string holds as they
# are. demo.js reads them with the same alphabet
ALPHABET = "".join(chr(code) for code in range(35, 127) if chr(code) != "\\")
LEVELS = (len(ALPHABET) - 1) // 2  # a value from -45 to 45 of its sheet's scale
# the many houses' map: the state's outline (california.json, which names its source), and the
# cities its houses cluster around, each with its longitude and latitude
OUTLINE = HERE / "california.json"
# the FIxLIP demo's photos, with their titles: the photos of "Describing images" are of them
FIXLIP_CONFIG = HERE.parent.parent / "fixlip" / "compute" / "config.toml"
PLACES = [
    ["San Francisco", -122.42, 37.77],
    ["Los Angeles", -118.24, 34.05],
    ["San Diego", -117.16, 32.72],
]
# the house's features as its tree shows them, left to right: the location in the middle, next to
# the income and the occupancy, with which it interacts the most
TREE_NAMES = ["Income", "Age", "Rooms", "Beds", "Pop.", "Occup.", "Lat", "Long"]
TREE_LAYOUT = [1, 3, 0, 6, 7, 5, 2, 4]
# the models, as the demo shows them: its icon, kind, details, what it was explained with and how,
# and its name
MODELS = {
    "graph": ["graph", "Graph neural network", "GCN with 2 layers", "GraphSHAP-IQ", "exact", "GCN"],
    "tree": [
        "tree",
        "Tree ensemble",
        "XGBoost, 300 trees of depth 5",
        "TreeExplainer",
        "exact",
        "XGBoost",
    ],
    "black box": [
        "box",
        "Black box",
        "SigLIP 2, a vision-language model",
        "ProxySHAP",
        "estimated",
        "SigLIP 2",
    ],
    "tabpfn": ["box", "Black box", "TabPFN", "TabPFNImputer", "exact", "TabPFN"],
}


def dollars(value: float) -> str:
    """Return a house's value, in $100,000, in thousands of dollars."""
    return f"${value * 100:,.0f}k"


def problem(**fields: Any) -> dict[str, Any]:
    """Return the problem of an example, as the demo's plan shows it.

    Its ``input`` x, and the prediction f(x) of its model f: the ``output``, and its ``value`` in a
    few words.
    """
    keys = ("input", "output", "value")
    if set(fields) != set(keys):
        msg = f"a problem has {', '.join(keys)}, not {', '.join(fields)}"
        raise ValueError(msg)
    return {key: fields[key] for key in keys}


def explanation(result: dict[str, Any], order: int) -> dict[str, Any]:
    """Return the explanation of an order: FSII of order n is the one of the highest order."""
    return result["orders"][str(min(order, result["largest_order"]))]


def model(kind: str) -> dict[str, str]:
    """Return the model of an example, as the demo's badge shows it."""
    return dict(
        zip(("icon", "title", "detail", "method", "how", "short"), MODELS[kind], strict=True)
    )


def split(
    values: dict[tuple[int, ...], float],
    n: int,
    strongest: int | None,
    keep: Collection[tuple[int, ...]] = (),
) -> tuple[list[float], list[list[Any]]]:
    """Return the main effects of values (one per player) and the ``strongest`` of their groups.

    The groups to ``keep`` are returned whatever their strength, where the values have them.
    """
    first = [0.0] * n
    groups = []
    for players, value in values.items():
        if len(players) == 1:
            first[players[0]] = value
        elif abs(value) > 1e-12:
            groups.append([list(players), value])
    groups.sort(key=lambda group: -abs(group[1]))
    if strongest is not None:
        groups = [
            group
            for rank, group in enumerate(groups)
            if rank < strongest or tuple(group[0]) in keep
        ]
    return [round(value, 5) for value in first], [
        [players, round(value, 5)] for players, value in groups
    ]


def sheets(
    result: dict[str, Any], strongest: int | None = STRONGEST, full: int | None = None
) -> list[dict[str, Any]]:
    """Return the sheets of an explanation: per order, its R^2, main effects and interactions.

    Of the interactions, a sheet keeps the ``strongest`` (all of them for ``None``). The last sheet
    is the full decomposition, of order ``full`` (n, the number of players, for ``None``).
    """
    n = result["n_players"]
    full = full or n
    out = []
    for order in (*ORDERS, full):
        values = {tuple(players): value for players, value in explanation(result, order)["values"]}
        first, groups = split(values, n, strongest)
        out.append(
            {
                "order": order,
                "full": order == full,
                "r2": round(explanation(result, order)["r2"], R2_DIGITS),
                "first": first,
                "groups": groups,
                # (as the result counts them, where it keeps only the strongest: fixlip.py)
                "count": explanation(result, order).get("count", len(values)),
            }
        )
    return out


def between(
    result: dict[str, Any], strongest: int | None, keep: Collection[tuple[int, ...]] = ()
) -> list[dict[str, Any]]:
    """Return the explanations of the orders between the last sheet and the highest with interactions.

    The orders the demo's blank sheets stand for, up to the highest order with an interaction (the
    orders above it equal the full decomposition), where the result keeps their values. Of the
    interactions, each keeps the ``strongest``, and the groups to ``keep``: what is picked by hand
    on the orders around them, so that its mark does not come and go from order to order.
    """
    n = result["n_players"]
    out = []
    for order in range(ORDERS[-1] + 1, result["largest_order"]):
        entry = explanation(result, order)
        if not entry.get("values"):
            continue
        first, groups = split(
            {tuple(players): value for players, value in entry["values"]}, n, strongest, keep
        )
        out.append(
            {
                "order": order,
                "full": False,
                "r2": round(entry["r2"], R2_DIGITS),
                "first": first,
                "groups": groups,
                "count": entry.get("count", len(entry["values"])),
            }
        )
    return out


def place_groups(sheet: dict[str, Any], coords: list[list[float]]) -> None:
    """Add to every group of three or more atoms of a sheet the center where its atoms' connections meet.

    The centers are where shapiq's SI graph plot (``si_graph_plot``) puts them: a spring layout of
    the explanation graph with the atoms fixed, in which every group's center is pulled to its atoms
    (all but its last, as there) by its value, and pushed off all atoms and other centers. They do
    not fall onto an atom the way a centroid does, the center of an atom and its three neighbors.
    The centers are in the units of the coordinates.
    """
    xy = np.array(coords)
    low, side = xy.min(axis=0), np.ptp(xy, axis=0).max()
    pos = {atom: (point - low) / side for atom, point in enumerate(xy)}  # as si_graph_plot scales
    graph = nx.Graph()
    graph.add_nodes_from(pos)
    for players, value in sheet["groups"]:
        if len(players) > 2:
            for player in players[:-1]:
                graph.add_edge(player, tuple(players), weight=abs(value) * COMPACTNESS)
    layout = nx.spring_layout(graph, weight="weight", seed=42, pos=pos, fixed=list(pos))
    for group in sheet["groups"]:
        if len(group[0]) > 2:
            group.append([round(float(c), 4) for c in layout[tuple(group[0])] * side + low])


def fits(result: dict[str, Any], full: int | None = None) -> list[list[float]]:
    """Return the order, R^2 and number of values of every order with interactions, and of ``full``.

    ``full`` is the order of the full decomposition: n, the number of players, for ``None``.
    """
    full = full or result["n_players"]
    orders = [*range(1, result["largest_order"] + 1)]
    if full > result["largest_order"]:
        orders.append(full)
    entries = [explanation(result, order) for order in orders]
    # the number of values: as the result counts them (fixlip.py does), else all of them
    return [
        [order, round(entry["r2"], R2_DIGITS), entry.get("count", len(entry.get("values", [])))]
        for order, entry in zip(orders, entries, strict=True)
    ]


def encode(numbers: np.ndarray, width: int = 1) -> str:
    """Return whole numbers from 0 on as text, ``width`` characters of the alphabet each."""
    base = len(ALPHABET)
    if numbers.min() < 0 or numbers.max() >= base**width:
        msg = f"numbers from {numbers.min()} to {numbers.max()} do not fit {width} characters"
        raise ValueError(msg)
    digits = [numbers // base**place % base for place in reversed(range(width))]
    return "".join(ALPHABET[digit] for number in zip(*digits, strict=True) for digit in number)


def houses_map(result: dict[str, Any], full: int) -> dict[str, Any]:
    """Return the map of the houses: where they are, their predictions and, per sheet, their rows.

    The houses are sorted by their prediction, each with its longitude and latitude, in hundredths
    of a degree from the westernmost and the southernmost house, two characters each. A row is a
    feature or a group of features, of a sheet's strongest by the mean of its absolute values: its
    players, its value for every house, a character each, for a whole number from -45 to 45 of the
    sheet's scale (the 98th percentile of its absolute values), and the mean of its absolute
    values, how much it moves a house's value on average. The last sheet is the full
    decomposition, of order ``full``. With them, the state's outline and the cities the map names.
    """
    columns = np.argsort(result["predictions"])
    x = np.array(result["x"])[columns]
    west, south = float(x[:, 7].min()), float(x[:, 6].min())

    def rows(values: dict[str, list[float]]) -> tuple[float, list[Any]]:
        """Return the scale and the strongest rows of values."""
        strongest = sorted(values.items(), key=lambda row: -np.mean(np.abs(row[1])))[:ROWS]
        matrix = np.array([row[1] for row in strongest])[:, columns]
        scale = float(np.quantile(np.abs(matrix), 0.98))
        quantized = np.round(np.clip(matrix / scale, -1, 1) * LEVELS).astype(int)
        return scale, [
            [
                [int(player) for player in key.split(",")],
                encode(row + LEVELS),
                round(float(np.mean(np.abs(values))), 4),
            ]
            for (key, values), row in zip(strongest, quantized, strict=True)
        ]

    out = []
    for order in (*ORDERS, full):
        values = explanation(result, order)["values"]
        scale, strongest = rows(values)
        out.append(
            {
                "order": order,
                "full": order == full,
                "r2": round(explanation(result, order)["r2"], R2_DIGITS),
                "count": len(values),
                "scale": round(scale, 4),
                "rows": strongest,
            }
        )
    return {
        "predictions": [round(result["predictions"][i], 2) for i in columns],
        # (longitude and latitude are the last two features)
        "coords": {
            "west": round(west, 2),
            "south": round(south, 2),
            "lon": encode(np.round((x[:, 7] - west) * 100).astype(int), width=2),
            "lat": encode(np.round((x[:, 6] - south) * 100).astype(int), width=2),
        },
        "outline": json.loads(OUTLINE.read_text(encoding="utf-8"))["outline"],
        "places": PLACES,
        "sheets": out,
        "fits": fits(result, full),
        "quartiles": [round(q, 3) for q in result["shapley_r2_quartiles"]],
    }


def house_values(x: list[float]) -> list[str]:
    """Return the features of a house as its tree shows them, under their names."""
    income, age, rooms, bedrooms, population, occupancy, latitude, longitude = x
    return [
        f"${income * 10:.1f}k",  # the median income of the block, in $10,000
        f"{age:.0f} yrs",
        f"{rooms:.1f}",  # per home
        f"{bedrooms:.1f}",  # per home
        f"{population:,.0f}",  # people
        f"{occupancy:.1f}",  # per home
        f"{latitude:.2f}°N",
        f"{-longitude:.2f}°W",
    ]


def pack_molecule(molecule: dict[str, Any]) -> dict[str, Any]:
    """Return the molecule's example: its atoms and bonds, and its sheets with their groups' centers."""
    molecule["n_players"] = len(molecule["atoms"])
    molecule_sheets = sheets(molecule)
    molecule_between = between(molecule, strongest=BETWEEN_STRONGEST)
    for sheet in (*molecule_sheets, *molecule_between):
        place_groups(sheet, molecule["coords"])
    return {
        "kind": "graph",
        "problem": problem(
            input="A molecule",
            output="Mutagenic",
            value=f"p = {molecule['probability']:.3f}, logit {molecule['logit']:.2f}",
        ),
        "atoms": molecule["atoms"],
        "bonds": molecule["bonds"],
        "coords": molecule["coords"],
        "probability": round(molecule["probability"], 3),
        "accuracy": round(molecule["test_accuracy"], 3),
        "budget": molecule["budget"],
        "largest": molecule["largest_order"],
        "sheets": molecule_sheets,
        "between": molecule_between,
        "fits": fits(molecule),
    }


def pack_house(house: dict[str, Any]) -> dict[str, Any]:
    """Return the house's example, valued by XGBoost: its features, and its sheets up to the depth."""
    return {
        "kind": "tree",
        "problem": problem(
            input="A house", output=dollars(house["prediction"]), value="its value, predicted"
        ),
        "features": TREE_NAMES,
        "values": house_values(house["x"]),
        "layout": TREE_LAYOUT,
        "prediction": round(house["prediction"], 4),
        "baseline": round(house["baseline"], 4),
        "accuracy": round(house["test_r2"], 3),
        "largest": house["largest_order"],
        # the trees' depth: FSII of that order is already the full decomposition, exact
        "sheets": sheets(house, strongest=None, full=house["largest_order"]),
        "fits": fits(house, full=house["largest_order"]),
    }


def pack_tabpfn(tabpfn: dict[str, Any]) -> dict[str, Any]:
    """Return the house's example, valued by TabPFN: its features, and its sheets of every order."""
    return {
        "kind": "tree",
        "problem": problem(
            input="A house", output=dollars(tabpfn["prediction"]), value="its value, predicted"
        ),
        "features": TREE_NAMES,
        "values": house_values(tabpfn["x"]),
        "layout": TREE_LAYOUT,
        "prediction": round(tabpfn["prediction"], 4),
        "baseline": round(tabpfn["baseline"], 4),
        "accuracy": round(tabpfn["test_r2"], 3),
        "context": tabpfn["context"],
        "largest": tabpfn["largest_order"],
        # no depth: FSII of order n, all 8 features, is the first that is exact
        "sheets": sheets(tabpfn, strongest=None),
        "between": between(tabpfn, strongest=None),
        "fits": fits(tabpfn),
    }


def pack_houses(houses: dict[str, Any]) -> dict[str, Any]:
    """Return the many houses' example: their map, and the rows of their sheets."""
    # the houses' predictions, from the lowest to the highest
    spread = f"{dollars(min(houses['predictions']))} to {dollars(max(houses['predictions']))}"
    return {
        "kind": "map",
        "problem": problem(
            input=f"{len(houses['predictions']):,} houses",
            output=spread,
            value="their values, predicted",
        ),
        "features": TREE_NAMES,  # as the one house names them
        "largest": houses["largest_order"],
        **houses_map(houses, full=houses["largest_order"]),
    }


def pack_photo(fixlip: dict[str, Any]) -> dict[str, Any]:
    """Return the example of a photo and its caption: its patches and tokens, and its sheets."""
    fixlip["n_players"] = fixlip["grid"] ** 2 + len(fixlip["tokens"])
    photo = fixlip.get("photo", "elephant")  # (the first results, the elephant's, do not name it)
    return {
        "kind": "image-text",
        "problem": problem(
            input="A photo and its caption",
            output=f"Similarity {fixlip['similarity']:.2f}",
            value="cosine, of the photo and the caption",
        ),
        "similarity": round(fixlip["similarity"], 4),
        "image": f"../fixlip/{photo}.jpg",  # the FIxLIP demo's photo, next to its assets
        "grid": fixlip["grid"],
        "tokens": fixlip["tokens"],
        "largest": fixlip["largest_order"],
        "sheets": sheets(fixlip),
        "between": between(
            fixlip,
            strongest=BETWEEN_STRONGEST,
            keep=picked_above(photo_picks(photo, fixlip["grid"], fixlip["tokens"])),
        ),
        "fits": fits(fixlip),
    }


# -- the insights: what is picked by hand on an order, a player or a group, and what the demo's tip
# says of it. One plain sentence, with one number in it: the sum of all that the picked adds on
# that order, its players' own values and those of every group of them. The sentence marks the
# number's place (``{up}``, ``{down}`` or ``{sum}``), and the demo writes the number there, from the
# order's values. A key is a pick's players, in their order, joined by commas ----------------------


def key(*players: int) -> str:
    """Return the key of a player or a group: its players, in their order, joined by commas."""
    return ",".join(str(player) for player in sorted(players))


FULL = "full"  # in a pick's orders: the full decomposition, whatever its order is
# in a pick's orders: every order above the last with a sheet of its own, the full decomposition
# among them (of the orders between, those whose sheets have the pick's value)
ABOVE = "above"
MOST_PICKED = 7  # the most that is picked on an order: more marks than that crowd the card's face
# the number's place in a sentence: what the picked adds, as an amount that raises the output
# ({up}: the sum is positive on every order of the row) or lowers it ({down}: negative), or with
# its sign ({sum}: whatever it is)
PLACES = ("{up}", "{down}", "{sum}")
# the digits the demo shows of a value (demo.js: VALUE), by the example's kind: an amount it shows
# must not round to zero (a house's value is in $100k, and shown in $1k)
DIGITS = {"graph": 2, "tree": 2, "image-text": 3}

Picks = list[tuple[list[Any], list[Any], str]]


def picked_above(table: Picks) -> set[tuple[int, ...]]:
    """Return the groups of a table of picks that are picked on the orders between, too.

    A row picked on ``ABOVE``: the sheets of the orders between the last sheet and the full
    decomposition keep these groups, however strong they are.
    """
    return {
        tuple(sorted(players))
        for players, marked, _ in table
        if len(players) > 1 and ABOVE in marked
    }


def plain(sentence: str) -> str | None:
    """Return what keeps a tip's sentence from being plain, or ``None``: it is one easy sentence.

    One sentence, with one comma at the most and no colon or semicolon, and one number: the place
    of the sum (one of ``PLACES``), and no other digit.
    """
    places = [place for place in PLACES if place in sentence]
    if len(places) != 1 or sentence.count(places[0]) != 1:
        return "does not have one place for its number"
    words = sentence.replace(places[0], "")
    if not words.endswith(".") or any(mark in words[:-1] for mark in ".:;!?"):
        return "is not one sentence without a colon or a semicolon"
    if words.count(",") > 1:
        return "has more than one comma"
    if any(letter.isdecimal() for letter in words):
        return "has a number besides its sum"
    return None


def curated(example: dict[str, Any], table: Picks) -> dict[str, Any]:
    """Return an example's picks and what the demo says of them, from its table of picks.

    A row of the table is a player or a group (its players), the orders it is picked on (``FULL``
    for the full decomposition, ``ABOVE`` for every order above 4), and its sentence: what is
    picked wears a mark on the demo's card, where a reader points at it, and the card's tip says
    the sentence. Few are picked on an order (3 to 5, and ``MOST_PICKED`` at the most), as many
    marks crowd the card. The picks of an order are the groups first, then the players, each the
    strongest first, as the marks number them.

    A sentence is plain (``plain``) and holds on every order of its row: its number is the sum of
    all that the picked adds on the order, and where the sentence says that this raises or lowers
    the output, it does. The sheets must have every value of that sum, the players' own and those
    of every group of them: a group that lacks one cannot be picked.
    """
    by_order = {
        sheet["order"]: (sheet, {tuple(group[0]): group[1] for group in sheet["groups"]})
        for sheet in (*example["sheets"], *example.get("between", []))
    }
    full = example["sheets"][-1]["order"]
    between = [sheet["order"] for sheet in example.get("between", [])]

    def value(order: int, group: list[int]) -> float | None:
        """Return a player's or a group's value on an order, if the order's sheet has it."""
        sheet, groups = by_order[order]
        return sheet["first"][group[0]] if len(group) == 1 else groups.get(tuple(group))

    def total(order: int, group: list[int]) -> float:
        """Return the sum of all that a group adds on an order: of every value among its players."""
        sheet, groups = by_order[order]
        parts = [
            groups.get(part)
            for size in range(2, min(len(group), order) + 1)
            for part in combinations(group, size)
        ]
        if None in parts:
            msg = f"{example['id']}: order {order} lacks {parts.count(None)} values among {group}"
            raise ValueError(msg)
        return sum(sheet["first"][player] for player in group) + sum(parts)

    picks: dict[int, list[list[int]]] = {}
    says: dict[int, dict[str, str]] = {}
    for players, marked, sentence in table:
        group = sorted(players)
        if fault := plain(sentence):
            msg = f"{example['id']}: the sentence on {group} {fault}"
            raise ValueError(msg)
        orders = [full if order == FULL else order for order in marked if order != ABOVE]
        if ABOVE in marked:
            # (the orders between: where their sheets have the group)
            orders += [order for order in between if value(order, group) is not None]
            orders += [full]
        for order in orders:
            if value(order, group) is None:
                msg = f"{example['id']}: order {order} has no value of {group} to pick"
                raise ValueError(msg)
            if key(*group) in says.setdefault(order, {}):
                msg = f"{example['id']}: {group} is picked twice on order {order}"
                raise ValueError(msg)
            amount = round(total(order, group), DIGITS[example["kind"]])
            if ("{up}" in sentence and amount <= 0) or ("{down}" in sentence and amount >= 0):
                msg = f"{example['id']}: on order {order}, {group} adds {amount}, not as its sentence says"
                raise ValueError(msg)
            picks.setdefault(order, []).append(group)
            says[order][key(*group)] = sentence
    for order, groups in picks.items():
        if len(groups) > MOST_PICKED:
            msg = f"{example['id']}: order {order} has {len(groups)} picks, {MOST_PICKED} at most"
            raise ValueError(msg)
    return {
        "picks": {
            str(order): [
                key(*group)
                for group in sorted(
                    groups, key=lambda group: (len(group) == 1, -abs(value(order, group)))
                )
            ]
            for order, groups in sorted(picks.items())
        },
        "says": {str(order): said for order, said in sorted(says.items())},
    }


# what is picked on the molecule, by hand: a player or a group (its atoms), the orders it is picked
# on, and its sentence. A few on every order, each with something of its own to say, and apart on
# the drawing: of what says the same (the three nitro groups, the ring carbons), one stands for the
# others. Above order 4, the sheets keep the strongest values only, and lack some among every
# group of three atoms or more: there, pairs and atoms are picked
MOLECULE_PICKS: Picks = [
    (
        [18],
        [1],
        "This nitrogen sits in a nitro group and raises the logit by {up}, the most of all atoms.",
    ),
    ([20], [1], "This oxygen of a nitro group raises the logit by {up}."),
    ([0], [1], "Two rings share this carbon and it raises the logit by {up}."),
    ([18, 21], [2, 3, 4, ABOVE], "Together this nitrogen and its oxygen raise the logit by {up}."),
    (
        [3, 9],
        [2],
        "Together this nitrogen and the ring carbon that carries it raise the logit by {up}.",
    ),
    ([0, 2], [2], "Together these two ring carbons raise the logit by {up}."),
    (
        [0, 2],
        [3],
        "The pair of these two carbons is negative here but together they still raise the logit "
        "by {up}.",
    ),
    ([13], [2], "Alone this ring carbon raises the logit by {up}."),
    (
        [7, 8, 13],
        [3, 4],
        "Together these three ring carbons raise the logit by {up}, although their group of three "
        "is negative.",
    ),
    ([9], [3, ABOVE], "Alone this nitrogen raises the logit by only {up}."),
    ([0, 2, 6, 7], [4], "Together this carbon and its three neighbors raise the logit by {up}."),
    (
        [5, 12, 17, 18],
        [4],
        "Together the carbon that carries this nitro group and its three neighbors raise the "
        "logit by {up}.",
    ),
    ([8, 13], [ABOVE], "Together these two ring carbons raise the logit by {up}."),
    (
        [6],
        [ABOVE],
        "Alone this ring carbon raises the logit by {up}, more than a nitrogen does alone.",
    ),
]

# the house's features, as the picks below name them
INCOME, AGE, ROOMS, BEDS, POP, OCCUP, LAT, LONG = range(8)
# what is picked on the house that XGBoost values, by hand: its latitude and its longitude carry
# the story (what one adds alone, the two take back together), and the income and the rooms raise
# its value (both are above California's medians: $35k, 5.2 rooms). A few on every order, and what
# the two models differ in on both of them
HOUSE_PICKS: Picks = [
    ([LAT], [1], "The latitude raises the house's value by {up}."),
    (
        [LONG],
        [1],
        "The longitude lowers the house's value by {down}, about as much as the latitude raises it.",
    ),
    (
        [INCOME],
        [1],
        "The income is above California's median and raises the house's value by {up}.",
    ),
    (
        [ROOMS],
        [1],
        "The house has more rooms than is typical and that raises its value by {up}.",
    ),
    (
        [LAT, LONG],
        [2, 3, 4, ABOVE],
        "Together the latitude and the longitude move the house's value by only {sum}.",
    ),
    ([INCOME, ROOMS], [2], "Together the income and the rooms raise the house's value by {up}."),
    (
        [LAT],
        [2, 3, 4, ABOVE],
        "Alone the latitude raises the house's value by {up}, which its pair with the longitude "
        "takes back.",
    ),
    ([INCOME], [2, 3, 4, ABOVE], "Alone the income raises the house's value by {up}."),
    (
        [OCCUP, LAT, LONG],
        [3, 4, ABOVE],
        "Together the occupancy, the latitude and the longitude lower the house's value by {down}.",
    ),
]
# and on the same house as TabPFN values it: the location (its latitude and its longitude) and the
# income with the rooms carry its stories. Its groups of many features have large values that
# cancel for the most part, and one of the largest stands for them
MANY = "This group's own value is large, yet together its {} features lower the house's value by only {{down}}."
TABPFN_PICKS: Picks = [
    ([LAT], [1], "The latitude lowers the house's value by {down}, the most of all features."),
    ([ROOMS], [1], "The rooms raise the house's value by {up}."),
    ([LONG], [1], "The longitude lowers the house's value by {down}."),
    ([INCOME], [1], "The income hardly counts here and lowers the house's value by {down}."),
    (
        [LAT, LONG],
        [2, 3, 4, ABOVE],
        "Together the latitude and the longitude lower the house's value by {down}.",
    ),
    (
        [INCOME, ROOMS],
        [2, 3, 4, ABOVE],
        "Together the income and the rooms raise the house's value by only {up}, far less than "
        "the rooms do alone.",
    ),
    ([ROOMS], [2, 3, 4], "Alone the rooms raise the house's value by {up}."),
    (
        [INCOME],
        [2],
        "Alone the income raises the house's value by {up}, which its pair with the rooms takes "
        "back.",
    ),
    ([AGE, ROOMS], [3], "Together the age and the rooms raise the house's value by {up}."),
    (
        [ROOMS, LAT, LONG],
        [4, ABOVE],
        "Together the rooms and the location lower the house's value by {down}.",
    ),
    (
        [INCOME, ROOMS, LAT, LONG],
        [4, ABOVE],
        "Together the income, the rooms and the location lower the house's value by {down}.",
    ),
    ([INCOME, ROOMS, OCCUP, LAT, LONG], [6], MANY.format("five")),
    ([INCOME, AGE, ROOMS, BEDS, POP, LAT, LONG], [7, FULL], MANY.format("seven")),
]

# what is picked on the photos, by the photo's id: on a photo, only what is picked can be pointed
# at. A row names its players as a word of the caption or a patch (its row and column in the
# photo, from 1), the orders it is picked on, and its sentence: what the patch shows, and what the
# values add on those orders, no more. A few on every order, each with something of its own to say
# (of the patches of the elephant that match its word, one), and apart on the card. The
# explanations are estimates: what is near zero is not picked, or its sentence says so
PHOTO_PICKS: dict[str, Picks] = {
    "elephant": [
        (
            ["elephant"],
            [1],
            "The word “elephant” names the photo's main subject and raises the similarity by {up}.",
        ),
        ([(6, 1)], [1], "This patch shows the egret's head and raises the similarity by {up}."),
        (["bird"], [1], "The word “bird” names the egret and raises the similarity by {up}."),
        (
            [(6, 1), "bird"],
            [2],
            "Together the egret's patch and the word “bird” raise the similarity by {up}.",
        ),
        (
            [(5, 4), "elephant"],
            [2, 3, 4, ABOVE],
            "Together the curled trunk and the word “elephant” raise the similarity by {up}.",
        ),
        (["elephant"], [2], "Alone the word “elephant” raises the similarity by {up}."),
        (["white"], [2, 3], "Alone the word “white” lowers the similarity by {down}."),
        (
            ["elephant", "bird"],
            [3, 4, ABOVE],
            "Together the words “elephant” and “bird” raise the similarity by {up}, less than the "
            "two add alone.",
        ),
        (["bird"], [3], "Alone the word “bird” raises the similarity by {up}."),
        (
            [(6, 1), "white"],
            [4, ABOVE],
            "Together the white egret's patch and the word “white” lower the similarity by only "
            "{down}.",
        ),
        (
            [(5, 5), "white"],
            [4, ABOVE],
            "Together this brown patch of the elephant and the word “white” lower the similarity "
            "by {down}.",
        ),
        (["elephant"], [4, ABOVE], "Alone the word “elephant” raises the similarity by only {up}."),
    ],
    "giraffe-tree": [
        (
            ["giraffe"],
            [1],
            "The word “giraffe” names the animal and raises the similarity by {up}.",
        ),
        (
            ["water"],
            [1],
            "A pond fills the lower part of the photo and the word “water” raises the similarity "
            "by {up}.",
        ),
        (
            [(3, 5)],
            [1],
            "This patch shows the giraffe's back and raises the similarity by {up}, the most of "
            "all patches.",
        ),
        (
            [(2, 2)],
            [1],
            "This patch shows a dead tree and still raises the similarity by {up}.",
        ),
        (
            ["water", "pond"],
            [2, 3],
            "Together the words “water” and “pond” raise the similarity by {up}, less than the two "
            "add alone.",
        ),
        (
            [(2, 2), "giraffe"],
            [2, 3, 4, ABOVE],
            "Together the dead tree's patch and the word “giraffe” raise the similarity by {up}, "
            "more than the word does alone.",
        ),
        (
            [(6, 5), "giraffe"],
            [2, 4, ABOVE],
            "Together the giraffe's head and the word “giraffe” raise the similarity by {up}.",
        ),
        (["water"], [2], "Alone the word “water” raises the similarity by {up}."),
        (["giraffe"], [2, 4, ABOVE], "Alone the word “giraffe” raises the similarity by {up}."),
        (
            ["drink", "water"],
            [3, 4],
            "Together the words “drink” and “water” raise the similarity by only {up}, less than "
            "“water” does alone.",
        ),
        (
            [(3, 5), (4, 5), "giraffe"],
            [3],
            "Together two patches of the giraffe's body and the word “giraffe” raise the "
            "similarity by {up}.",
        ),
        (
            [(7, 2), "water"],
            [3, 4, ABOVE],
            "Together this patch of the pond and the word “water” raise the similarity by {up}.",
        ),
    ],
    "juice-theory": [
        (
            ["traffic"],
            [1],
            "The photo shows a traffic light and the word “traffic” raises the similarity by {up}.",
        ),
        ([(1, 7)], [1], "This patch shows the red light and raises the similarity by {up}."),
        (
            [(5, 4)],
            [1],
            "This patch shows letters of the sign and raises the similarity by {up}.",
        ),
        (
            [(5, 4), "sign"],
            [2],
            "Together the sign's letters and the word “sign” raise the similarity by {up}.",
        ),
        (
            [(1, 7), "traffic"],
            [2, 3, 4, ABOVE],
            "Together the red light and the word “traffic” raise the similarity by {up}.",
        ),
        (
            [(5, 2), "juice"],
            [2],
            "Together the “u” of the sign's “Juice” and the word “juice” raise the similarity by "
            "only {up}.",
        ),
        (["juice"], [2, 3, 4, ABOVE], "Alone the word “juice” lowers the similarity by {down}."),
        (
            [(5, 1), "juice"],
            [3],
            "Together the “J” of the sign's “Juice” and the word “juice” lower the similarity by "
            "only {down}.",
        ),
        (
            [(5, 1), "juice"],
            [4, ABOVE],
            "Together the “J” of the sign's “Juice” and the word “juice” nearly cancel and move "
            "the similarity by {sum}.",
        ),
        (
            [(5, 3), "theory"],
            [3],
            "Together letters of the sign's “THEORY” and the word “theory” lower the similarity "
            "by {down}.",
        ),
        (
            ["traffic", "light"],
            [4, ABOVE],
            "Together the words “traffic” and “light” raise the similarity by {up}, less than the "
            "two add alone.",
        ),
        (
            [(7, 2), "juice"],
            [4, ABOVE],
            "Together lettering of “The Pantry” and the word “juice” lower the similarity by "
            "{down}, less than the word does alone.",
        ),
    ],
}


def photo_picks(photo: str, grid: int, tokens: list[str]) -> Picks:
    """Return what is picked on a photo (``PHOTO_PICKS``), its patches and words as players."""

    def player(named: tuple[int, int] | str) -> int:
        if isinstance(named, str):
            return grid * grid + tokens.index(named)
        row, column = named
        return (row - 1) * grid + column - 1

    return [
        ([player(named) for named in players], on, sentence)
        for players, on, sentence in PHOTO_PICKS.get(photo, [])
    ]


def curated_photo(example: dict[str, Any]) -> dict[str, Any]:
    """Return the picks of a photo and its caption and what the demo says of them, by its file."""
    photo = Path(example["image"]).stem
    return curated(example, photo_picks(photo, example["grid"], example["tokens"]))


def photo_words(photo: str) -> dict[str, Any]:
    """Return the words of a photo's example: its tab's, and the photo's title in the FIxLIP demo."""
    config = tomllib.loads((FIXLIP_CONFIG).read_text(encoding="utf-8"))
    title = next(entry["title"] for entry in config["photos"] if entry["id"] == photo)
    words = {"label": "Describing images", "detail": "SigLIP 2", "model": model("black box")}
    return {**words, "photo": {"title": title}}


# the examples: the file of each one's results, its words, which no result gives (its tab's name,
# its model's, and the model as the demo shows it), what packs its results, and what picks its
# insights. They are in the order of the demo's tabs; an example ``of`` another one has no tab: it
# is another example of that tab (another photo and its caption), which the tab's picker shows
EXAMPLES: dict[str, tuple[str, dict[str, Any], Any, Any]] = {
    "molecule": (
        "molecule",
        {"label": "Molecule", "detail": "GCN", "model": model("graph")},
        pack_molecule,
        partial(curated, table=MOLECULE_PICKS),
    ),
    "house": (
        "house",
        {"label": "One house", "detail": "XGBoost", "model": model("tree")},
        pack_house,
        partial(curated, table=HOUSE_PICKS),
    ),
    "tabpfn": (
        "house_tabpfn",
        {"label": "One house, TabPFN", "detail": "TabPFN", "model": model("tabpfn")},
        pack_tabpfn,
        partial(curated, table=TABPFN_PICKS),
    ),
    # (the same model as the house's, with the same average)
    "houses": (
        "houses",
        {"label": "All houses", "detail": "XGBoost", "model": model("tree")},
        pack_houses,
        None,
    ),
    "fixlip": ("fixlip", photo_words("elephant"), pack_photo, curated_photo),
    "giraffe": (
        "fixlip_giraffe-tree",
        {**photo_words("giraffe-tree"), "of": "fixlip"},
        pack_photo,
        curated_photo,
    ),
    "sign": (
        "fixlip_juice-theory",
        {**photo_words("juice-theory"), "of": "fixlip"},
        pack_photo,
        curated_photo,
    ),
}
HEADER = "// The orders demo's explanations (see demo.js), written by ../compute/pack.py.\n"


def main() -> None:
    """Write ``../assets/data.js`` from the results."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--assets", type=Path, default=HERE.parent / "assets")
    parser.add_argument(
        "--examples",
        nargs="+",
        choices=list(EXAMPLES),
        default=list(EXAMPLES),
        metavar="EXAMPLE",
        help=f"the examples to pack anew, of {', '.join(EXAMPLES)} (all of them); the others are "
        "kept as the data file has them",
    )
    args = parser.parse_args()
    target = args.assets / "data.js"

    # the examples that are not packed anew, as the data file has them
    kept: dict[str, dict[str, Any]] = {}
    indices = set()
    if set(args.examples) != set(EXAMPLES):
        text = target.read_text(encoding="utf-8")
        old = json.loads(re.search(r"window\.SHAPIQ_ORDERS = (.*);\s*$", text, re.DOTALL).group(1))
        kept = {example["id"]: example for example in old["examples"]}
        indices.add(old["index"])

    examples = []
    for name, (file, words, packer, picker) in EXAMPLES.items():
        if name in args.examples:
            result = json.loads((args.results / f"{file}.json").read_text(encoding="utf-8"))
            indices.add(result["index"])
            packed = packer(result)
        else:
            # (anew, whatever the data has)
            written = ("id", "says", "picks", "of", "photo", *words)
            packed = {name: value for name, value in kept[name].items() if name not in written}
        example = {"id": name, **words, **packed}
        if picker:
            example.update(picker(example))  # (what is picked by hand, and what is said of it)
        examples.append(example)
    (index,) = indices  # of every order of every example, for the demo's labels

    data = {"index": index, "examples": examples}
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    target.write_text(f"{HEADER}window.SHAPIQ_ORDERS = {text};\n", encoding="utf-8")


if __name__ == "__main__":
    main()
