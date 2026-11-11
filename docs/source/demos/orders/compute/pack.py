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
as the data file has them (but for their words and their notes, which are written anew), so that
one example can change without computing all of them again. The examples are the demo's tabs
(molecule, house, tabpfn, houses, fixlip) and the other photos of the last one (giraffe, sign).
"""

from __future__ import annotations

import argparse
import json
import re
import tomllib
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

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
    values: dict[tuple[int, ...], float], n: int, strongest: int | None
) -> tuple[list[float], list[list[Any]]]:
    """Return the main effects of values (one per player) and the ``strongest`` of their groups."""
    first = [0.0] * n
    groups = []
    for players, value in values.items():
        if len(players) == 1:
            first[players[0]] = value
        elif abs(value) > 1e-12:
            groups.append([list(players), value])
    groups.sort(key=lambda group: -abs(group[1]))
    return [round(value, 5) for value in first], [
        [players, round(value, 5)] for players, value in groups[:strongest]
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


def between(result: dict[str, Any], strongest: int | None) -> list[dict[str, Any]]:
    """Return the explanations of the orders between the last sheet and the highest with interactions.

    The orders the demo's blank sheets stand for, up to the highest order with an interaction (the
    orders above it equal the full decomposition), where the result keeps their values. Of the
    interactions, each keeps the ``strongest``.
    """
    n = result["n_players"]
    out = []
    for order in range(ORDERS[-1] + 1, result["largest_order"]):
        entry = explanation(result, order)
        if not entry.get("values"):
            continue
        first, groups = split(
            {tuple(players): value for players, value in entry["values"]}, n, strongest
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
        "between": between(fixlip, strongest=BETWEEN_STRONGEST),
        "fits": fits(fixlip),
    }


# -- the notes: what the demo's tip says, in a sentence, of a player or a group pointed at. Most
# have none: not every value has a story, and a note only says what the example shows. A note's key
# is its players, in their order, joined by commas; "many" is the note of every group of that many
# players at the least --------------------------------------------------------------------------------


def key(*players: int) -> str:
    """Return the key of a player or a group: its players, in their order, joined by commas."""
    return ",".join(str(player) for player in sorted(players))


FULL = "full"  # in a pick's orders: the full decomposition, whatever its order is


def curated(
    example: dict[str, Any], table: list[tuple[list[int], list[Any], str | None]]
) -> dict[str, Any]:
    """Return an example's picks and notes from its table of what was picked by hand.

    A row of the table is a player or a group (its players), the orders it is picked on (``FULL``
    for the full decomposition), and its note, or ``None``: what is picked wears a mark on the
    demo's card, where a reader points at it, and not everything picked has a story. A row picked
    on order 4 and on the full decomposition is picked on the orders between them too. The picks
    of an order are the groups first, then the players, each the strongest first, as the marks
    number them; a note holds on the orders its row is picked on, and says so (``[note, orders]``).
    """
    by_order = {
        sheet["order"]: sheet for sheet in (*example["sheets"], *example.get("between", []))
    }
    full, last = example["sheets"][-1]["order"], example["sheets"][-2]["order"]
    between = [sheet["order"] for sheet in example.get("between", [])]

    def value(order: int, group: list[int]) -> float | None:
        """Return a player's or a group's value on an order, if the order's sheet has it."""
        sheet = by_order[order]
        if len(group) == 1:
            return sheet["first"][group[0]]
        return next((entry[1] for entry in sheet["groups"] if entry[0] == group), None)

    picks: dict[int, list[list[int]]] = {}
    notes: dict[str, Any] = {}
    for players, marked, note in table:
        group = sorted(players)
        orders = [full if order == FULL else order for order in marked]
        if last in orders and full in orders:
            # (the orders between: as their neighbors, where their sheets have the group)
            orders += [
                order
                for order in between
                if order not in orders and value(order, group) is not None
            ]
        for order in orders:
            if value(order, group) is None:
                msg = f"{example['id']}: order {order} has no value of {group} to pick"
                raise ValueError(msg)
            picks.setdefault(order, []).append(group)
        if note:
            # (a group's note: on orders of one sign, unless the note is that the sign flips; a
            # player's says what the player is, whatever its own value)
            signs = {value(order, group) > 0 for order in orders}
            if len(group) > 1 and len(signs) > 1 and "flips" not in note:
                msg = f"{example['id']}: the note on {group} is on orders of both signs"
                raise ValueError(msg)
            notes[key(*group)] = [note, sorted(orders)]
    return {
        "notes": notes,
        "picks": {
            str(order): [
                key(*group)
                for group in sorted(
                    groups, key=lambda group: (len(group) == 1, -abs(value(order, group)))
                )
            ]
            for order, groups in sorted(picks.items())
        },
    }


# what is picked on the molecule, by hand: a player or a group (its atoms), the orders it is picked
# on, and its note, where it has one that the atoms' bonds do not already give (notes_molecule).
# The strong positive groups are whole neighborhoods of an atom, and have a reading; the negative
# ones, and what flips its sign from order to order, have none
MOLECULE_PICKS: list[tuple[list[int], list[Any], str | None]] = [
    ([9], [1, 2, 3, 4, FULL], None),
    ([16], [1], None),
    ([18], [1], None),
    ([0], [1], None),
    ([13], [2], None),
    ([17], [2], None),
    ([8], [2, 4, FULL], None),
    ([7], [3], None),
    ([11], [3], None),
    ([5], [3, 4, FULL], None),
    ([6], [4, FULL], None),
    ([9, 14], [2, 3, 4, FULL], None),
    ([18, 21], [2, 3], None),
    ([3, 9], [2], None),
    (
        [0, 2],
        [2, 3],
        "A bond shared by two of the rings. No story: its value even flips sign from one order "
        "to the next.",
    ),
    (
        [1, 4],
        [3],
        "A bond shared by two of the rings. No story: its value even flips sign from one order "
        "to the next.",
    ),
    (
        [8, 26],
        [2],
        "A hydrogen with its only neighbor, its carbon. No story for the negative value.",
    ),
    (
        [0, 1, 4],
        [3],
        "Three carbons where the rings meet. No story: its value flips sign as the order grows.",
    ),
    (
        [7, 8, 13],
        [3, 4, FULL],
        "A ring carbon with its two ring neighbors, but without its hydrogen. Negative, with no "
        "simple reading.",
    ),
    ([0, 2, 6, 7], [4, FULL], None),
    ([0, 1, 4, 5], [4, FULL], None),
    ([0, 3, 8, 9], [4, FULL], None),
    ([3, 9, 14, 15], [4, FULL], None),
    (
        [0, 1, 2, 3, 4, 5, 6, 7, 8],
        [FULL],
        "A carbon and the atoms within two bonds of it, but one: about as far as the GCN's two "
        "layers reach. No simple reading.",
    ),
]


def notes_molecule(example: dict[str, Any]) -> dict[str, Any]:
    """Return the picks and the notes of the molecule.

    The notes that the atoms' bonds give hold on every order: on its nitro groups, and on an atom
    with all of its neighbors, the atoms that one layer of the GCN combines.
    """
    atoms = example["atoms"]
    bonded: dict[int, list[int]] = {atom: [] for atom in range(len(atoms))}
    for a, b in example["bonds"]:
        bonded[a].append(b)
        bonded[b].append(a)
    notes: dict[str, Any] = {}
    for atom, element in enumerate(atoms):
        near = [atoms[other] for other in bonded[atom]]
        oxygens = [other for other in bonded[atom] if atoms[other] == "O"]
        if element == "N" and len(oxygens) == 2:
            carbons = [other for other in bonded[atom] if atoms[other] == "C"]
            notes[key(atom)] = (
                "The nitrogen of a nitro group (NO₂). Nitro groups on aromatic rings are a classic "
                "structural alert for mutagenicity."
            )
            for oxygen in oxygens:
                notes[key(oxygen)] = "An oxygen of a nitro group (NO₂)."
                notes[key(atom, oxygen)] = (
                    "A bond of a nitro group (NO₂). Much of the nitrogen's Shapley value is what "
                    "it does together with its oxygens."
                )
            for carbon in carbons:
                notes[key(atom, carbon)] = "The bond that attaches a nitro group to the ring."
            notes[key(atom, *oxygens, *carbons)] = (
                "A nitro group (NO₂) with the ring carbon it sits on: the nitrogen's whole "
                "neighborhood, which one GCN layer combines."
            )
        elif element == "C" and len(bonded[atom]) == 3:
            notes[key(atom, *bonded[atom])] = (
                "The ring carbon carrying a nitro group, with its three neighbors: the atoms one "
                "GCN layer combines at it."
                if "N" in near
                else "A carbon with all three of its neighbors: exactly the atoms one GCN layer "
                "combines at that carbon."
            )
            if "H" in near:
                notes[key(atom)] = (
                    "A ring carbon that carries a hydrogen. No story for its own value: the model "
                    "is what it is."
                )
    picked = curated(example, MOLECULE_PICKS)
    # (a note the bonds give holds on every order, whatever orders its group is picked on)
    return {"notes": {**picked["notes"], **notes}, "picks": picked["picks"]}


# the house's features, as the picks below name them
INCOME, AGE, ROOMS, BEDS, POP, OCCUP, LAT, LONG = range(8)
# what is picked on the house that XGBoost values, by hand: the place carries its story, the
# income and the rooms raise its value (both are above California's medians: $35k, 5.2 rooms), and
# the rest has none
HOUSE_PICKS: list[tuple[list[int], list[Any], str | None]] = [
    (
        [LAT],
        [1, 2, 3, 4, FULL],
        "Alone, the latitude seems to raise the value; the longitude takes that back: by its own "
        "value at order 1, through their pair from order 2.",
    ),
    (
        [LONG],
        [1],
        "At order 1 the longitude takes back what the latitude adds; from order 2 the pair does, "
        "and the longitude alone is near zero.",
    ),
    ([INCOME], [1, 2, 3, 4, FULL], "An income above the California median raises the value."),
    (
        [ROOMS],
        [1, 2, 3, 4, FULL],
        "More rooms per home than is typical in California raise the value.",
    ),
    ([OCCUP], [2, 3, 4, FULL], None),
    (
        [LAT, LONG],
        [2, 3, 4, FULL],
        "Where the house is. Alone, the latitude seems to raise its value; with the longitude, "
        "that is taken back: the place as a whole hardly moves it.",
    ),
    (
        [INCOME, ROOMS],
        [2],
        "Small for XGBoost: income and rooms add up almost independently, unlike for TabPFN.",
    ),
    ([ROOMS, POP], [2], None),
    ([AGE, OCCUP], [2], None),
    ([INCOME, OCCUP], [2], None),
    ([INCOME, POP], [2, 3, 4, FULL], None),
    ([INCOME, LONG], [3, 4, FULL], None),
    (
        [OCCUP, LAT, LONG],
        [3, 4, FULL],
        "No story: a small correction to the place, next to the pair of latitude and longitude.",
    ),
    (
        [AGE, LAT, LONG],
        [3, 4, FULL],
        "No story: a small correction to the place, next to the pair of latitude and longitude.",
    ),
    (
        [INCOME, LAT, LONG],
        [3, 4, FULL],
        "No story: a small correction to the place, next to the pair of latitude and longitude.",
    ),
]
# and on the same house as TabPFN values it: the place and the income with the rooms carry its
# stories, read off the values' sums; its groups of many features have none
TABPFN_PICKS: list[tuple[list[int], list[Any], str | None]] = [
    (
        [LAT],
        [1, 2, 3, 4, 5, 6, 7, FULL],
        "On its own the latitude lowers the value; with the longitude, the place lowers it more "
        "(see their pair).",
    ),
    (
        [ROOMS],
        [1, 2, 3, 4, 5, 6, 7, FULL],
        "Alone, more rooms per home than is typical raise the value.",
    ),
    ([LONG], [1], None),
    (
        [LONG],
        [2, 3, 4, 5, 6, 7, FULL],
        "Alone the longitude raises the value; with the latitude, that is taken back and more.",
    ),
    (
        [INCOME],
        [2, 3, 4, 5, 6, 7, FULL],
        "Alone, the income raises the value; with the rooms, that is taken back (see their pair).",
    ),
    (
        [LAT, LONG],
        [2, 3, 4, 5, 6, 7, FULL],
        "Where the house is: for TabPFN, the place as a whole lowers this house's value.",
    ),
    (
        [INCOME, ROOMS],
        [2, 3, 4, 5, 6, 7, FULL],
        "Alone, the income and the rooms each raise the value; together, they take most of that "
        "back.",
    ),
    ([OCCUP, LAT], [2, 3], "With the occupancy, the latitude's lowering is mostly taken back."),
    ([AGE, LONG], [2, 3], "With the age, the longitude's raise is mostly taken back."),
    ([AGE, ROOMS], [2, 3, 4], "With the age, the rooms raise the value even more."),
    ([INCOME, LONG], [2], "No story: its sign changes from order to order."),
    ([AGE, LAT, LONG], [3], None),
    (
        [INCOME, ROOMS, LONG],
        [3],
        "No story: it raises at order 3 and lowers from order 5 on.",
    ),
    (
        [ROOMS, LAT, LONG],
        [4, 5, 6, 7, FULL],
        "It takes back, and more, what the rooms' pairs with the latitude and the longitude add: "
        "in sum, rooms and place lower the value.",
    ),
    (
        [INCOME, ROOMS, LAT, LONG],
        [4, 5, 6, 7, FULL],
        "In sum, once the rooms and the place are known, adding the income hardly moves the value.",
    ),
    ([INCOME, ROOMS, OCCUP], [4], None),
    ([AGE, ROOMS, LONG], [4, 5, 6], None),
    ([INCOME, AGE, ROOMS, LONG], [5], None),
    ([INCOME, BEDS, POP, LAT, LONG], [5], None),
    ([INCOME, ROOMS, OCCUP, LAT, LONG], [6], None),
    ([INCOME, ROOMS, BEDS, OCCUP, LAT, LONG], [6], None),
    ([INCOME, AGE, ROOMS, BEDS, POP, LAT, LONG], [7, FULL], None),
    ([AGE, ROOMS, BEDS, POP, LAT, LONG], [7, FULL], None),
    ([INCOME, ROOMS, BEDS, POP, OCCUP, LAT, LONG], [7], None),
    ([INCOME, AGE, ROOMS, BEDS, POP, LONG], [FULL], None),
]


def notes_house(example: dict[str, Any]) -> dict[str, Any]:
    """Return the picks and the notes of the house that XGBoost values."""
    picked = curated(example, HOUSE_PICKS)
    picked["notes"]["many"] = [
        4,
        "Small print. The trees are 5 deep, so groups of 4 and 5 features have values too, but "
        "hardly any: this model's interactions are pairs and a few triples.",
    ]
    return picked


def notes_tabpfn(example: dict[str, Any]) -> dict[str, Any]:
    """Return the picks and the notes of the house that TabPFN values."""
    picked = curated(example, TABPFN_PICKS)
    picked["notes"]["many"] = [
        5,
        "No story here. TabPFN's values of many features are large, of both signs, and cancel "
        "for the most part: the model is what it is.",
    ]
    return picked


# what is picked on the photos, by the photo's id: on a photo, only what is picked can be pointed
# at. A row names its players as a word of the caption or a patch (its row and column in the
# photo, from 1), the orders it is picked on, and its note, where it has a story: what the patch
# shows, and what the values say on those orders, no more. The explanations are estimates, and
# many strong values have no reading: word with word, a function word, a group of three
PHOTO_PICKS: dict[str, list[tuple[list[Any], list[Any], str | None]]] = {
    "elephant": [
        (
            ["elephant"],
            [1, 2, 3, 4, FULL],
            "The word for the photo's main subject: it raises the similarity on every order.",
        ),
        ([(6, 1)], [1, 2], "The egret's patch: the white bird's head and neck."),
        (["bird"], [1, 2, 3, 4, FULL], None),
        (["white"], [2, 3, 4, FULL], None),
        (["small"], [4, FULL], None),
        (
            [(6, 1), "bird"],
            [2, 3],
            "The egret's patch and the word for it: the model matches the two.",
        ),
        (
            [(6, 1), "white"],
            [2, 3, 4, FULL],
            "The egret's patch and “white”: the color word matches the bird.",
        ),
        (
            [(5, 4), "elephant"],
            [2, 3, 4, FULL],
            "The curled trunk and the word “elephant”: the model matches the two.",
        ),
        (
            [(3, 5), "elephant"],
            [2, 3, 4, FULL],
            "The top of the elephant's head and the word “elephant”: they match.",
        ),
        (
            [(4, 6), "elephant"],
            [4, FULL],
            "The elephant's ear and shoulder and the word “elephant”: they match.",
        ),
        (
            [(5, 5), "white"],
            [4, FULL],
            "A brown patch of the elephant and the word “white”: they do not match.",
        ),
        (["elephant", "standing"], [2], None),
        (["elephant", "bird"], [3, 4, FULL], None),
        ([(6, 1), "elephant"], [3], None),
        ([(6, 1), "elephant", "bird"], [4, FULL], None),
    ],
    "giraffe-tree": [
        (
            ["giraffe"],
            [1, 2, 3, 4, FULL],
            "The animal the caption is about. Its word adds to the match on every order.",
        ),
        (["water"], [1, 2, 3, 4, FULL], "“water”: the pond fills the lower part of the photo."),
        (["pond"], [2, 3], "“pond”: the water at the bottom of the photo."),
        (
            [(3, 5)],
            [1],
            "The top of the giraffe's back: the patch the Shapley values credit most.",
        ),
        (
            [(2, 2)],
            [1],
            "The dead tree in the background. No giraffe is there, yet its patch gets credit: "
            "from order 2 on, see it with “giraffe”.",
        ),
        (["legs"], [2, 3, 4, FULL], None),
        (["at"], [4, FULL], None),
        (
            [(2, 2), "giraffe"],
            [2, 3, 4, FULL],
            "The dead tree in the background and the word “giraffe”: together they raise the "
            "match, though no giraffe is there.",
        ),
        (
            [(6, 5), "giraffe"],
            [2, 3, 4, FULL],
            "The giraffe's head, lowered to the water, and the word “giraffe”.",
        ),
        (
            [(4, 5), "giraffe"],
            [2],
            "The giraffe's chest and a front leg, and the word “giraffe”.",
        ),
        ([(3, 5), "giraffe"], [2], "The top of the giraffe's back and the word “giraffe”."),
        (
            [(3, 5), (4, 5), "giraffe"],
            [3],
            "Two patches of the giraffe's body with the word “giraffe”: the body counts when "
            "both are there.",
        ),
        (
            ["water", "pond"],
            [2, 3, 4, FULL],
            "Two words for the same thing: each adds less once the other is there.",
        ),
        (
            ["drink", "water"],
            [2, 3, 4, FULL],
            "“drink” already suggests water: each adds less once the other is there.",
        ),
        ([(7, 2), "water"], [3, 4, FULL], "A patch of the pond and the word “water”."),
        (["giraffe", "legs"], [2, 3, 4, FULL], None),
        (["spread", "pond"], [4, FULL], None),
    ],
    "juice-theory": [
        (
            ["traffic"],
            [1, 2, 3, 4, FULL],
            "The photo shows a traffic light, so the word “traffic” fits it.",
        ),
        (
            ["light"],
            [1, 2, 3, 4, FULL],
            "The photo shows a traffic light, so the word “light” fits it.",
        ),
        ([(1, 7)], [1, 2, 3, 4, FULL], "The red light."),
        ([(5, 4)], [1], "The letters “RY” of “THEORY”, below the tail of “Juice”."),
        (["juice"], [3, 4, FULL], None),
        (["sign", "traffic"], [2, 3], None),
        (
            [(5, 4), "theory"],
            [2, 4, FULL],
            "The letters “RY” of “THEORY”, and the word “theory”.",
        ),
        ([(5, 4), "sign"], [2], "The sign's lettering and the word “sign”."),
        ([(1, 7), "traffic"], [2, 3, 4, FULL], "The red light and the word “traffic”."),
        ([(1, 7), "light"], [2, 3, 4, FULL], "The red light and the word “light”."),
        (
            [(5, 2), "juice"],
            [2, 3],
            "The “u” of “Juice” on the sign, and the word “juice”.",
        ),
        (
            [(5, 1), "juice"],
            [3, 4, FULL],
            "The “J” of “Juice” on the sign, and the word “juice”.",
        ),
        ([(5, 4), "sign", "theory"], [3], None),
        (
            [(5, 3), "theory"],
            [3],
            "Letters of “THEORY” too, the “E” and the “O”: with the word “theory”, they lower "
            "the match. The model does not spell.",
        ),
        (["traffic", "light"], [4, FULL], None),
        (["sign", "traffic", "light"], [4, FULL], None),
        (
            [(7, 3), "juice"],
            [4, FULL],
            "Lettering of “The Pantry”, and the word “juice”: linked nearly as strongly as the "
            "word's own letters.",
        ),
    ],
}


def notes_photo(example: dict[str, Any]) -> dict[str, Any]:
    """Return the picks and the notes of a photo and its caption, by the photo's file name."""
    grid, tokens = example["grid"], example["tokens"]

    def player(named: tuple[int, int] | str) -> int:
        if isinstance(named, str):
            return grid * grid + tokens.index(named)
        row, column = named
        return (row - 1) * grid + column - 1

    table = PHOTO_PICKS.get(Path(example["image"]).stem, [])
    return curated(
        example, [([player(named) for named in players], on, note) for players, on, note in table]
    )


def photo_words(photo: str) -> dict[str, Any]:
    """Return the words of a photo's example: its tab's, and the photo's title in the FIxLIP demo."""
    config = tomllib.loads((FIXLIP_CONFIG).read_text(encoding="utf-8"))
    title = next(entry["title"] for entry in config["photos"] if entry["id"] == photo)
    words = {"label": "Describing images", "detail": "SigLIP 2", "model": model("black box")}
    return {**words, "photo": {"title": title}}


# the examples: the file of each one's results, its words, which no result gives (its tab's name,
# its model's, and the model as the demo shows it), what packs its results, and what notes it. They
# are in the order of the demo's tabs; an example ``of`` another one has no tab: it is another
# example of that tab (another photo and its caption), which the tab's picker shows
EXAMPLES: dict[str, tuple[str, dict[str, Any], Any, Any]] = {
    "molecule": (
        "molecule",
        {"label": "Molecule", "detail": "GCN", "model": model("graph")},
        pack_molecule,
        notes_molecule,
    ),
    "house": (
        "house",
        {"label": "One house", "detail": "XGBoost", "model": model("tree")},
        pack_house,
        notes_house,
    ),
    "tabpfn": (
        "house_tabpfn",
        {"label": "One house, TabPFN", "detail": "TabPFN", "model": model("tabpfn")},
        pack_tabpfn,
        notes_tabpfn,
    ),
    # (the same model as the house's, with the same average)
    "houses": (
        "houses",
        {"label": "All houses", "detail": "XGBoost", "model": model("tree")},
        pack_houses,
        None,
    ),
    "fixlip": ("fixlip", photo_words("elephant"), pack_photo, notes_photo),
    "giraffe": (
        "fixlip_giraffe-tree",
        {**photo_words("giraffe-tree"), "of": "fixlip"},
        pack_photo,
        notes_photo,
    ),
    "sign": (
        "fixlip_juice-theory",
        {**photo_words("juice-theory"), "of": "fixlip"},
        pack_photo,
        notes_photo,
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
    for name, (file, words, packer, noter) in EXAMPLES.items():
        if name in args.examples:
            result = json.loads((args.results / f"{file}.json").read_text(encoding="utf-8"))
            indices.add(result["index"])
            packed = packer(result)
        else:
            # (anew, whatever the data has)
            written = ("id", "notes", "picks", "of", "photo", *words)
            packed = {name: value for name, value in kept[name].items() if name not in written}
        example = {"id": name, **words, **packed}
        if noter:
            example.update(noter(example))  # (its notes, and what is picked by hand)
        examples.append(example)
    (index,) = indices  # of every order of every example, for the demo's labels

    data = {"index": index, "examples": examples}
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    target.write_text(f"{HEADER}window.SHAPIQ_ORDERS = {text};\n", encoding="utf-8")


if __name__ == "__main__":
    main()
