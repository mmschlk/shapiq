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
                "count": len(values),
            }
        )
    return out


def between(result: dict[str, Any], strongest: int | None) -> list[dict[str, Any]]:
    """Return the explanations of the orders between the last sheet and the highest with interactions.

    The orders the demo's blank sheets stand for, up to the highest order with an interaction (the
    orders above it equal the full decomposition), where the result keeps their values (fixlip.py
    keeps only those of the orders with a sheet). Of the interactions, each keeps the ``strongest``.
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
                "count": len(entry["values"]),
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
    # the number of values: counted where the values were left out (fixlip.py keeps only those of
    # the orders with a sheet), else all of them
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
        "fits": fits(fixlip),
    }


# -- the notes: what the demo's tip says, in a sentence, of a player or a group pointed at. Most
# have none: not every value has a story, and a note only says what the example shows. A note's key
# is its players, in their order, joined by commas; "many" is the note of every group of that many
# players at the least --------------------------------------------------------------------------------


def key(*players: int) -> str:
    """Return the key of a player or a group: its players, in their order, joined by commas."""
    return ",".join(str(player) for player in sorted(players))


def notes_molecule(example: dict[str, Any]) -> dict[str, Any]:
    """Return the notes on the molecule: its nitro groups, and the carbons with their neighbors."""
    atoms = example["atoms"]
    bonded: dict[int, list[int]] = {atom: [] for atom in range(len(atoms))}
    for a, b in example["bonds"]:
        bonded[a].append(b)
        bonded[b].append(a)
    notes = {}
    for atom, element in enumerate(atoms):
        oxygens = [other for other in bonded[atom] if atoms[other] == "O"]
        if element == "N" and len(oxygens) == 2:
            carbons = [other for other in bonded[atom] if atoms[other] == "C"]
            notes[key(atom)] = (
                "The nitrogen of a nitro group (NO₂). Nitro groups on aromatic rings are a classic "
                "sign of a mutagen."
            )
            for oxygen in oxygens:
                notes[key(oxygen)] = "An oxygen of a nitro group (NO₂)."
                notes[key(atom, oxygen)] = (
                    "A bond of a nitro group (NO₂). Much of what the Shapley values credit to the "
                    "nitrogen alone is what it does with its oxygens."
                )
            notes[key(atom, *oxygens)] = "A nitro group (NO₂), whole."
            notes[key(atom, *oxygens, *carbons)] = (
                "A nitro group (NO₂) with the carbon of the ring it sits on."
            )
        elif element == "C" and len(bonded[atom]) == 3:
            notes[key(atom, *bonded[atom])] = (
                "A carbon with its three neighbors: the atoms that one layer of the GCN reads "
                "together."
            )
    return notes


def notes_house(example: dict[str, Any]) -> dict[str, Any]:
    """Return the notes on the house that XGBoost values: its location, and its largest groups."""
    latitude, longitude = (example["features"].index(name) for name in ("Lat", "Long"))
    return {
        key(latitude, longitude): (
            "Where the house is. Alone, the latitude seems to raise its value; with the longitude, "
            "that is taken back: the place as a whole hardly moves it."
        ),
        "many": [
            4,
            "Small print. The trees are 5 deep, so groups of 4 and 5 features have values too, but "
            "hardly any: this model's interactions are pairs and triples.",
        ],
    }


def notes_tabpfn(example: dict[str, Any]) -> dict[str, Any]:
    """Return the notes on the house that TabPFN values: two pairs, and its large groups."""
    index = example["features"].index
    return {
        key(index("Lat"), index("Long")): (
            "Where the house is: for TabPFN, the place as a whole lowers this house's value."
        ),
        key(index("Income"), index("Rooms")): (
            "Alone, the income and the rooms each raise the value; together, they take most of it "
            "back."
        ),
        "many": [
            5,
            "No story here. A black box's values of many features are large, of both signs, and "
            "cancel for the most part: the model is what it is.",
        ],
    }


# the notes on the photos, by the photo's id: the players of each as a word of the caption (its
# first) or a patch (its row and column in the photo, from 1), and the note
PHOTO_NOTES: dict[str, list[tuple[list[Any], str]]] = {
    "elephant": [
        (
            ["elephant"],
            "The word the match hangs on: no other player raises the similarity as much.",
        ),
        ([(6, 1)], "The egret's patch."),
        ([(6, 1), "bird"], "The egret's patch and the word for it: the model matches the two."),
        ([(6, 1), "white"], "The egret's patch and its color in the caption."),
        ([(5, 4), "elephant"], "A patch of the elephant and the word for it."),
        ([(3, 5), "elephant"], "A patch of the elephant and the word for it."),
        ([(4, 6), "elephant"], "A patch of the elephant and the word for it."),
    ],
    "giraffe-tree": [
        (["giraffe"], "The word the Shapley values credit the most."),
        ([(2, 2)], "The dead tree in the background. No giraffe, but its patch raises the match."),
        (
            [(2, 2), "giraffe"],
            "The dead tree in the background and the word “giraffe”: the model sees some giraffe "
            "in it.",
        ),
        ([(3, 5), "giraffe"], "A patch of the giraffe and the word for it."),
        ([(4, 5), "giraffe"], "A patch of the giraffe and the word for it."),
        ([(6, 5), "giraffe"], "A patch of the giraffe and the word for it."),
        ([(3, 6), "giraffe"], "A patch of the giraffe and the word for it."),
        (
            ["water", "pond"],
            "Two words for the same thing: each adds less once the other is there.",
        ),
    ],
    "juice-theory": [
        ([(5, 4)], "The end of “THEORY” on the sign."),
        (
            [(5, 4), "theory"],
            "The model reads: the end of “THEORY” on the sign, and the word “theory”.",
        ),
        ([(5, 4), "sign"], "The sign's lettering and the word “sign”."),
        (
            [(5, 1), "juice"],
            "The model reads: letters of “Juice” on the sign, and the word “juice”.",
        ),
        (
            [(5, 2), "juice"],
            "The model reads: letters of “Juice” on the sign, and the word “juice”.",
        ),
        ([(1, 7)], "The red light."),
        ([(1, 7), "traffic"], "The red light and the word “traffic”."),
        ([(1, 7), "light"], "The red light and the word “light”."),
    ],
}


def notes_photo(example: dict[str, Any]) -> dict[str, Any]:
    """Return the notes on a photo and its caption, by the photo's id (the name of its file)."""
    grid, tokens = example["grid"], example["tokens"]

    def player(named: tuple[int, int] | str) -> int:
        if isinstance(named, str):
            return grid * grid + tokens.index(named)
        row, column = named
        return (row - 1) * grid + column - 1

    photo = Path(example["image"]).stem
    return {
        key(*(player(named) for named in players)): note
        for players, note in PHOTO_NOTES.get(photo, [])
    }


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
            written = ("id", "notes", "of", "photo", *words)  # (anew, whatever the data has)
            packed = {name: value for name, value in kept[name].items() if name not in written}
        example = {"id": name, **words, **packed}
        if noter:
            example["notes"] = noter(example)
        examples.append(example)
    (index,) = indices  # of every order of every example, for the demo's labels

    data = {"index": index, "examples": examples}
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    target.write_text(f"{HEADER}window.SHAPIQ_ORDERS = {text};\n", encoding="utf-8")


if __name__ == "__main__":
    main()
