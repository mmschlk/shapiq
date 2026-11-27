"""The explorer's GNNs in numpy, their graph games, and GraphSHAP-IQ on them, for the page's figures.

What the figure scripts of this folder share. Everything comes from the explorer's own data,
``../assets/data.js``: the three folded GNNs (B-XAIC's GIN: three GIN layers, their batch norms
folded into the linear layers, a sum over the atoms and a linear readout of the log-odds), the seven
molecules and their explanations. Nothing needs torch, the datasets or a trained model:

- :func:`data`, :func:`task`, :func:`molecule`: the explorer's data, a question (task) and a
  molecule by id. A molecule is a dict as in data.js (``atoms`` [[element, x, y, hydrogens,
  charge]], ``bonds`` [[i, j, order]], ``rings``); :func:`from_smiles` makes one from a SMILES as
  the demos draw them (RDKit, imported only there).
- :class:`GNN`: the forward pass on a batch of node features ``(batch, n, 11)``, in float32 as the
  demos compute it, with the node embeddings after every layer and the readout's term of every atom.
- :func:`coalition_features`: the node features of maskings (masked atoms take the question's
  baseline, ``task["baseline"]``, the mean atom of its training set).
- :func:`graph_game`: the GNN-induced graph game as a shapiq ``Game``
  (``GraphBaselineImputer``) with its ``GraphModel``, and :func:`explain`, GraphSHAP-IQ on it.
- :func:`all_values` and :func:`moebius_brute_force`: the game on all 2^n maskings and its Möbius
  transform, by brute force (n up to about 22).

Run it to check that it reproduces the explorer's data::

    python gnn.py           # log-odds, empty prediction, budget and Shapley values of all 21
                            # explanations against data.js, and brute force against GraphSHAP-IQ
"""

from __future__ import annotations

import base64
import json
import time
import warnings
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

from shapiq.graph import GraphBaselineImputer, GraphModel, GraphSHAPIQ

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent / "assets"
PREFIX = "window.SHAPIQ_GRAPHSHAPIQ_EXPLORER = "
HOPS = 3  # the GNNs' layers: the radius of the receptive fields

Molecule = dict[str, Any]
Task = dict[str, Any]


# -- the explorer's data --------------------------------------------------------------------------


@cache
def data() -> dict[str, Any]:
    """Return the explorer's data, ``../assets/data.js`` (``window.SHAPIQ_GRAPHSHAPIQ_EXPLORER``)."""
    text = (ASSETS / "data.js").read_text(encoding="utf-8")
    return json.loads(text[text.index(PREFIX) + len(PREFIX) :].rstrip().removesuffix(";"))


def symbols() -> list[str]:
    """Return the elements of the one-hot node features, ``"Unk"`` last."""
    return data()["symbols"]


def task(task_id: str) -> Task:
    """Return a question (``indole``, ``mutagenic`` or ``PAINS``) with its folded GNN."""
    return next(t for t in data()["tasks"] if t["id"] == task_id)


def molecule(molecule_id: str) -> Molecule:
    """Return one of the explorer's seven molecules by its id."""
    return next(m for m in data()["molecules"] if m["id"] == molecule_id)


def molecules() -> list[Molecule]:
    """Return the explorer's molecules, in the order of data.js."""
    return data()["molecules"]


def explanation(molecule_id: str, task_id: str) -> dict[str, Any]:
    """Return the explorer's explanation of a molecule for a question (only its strongest terms)."""
    return data()["explanations"][f"{molecule_id}/{task_id}"]


def atoms_of(mask: int) -> tuple[int, ...]:
    """Return the atoms of a coalition written as a bitmask (as data.js writes them)."""
    return tuple(k for k in range(mask.bit_length()) if mask >> k & 1)


def mask_of(atoms: tuple[int, ...] | list[int]) -> int:
    """Return a coalition of atoms as a bitmask."""
    return sum(1 << k for k in atoms)


# -- the GNN ------------------------------------------------------------------------------------


def features(m: Molecule) -> np.ndarray:
    """Return a molecule's node features, one-hot over the elements, of shape ``(n, 11)``."""
    sym = symbols()
    x = np.zeros((len(m["atoms"]), len(sym)), dtype=np.float32)
    for k, atom in enumerate(m["atoms"]):
        x[k, sym.index(atom[0]) if atom[0] in sym else len(sym) - 1] = 1
    return x


def edge_index(m: Molecule) -> np.ndarray:
    """Return a molecule's bonds as an edge index of shape ``(2, n_bonds)``, one direction each."""
    return np.array([[b[0] for b in m["bonds"]], [b[1] for b in m["bonds"]]], dtype=int).reshape(
        2, -1
    )


class GNN:
    """A question's folded GNN on one molecule's graph, in numpy and float32.

    The steps of ``task["gnn"]["steps"]`` are affine maps, each optionally after a gather (an atom
    adds its neighbours' rows, GIN's aggregation with eps = 0) or a sum over the atoms, and
    optionally followed by a ReLU. Two steps make one GIN layer; the last step is the readout.
    """

    def __init__(self, t: Task, m: Molecule, chunk: int = 32768) -> None:
        """Read the GNN's weights and the molecule's adjacency (with self-loops)."""
        self.task, self.molecule, self.chunk = t, m, chunk
        self.n = len(m["atoms"])
        self.steps = [
            {
                "w": np.frombuffer(base64.b64decode(s["w"]), dtype="<f4").reshape(s["shape"]),
                "b": np.frombuffer(base64.b64decode(s["b"]), dtype="<f4"),
                "relu": s["relu"],
                "gather": s["gather"],
                "sum": s["sum"],
            }
            for s in t["gnn"]["steps"]
        ]
        self.adjacency = np.eye(self.n, dtype=np.float32)
        for i, j, *_ in m["bonds"]:
            self.adjacency[i, j] = self.adjacency[j, i] = 1
        self.baseline = np.asarray(t["baseline"], dtype=np.float32)
        readout = self.steps[-1]
        self.readout_w, self.readout_b = readout["w"][0], float(readout["b"][0])

    def layers(self, x: np.ndarray) -> list[np.ndarray]:
        """Return the node embeddings after each GIN layer, each ``(batch, n, 32)``.

        ``x`` has the shape ``(batch, n, 11)``, or ``(n, 11)`` and then the embeddings have no
        batch axis.
        """
        single = x.ndim == 2
        h = np.asarray(x, dtype=np.float32)[None] if single else np.asarray(x, dtype=np.float32)
        out = []
        for k, step in enumerate(self.steps[:-1]):
            if step["gather"]:
                h = self.adjacency @ h
            h = h @ step["w"].T + step["b"]
            if step["relu"]:
                h = np.maximum(h, 0)
            if k % 2 == 1:
                out.append(h[0] if single else h)
        return out

    def atom_terms(self, x: np.ndarray) -> np.ndarray:
        """Return each atom's term of the log-odds: its last embedding through the readout.

        The log-odds are the sum of these terms plus the readout's bias (:attr:`readout_b`).
        """
        return self.layers(x)[-1] @ self.readout_w

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Return the log-odds for node features of shape ``(batch, n, 11)``, as float64."""
        x = np.asarray(x, dtype=np.float32)
        out = np.empty(len(x))
        for start in range(0, len(x), self.chunk):
            h = x[start : start + self.chunk]
            for step in self.steps:
                if step["gather"]:
                    h = self.adjacency @ h
                if step["sum"]:
                    h = h.sum(axis=1, keepdims=True)
                h = h @ step["w"].T + step["b"]
                if step["relu"]:
                    h = np.maximum(h, 0)
            out[start : start + self.chunk] = h[:, 0, 0]
        return out

    def predict(self, present: np.ndarray | list[int] | None = None) -> float:
        """Return the log-odds with only the atoms ``present`` (default: all) unmasked."""
        coalition = np.ones(self.n, dtype=bool)
        if present is not None:
            coalition = np.zeros(self.n, dtype=bool)
            coalition[list(present)] = True
        return float(self(coalition_features(self, coalition[None]))[0])


def coalition_features(gnn: GNN, coalitions: np.ndarray) -> np.ndarray:
    """Return the node features of maskings.

    ``coalitions`` (batch, n) is True where an atom is present; a masked atom takes the question's
    baseline features (as GraphBaselineImputer).
    """
    coalitions = np.asarray(coalitions, dtype=bool)
    x = features(gnn.molecule)
    return np.where(coalitions[:, :, None], x[None], gnn.baseline[None, None]).astype(np.float32)


# -- the graph game and GraphSHAP-IQ ------------------------------------------------------------


def graph_game(m: Molecule, t: Task, n_hops: int = HOPS) -> tuple[GraphBaselineImputer, GraphModel]:
    """Return the GNN-induced graph game of a molecule for a question, and its ``GraphModel``.

    The game's value of a coalition is the GNN's log-odds with the other atoms masked (their
    features replaced by the baseline), not centred: the empty coalition's value is the prediction
    on the all-baseline graph (``empty`` in data.js).
    """
    x = features(m).astype(float)
    graph = GraphModel(edge_index=edge_index(m), n_hops=n_hops, x=x)
    game = GraphBaselineImputer(
        GNN(t, m), np.asarray(t["baseline"], dtype=float), x=graph, normalize=False
    )
    return game, graph


def explain(
    m: Molecule, t: Task, max_interaction_size: int | None = None, n_hops: int = HOPS
) -> GraphSHAPIQ:
    """Return GraphSHAP-IQ on the molecule's graph game (exact without ``max_interaction_size``).

    ``.moebius`` holds the Möbius interactions (InteractionValues), ``.budget`` the maskings
    evaluated, and ``(index, order)`` converts, e.g. ``explain(m, t)("k-SII", 2)``.
    """
    game, graph = graph_game(m, t, n_hops)
    return GraphSHAPIQ(game, graph, max_interaction_size=max_interaction_size)


# -- brute force over all 2^n maskings ------------------------------------------------------------


def all_values(m: Molecule, t: Task) -> np.ndarray:
    """Return the game's value of every coalition, indexed by its bitmask (length 2^n)."""
    gnn = GNN(t, m)
    n = gnn.n
    values = np.empty(2**n)
    step = 2**15
    for start in range(0, 2**n, step):
        masks = np.arange(start, min(start + step, 2**n))
        coalitions = (masks[:, None] >> np.arange(n)[None]) & 1
        values[start : start + len(masks)] = gnn(coalition_features(gnn, coalitions))
    return values


def moebius_brute_force(values: np.ndarray) -> np.ndarray:
    """Return the Möbius transform of a game given on all coalitions (by bitmask), in O(n 2^n)."""
    out = np.array(values, dtype=float)
    n = int(np.log2(len(out)))
    for k in range(n):
        view = out.reshape(-1, 2, 2**k)
        view[:, 1, :] -= view[:, 0, :]
    return out


# -- a molecule from a SMILES ---------------------------------------------------------------------


def from_smiles(smiles: str, rotate: float = 0, name: str = "") -> Molecule:
    """Return a molecule drawn as the demos draw theirs, from a SMILES (needs RDKit).

    A copy of ``depict`` in ``../../graphshapiq-common/compute/pack.py`` (which imports torch):
    RDKit's 2D coordinates (CoordGen), bonds of length 1, Kekulé bonds, y pointing down, atoms in
    RDKit's order for the SMILES (the order of the explanations).
    """
    from rdkit import Chem  # only this function needs RDKit
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula

    mol = Chem.MolFromSmiles(smiles)
    rdDepictor.SetPreferCoordGen(True)  # noqa: FBT003  (RDKit's API)
    rdDepictor.Compute2DCoords(mol)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    xy = mol.GetConformer().GetPositions()[:, :2]
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    xy = (xy - xy.mean(axis=0)) / np.median([np.linalg.norm(xy[i] - xy[j]) for i, j in bonds])
    turn = np.deg2rad(rotate)
    xy = xy @ np.array([[np.cos(turn), np.sin(turn)], [-np.sin(turn), np.cos(turn)]])
    xy *= [1, -1]  # SVG's y points down
    return {
        "id": name.lower().replace(" ", "-") or smiles,
        "name": name or smiles,
        "smiles": smiles,
        "atoms": [
            [
                a.GetSymbol(),
                *np.round(xy[a.GetIdx()], 3).tolist(),
                a.GetTotalNumHs(),
                a.GetFormalCharge(),
            ]
            for a in mol.GetAtoms()
        ],
        "bonds": [
            [b.GetBeginAtomIdx(), b.GetEndAtomIdx(), int(b.GetBondTypeAsDouble())]
            for b in mol.GetBonds()
        ],
        "rings": [list(ring) for ring in mol.GetRingInfo().AtomRings()],
        "formula": CalcMolFormula(mol),
    }


# -- the check ------------------------------------------------------------------------------------


def check() -> None:
    """Compare GraphSHAP-IQ on the numpy GNNs with the explorer's data, and with brute force."""
    worst = {"logOdds": 0.0, "empty": 0.0, "SV": 0.0}
    for m in molecules():
        for t in data()["tasks"]:
            ref = explanation(m["id"], t["id"])
            start = time.perf_counter()
            g = explain(m, t)
            moebius = g.moebius
            sv = g("SV", 1)
            seconds = time.perf_counter() - start
            log_odds = float(sum(moebius.dict_values.values()))
            empty = float(moebius[()]) if () in moebius.interaction_lookup else 0.0
            n = len(m["atoms"])
            sv_err = max(abs(sv[atoms_of(mask)] - v) for mask, v in ref["kSII"]["1"])
            errors = {
                "logOdds": abs(log_odds - ref["logOdds"]),
                "empty": abs(empty - ref["empty"]),
                "SV": sv_err,
            }
            for key, value in errors.items():
                worst[key] = max(worst[key], value)
            if g.budget != ref["budget"]:
                msg = f"{m['id']}/{t['id']}: budget {g.budget}, data.js {ref['budget']}"
                raise ValueError(msg)
            print(
                f"{m['id']}/{t['id']}: n={n} budget={g.budget:,} log-odds {log_odds:.4f} "
                f"(data.js {ref['logOdds']}) empty {empty:.4f} ({ref['empty']}) "
                f"SV error {sv_err:.1e} [{seconds:.2f} s]"
            )
    print(
        "largest differences to data.js (which rounds to 4 decimals):",
        {k: f"{v:.1e}" for k, v in worst.items()},
    )
    # brute force on all 2^n maskings: the same Möbius interactions inside the receptive fields,
    # and zero (up to float32 noise) outside them
    for molecule_id in ("benzimidazole", "dopamine"):
        m = molecule(molecule_id)
        t = task(m["task"])
        g = explain(m, t)
        brute = moebius_brute_force(all_values(m, t))
        inside = {mask_of(s) for s in g.moebius.interaction_lookup}
        diff = max(abs(brute[mask_of(s)] - g.moebius[s]) for s in g.moebius.interaction_lookup)
        outside = max((abs(brute[k]) for k in range(len(brute)) if k not in inside), default=0.0)
        print(
            f"brute force {molecule_id}: max difference inside {diff:.1e}, max |value| outside {outside:.1e}"
        )


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    check()
