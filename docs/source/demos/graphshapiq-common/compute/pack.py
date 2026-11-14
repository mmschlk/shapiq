"""Write the data of the GraphSHAP-IQ demos: ``../../graphshapiq-<demo>/assets/data.js``.

Reads the GNNs of train.py and the explanations of explain.py (in ``results/``) and the texts of
config.toml, draws the molecules (2D coordinates of RDKit), keeps what the demos show, rounded, and
writes each demo's data as a script that sets a global (a script loads from disk too, where the
browser would not fetch a JSON file):

- ``explorer`` (``window.SHAPIQ_GRAPHSHAPIQ_EXPLORER``): the three GNNs, the molecules, and the
  explanation of every molecule for every question: the k-SII of orders 1 to 4 and the Möbius
  interactions, the strongest of every order overall and of every atom (the GNNs also run in the
  reader's browser, where the demo computes GraphSHAP-IQ itself for the molecule a reader builds);
- ``exact`` (``window.SHAPIQ_GRAPHSHAPIQ_EXACT``), the figure of receptive fields on the explorer's
  page: molecules of growing size, and the number of maskings GraphSHAP-IQ evaluates for GNNs of
  1 to 4 layers (no GNN is evaluated).

The demos run the GNNs themselves, for the maskings a reader picks: every GNN is written as a chain
of affine maps and ReLUs, its batch normalizations folded into its linear layers, in float32
(base64), and checked against PyTorch on random maskings. A coalition of atoms is written as a
bitmask of the atoms' indices (the molecules have at most 31 atoms).

Usage::

    python pack.py                    # all of them
    python pack.py --demos exact      # one of them
    python pack.py --texts            # only the texts of config.toml, into the data that is there

``--texts`` needs no ``results/``: it reads the explorer's ``data.js``, and writes it again with the
tasks' and the molecules' texts as config.toml has them now (and the drawings of the tasks'
patterns). The GNNs, the molecules' drawings and the explanations stay as they are.
"""

from __future__ import annotations

import argparse
import base64
import json
import tomllib
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import torch
from molecules import GNN, SYMBOLS, LogOdds, to_graph
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from shapiq import Game
from shapiq.graph import GraphModel, GraphSHAPIQ

HERE = Path(__file__).resolve().parent
DEMOS = HERE.parent.parent  # docs/source/demos
OVERALL = 24  # explorer: the strongest interactions of an order it gets
PER_ATOM = 8  # and the strongest interactions of an order with every atom
HOPS = range(1, 5)  # exact: the GNN depths it compares


def mask(atoms: list[int] | tuple[int, ...]) -> int:
    """Return a coalition of atoms as a bitmask."""
    return sum(1 << atom for atom in atoms)


def depict(entry: dict[str, Any]) -> dict[str, Any]:
    """Return a molecule's drawing: atoms, bonds (Kekulé) and rings, bonds of length 1."""
    mol = Chem.MolFromSmiles(entry["smiles"])
    rdDepictor.SetPreferCoordGen(True)  # noqa: FBT003  (RDKit's API)
    rdDepictor.Compute2DCoords(mol)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    xy = mol.GetConformer().GetPositions()[:, :2]
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    xy = (xy - xy.mean(axis=0)) / np.median([np.linalg.norm(xy[i] - xy[j]) for i, j in bonds])
    turn = np.deg2rad(entry.get("rotate", 0))
    xy = xy @ np.array([[np.cos(turn), np.sin(turn)], [-np.sin(turn), np.cos(turn)]])
    xy *= [1, -1]  # SVG's y points down
    return {
        "atoms": [  # element, x, y, hydrogens, charge
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


def fold(gnn: GNN) -> list[dict[str, Any]]:
    """Return a GNN's layers as affine maps (``w``, ``b``), the batch normalizations folded in.

    Every GIN layer sums each atom's features with its neighbors' (``"gather"``), then applies
    its MLP (linear, batch normalization, ReLU, linear) and, but for the last layer, a batch
    normalization and a ReLU. The readout sums the atoms and takes the difference of the logits.
    """

    def affine(linear: torch.nn.Linear, norm: torch.nn.Module | None = None) -> dict[str, Any]:
        w, b = linear.weight.detach().double(), linear.bias.detach().double()
        if norm is not None:  # in evaluation mode, a batch normalization is affine
            bn = norm.module if hasattr(norm, "module") else norm
            scale = bn.weight.detach().double() / torch.sqrt(bn.running_var.double() + bn.eps)
            w, b = scale[:, None] * w, scale * (b - bn.running_mean.double()) + bn.bias.detach()
        return {"w": w.numpy(), "b": b.numpy()}

    steps = []
    gin = gnn.gin
    for k, conv in enumerate(gin.convs):
        last = k == len(gin.convs) - 1
        lin1, lin2 = conv.nn.lins
        first = affine(lin1, conv.nn.norms[0]) | {"relu": True, "gather": True}
        second = affine(lin2, None if last else gin.norms[k]) | {"relu": not last, "gather": False}
        steps += [first, second]
    w = gnn.readout.weight.detach().double()
    b = gnn.readout.bias.detach().double()
    steps.append(
        {"w": (w[1:] - w[:1]).numpy(), "b": (b[1:] - b[:1]).numpy(), "relu": False, "sum": True}
    )
    return steps


def run(steps: list[dict[str, Any]], x: np.ndarray, edges: np.ndarray) -> float:
    """Return the log-odds of a folded GNN, in float32, as the demos compute them."""
    h = x.astype(np.float32)
    for step in steps:
        if step.get("gather"):
            h = h + np.array([h[edges[1][edges[0] == k]].sum(axis=0) for k in range(len(h))])
        if step.get("sum"):
            h = h.sum(axis=0, keepdims=True)
        h = h @ step["w"].astype(np.float32).T + step["b"].astype(np.float32)
        if step["relu"]:
            h = np.maximum(h, 0)
    return float(h[0, 0])


def pack_gnn(task: dict[str, Any], smiles: list[str], results: Path) -> dict[str, Any]:
    """Return a task's GNN for the demos, checked against PyTorch on random maskings."""
    gnn = GNN()
    gnn.load_state_dict(torch.load(results / "models" / f"{task['id']}.pt"))
    model = LogOdds(gnn.eval())
    steps = fold(gnn)
    rng = np.random.default_rng(0)
    error = 0.0
    for code in smiles:
        graph = to_graph(Chem.MolFromSmiles(code))
        edges = graph.edge_index.numpy()
        for _ in range(20):
            x = graph.x.numpy() * (rng.random((len(graph.x), 1)) < 0.6)
            with torch.no_grad():
                expected = float(
                    model(
                        torch.as_tensor(x), graph.edge_index, torch.zeros(len(x), dtype=torch.long)
                    )
                )
            error = max(error, abs(run(steps, x, edges) - expected))
    if error > 1e-3:
        msg = f"the folded GNN of {task['id']} differs from PyTorch by {error:.1e}"
        raise ValueError(msg)

    def encode(array: np.ndarray) -> str:
        return base64.b64encode(np.ascontiguousarray(array, dtype="<f4").tobytes()).decode()

    return {
        "steps": [
            {
                "shape": list(step["w"].shape),
                "w": encode(step["w"]),
                "b": encode(step["b"]),
                "relu": step["relu"],
                "gather": step.get("gather", False),
                "sum": step.get("sum", False),
            }
            for step in steps
        ],
        "check": error,
    }


def task_texts(task: dict[str, Any]) -> dict[str, Any]:
    """Return what the demos say of a task: its texts, and its pattern's drawing (``motif``)."""
    keys = ("id", "label", "question", "meaning", "yes", "no", "dataset", "truthLabel")
    drawing = depict({"smiles": task["motif"]})
    motif = {"atoms": drawing["atoms"], "bonds": drawing["bonds"]}
    return {key: task[key] for key in keys} | {"motif": motif}


def molecule_texts(molecule: dict[str, Any]) -> dict[str, Any]:
    """Return what the explorer says of a molecule: its names and texts, and its first question."""
    keys = ("id", "name", "short", "blurb", "task", "takeaway")
    return {key: molecule[key] for key in keys if key in molecule}


def task_entry(task: dict[str, Any], results: Path, gnn: dict[str, Any] | None) -> dict[str, Any]:
    """Return what the demos show of a task: its texts, its GNN's score, its GNN and baseline."""
    scores = json.loads((results / "models" / f"{task['id']}.json").read_text())
    entry = task_texts(task) | {
        "f1": round(scores["test"]["f1"], 3),
        "trained": scores["sizes"]["train"],
    }
    if gnn is not None:
        # a masked atom's features, the same in every explanation of the task
        explained = sorted(results.glob(f"*_{task['id']}.json"))
        if not explained:
            msg = f"no explanation for {task['id']} in {results}: run explain.py first"
            raise FileNotFoundError(msg)
        baseline = json.loads(explained[0].read_text())["baseline"]
        entry |= {"gnn": gnn, "baseline": [round(value, 5) for value in baseline]}
    return entry


def strongest(values: list[list[Any]], n_atoms: int) -> list[list[Any]]:
    """Return the interactions the explorer shows, of the values sorted by size, the largest first.

    Every atom alone, the strongest groups overall, and the strongest groups of every atom.
    """
    groups = [(s, v) for s, v in values if len(s) > 1]
    keep = {tuple(s) for s, _ in groups[:OVERALL]}
    for atom in range(n_atoms):
        keep |= {tuple(s) for s, _ in [g for g in groups if atom in g[0]][:PER_ATOM]}
    return [[mask(s), round(v, 4)] for s, v in values if len(s) == 1 or tuple(s) in keep]


def compact(result: dict[str, Any]) -> dict[str, Any]:
    """Return what the explorer shows of an explanation, the coalitions as bitmasks."""
    n = result["n_atoms"]
    if n > 31:
        msg = f"{result['molecule']} has {n} atoms, more than the demos' 31"
        raise ValueError(msg)
    return {
        "logOdds": round(result["log_odds"], 4),
        "p": round(float(1 / (1 + np.exp(-result["log_odds"]))), 4),
        "empty": round(result["empty"], 4),
        "budget": result["budget"],
        "nAtoms": n,
        "truth": result["truth"],
        "kSII": {order: strongest(values, n) for order, values in result["k_sii"].items()},
        "moebius": strongest(result["moebius"], n),
        "moebiusCount": len(result["moebius"]),
    }


def write(demo: str, data: dict[str, Any]) -> None:
    """Write a demo's data, ``../../graphshapiq-<demo>/assets/data.js``."""
    text = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    path = DEMOS / f"graphshapiq-{demo}" / "assets" / "data.js"
    path.write_text(
        f"// The GraphSHAP-IQ {demo} demo's data (see demo.js), written by "
        "../../graphshapiq-common/compute/pack.py.\n"
        f"window.SHAPIQ_GRAPHSHAPIQ_{demo.upper()} = {text};\n",
        encoding="utf-8",
    )
    print(f"{demo}: {len(text) / 1024:.0f} KiB")


def read(demo: str) -> dict[str, Any]:
    """Return a demo's data as :func:`write` wrote it."""
    path = DEMOS / f"graphshapiq-{demo}" / "assets" / "data.js"
    text = path.read_text(encoding="utf-8")
    return json.loads(text[text.index(" = ") + 3 :].rstrip().removesuffix(";"))


def refresh_texts(config: dict[str, Any], demos: list[str]) -> None:
    """Write the texts of config.toml into the data of the explorer as it is.

    A task's and a molecule's entry keeps the order of its keys: the texts, then what was computed.
    """
    tasks = {task["id"]: task for task in config["tasks"]}
    molecules = {molecule["id"]: molecule for molecule in config["molecules"]}

    def refreshed(entry: dict[str, Any], texts: dict[str, Any], old: set[str]) -> dict[str, Any]:
        return texts | {key: value for key, value in entry.items() if key not in old | set(texts)}

    task_keys = {"label", "question", "meaning", "yes", "no", "dataset", "truthLabel", "motif"}
    molecule_keys = {"name", "short", "blurb", "task", "takeaway"}
    if "explorer" not in demos:  # the figure of receptive fields has no texts of config.toml
        return
    data = read("explorer")
    data["tasks"] = [
        refreshed(entry, task_texts(tasks[entry["id"]]), task_keys) for entry in data["tasks"]
    ]
    data["molecules"] = [
        refreshed(entry, molecule_texts(molecules[entry["id"]]), molecule_keys)
        for entry in data["molecules"]
    ]
    write("explorer", data)


def explanation(results: Path, molecule: str, task: str) -> dict[str, Any]:
    """Return a molecule's explanation for a task, by explain.py."""
    path = results / f"{molecule}_{task}.json"
    if not path.exists():
        msg = f"{path} is missing: run explain.py first"
        raise FileNotFoundError(msg)
    return json.loads(path.read_text())


def pack_explorer(config: dict[str, Any], results: Path) -> None:
    """Write the explorer's data: GNNs, molecules, and every molecule's explanations."""
    smiles = [m["smiles"] for m in config["molecules"]]
    tasks = [task_entry(task, results, pack_gnn(task, smiles, results)) for task in config["tasks"]]
    molecules = []
    explanations = {}
    for molecule in config["molecules"]:
        molecules.append(molecule_texts(molecule) | depict(molecule))
        for task in config["tasks"]:
            key = f"{molecule['id']}/{task['id']}"
            explanations[key] = compact(explanation(results, molecule["id"], task["id"]))
    data = {
        "symbols": SYMBOLS,  # the elements of the GNNs' one-hot features, in their order
        "tasks": tasks,
        "molecules": molecules,
        "explanations": explanations,
    }
    write("explorer", data)


def pack_exact(config: dict[str, Any]) -> None:
    """Write the exact demo's data: molecules and GraphSHAP-IQ's maskings for 1 to 4 layers."""
    molecules = []
    for molecule in config["exact_demo"]:
        mol = Chem.MolFromSmiles(molecule["smiles"])
        edges = np.array([[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()]).T
        n = mol.GetNumAtoms()
        budgets = {}
        for hops in HOPS:
            graph = GraphModel(edge_index=edges, n_hops=hops, n_nodes=n)
            with warnings.catch_warnings():  # the warning of a large budget: here it is the point
                warnings.simplefilter("ignore")
                # GraphSHAP-IQ plans its maskings without evaluating the game
                budgets[hops] = GraphSHAPIQ(Game(n), graph).budget
        keys = ("id", "name", "blurb")
        molecules.append(
            {key: molecule[key] for key in keys} | depict(molecule) | {"n": n, "budgets": budgets}
        )
        print(f"  {molecule['id']}: {n} atoms, 2^n = {2**n:,}, maskings {budgets}")
    write("exact", {"molecules": molecules})


def main() -> None:
    """Write the data of the demos named on the command line (all by default)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    demos = ["explorer", "exact"]
    parser.add_argument("--demos", nargs="+", choices=demos, default=demos)
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument(
        "--texts",
        action="store_true",
        help="only write config.toml's texts into the explorer's data as it is",
    )
    args = parser.parse_args()
    config = tomllib.loads((HERE / "config.toml").read_text(encoding="utf-8"))
    if args.texts:
        refresh_texts(config, args.demos)
        return
    if "explorer" in args.demos:
        pack_explorer(config, args.results)
    if "exact" in args.demos:
        pack_exact(config)


if __name__ == "__main__":
    main()
