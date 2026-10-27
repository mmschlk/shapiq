"""Write the GraphSHAP-IQ demo's data, ``../assets/data.js``.

Reads the GNNs of train.py and the explanations of explain.py (in ``results/``) and the texts of
config.toml, draws the molecules (2D coordinates of RDKit), keeps what the demo shows, rounded,
and writes them as a script that sets ``window.SHAPIQ_GRAPHSHAPIQ`` (a script loads from disk too,
where the browser would not fetch a JSON file). A molecule and task without an explanation is left
out; the demo disables it.

The demo runs the GNNs themselves, to predict for the maskings a reader picks: every GNN is written
as a chain of affine maps and ReLUs, its batch normalizations folded into its linear layers, in
float32 (base64), and checked against PyTorch on random maskings. A coalition of atoms is written
as a bitmask of the atoms' indices (the demo's molecules have at most 31 atoms). Of the
interactions of every order, the demo gets the strongest ones overall and the strongest ones of
every atom, which is all it draws and lists.

Usage::

    python pack.py
"""

from __future__ import annotations

import argparse
import base64
import json
import tomllib
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
OVERALL = 24  # the strongest interactions of an order the demo gets
PER_ATOM = 8  # and the strongest interactions of an order with every atom
HOPS = range(1, 5)  # the GNN depths whose numbers of maskings the demo compares


def mask(atoms: list[int]) -> int:
    """Return a coalition of atoms as a bitmask."""
    return sum(1 << atom for atom in atoms)


def depict(molecule: dict[str, Any]) -> dict[str, Any]:
    """Return a molecule's drawing: atoms, bonds (Kekulé) and rings, bonds of length 1."""
    mol = Chem.MolFromSmiles(molecule["smiles"])
    rdDepictor.SetPreferCoordGen(True)  # noqa: FBT003  (RDKit's API)
    rdDepictor.Compute2DCoords(mol)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    xy = mol.GetConformer().GetPositions()[:, :2]
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    xy = (xy - xy.mean(axis=0)) / np.median([np.linalg.norm(xy[i] - xy[j]) for i, j in bonds])
    turn = np.deg2rad(molecule.get("rotate", 0))
    xy = xy @ np.array([[np.cos(turn), np.sin(turn)], [-np.sin(turn), np.cos(turn)]])
    xy *= [-1 if molecule.get("flip", False) else 1, -1]  # SVG's y points down
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
        "budgets": budgets(mol),
    }


def budgets(mol: Chem.Mol) -> dict[int, int]:
    """Return the number of maskings GraphSHAP-IQ evaluates for a GNN of 1 to 4 layers."""
    edges = np.array([[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()]).T
    counts = {}
    for hops in HOPS:
        graph = GraphModel(edge_index=edges, n_hops=hops, n_nodes=mol.GetNumAtoms())
        # GraphSHAP-IQ plans its maskings without evaluating the game
        counts[hops] = GraphSHAPIQ(Game(graph.n_nodes), graph).budget
    return counts


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
    """Return the log-odds of a folded GNN, in float32, as the demo computes them."""
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


def pack_gnn(
    task: dict[str, Any], molecules: list[dict[str, Any]], results: Path
) -> dict[str, Any]:
    """Return a task's GNN for the demo, checked against PyTorch on random maskings."""
    gnn = GNN()
    gnn.load_state_dict(torch.load(results / "models" / f"{task['id']}.pt"))
    model = LogOdds(gnn.eval())
    steps = fold(gnn)
    rng = np.random.default_rng(0)
    error = 0.0
    for molecule in molecules:
        graph = to_graph(Chem.MolFromSmiles(molecule["smiles"]))
        edges = graph.edge_index.numpy()
        for _ in range(20):
            x = graph.x.numpy() * (rng.random((len(graph.x), 1)) < 0.6)
            with torch.no_grad():
                torch_value = float(
                    model(
                        torch.as_tensor(x), graph.edge_index, torch.zeros(len(x), dtype=torch.long)
                    )
                )
            error = max(error, abs(run(steps, x, edges) - torch_value))
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


def strongest(values: list[list[Any]], n_atoms: int) -> list[list[Any]]:
    """Return the interactions the demo shows, of the values sorted by size, the largest first.

    Every atom alone, the strongest groups overall, and the strongest groups of every atom.
    """
    groups = [(s, v) for s, v in values if len(s) > 1]
    keep = {tuple(s) for s, _ in groups[:OVERALL]}
    for atom in range(n_atoms):
        keep |= {tuple(s) for s, _ in [g for g in groups if atom in g[0]][:PER_ATOM]}
    return [[mask(s), round(v, 4)] for s, v in values if len(s) == 1 or tuple(s) in keep]


def compact(result: dict[str, Any]) -> dict[str, Any]:
    """Return what the demo shows of an explanation, the coalitions as bitmasks."""
    n = result["n_atoms"]
    if n > 31:
        msg = f"{result['molecule']} has {n} atoms, more than the demo's 31"
        raise ValueError(msg)
    return {
        "logOdds": round(result["log_odds"], 4),
        "empty": round(result["empty"], 4),
        "budget": result["budget"],
        "truth": result["truth"],
        "training": result.get("training"),
        "kSII": {order: strongest(values, n) for order, values in result["k_sii"].items()},
        "moebius": strongest(result["moebius"], n),
        "moebiusCount": len(result["moebius"]),
    }


def main() -> None:
    """Write ``../assets/data.js`` from the results and config.toml."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results", type=Path, default=HERE / "results")
    parser.add_argument("--assets", type=Path, default=HERE.parent / "assets")
    args = parser.parse_args()
    config = tomllib.loads((HERE / "config.toml").read_text(encoding="utf-8"))

    tasks = []
    for task in config["tasks"]:
        scores = json.loads((args.results / "models" / f"{task['id']}.json").read_text())
        # a masked atom's features, the same in every explanation of the task
        explained = sorted(args.results.glob(f"*_{task['id']}.json"))
        baseline = json.loads(explained[0].read_text())["baseline"] if explained else None
        keys = (
            "id",
            "label",
            "question",
            "yes",
            "no",
            "probability",
            "dataset",
            "truthLabel",
            "truthYes",
        )
        tasks.append(
            {key: task[key] for key in (*keys, "truthNo") if key in task}
            | {"f1": round(scores["test"]["f1"], 3), "trained": scores["sizes"]["train"]}
            | {"gnn": pack_gnn(task, config["molecules"], args.results)}
            | {"baseline": [round(value, 5) for value in baseline] if baseline else None}
        )
    molecules, explanations = [], {}
    for molecule in config["molecules"]:
        keys = ("id", "name", "short", "blurb", "task")
        molecules.append({key: molecule[key] for key in keys if key in molecule} | depict(molecule))
        for task in config["tasks"]:
            path = args.results / f"{molecule['id']}_{task['id']}.json"
            if not path.exists():
                print(f"{molecule['id']}, {task['id']}: not explained, left out")
                continue
            explanations[f"{molecule['id']}/{task['id']}"] = compact(json.loads(path.read_text()))
    gnn = GNN()
    data = {
        "model": {"layers": gnn.num_layers, "hidden": gnn.gin.hidden_channels},
        "symbols": SYMBOLS,  # the elements of the GNNs' one-hot features, in their order
        "orders": config["orders"],
        "tasks": tasks,
        "molecules": molecules,
        "explanations": explanations,
    }
    text = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
    (args.assets / "data.js").write_text(
        "// The GraphSHAP-IQ demo's GNNs, molecules and explanations (see demo.js), written by "
        "../compute/pack.py.\n"
        f"window.SHAPIQ_GRAPHSHAPIQ = {text};\n",
        encoding="utf-8",
    )
    print(f"{len(explanations)} explanations, {len(text) / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
