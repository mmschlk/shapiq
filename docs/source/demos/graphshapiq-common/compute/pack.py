"""Write the data of the four GraphSHAP-IQ demos: ``../../graphshapiq-<demo>/assets/data.js``.

Reads the GNNs of train.py and the explanations of explain.py (in ``results/``) and the texts of
config.toml, draws the molecules (2D coordinates of RDKit), keeps what the demos show, rounded, and
writes each demo's data as a script that sets a global (a script loads from disk too, where the
browser would not fetch a JSON file):

- ``explorer`` (``window.SHAPIQ_GRAPHSHAPIQ_EXPLORER``): the three GNNs, the molecules, and the
  explanation of every molecule for every question: the k-SII of orders 1 to 4 and the Möbius
  interactions, the strongest of every order overall and of every atom;
- ``lab`` (``window.SHAPIQ_GRAPHSHAPIQ_LAB``): the GNNs that run in the reader's browser, where the
  demo computes GraphSHAP-IQ itself, for the molecules the reader builds;
- ``orders`` (``window.SHAPIQ_GRAPHSHAPIQ_ORDERS``): a few molecules explained at every order with
  the Faithful Shapley Interactions (FSII), and how faithful every order is (R^2 beyond size, as in
  the orders demo, ``../../orders/compute/faithfulness.py``);
- ``exact`` (``window.SHAPIQ_GRAPHSHAPIQ_EXACT``): molecules of growing size, and the number of
  maskings GraphSHAP-IQ evaluates for GNNs of 1 to 4 layers (no GNN is evaluated).

The demos run the GNNs themselves, for the maskings a reader picks: every GNN is written as a chain
of affine maps and ReLUs, its batch normalizations folded into its linear layers, in float32
(base64), and checked against PyTorch on random maskings. A coalition of atoms is written as a
bitmask of the atoms' indices (the molecules have at most 31 atoms).

Usage::

    python pack.py                    # all four demos
    python pack.py --demos lab exact  # some of them
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
from scipy.special import comb

from shapiq import Game, InteractionValues
from shapiq.game_theory.moebius_converter import MoebiusConverter
from shapiq.graph import GraphModel, GraphSHAPIQ

HERE = Path(__file__).resolve().parent
DEMOS = HERE.parent.parent  # docs/source/demos
OVERALL = 24  # explorer: the strongest interactions of an order it gets
PER_ATOM = 8  # and the strongest interactions of an order with every atom
GROUPS = 60  # orders: the strongest groups of 2+ atoms of every order (and all atoms alone)
CLOSE = 0.999  # orders: the R^2 from which the next order is the full decomposition
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


def task_entry(task: dict[str, Any], results: Path, gnn: dict[str, Any] | None) -> dict[str, Any]:
    """Return what the demos show of a task: its texts, its GNN's score, its GNN and baseline."""
    scores = json.loads((results / "models" / f"{task['id']}.json").read_text())
    keys = ("id", "label", "question", "yes", "no", "dataset", "truthLabel")
    entry = {key: task[key] for key in keys} | {
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


def zeta(values: np.ndarray, n: int) -> np.ndarray:
    """Return the sum of ``values`` over the subsets of every coalition (codes 0 to 2^n - 1)."""
    out = values.copy()
    for player in range(n):
        out = out.reshape(-1, 2, 1 << player)
        out[:, 1, :] += out[:, 0, :]
        out = out.reshape(-1)
    return out


def at_every_order(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Return a molecule's FSII of orders 1, 2, ..., and its full decomposition, with their R^2.

    The R^2 is the orders demo's R^2 beyond size, exactly, over all coalitions of the Shapley
    distribution: how much of what the number of atoms in a coalition does not predict the
    explanation reproduces. It does not fall with the order, and reaches 1 with the full
    decomposition (the Möbius interactions, FSII of the highest order).
    """
    n = result["n_atoms"]
    size = 1 << n
    moebius = np.zeros(size)
    moebius[0] = result["empty"]
    for players, value in result["moebius"]:
        moebius[mask(players)] = value
    values = zeta(moebius, n)  # the GNN's log-odds for every coalition
    sizes = np.array([code.bit_count() for code in range(size)])
    weights = np.zeros(n + 1)
    weights[1:n] = 1 / (np.arange(1, n) * (n - np.arange(1, n)))
    weights = weights / weights.sum() / comb(n, np.arange(n + 1))
    weight = weights[sizes]  # the empty and the full coalition have no weight
    means = np.array([values[sizes == s].mean() for s in range(n + 1)])
    spread = np.sum(weight * (values - means[sizes]) ** 2)

    lookup = {(): 0}
    lookup |= {tuple(players): k + 1 for k, (players, _) in enumerate(result["moebius"])}
    coefficients = np.array([result["empty"]] + [value for _, value in result["moebius"]])
    converter = MoebiusConverter(
        InteractionValues(
            values=coefficients,
            index="Moebius",
            max_order=n,
            min_order=0,
            n_players=n,
            interaction_lookup=lookup,
            baseline_value=result["empty"],
        )
    )

    def entry(order: int | str, explanation: dict[tuple[int, ...], float]) -> dict[str, Any]:
        dense = np.zeros(size)
        dense[0] = result["empty"]
        for players, value in explanation.items():
            dense[mask(players)] = value
        r2 = 1 - np.sum(weight * (values - zeta(dense, n)) ** 2) / spread
        alone = [[mask(s), round(v, 4)] for s, v in explanation.items() if len(s) == 1]
        groups = sorted(
            ([mask(s), round(v, 4)] for s, v in explanation.items() if len(s) > 1),
            key=lambda group: -abs(group[1]),
        )
        return {"k": order, "r2": round(float(r2), 4), "groups": alone + groups[:GROUPS]}

    orders = []
    for order in range(1, n):
        fsii = converter.compute("FSII", order=order)
        explanation = {
            tuple(players): float(fsii.values[k])
            for players, k in fsii.interaction_lookup.items()
            if players and fsii.values[k] != 0
        }
        orders.append(entry(order, explanation))
        if orders[-1]["r2"] >= CLOSE:
            break
    full = {tuple(players): value for players, value in result["moebius"]}
    orders.append(entry("all", full))
    return orders


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
        keys = ("id", "name", "short", "blurb", "task", "takeaway")
        molecules.append({key: molecule[key] for key in keys if key in molecule} | depict(molecule))
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


def pack_lab(config: dict[str, Any], results: Path) -> None:
    """Write the lab's data: the GNNs that run in the reader's browser."""
    smiles = [m["smiles"] for m in config["molecules"]]
    by_id = {task["id"]: task for task in config["tasks"]}
    tasks = [
        task_entry(by_id[tid], results, pack_gnn(by_id[tid], smiles, results))
        for tid in config["lab_demo"]["tasks"]
    ]
    write("lab", {"symbols": SYMBOLS, "tasks": tasks})


def pack_orders(config: dict[str, Any], results: Path) -> None:
    """Write the orders demo's data: molecules explained at every order, and how faithfully."""
    by_id = {m["id"]: m for m in config["molecules"]}
    tasks = {task["id"]: task_entry(task, results, None) for task in config["tasks"]}
    molecules = []
    for mid in config["orders_demo"]["molecules"]:
        molecule = by_id[mid]
        result = explanation(results, mid, molecule["task"])
        if result["n_atoms"] > 16:
            msg = f"{mid} has {result['n_atoms']} atoms, more than the orders demo's 16"
            raise ValueError(msg)
        orders = at_every_order(result)
        keys = ("id", "name", "short", "blurb", "task")
        molecules.append(
            {key: molecule[key] for key in keys if key in molecule}
            | depict(molecule)
            | {
                "logOdds": round(result["log_odds"], 4),
                "empty": round(result["empty"], 4),
                "truth": result["truth"],
                "orders": orders,
            }
        )
        print(f"  {mid}: R^2 " + ", ".join(f"{o['k']}: {o['r2']:.3f}" for o in orders))
    write("orders", {"tasks": tasks, "molecules": molecules})


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
    """Write the data of the demos named on the command line (all four by default)."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    demos = ["explorer", "lab", "orders", "exact"]
    parser.add_argument("--demos", nargs="+", choices=demos, default=demos)
    parser.add_argument("--results", type=Path, default=HERE / "results")
    args = parser.parse_args()
    config = tomllib.loads((HERE / "config.toml").read_text(encoding="utf-8"))
    if "explorer" in args.demos:
        pack_explorer(config, args.results)
    if "lab" in args.demos:
        pack_lab(config, args.results)
    if "orders" in args.demos:
        pack_orders(config, args.results)
    if "exact" in args.demos:
        pack_exact(config)


if __name__ == "__main__":
    main()
