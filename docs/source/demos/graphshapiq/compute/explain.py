"""Explain the GraphSHAP-IQ demo's molecules with GraphSHAP-IQ: one JSON file per molecule and task.

The players are the molecule's heavy atoms. The value of a coalition is the GNN's log-odds of the
task (its logit of yes minus its logit of no) for the molecule with only the coalition's atoms
known: the other atoms keep their bonds, but their element is masked: replaced by the average atom
of the GNN's training molecules (the mean of their atoms' one-hot elements, mostly carbon), as the
GraphBaselineImputer does with background data. The GNN's three layers make every atom's embedding
depend only on the atoms within three bonds, its receptive field; the sum over the atoms and the
linear readout keep the log-odds local, so GraphSHAP-IQ computes the Möbius interactions of all
orders exactly, from the maskings within the receptive fields only. From them, it converts the
k-SII of every order the demo shows (order 1: the Shapley values).

The exactness is checked: the Möbius interactions must reproduce the GNN's log-odds on random
maskings of the molecule, to floating point precision.

Usage::

    python explain.py                                     # every molecule and task of config.toml
    python explain.py --molecules serotonin --tasks indole

The explanations go to ``results/<molecule>_<task>.json``; explanations that exist are not
computed again. Each takes less than 20 seconds on four CPU cores (after loading the datasets).
"""

from __future__ import annotations

import argparse
import json
import time
import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np
import torch
from molecules import GNN, LogOdds, load_bxaic, load_mutagenicity, to_graph
from rdkit import Chem

from shapiq.graph import (
    GraphBaselineImputer,
    GraphSHAPIQ,
    convert_graph,
    get_graph_predict_function,
)

if TYPE_CHECKING:
    from torch_geometric.data import Data

HERE = Path(__file__).resolve().parent
N_CHECK = 2_000  # the random maskings the Möbius interactions are checked on


def truth(mol: Chem.Mol, pattern: str) -> list[int]:
    """Return the atoms of every match of the task's pattern (its ground truth), sorted."""
    query = Chem.MolFromSmarts(pattern)
    return sorted({atom for match in mol.GetSubstructMatches(query) for atom in match})


def as_networkx(graph: Data) -> nx.Graph:
    """Return a molecule's graph as the GNN sees it: its atoms' elements and its bonds."""
    elements = graph.x.argmax(dim=1).tolist()
    nx_graph = nx.Graph(graph.edge_index.T.tolist())
    nx_graph.add_nodes_from(range(len(elements)))
    nx.set_node_attributes(nx_graph, dict(enumerate(elements)), "element")
    return nx_graph


class Dataset:
    """A task's dataset, to look up whether the GNN saw a molecule in training."""

    def __init__(self, task: dict[str, Any], data: Path) -> None:
        """Load the dataset and hash every molecule's graph."""
        if task["dataset"] == "B-XAIC":
            self.graphs, self.splits = load_bxaic(task["id"], data)
        else:
            self.graphs, self.splits = load_mutagenicity(data)
        train = [
            graph.x
            for graph, split in zip(self.graphs, self.splits, strict=True)
            if split == "train"
        ]
        self.atoms = torch.cat(train).numpy()  # the background data of the masked atoms
        self.hashes: dict[str, list[int]] = {}
        for k, graph in enumerate(self.graphs):
            self.hashes.setdefault(self.hash(as_networkx(graph)), []).append(k)

    @staticmethod
    def hash(graph: nx.Graph) -> str:
        """Return the Weisfeiler-Lehman hash of a graph with its atoms' elements."""
        return nx.weisfeiler_lehman_graph_hash(graph, node_attr="element", iterations=4)

    def find(self, graph: Data) -> dict[str, Any] | None:
        """Return the split and the label of a molecule the GNN cannot tell from this one."""
        query = as_networkx(graph)
        for k in self.hashes.get(self.hash(query), []):
            match = nx.is_isomorphic(
                query, as_networkx(self.graphs[k]), node_match=lambda a, b: a == b
            )
            if match:
                return {"split": self.splits[k], "label": int(self.graphs[k].y)}
        return None


def explain(
    molecule: dict[str, Any], task: dict[str, Any], dataset: Dataset, orders: int, results: Path
) -> None:
    """Explain a molecule for a task's GNN, and write the explanation to ``results``."""
    gnn = GNN()
    gnn.load_state_dict(torch.load(results / "models" / f"{task['id']}.pt"))
    model = LogOdds(gnn).eval()
    mol = Chem.MolFromSmiles(molecule["smiles"])
    graph = convert_graph(to_graph(mol), n_hops=model.num_layers)
    predict = get_graph_predict_function(model, graph, batch_size=512)
    # the atoms of the training molecules as background data: their mean is the masked atoms'
    game = GraphBaselineImputer(predict, dataset.atoms, x=graph, normalize=False)
    n = game.n_players

    start = time.time()
    graphshapiq = GraphSHAPIQ(game, graph)
    moebius = graphshapiq.moebius  # evaluates the GNN on every masking within receptive fields
    k_sii = {k: graphshapiq("k-SII", k) for k in range(1, min(orders, n) + 1)}
    seconds = time.time() - start

    # the Möbius interactions add up to the GNN's log-odds on every masking
    coalitions = np.random.default_rng(0).random((N_CHECK, n)) < 0.5
    members = np.zeros((len(moebius.dict_values), n))
    values = np.zeros(len(moebius.dict_values))
    for row, (players, value) in enumerate(moebius.dict_values.items()):
        members[row, list(players)] = 1
        values[row] = value
    sums = np.concatenate(
        [((~chunk) @ members.T == 0) @ values for chunk in np.array_split(coalitions, 20)]
    )  # a term counts for a coalition if none of its atoms is masked
    error = float(np.max(np.abs(sums - game(coalitions))))
    log_odds = float(game(np.ones((1, n), dtype=bool))[0])

    def listed(values: Any) -> list[list[Any]]:  # noqa: ANN401
        """The non-empty coalitions and their values, the largest first."""
        items = [(list(s), float(v)) for s, v in values.dict_values.items() if s and v != 0]
        return [[s, v] for s, v in sorted(items, key=lambda item: -abs(item[1]))]

    result = {
        "molecule": molecule["id"],
        "task": task["id"],
        "n_atoms": n,
        "n_hops": graph.n_hops,
        "baseline": [float(v) for v in np.ravel(game.baseline_values)],  # a masked atom's features
        "budget": graphshapiq.budget,
        "max_receptive_field": graph.max_receptive_field_size,
        "seconds": round(seconds, 1),
        "log_odds": log_odds,
        "empty": float(moebius[()]),  # the log-odds with every element masked
        "check_error": error,
        "truth": truth(mol, task["pattern"]),
        "training": dataset.find(to_graph(mol)),  # the molecule in the GNN's dataset
        "moebius": listed(moebius),
        "k_sii": {str(k): listed(values) for k, values in k_sii.items()},
    }
    (results / f"{molecule['id']}_{task['id']}.json").write_text(json.dumps(result))
    top = result["moebius"][0]
    print(
        f"{molecule['id']}, {task['id']}: {n} atoms, p = {1 / (1 + np.exp(-log_odds)):.3f}, "
        f"{graphshapiq.budget:,} of 2^{n} maskings, {len(result['moebius']):,} Möbius "
        f"interactions (largest {top[1]:+.2f} of {top[0]}), check {error:.1e}, {seconds:.0f} s",
        flush=True,
    )


def main() -> None:
    """Explain the molecules and tasks of config.toml that are not explained yet."""
    config = tomllib.loads((HERE / "config.toml").read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--molecules", nargs="+", default=[m["id"] for m in config["molecules"]])
    parser.add_argument("--tasks", nargs="+", default=[t["id"] for t in config["tasks"]])
    parser.add_argument("--results", type=Path, default=HERE / "results")
    args = parser.parse_args()
    torch.set_num_threads(4)
    for task in (t for t in config["tasks"] if t["id"] in args.tasks):
        todo = [
            molecule
            for molecule in config["molecules"]
            if molecule["id"] in args.molecules
            and not (args.results / f"{molecule['id']}_{task['id']}.json").exists()
        ]
        if todo:
            dataset = Dataset(task, args.results / "data")
            for molecule in todo:
                explain(molecule, task, dataset, config["orders"], args.results)


if __name__ == "__main__":
    main()
