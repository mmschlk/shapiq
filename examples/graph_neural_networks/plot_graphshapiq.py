"""
Exact Shapley Interactions of a Graph Neural Network
====================================================

A graph neural network (GNN) predicts a property of a whole graph, such as whether a molecule is
mutagenic, from the features of its nodes. Its players are the nodes: absent nodes keep their
edges, but their features are replaced by a baseline (here zeros).

A GNN with :math:`L` message passing layers computes the embedding of a node from the nodes within
:math:`L` hops only, the node's *receptive field*. If the embeddings are pooled and read out
linearly, every interaction of nodes that do not share a receptive field is zero. GraphSHAP-IQ
:cite:p:`Muschalik.2025` therefore evaluates the GNN only on subsets of receptive fields, and still
computes Shapley values and interactions of any order exactly.

This example explains a small molecule with :class:`~shapiq.graph.GraphExplainer`, checks the
values against all :math:`2^n` coalitions, and approximates them with fewer evaluations.
"""

# sphinx_gallery_thumbnail_number = 1

from __future__ import annotations

import time
from itertools import pairwise

import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_add_pool

import shapiq
from shapiq.graph import GraphExplainer, GraphGame, GraphSHAPIQ

# %%
# A Molecule and a GNN
# --------------------
# The molecule is nitrotoluene without its hydrogens: a carbon ring with a methyl group and a
# nitro group (NO\ :sub:`2`). The node features are the one-hot atom types. The GNN has two
# graph convolutions, a sum pooling, and a linear readout; its weights are random, standing in for
# a trained mutagenicity classifier.

atoms = ["C", "C", "C", "C", "C", "C", "C", "N", "O", "O"]
bonds = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 6), (3, 7), (7, 8), (7, 9)]
atom_types = ["C", "N", "O"]
x = torch.tensor([[float(atom == kind) for kind in atom_types] for atom in atoms])
edge_index = torch.tensor(bonds + [(j, i) for i, j in bonds]).T  # both directions
molecule = Data(x=x, edge_index=edge_index)


class GCN(torch.nn.Module):
    """Graph convolutions, a sum over the nodes, and a linear readout of two class logits."""

    def __init__(self, num_layers: int = 2, hidden: int = 16) -> None:
        super().__init__()
        self.num_layers = num_layers  # GraphSHAP-IQ reads the number of hops from it
        sizes = [len(atom_types)] + [hidden] * num_layers
        self.convs = torch.nn.ModuleList(GCNConv(a, b) for a, b in pairwise(sizes))
        self.readout = torch.nn.Linear(hidden, 2)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        return self.readout(global_add_pool(x, batch))


torch.manual_seed(0)
model = GCN()

# %%
# Explaining the Molecule
# -----------------------
# :class:`~shapiq.graph.GraphExplainer` explains the logit of class ``1`` by default. The
# pairwise k-SII interactions show which atoms the GNN considers together; the SI graph draws
# them on the bonds of the molecule.

explainer = GraphExplainer(model, index="k-SII", max_order=2)
interactions = explainer.explain(molecule)
print(interactions)

names = [f"{atom}{node}" for node, atom in enumerate(atoms)]
shapiq.si_graph_plot(interactions, graph=bonds, feature_names=names, show=True)

# %%
# The Receptive Fields Save Evaluations
# -------------------------------------
# With two layers, the receptive field of an atom are the atoms within two bonds. GraphSHAP-IQ
# evaluates the GNN on the subsets of the largest receptive fields only, instead of on all
# :math:`2^{10}` coalitions. The values are the same as those of brute force, which evaluates all
# of them with :class:`~shapiq.game_theory.ExactComputer`.

game = GraphGame(model, molecule, normalize=False)
graphshapiq = GraphSHAPIQ(game, game.edge_index, n_hops=game.n_hops)
print(f"largest receptive field: {graphshapiq.max_receptive_field_size} atoms")
print(f"evaluations: {graphshapiq.budget} instead of {2**game.n_players}")

start = time.perf_counter()
brute_force = shapiq.ExactComputer(game=game, n_players=game.n_players)("k-SII", 2)
print(f"brute force: {time.perf_counter() - start:.2f} s")
difference = max(
    abs(interactions[key] - brute_force[key]) for key in brute_force.interaction_lookup
)
print(f"largest difference to brute force: {difference:.1e}")

# %%
# Approximating with Smaller Interactions
# ---------------------------------------
# Large receptive fields make the exact computation expensive. With ``max_interaction_size``,
# GraphSHAP-IQ evaluates only the coalitions of up to that many atoms, plus, for the efficiency
# routine, the larger receptive fields and their intersections. The Shapley values still add up
# to the prediction, and are exact again for one atom less than the largest receptive field.

exact = GraphExplainer(model).explain(molecule)
for size in range(1, graphshapiq.max_receptive_field_size):
    approximation = GraphExplainer(model, max_interaction_size=size).explain(molecule)
    error = max(abs(approximation[(node,)] - exact[(node,)]) for node in range(len(atoms)))
    print(
        f"max_interaction_size={size}: {approximation.estimation_budget:4d} evaluations, "
        f"largest error {error:.4f}, estimated: {approximation.estimated}"
    )
