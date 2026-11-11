GraphSHAP-IQ: Which Atoms Act Together?
=======================================

.. rst-class:: shapiq-examples-lead

A graph neural network (GNN) reads a molecule and answers a yes/no question about it. The Shapley
values tell which atoms swayed it. The interactions tell which atoms only matter together. This
demo explains three GNNs for eight molecules by the Shapley interactions of their atoms, of any
order and exactly, as in GraphSHAP-IQ :cite:p:`Muschalik.2025`, computed with ``shapiq``.

.. shapiq-demo:: graphshapiq-explorer
   :fallback: images/hero_graphshapiq_explorer_demo.png
   :alt: The molecule 1,3-dinitrobenzene explained by a GNN for the question whether it is mutagenic: each nitrogen and its two oxygens are joined by red bands, the pairs of atoms that together raise the odds the most.

What you see
------------

**The questions.** Each question has its own GNN, trained to answer it. *Indole?* and *PAINS?*
come from B-XAIC :cite:p:`Proszewska.2025`, a benchmark of ChEMBL molecules whose labels follow
from substructures, so that every label has a ground truth: an indole (a benzene ring fused to a
five-membered ring with a nitrogen), or a pan-assay interference compound (PAINS)
:cite:p:`Baell.2010`, here a catechol, which fools many lab assays. *Mutagenic?* comes from the
Ames-test molecules of Kazius et al. :cite:p:`Kazius.2005`, in TUDataset's copy
:cite:p:`Morris.2020`, as in GraphSHAP-IQ's figures; its ground truth is the nitro group, a known
structural alert. Every GNN is B-XAIC's GIN :cite:p:`Xu.2019`: three message passing layers, a sum
over the atoms, and a linear readout. It sees an atom only as its element and a bond only as a
link, not as single, double or aromatic.

**The players** are the atoms. A coalition of atoms is what the GNN gets to know: the other atoms
keep their bonds, but their element is masked, replaced by the average atom of the GNN's training
molecules. The explanation splits the GNN's log-odds of *yes* into what every atom adds on its
own and what groups of atoms add together, their **interactions**. An interaction is red when the
atoms raise the odds of *yes* together, and blue when they lower them.

**The views.** *Prediction* shows the GNN's answer; click atoms to hide them and see what it
predicts without them, as the GNN runs in your browser. *Atoms alone* shows the Shapley value of
every atom as a ring around it. *Atoms together* adds the interactions, drawn as the SI-graphs of
the paper: a band joins two atoms, a shape joins more. − and + change the order: the k-SII
:cite:p:`Bordt.2023` with groups of up to 2, 3 or 4 atoms, and the Möbius interactions of all
orders, from which GraphSHAP-IQ computes the others. Point at an atom (or tap it) to see its
interactions; a thin outline marks the ground truth.

What it shows
-------------

- **A nitro group is a team.** For 1,3-dinitrobenzene, the Shapley values of the *Mutagenic?* GNN
  credit the two nitrogens (+1.86 each) and all but ignore the oxygens (+0.07). The interactions
  show why: an oxygen on its own lowers the odds (−0.70), and together with its nitrogen it raises
  them (+1.83), as GraphSHAP-IQ found for the mutagen of its Figure 1.
- **So is an indole.** For serotonin, the indole's nitrogen leads the Shapley values (+4.75), but on
  its own it lowers the odds (−1.27): all it adds, it adds in pairs with the carbons of its rings.
  For LSD, groups of nine atoms add and subtract more than 10 each at all orders, and cancel: the
  Möbius interactions are exact, but not simple, which is why lower orders summarize them.
- **Indigo fools the GNN.** Indole is named after the dye, but indigo is no indole: its
  five-membered rings are not aromatic. The GNN, which sees no bond orders, calls it an indole
  anyway (97%). Its nitrogens speak for it (+2.09 each), and its oxygens against it, most of all
  in pairs with the nitrogens (−3.18), but not loud enough.
- **No atom makes a catechol alone.** Dopamine's two neighbouring OH groups make it a PAINS. At
  order 2, every one of the catechol's eight atoms lowers the odds on its own, and every pair of
  them raises them.
- **Sometimes the atoms are enough.** For aspirin, the four oxygens speak against "mutagenic", each
  on its own, and the pairs are weak (at most 0.31): the Shapley values tell the story already.

How it was computed
-------------------

The graph game
^^^^^^^^^^^^^^

The value of a coalition of atoms is the GNN's output for the molecule with only those atoms known:
the log-odds of *yes*, its logit of yes minus its logit of no. A GNN with three layers passes
messages along bonds three times, so what it computes for an atom depends only on the atoms within
three bonds, its **receptive field**. As the GNN sums its atoms up and reads the sum out linearly,
atoms that are never together in a receptive field do not interact: all their interactions are zero
:cite:p:`Muschalik.2025`. GraphSHAP-IQ therefore evaluates the GNN only on the maskings within
receptive fields, computes their Möbius interactions, and converts them into any index, exactly
(:doc:`../graphshapiq-exact/index` shows how many maskings that saves). Each explanation took less
than 15 seconds on four CPU cores, and its Möbius interactions reproduce the GNN's output on random
maskings to :math:`10^{-5}`.

Explain it yourself
^^^^^^^^^^^^^^^^^^^

:class:`~shapiq.explainer.GraphExplainer` explains a GNN of ``torch_geometric`` with GraphSHAP-IQ.
Here, a GIN learns from TUDataset's copy of the Mutagenicity dataset :cite:p:`Morris.2020`, with
hydrogens, as in the paper's figures, and explains 1,3-dinitrobenzene (about a minute on a laptop):

.. code-block:: python

   import torch
   from rdkit import Chem
   from torch_geometric.data import Data
   from torch_geometric.datasets import TUDataset
   from torch_geometric.loader import DataLoader
   from torch_geometric.nn import GIN, global_add_pool

   from shapiq.explainer import GraphExplainer

   # Kazius et al.'s 4,337 molecules: their atoms (hydrogens too) one-hot, class 0 for "mutagen"
   dataset = TUDataset(root="data", name="Mutagenicity")
   ELEMENTS = ["C", "O", "Cl", "H", "N", "F", "Br", "S", "P", "I", "Na", "K", "Li", "Ca"]


   class GNN(torch.nn.Module):
       """Three GIN layers, a sum over the atoms, and a linear readout: the log-odds of "mutagen"."""

       def __init__(self) -> None:
           super().__init__()
           self.num_layers = 3  # GraphSHAP-IQ reads the radius of the receptive fields from it
           self.gin = GIN(len(ELEMENTS), 32, num_layers=3, norm="batch_norm")
           self.readout = torch.nn.Linear(32, 1)

       def forward(self, x, edge_index, batch):
           return self.readout(global_add_pool(self.gin(x, edge_index), batch))


   torch.manual_seed(0)
   model = GNN()
   optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
   for _ in range(20):
       for batch in DataLoader(dataset, batch_size=64, shuffle=True):
           optimizer.zero_grad()
           log_odds = model(batch.x, batch.edge_index, batch.batch).squeeze(1)
           mutagen = (batch.y == 0).float()
           torch.nn.functional.binary_cross_entropy_with_logits(log_odds, mutagen).backward()
           optimizer.step()
   model.eval()

   # 1,3-dinitrobenzene with its hydrogens, as a graph of the dataset's kind
   mol = Chem.AddHs(Chem.MolFromSmiles("c1cc(cc(c1)[N+](=O)[O-])[N+](=O)[O-]"))
   atoms = [f"{atom.GetSymbol()}{atom.GetIdx()}" for atom in mol.GetAtoms()]
   bonds = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()]
   elements = torch.tensor([ELEMENTS.index(atom.GetSymbol()) for atom in mol.GetAtoms()])
   graph = Data(
       x=torch.nn.functional.one_hot(elements, len(ELEMENTS)).float(),
       edge_index=torch.tensor(bonds + [(j, i) for i, j in bonds]).T,
   )

   # a masked atom becomes the average atom of the dataset; the explanation is exact
   explainer = GraphExplainer(model, data=dataset.x.numpy(), index="k-SII", max_order=2)
   interactions = explainer.explain(graph)
   print(f"{interactions.estimation_budget:,} of 2^{len(atoms)} maskings")
   pairs = {s: value for s, value in interactions.dict_values.items() if len(s) == 2}
   for (i, j), value in sorted(pairs.items(), key=lambda pair: -abs(pair[1]))[:6]:
       print(f"{atoms[i]} and {atoms[j]}: {value:+.2f}")

.. code-block:: text

   43,968 of 2^16 maskings
   O10 and O11: -1.72
   O7 and O8: -1.72
   N6 and O7: +1.32
   N6 and O8: +1.32
   N9 and O11: +1.32
   N9 and O10: +1.32

Of the 16 atoms of 1,3-dinitrobenzene with its hydrogens, the GNN is evaluated on 43,968
maskings instead of all 65,536. Its strongest pairs are in the nitro groups: a nitrogen and an
oxygen raise the odds of a mutagen together, while the two oxygens of a group lower them together,
as either of them is enough. Try another molecule, ``index="Moebius"`` with ``max_order=len(atoms)`` for the
interactions of all orders, or :class:`~shapiq.graph.GraphSHAPIQ` with ``max_interaction_size``
for fewer maskings.

Run it again
------------

The scripts that computed the data of this demo and the other GraphSHAP-IQ demos are in
`docs/source/demos/graphshapiq-common/compute <https://github.com/mmschlk/shapiq/tree/main/docs/source/demos/graphshapiq-common/compute>`__,
with a README on how to run them. ``config.toml`` lists the questions and the molecules, as
SMILES; add a molecule there, and then run:

.. code-block:: bash

   python train.py    # a GNN per question (once)
   python explain.py  # the explanations of every molecule and question
   python pack.py     # the demos' data, ../../graphshapiq-*/assets/data.js
