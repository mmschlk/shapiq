GraphSHAP-IQ: Which Atoms Does a Graph Neural Network See Together?
=====================================================================

.. rst-class:: shapiq-examples-lead

Does a molecule contain an indole? Is it mutagenic? A graph neural network (GNN) answers from the
molecule's atoms and bonds. This demo explains its answers for eight molecules, from serotonin to
TNT, by the Shapley interactions of their atoms, of any order and exactly, as in GraphSHAP-IQ
:cite:p:`Muschalik.2025`, computed with ``shapiq``.

.. shapiq-demo:: graphshapiq
   :fallback: images/hero_graphshapiq_demo.png
   :alt: The molecule TNT, explained by a GNN's interactions of its atoms: the nitrogen and the oxygens of each of its three nitro groups are joined by red bands, the pairs of atoms that together raise the odds that it is mutagenic the most.

What you see
------------

**The questions.** Each question has its own GNN, trained to answer it. *Indole?* and *PAINS?*
come from B-XAIC :cite:p:`Proszewska.2025`, a benchmark of 50,000 molecules of ChEMBL whose labels
follow from substructures, so that every label has a ground truth: an indole (a benzene ring fused
to a five-membered ring with a nitrogen), or a pan-assay interference compound (PAINS)
:cite:p:`Baell.2010`, here a catechol, which fools many lab tests. *Mutagenic?* comes from the
4,337 molecules of Kazius et al. :cite:p:`Kazius.2005`, tested for mutagenicity in bacteria (the
Ames test), as in GraphSHAP-IQ's figures; its outlined ground truth is the nitro group, a known
toxicophore. Every GNN is B-XAIC's GIN :cite:p:`Xu.2019`: three message passing layers, a sum over
the atoms, and a linear readout. It sees an atom only as its element and a bond only as a link,
not as single, double or aromatic.

**The players** are the atoms. A coalition of atoms is what the GNN gets to know: the other atoms
keep their bonds, but their element is masked, replaced by the average atom of the GNN's training
molecules (mostly carbon). The explanation splits the GNN's log-odds of *yes* into what every atom
adds on its own and what groups of atoms add together, their **interactions**. An interaction is
red when the atoms raise the odds of *yes* together, and blue when they lower them.

**The views.** *No explanation* shows the GNN's prediction. Click atoms to mask them and see what
the GNN predicts without them: the GNN runs in your browser. *Interactions* draws the explanation
on the molecule, as the SI-graphs of the paper: every atom tinted by its own value, the strongest
groups of atoms joined; − and + change the highest order. Order 1 shows the Shapley values, orders
2 to 4 the k-SII with interactions of up to as many atoms, and *all orders* the Möbius
interactions, from which GraphSHAP-IQ computes the others. Point at an atom (or tap it) to see its
interactions; the green outline marks the ground truth. *Receptive fields* shows what makes the
computation exact and fast.

What it shows
-------------

- **Nitro groups act as pairs.** For TNT and nitrofurantoin, the GNN for *Mutagenic?* gives the
  nitrogens of the nitro groups the largest Shapley values, and the pairs of such a nitrogen and an
  oxygen the strongest interactions, as GraphSHAP-IQ found for the mutagen of its Figure 1.
- **An indole is a team.** For serotonin, psilocybin and LSD, the indole's nitrogen leads the
  Shapley values, and its pairs with the carbons of its ring lead the interactions. At higher
  orders, large groups of the indole's atoms raise and lower the odds by a lot each, and offset
  each other: the Möbius interactions are exact, but not simple, which is why lower orders
  summarize them.
- **Indigo fools the GNN.** Indole is named after the dye, but indigo is no indole: its
  five-membered rings are not aromatic. The GNN, which sees no bond orders, calls it an indole
  anyway (97%). Its nitrogens say yes; its oxygens say no, most of all in pairs with the
  nitrogens, but not loud enough. Caffeine, with nitrogens in both of its rings, the GNN rightly
  rejects (1%).
- **A catechol for PAINS.** Dopamine's two neighboring OH groups make it a PAINS for the GNN
  (more than 99%). At order 2, the catechol's atoms lower the odds alone and raise them in pairs:
  none of them makes a catechol on its own.

How it was computed
-------------------

The graph game
^^^^^^^^^^^^^^

The value of a coalition of atoms is the GNN's output for the molecule with only those atoms
known: the log-odds of *yes*, its logit of yes minus its logit of no. A GNN with three layers
passes messages along bonds three times, so what it computes for an atom depends only on the atoms
within three bonds: the atom's **receptive field**. As the GNN sums its atoms up and reads the sum
out linearly, atoms that are never together in a receptive field do not interact: all their
interactions are zero :cite:p:`Muschalik.2025`. GraphSHAP-IQ therefore evaluates the GNN only on the
maskings within receptive fields, computes their Möbius interactions, and converts them into any
index, exactly. For TNT's 16 atoms, that takes 23,680 instead of :math:`2^{16} = 65,536`
maskings; for LSD's 24 atoms, 84,864 instead of :math:`2^{24}`, about 16.8 million. Each
explanation took less than 20 seconds on four CPU cores, and its Möbius interactions reproduce the
GNN's output on random maskings to :math:`10^{-5}`.

Explain it yourself
^^^^^^^^^^^^^^^^^^^

:class:`~shapiq.explainer.GraphExplainer` explains a GNN of ``torch_geometric`` with GraphSHAP-IQ.
Here, a GIN learns from TUDataset's copy of the Mutagenicity dataset :cite:p:`Morris.2020`, with
hydrogens, as in the paper's figures, and explains TNT (a few minutes on a laptop):

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

   # TNT with its hydrogens, as a graph of the dataset's kind
   tnt = Chem.AddHs(Chem.MolFromSmiles("Cc1c(cc(cc1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]"))
   atoms = [f"{atom.GetSymbol()}{atom.GetIdx()}" for atom in tnt.GetAtoms()]
   bonds = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in tnt.GetBonds()]
   elements = torch.tensor([ELEMENTS.index(atom.GetSymbol()) for atom in tnt.GetAtoms()])
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

   374,720 of 2^21 maskings
   O11 and O12: -1.69
   O14 and O15: -1.69
   O8 and O9: -1.69
   N10 and O11: +1.31
   N10 and O12: +1.31
   N7 and O8: +1.19

Of TNT's 21 atoms, the GNN is evaluated on 374,720 maskings instead of all 2,097,152. Its strongest
pairs are in the nitro groups: the nitrogen and an oxygen raise the odds of a mutagen together,
while the two oxygens of a group lower them together, as either of them is enough. Try another
molecule, ``index="Moebius"`` with ``max_order=len(atoms)`` for the interactions of all orders, or
:class:`~shapiq.graph.GraphSHAPIQ` with ``max_interaction_size`` for fewer maskings.

Run it again
------------

The scripts that computed the demo's data are in
`docs/source/demos/graphshapiq/compute <https://github.com/mmschlk/shapiq/tree/main/docs/source/demos/graphshapiq/compute>`__,
with a README on how to run them. ``config.toml`` lists the questions and the molecules, as
SMILES; add a molecule there, and then run:

.. code-block:: bash

   python train.py    # a GNN per question (once)
   python explain.py  # the explanations of every molecule and question
   python pack.py     # the demo's data, ../assets/data.js
