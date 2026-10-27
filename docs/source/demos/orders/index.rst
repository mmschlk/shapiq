From Shapley Values to the Möbius Transform
===========================================

.. rst-class:: shapiq-examples-lead

How much of a model does an explanation explain? This demo explains one prediction at every order
of interaction, from the Shapley values, one number per player, to the Möbius transform, which adds
up to the model exactly, and shows how well each order fits the model. Its molecule is the one of
GraphSHAP-IQ's Figure 1 :cite:p:`Muschalik.2025`, explained exactly with ``shapiq``.

.. shapiq-demo:: orders
   :fallback: images/hero_orders.png
   :alt: Five sheets stacked on top of each other, each a molecule explained at one order of interaction: the Shapley values on top, then the Shapley interactions of order 2, 3 and 4, and the Möbius interactions at the bottom. Beside every sheet, its fit to the model rises from R² = 0.76 to 1.

What you see
------------

**The sheets.** Every sheet explains the same prediction. The top one holds the **Shapley values**:
one number per player, its main effect. Every sheet below holds the **Shapley interactions** of one
order more, k-SII :cite:p:`Bordt.2023`: the effects of the players and of all groups of up to
:math:`k` of them. The bottom one holds the **Möbius transform** :cite:p:`Grabisch.1999`, the
interactions of all orders, which is k-SII of the highest order. Red raises the prediction, blue
lowers it; the larger and the more opaque, the stronger. Point at a sheet (or tap it) to lift it
out of the stack.

**The fit.** An explanation predicts what the model does with only some of the players: the sum of
the values of their groups. Beside every sheet, :math:`R^2` measures how much of the model's output
on random coalitions, every player present with probability 1/2, this prediction reproduces. The
Shapley values miss everything that players do only together. Every order recovers more of it, and
the Möbius transform is the model itself, written as a sum over groups of players: :math:`R^2 = 1`.

**The molecule** is graph 71 of the Mutagenicity dataset of the TUDataset collection
:cite:p:`Morris.2020`, 30 atoms with three nitro groups (NO\ :sub:`2`), which a graph
convolutional network (GCN) classifies as mutagenic. Its players are the atoms; an absent atom
keeps its bonds, and its features are zeros. The Shapley values credit the nitrogen atoms, the
interactions of order 2 the bonds of the nitro groups, and those of order 4 every nitro group with
its carbon, and pieces of the carbon rings. With them, the explanation fits the model almost
exactly.

**The photo and caption** are those of the :doc:`FIxLIP demo <../fixlip/index>`: SigLIP 2
:cite:p:`Tschannen.2025` matches a photo and its caption, and the players are the photo's patches
and the caption's tokens. Its values are **illustrative**: a toy game, defined by interactions that
the FIxLIP demo's explanation suggests (e.g. "white" and "bird" with the egret, and "next to" with
both animals), shows what higher orders add to the explanation of a vision-language model
:cite:p:`Baniecki.2025`. Its :math:`R^2` are illustrative too.

How it was computed
-------------------

A GCN with two layers, a linear pooling, and a linear readout computes the embedding of an atom
from the atoms within two bonds of it, its receptive field. Every interaction of atoms that share no
receptive field is zero, so :class:`~shapiq.graph.GraphSHAPIQ` :cite:p:`Muschalik.2025` computes
the Möbius transform exactly from the subsets of the receptive fields: 7,693 evaluations of the
model instead of :math:`2^{30}`. Every order of k-SII follows from it without evaluating the model
again:

.. code-block:: python

   from shapiq.graph import GraphSHAPIQ
   from shapiq_games import GraphGame

   game = GraphGame(model, molecule, class_index=0)  # the logit of "mutagen"
   graphshapiq = GraphSHAPIQ(game, game.graph)
   print(graphshapiq.budget)  # 7693

   moebius = graphshapiq("Moebius", order=game.n_players)  # all orders, exactly
   explanations = {k: graphshapiq("k-SII", order=k) for k in (1, 2, 3, 4)}  # SV for k = 1

The GCN is trained as in GraphSHAP-IQ's experiments: two GCN layers, each followed by batch
normalization, LeakyReLU, and dropout, the embeddings of both layers summed over the atoms, and a
linear readout, trained with Adam on 80 % of Mutagenicity. It classifies 78 % of the test molecules
correctly, about as many as in the paper. It is not the paper's network, so its values differ from
those of Figure 1, but they highlight the same nitro groups.

Run it again
------------

The scripts that computed the demo's data are in
`docs/source/demos/orders/compute <https://github.com/mmschlk/shapiq/tree/main/docs/source/demos/orders/compute>`__,
with a README on how to run them:

.. code-block:: bash

   python molecule.py  # trains the GCN and explains the molecule at every order
   python fixlip.py    # the illustrative explanation of the photo and caption
   python pack.py      # the demo's data, ../assets/data.js
