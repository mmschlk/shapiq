From Shapley Values to the Full Decomposition
=============================================

.. rst-class:: shapiq-examples-lead

How much of a model does an explanation explain? This demo explains one prediction at every order
of interaction, from the Shapley values, one number per player, to the full decomposition, which
adds up to the model exactly, and shows how well each order fits the model. Its molecule is the one
of GraphSHAP-IQ's Figure 1 :cite:p:`Muschalik.2025`, its houses those of California housing, all
explained exactly with ``shapiq``.

.. shapiq-demo:: orders
   :fallback: images/hero_orders.png
   :alt: Sheets stacked on top of each other, each a molecule explained at one order of interaction: the Shapley values on top, then the Shapley interactions of order 2, 3 and 4, blank sheets for the orders 5 to 29, and the full decomposition of all 30 atoms at the bottom. Beside every sheet, its fit to the model rises from R² = 0.76 to 1.

What you see
------------

**The sheets.** Every sheet explains the same prediction. The top one holds the **Shapley values**:
one number per player, its main effect. Every sheet below holds the **Shapley interactions** of one
order more, k-SII :cite:p:`Bordt.2023`: the effects of the players and of all groups of up to
:math:`k` of them. Blank sheets stand for the orders after order 4, and the bottom sheet holds the
**full decomposition**: k-SII of order :math:`n`, the number of players, which gives every group of
players its own value, the Möbius transform :cite:p:`Grabisch.1999`. Red raises the prediction,
blue lowers it; the larger and the more opaque, the stronger. Point at a sheet (or tap it) to lift
it out of the stack.

**The fit.** An explanation predicts what the model does with only some of the players: the sum of
the values of their groups. Beside every sheet, :math:`R^2` measures how much of the model's output
on random coalitions, every player present with probability 1/2, this prediction reproduces. The
Shapley values miss everything that players do only together. Every order recovers more of it, and
the full decomposition is the model itself, written as a sum over groups of players:
:math:`R^2 = 1`. Many models have no interactions above some order: the trees of a given depth, the
receptive fields of a graph neural network. From that order on, k-SII equals the full
decomposition, and the plot below the stack breaks its axis before :math:`n`. Point along the plot
to bring out the sheet of an order.

**The molecule** is graph 71 of the Mutagenicity dataset of the TUDataset collection
:cite:p:`Morris.2020`, 30 atoms with three nitro groups (NO\ :sub:`2`), which a graph
convolutional network (GCN) classifies as mutagenic. Its players are the atoms; an absent atom
keeps its bonds, and its features are zeros. The Shapley values credit the nitrogen atoms, the
interactions of order 2 the bonds of the nitro groups, and those of order 4 every nitro group with
its carbon, and pieces of the carbon rings. With them, the explanation fits the model almost
exactly. No group of more than 10 atoms has a value: the GCN's receptive fields hold at most 10.

**One house** in Los Angeles, valued by an XGBoost model of depth 7. Its players are the 8
features of the house; an absent feature takes its values from 100 training houses (interventional
explanations). The Shapley values split the location into a negative longitude and a positive
latitude and fit the model with :math:`R^2 = 0.30`. Latitude and longitude only together locate the
house: with their interaction, the explanation of order 2 fits with :math:`R^2 = 0.995`. Among the
test houses near downtown Los Angeles, this is the one whose Shapley values fit closest to
:math:`R^2 = 0.3`; many fit worse.

**150 houses**, explained in the same way, as a heatmap: a column per house, sorted by the
model's prediction, and a row per feature or group of features. On average, the Shapley values fit
with :math:`R^2 = 0.77`, and those of a quarter of all test houses below 0.66; with pairs of
features, led by the interaction of latitude and longitude, the fit rises to 0.99. The trees have
depth 7, so no group of all 8 features has a value.

**The photo and caption** are those of the :doc:`FIxLIP demo <../fixlip/index>`: SigLIP 2
:cite:p:`Tschannen.2025` matches a photo and its caption, and the players are the photo's patches
and the caption's tokens. Its values are **illustrative**: a toy game, defined by interactions that
the FIxLIP demo's explanation suggests (e.g. "white" and "bird" with the egret, and "next to" with
both animals), shows what higher orders add to the explanation of a vision-language model
:cite:p:`Baniecki.2025`. Its :math:`R^2` are illustrative too.

How it was computed
-------------------

**The molecule.** A GCN with two layers, a linear pooling, and a linear readout computes the
embedding of an atom from the atoms within two bonds of it, its receptive field. Every interaction
of atoms that share no receptive field is zero, so :class:`~shapiq.graph.GraphSHAPIQ`
:cite:p:`Muschalik.2025` computes the full decomposition exactly from the subsets of the receptive
fields: 7,693 evaluations of the model instead of :math:`2^{30}`. Every order of k-SII follows from
it without evaluating the model again:

.. code-block:: python

   from shapiq.graph import GraphSHAPIQ
   from shapiq_games import GraphGame

   game = GraphGame(model, molecule, class_index=0)  # the logit of "mutagen"
   graphshapiq = GraphSHAPIQ(game, game.graph)
   print(graphshapiq.budget)  # 7693

   full = graphshapiq("Moebius", order=game.n_players)  # the full decomposition, exactly
   explanations = {k: graphshapiq("k-SII", order=k) for k in (1, 2, 3, 4)}  # SV for k = 1

The GCN is trained as in GraphSHAP-IQ's experiments: two GCN layers, each followed by batch
normalization, LeakyReLU, and dropout, the embeddings of both layers summed over the atoms, and a
linear readout, trained with Adam on 80 % of Mutagenicity. It classifies 78 % of the test molecules
correctly, about as many as in the paper. It is not the paper's network, so its values differ from
those of Figure 1, but they highlight the same nitro groups.

**The houses.** :class:`~shapiq.TreeExplainer` explains tree ensembles exactly. In interventional
mode, it computes interactions of any order for many houses at once, with the Woodelf algorithm
when ``shapiq[tree]`` is installed:

.. code-block:: python

   import numpy as np
   from sklearn.datasets import fetch_california_housing
   from sklearn.model_selection import train_test_split
   from xgboost import XGBRegressor

   import shapiq

   X, y = fetch_california_housing(return_X_y=True)
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
   model = XGBRegressor(n_estimators=300, max_depth=7, learning_rate=0.1, random_state=0)
   model.fit(X_train, y_train)

   # an absent feature takes its values from 100 training houses
   background = X_train[np.random.default_rng(0).choice(len(X_train), 100, replace=False)]
   explainer = shapiq.TreeExplainer(
       model, mode="interventional", reference_dataset=background, index="k-SII", max_order=2
   )
   pairs = explainer.explain_X(X_test[:150])  # k-SII of order 2 of 150 houses, in a second
   print(pairs[0][(6, 7)])  # the interaction of latitude and longitude of the first house

With ``index="Moebius"`` and ``max_order=8``, the explainer computes the full decomposition, from
which the demo's scripts convert every order of k-SII. On the test houses, the model reaches
:math:`R^2 = 0.85`.

Run it again
------------

The scripts that computed the demo's data are in
`docs/source/demos/orders/compute <https://github.com/mmschlk/shapiq/tree/main/docs/source/demos/orders/compute>`__,
with a README on how to run them:

.. code-block:: bash

   python molecule.py  # trains the GCN and explains the molecule at every order
   python housing.py   # trains the XGBoost model and explains the houses at every order
   python fixlip.py    # the illustrative explanation of the photo and caption
   python pack.py      # the demo's data, ../assets/data.js
