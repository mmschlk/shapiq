How Much Does an Explanation Explain?
=====================================

.. rst-class:: shapiq-examples-lead

An explanation of a prediction can say little or everything: one number per player, the Shapley
values, or a value for every group of players, the full decomposition, which is the model itself.
Between them lie the Shapley interactions of every order. This page builds one figure up, step by
step. Every figure shows the same example: choose another in any of them, and all of them follow.

.. the figure of demos/orders, a step of it at a time (:step:); the figures of a page share their
   example. Without JavaScript, the last one shows its static image.


A Model and Its Prediction
--------------------------

A model :math:`f` takes an input :math:`x` and returns a prediction :math:`f(x)`: whether a
molecule is mutagenic, for a graph neural network; a house's price, for XGBoost or TabPFN; the
prices of 150 houses across California; how well a photo matches its caption, for SigLIP 2. The
input is made of players: atoms, features, or the photo's patches and the caption's words. Why this
prediction?

.. shapiq-demo:: orders
   :step: problem


The Mechanism
-------------

Below the input lies what the model does with it, its mechanism. However complex the model, its
prediction adds up from the effects of its players and of every group of them, each a term of its
own: the full decomposition, the Möbius transform :cite:p:`Grabisch.1999`. With :math:`n` players,
there are up to :math:`2^n` of them, for 30 atoms over a billion. The mechanism is exact, and
nobody can read it. The card above shows it on the input; where it is only estimated, it is
blurred.

.. shapiq-demo:: orders
   :step: mechanism


The Shapley Values
------------------

The simplest explanation gives every player one number, its Shapley value: red raises the
prediction, blue lowers it. It floats above the input. But how much of the mechanism does it
explain? Its faithfulness, :math:`R^2`, says how well it predicts what the model does when only
some of the players are there, beside the layer: 1 is all of it. Point at the mechanism to compare.
The card above names the strongest player; point at any player on it, an atom, a feature, a house,
a patch or a word, to read its value.

.. shapiq-demo:: orders
   :step: shapley


Interactions, Order by Order
----------------------------

What the Shapley values miss is what players do together. The Shapley interactions of order 2 add
a value for every pair of players, those of order 3 for every three, and so on: FSII, the Faithful
Shapley Interactions :cite:p:`Tsai.2022`, the explanation of each order that fits the model best.
Every order is more detailed than the one above it, and more faithful. The pairs alone recover most
of what the Shapley values miss. Point at the other orders, or drag the slider down through them,
and see the fit rise. On the card, a player now also says the strongest groups it is in: click a
layer to keep it there, then point at its players.

.. shapiq-demo:: orders
   :step: interactions


Every Order
-----------

Between order 4 and the last order, as many as there are players, lie blank layers, standing for
the orders in between; point at one to see an order's explanation. The last layer is the full decomposition: the
mechanism, explained exactly, with :math:`R^2 = 1`, unless it is estimated, when what it falls
short of is the estimation gap.

.. shapiq-demo:: orders
   :step: full
   :fallback: images/hero_orders.png
   :alt: Layers floating above a molecule, each the molecule's explanation at one order of interaction: the Shapley values on top, then the Shapley interactions of order 2, 3 and 4, blank layers standing for the orders 5 to 10, and the full decomposition of all 30 atoms the last; below the molecule, the mechanism, the model itself. Beside them, a plot of their fit to the model rises from R² = 0.36 to 1, a point at the height of every layer.


----

Further Information
-------------------

How the fit of every order is measured, what the examples are, and how their explanations were
computed with ``shapiq``. What Shapley interactions are, and when they tell more than Shapley
values, is in :doc:`index`.

How the Fit Is Measured
~~~~~~~~~~~~~~~~~~~~~~~

An explanation predicts what the model does with only some of the players: the sum of the values
of their groups. The plot beside the layers shows, for every order, how much of the model's output
on random coalitions this prediction reproduces, its :math:`R^2`: the explanation's faithfulness.
The coalitions are drawn from the distribution of the Shapley values: a size :math:`s` from 1 to
:math:`n - 1` with probability proportional to :math:`1 / (s (n - s))`, and every coalition of that
size alike, the weights of KernelSHAP. Much of what the model does with them follows from how many
players they hold, not from which (for the molecule, 98 % of the spread of its output), so
:math:`R^2` measures the fit beyond size: it weighs the explanation's errors against the spread of
the model's output among coalitions of the same size,

.. math::

   R^2 = 1 - \frac{\sum_S w(S) \, \big(v(S) - \hat{v}(S)\big)^2}{\sum_S w(S) \, \big(v(S) - \bar{v}_{|S|}\big)^2},

with :math:`w` the distribution, :math:`v` the model, :math:`\hat{v}` the explanation, and
:math:`\bar{v}_s` the model's mean output on the coalitions of size :math:`s`. At 0, the explanation
tells no more than the number of players; at 1, it reproduces the model exactly. Under this
distribution, FSII of every order is the best fit of its order to the model it explains, so for a
model explained exactly, the fit never falls from one order to the next. The Shapley values give
every player its share; what they miss is what players do only together. Every order recovers more
of it, and the full decomposition is the model itself,
written as a sum over groups of players: :math:`R^2 = 1`. Unless it is estimated: for a black box of
many players, the full decomposition, too, is fit to a sample of the coalitions, and what its fit
falls short of 1, hatched in gray in the plot, is the **estimation gap**. Many models have no
interactions above some order: the trees of a given depth, the receptive fields of a graph neural
network. From that order on, FSII equals the full decomposition: the houses' layers end there, at
order 5, where the fit is exact, and the molecule's plot runs dashed from order 10 to :math:`n = 30`.

The Examples
~~~~~~~~~~~~

**The molecule** is graph 71 of the Mutagenicity dataset of the TUDataset collection
:cite:p:`Morris.2020`, 30 atoms with three nitro groups (NO\ :sub:`2`), which a graph convolutional
network (GCN) classifies as mutagenic. Its players are the atoms; an absent atom keeps its bonds,
and its features are zeros. The Shapley values credit the nitrogen atoms and fit the GCN with
:math:`R^2 = 0.36`; the interactions of order 2 add the bonds of the nitro groups,
:math:`R^2 = 0.88`, and those of order 4 pieces of the carbon rings and every nitro group with its
carbon, :math:`R^2 = 0.998`: the explanation fits the model almost exactly. No group of more than
10 atoms has a value: the GCN's receptive fields hold at most 10.

**One house** in Los Angeles, valued by an XGBoost model of depth 5. Its players are the 8 features
of the house; an absent feature takes its values from 100 training houses (interventional
explanations). The Shapley values split the location into a negative longitude and a positive
latitude and fit the model with :math:`R^2 = 0.36`. Latitude and longitude only together locate the
house: with their interaction, the explanation of order 2 fits with :math:`R^2 = 0.995`. Every layer
draws the explanation as a tree that grows one level per order: the features on top, and every group
of :math:`k` features at level :math:`k`, joined to its features. The full decomposition is a tree
of depth 5, as deep as the model's trees, whose paths split on at most 5 features: no blank layers,
they end at 5-FSII, exact. Of the 494 test houses near downtown Los Angeles, the Shapley
values of this one fit worse than those of 94 %.

**One house, TabPFN**: the same house, valued by TabPFN :cite:p:`Hollmann.2025`, a transformer
pretrained on synthetic tables, which predicts by reading 2,000 training houses together with the
house (on the test houses, :math:`R^2 = 0.85`, as the trees). TabPFN is a black box, and nothing in
it limits how many features act together: its full decomposition has a value for groups of every
size, up to all 8 features, and only 8-FSII is exact. shapiq's
:class:`~shapiq.imputer.TabPFNImputer` removes an absent feature from TabPFN's context instead of
imputing it :cite:p:`Rundel.2024`, and with 8 features, all 256 subsets of them are evaluated. The
Shapley values fit TabPFN with :math:`R^2 = 0.06`, barely more than the number of features alone:
TabPFN weighs this house's features together. Pairs, latitude with longitude and the income with
the rooms, lift the fit to 0.84, and every order more a little further, 0.95, 0.98, up to the full
decomposition, which fits exactly.

**All houses**: 150 test houses, explained in the same way, on the map of California: a dot where
every house is, in the color of what one feature, or one group of features, adds to its value or
takes off it, and beside the map the explanation's features and groups, by how much each moves a
house's value on average. The Shapley values credit the latitude most, by $91,000 on average: it
raises the houses of the south and lowers those of the north, and the longitude does the opposite.
On average, they fit with :math:`R^2 = 0.83`, and those of a quarter of all test houses below 0.73;
with pairs of features, led by the interaction of latitude and longitude, which the map shows from
order 2 on, the fit rises to 0.996. Point at another row to see it on the map, and at a house for
its own values. The trees have depth 5, so no group of more than 5 features has a value, and
5-FSII is exact.

**Describing images**: the photo and caption are those of the
:doc:`FIxLIP demo <../demos/fixlip/index>`: SigLIP 2 :cite:p:`Tschannen.2025` matches a photo and
its caption, and the players are the photo's
64 patches and the caption's 10 tokens, as in FIxLIP :cite:p:`Baniecki.2025`. SigLIP 2 is a black
box: nothing in it gives its interactions exactly. :class:`~shapiq.approximator.ProxySHAP`
:cite:p:`Thies.2026` fits an XGBoost proxy of depth 6 to 10,000 coalitions of the game, drawn from
the distribution of the Shapley values; without its residual adjustment, the explanation is the
proxy's full decomposition, read off its trees exactly, and every order of FSII follows from it.
The Shapley values fit SigLIP 2 with :math:`R^2 = 0.68`; pairs, which link "bird" and "white" to the
egret's patch, lift the fit to 0.85, groups of three to 0.89, and of four to 0.91. From there, the
fit stays at 0.91, up to the full decomposition, the proxy itself; a higher order fits the proxy
better, but SigLIP 2 only as far as the proxy does. The full decomposition falls short of 1 by 0.09,
the estimation gap: what the proxy, fit to 10,000 of the :math:`2^{74}` coalitions, misses of
SigLIP 2, which only all of them would spare.

How They Were Computed
~~~~~~~~~~~~~~~~~~~~~~

**The molecule.** A GCN with two layers, a linear pooling, and a linear readout computes the
embedding of an atom from the atoms within two bonds of it, its receptive field. Every interaction
of atoms that share no receptive field is zero, so :class:`~shapiq.graph.GraphSHAPIQ`
:cite:p:`Muschalik.2025` computes the full decomposition exactly from the subsets of the receptive
fields: 7,693 evaluations of the model instead of :math:`2^{30}`. Every order of FSII follows from
it without evaluating the model again:

.. code-block:: python

   from shapiq.graph import GraphSHAPIQ
   from shapiq_games import GraphGame

   game = GraphGame(model, molecule, class_index=0)  # the logit of "mutagen"
   graphshapiq = GraphSHAPIQ(game, game.graph)
   print(graphshapiq.budget)  # 7693

   full = graphshapiq("Moebius", order=game.n_players)  # the full decomposition, exactly
   explanations = {k: graphshapiq("FSII", order=k) for k in (1, 2, 3, 4)}  # SV for k = 1

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
   model = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.1, random_state=0)
   model.fit(X_train, y_train)

   # an absent feature takes its values from 100 training houses
   background = X_train[np.random.default_rng(0).choice(len(X_train), 100, replace=False)]
   explainer = shapiq.TreeExplainer(
       model, mode="interventional", reference_dataset=background, index="FSII", max_order=2
   )
   pairs = explainer.explain_X(X_test[:150])  # FSII of order 2 of 150 houses, in a second
   print(pairs[0][(6, 7)])  # the interaction of latitude and longitude of the first house

With ``index="Moebius"`` and ``max_order=8``, the explainer computes the full decomposition, from
which the demo's scripts convert every order of FSII. On the test houses, the model reaches
:math:`R^2 = 0.85`.

**One house, TabPFN.** A black box with 8 features has only :math:`2^8 = 256` subsets of them:
:class:`~shapiq.ExactComputer` evaluates the game on all of them and computes the full
decomposition exactly. TabPFN makes one pass, with the features in their order, so that the game
is the same on every run:

.. code-block:: python

   from tabpfn import TabPFNRegressor
   from tabpfn.constants import ModelVersion

   import shapiq

   # x_context, y_context: 2,000 training houses; x_test: the test houses (for the mean prediction)
   model = TabPFNRegressor.create_default_for_version(
       ModelVersion.V2,
       n_estimators=1,
       inference_config={"FEATURE_SHIFT_METHOD": None, "FINGERPRINT_FEATURE": False},
       random_state=0,
       ignore_pretraining_limits=True,  # a context of more than 1,000 houses on a CPU
   ).fit(x_context, y_context)
   game = shapiq.TabPFNImputer(
       model, x_context, y_context, x_test=x_test, predict_function=lambda model, x: model.predict(x)
   ).fit(house)
   full = shapiq.ExactComputer(game)("Moebius")  # the full decomposition, from all 256 subsets

**Describing images.** For a black box, :class:`~shapiq.approximator.ProxySHAP` explains a
proxy of the model instead. Without its residual adjustment, the explanation is the proxy's full
decomposition, which shapiq's :class:`~shapiq.tree.InterventionalTreeSHAPIQ` reads off its trees
exactly, as ProxySHAP does, and every order of FSII follows from it:

.. code-block:: python

   import numpy as np
   from xgboost import XGBRegressor

   import shapiq
   from shapiq.tree import InterventionalTreeSHAPIQ

   # game: the FIxLIP demo's game of SigLIP 2, the elephant and its caption (74 players)
   n = game.n_players
   # the coalitions' sizes as the Shapley values weigh them: 1 / (s (n - s)), but not 0 and n
   sizes = np.arange(n + 1)
   weights = np.where((sizes > 0) & (sizes < n), 1 / np.maximum(sizes * (n - sizes), 1), 0)
   approximator = shapiq.approximator.ProxySHAP(
       n=n,
       index="FSII",
       sampling_weights=weights,
       proxy_model=XGBRegressor(n_estimators=1000, max_depth=6, learning_rate=0.05, reg_lambda=5.0),
       adjustment=False,
   )
   approximator.approximate(budget=10_000, game=game)  # draws the coalitions, fits the proxy
   # the proxy's full decomposition: its trees split on at most 6 players
   explainer = InterventionalTreeSHAPIQ(
       approximator.proxy_model, data=np.zeros((1, n)), index="Moebius", max_order=6
   )
   full = explainer.explain_function(np.ones((1, n)))

Run It Again
~~~~~~~~~~~~

The scripts that computed the figure's data are in
`docs/source/demos/orders/compute <https://github.com/mmschlk/shapiq/tree/main/docs/source/demos/orders/compute>`__,
with a README on how to run them:

.. code-block:: bash

   python molecule.py  # trains the GCN and explains the molecule at every order
   python housing.py   # trains the XGBoost model and explains the houses at every order
   python house_tabpfn.py  # explains the house with TabPFN, from all 256 subsets of its features
   python fixlip.py    # evaluates SigLIP 2 on 10,512 coalitions and explains it with ProxySHAP
   python pack.py      # the figure's data, ../assets/data.js
