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
   :alt: Sheets stacked on top of each other, each a molecule explained at one order of interaction: the Shapley values on top, then the Shapley interactions of order 2, 3 and 4, blank sheets for the orders 5 to 29, and the full decomposition of all 30 atoms at the bottom. Beside every sheet, its fit to the model rises from R² = 0.36 to 1.

What you see
------------

**The sheets.** Every sheet explains the same prediction. The top one holds the **Shapley values**:
one number per player, its main effect. Every sheet below holds the **Shapley interactions** of one
order more, FSII, the Faithful Shapley Interactions :cite:p:`Tsai.2022`: of all explanations with
values for the players and for groups of up to :math:`k` of them, the one that fits the model best
(see the fit, below); FSII of order 1 is the Shapley values. Blank sheets stand for the orders
between order 4 and the bottom sheet, which holds the **full decomposition**: FSII of order
:math:`n`, the number of players, which gives every group of players its own value, the Möbius
transform :cite:p:`Grabisch.1999`; for trees of depth 5, FSII of order 5, which already is. Red
raises the prediction, blue lowers it; the larger and the more opaque, the stronger. The slider
beside the stack, an arrow down, points to the more complex explanations: every order below has more
values. Point at a sheet to slide it out of the stack, or drag the slider's handle down through the
orders (on a touch screen, tap a sheet or drag the handle). Above the stack, an icon shows the kind
of model: a tree for tree ensembles, a graph for graph neural networks, and a black box for any
other model.

**The fit.** An explanation predicts what the model does with only some of the players: the sum of
the values of their groups. Beside every sheet, :math:`R^2` measures how much of the model's output
on random coalitions this prediction reproduces, its faithfulness. The coalitions are drawn from
the distribution of the Shapley values: a size :math:`s` from 1 to :math:`n - 1` with probability
proportional to :math:`1 / (s (n - s))`, and every coalition of that size alike, the weights of
KernelSHAP. Much of what the model does with them follows from how many players they hold, not from
which (for the molecule, 98 % of the spread of its output), so :math:`R^2` measures the fit beyond
size: it weighs the explanation's errors against the spread of the model's output among coalitions
of the same size,

.. math::

   R^2 = 1 - \frac{\sum_S w(S) \, \big(v(S) - \hat{v}(S)\big)^2}{\sum_S w(S) \, \big(v(S) - \bar{v}_{|S|}\big)^2},

with :math:`w` the distribution, :math:`v` the model, :math:`\hat{v}` the explanation, and
:math:`\bar{v}_s` the model's mean output on the coalitions of size :math:`s`. At 0, the explanation
tells no more than the number of players; at 1, it reproduces the model exactly. Under this
distribution, FSII of every order is the best fit of its order, so the fit never falls from one
order to the next. The Shapley values give every player its share; what they miss is what players
do only together. Every order recovers more of it, and the full decomposition is the model itself,
written as a sum over groups of players: :math:`R^2 = 1`. Unless it is estimated: for a black box of
many players, the full decomposition, too, is fit to a sample of the coalitions, and what its fit
falls short of 1, shaded in the plot, is the **estimation tax**. Many models have no interactions
above some order: the trees of a given depth, the receptive fields of a graph neural network. From
that order on, FSII equals the full decomposition: the houses' stacks end there, at order 5, where
the fit is exact, and the plot of the molecule breaks its axis before :math:`n = 30`. Point along
the plot to bring out the sheet of an order.

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
house: with their interaction, the explanation of order 2 fits with :math:`R^2 = 0.995`. Every sheet
draws the explanation as a tree that grows one level per order: the features on top, and every group
of :math:`k` features at level :math:`k`, joined to its features. The full decomposition is a tree
of depth 5, as deep as the model's trees, whose paths split on at most 5 features: no filler sheets,
the stack ends at 5-FSII, exact. Of the 494 test houses near downtown Los Angeles, the Shapley
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

**All houses**: 150 test houses, explained in the same way, as a heatmap: a column per house, sorted
by the model's prediction, and a row per feature or group of features. On average, the Shapley
values fit with :math:`R^2 = 0.83`, and those of a quarter of all test houses below 0.73; with pairs
of features, led by the interaction of latitude and longitude, the fit rises to 0.996. The trees
have depth 5, so no group of more than 5 features has a value, and 5-FSII is exact.

**Describing images**: the photo and caption are those of the :doc:`FIxLIP demo <../fixlip/index>`:
SigLIP 2 :cite:p:`Tschannen.2025` matches a photo and its caption, and the players are the photo's
64 patches and the caption's 10 tokens, as in FIxLIP :cite:p:`Baniecki.2025`. SigLIP 2 is a black
box: nothing in it gives its interactions exactly. :class:`~shapiq.approximator.ProxySHAP`
:cite:p:`Thies.2026` fits an XGBoost proxy of depth 6 to 10,000 coalitions of the game, drawn from
the distribution of the Shapley values; without its residual adjustment, the explanation is the
proxy's full decomposition, read off its trees exactly, and every order of FSII follows from it.
The Shapley values fit SigLIP 2 with :math:`R^2 = 0.68`; pairs, which link "bird" and "white" to the
egret's patch, lift the fit to 0.85, groups of three to 0.89, and of four to 0.91. From there, the
fit stays at 0.91, up to the full decomposition, the proxy itself; a higher order fits the proxy
better, but SigLIP 2 only as far as the proxy does. The full decomposition falls short of 1 by 0.09,
the estimation tax: what the proxy, fit to 10,000 of the :math:`2^{74}` coalitions, misses of
SigLIP 2, which only all of them would spare.

How it was computed
-------------------

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

   import shapiq

   # x_context, y_context: 2,000 training houses; x_test: the test houses (for the mean prediction)
   model = TabPFNRegressor(
       n_estimators=1, inference_config={"FEATURE_SHIFT_METHOD": None, "FINGERPRINT_FEATURE": False}
   ).fit(x_context, y_context)
   game = shapiq.TabPFNImputer(model, x_context, y_context, x_test=x_test).fit(house)
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

Run it again
------------

The scripts that computed the demo's data are in
`docs/source/demos/orders/compute <https://github.com/mmschlk/shapiq/tree/main/docs/source/demos/orders/compute>`__,
with a README on how to run them:

.. code-block:: bash

   python molecule.py  # trains the GCN and explains the molecule at every order
   python housing.py   # trains the XGBoost model and explains the houses at every order
   python house_tabpfn.py  # explains the house with TabPFN, from all 256 subsets of its features
   python fixlip.py    # evaluates SigLIP 2 on 10,512 coalitions and explains it with ProxySHAP
   python pack.py      # the demo's data, ../assets/data.js
