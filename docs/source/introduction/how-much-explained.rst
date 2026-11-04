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
prices of 150 houses; how well a photo matches its caption, for SigLIP 2. The input is made of
players: atoms, features, or the photo's patches and the caption's words. Why this prediction?

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

.. shapiq-demo:: orders
   :step: shapley


Interactions, Order by Order
----------------------------

What the Shapley values miss is what players do together. The Shapley interactions of order 2 add
a value for every pair of players, those of order 3 for every three, and so on: FSII, the Faithful
Shapley Interactions :cite:p:`Tsai.2022`, the explanation of each order that fits the model best.
Every order is more detailed than the one above it, and more faithful. The pairs alone recover most
of what the Shapley values miss. Point at the other orders, or drag the slider down through them,
and see the fit rise.

.. shapiq-demo:: orders
   :step: interactions


Every Order
-----------

Between order 4 and the last order, as many as there are players, lie blank layers, one for every
order; point at one to see its explanation. The last layer is the full decomposition: the
mechanism, explained exactly, with :math:`R^2 = 1`, unless it is estimated, when what it falls
short of is the estimation gap.

.. shapiq-demo:: orders
   :step: full
   :fallback: images/hero_orders.png
   :alt: Layers floating above a molecule, each the molecule's explanation at one order of interaction: the Shapley values on top, then the Shapley interactions of order 2, 3 and 4, blank layers for the orders 5 to 29, and the full decomposition of all 30 atoms the last; below the molecule, the mechanism, the model itself. Beside them, a plot of their fit to the model rises from R² = 0.36 to 1, a point at the height of every layer.


Read More
---------

How the explanations of the figure were computed, and what every example shows, is on the page of
:doc:`its demo <../demos/orders/index>`. What Shapley interactions are, and when they tell more
than Shapley values, is in :doc:`index`.
