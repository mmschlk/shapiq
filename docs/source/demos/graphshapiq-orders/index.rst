GraphSHAP-IQ: One Prediction at Every Order
===========================================

.. rst-class:: shapiq-examples-lead

The Shapley values give every atom one number. But a molecule is more than its atoms: it is their
groups. GraphSHAP-IQ :cite:p:`Muschalik.2025` computes the interactions of every order exactly.
Raise the order, and watch the explanation go from a rough sketch to a perfect reconstruction of
the graph neural network (GNN).

.. shapiq-demo:: graphshapiq-orders
   :fallback: images/hero_graphshapiq_orders_demo.png
   :alt: 1,3-dinitrobenzene explained by pairs of atoms, with a faithfulness of 0.95: the curve of faithfulness climbs from below zero for the Shapley values to one for the full decomposition.

How to read it
--------------

Every order is the Faithful Shapley Interaction index (FSII) :cite:p:`Tsai.2022` of that order: of
all explanations with values for groups of up to :math:`k` atoms, the one that fits the GNN best.
Order 1 is the Shapley values, and the last step is the full decomposition, the Möbius
interactions. The number above the curve is how faithful the explanation is, as in
:doc:`../../introduction/how-much-explained`: its :math:`R^2` beyond size, the share of what the
atoms of a masking decide, beyond how many atoms it keeps, that the explanation reproduces. It
never falls with the order, and it is 1 for the full decomposition.

What it shows
-------------

- **The Shapley values are not enough.** For every molecule here, they score below zero: one number
  per atom predicts the GNN on maskings worse than counting how many atoms a masking keeps.
- **For 1,3-dinitrobenzene, pairs suffice.** Order 2 already reaches 0.95: the pairs of a
  nitrogen and its oxygens carry the answer.
- **Dopamine needs order 3.** Its pairs score below zero too, and only groups of three atoms reach
  0.89: the catechol is a team of three and more.
- **Serotonin's indole takes longer.** It climbs from 0.76 at order 2 to 0.94 and 0.98: a larger
  structure, spread across many orders.

Run it again
------------

``python pack.py --demos orders`` in
`docs/source/demos/graphshapiq-common/compute <https://github.com/mmschlk/shapiq/tree/main/docs/source/demos/graphshapiq-common/compute>`__
converts GraphSHAP-IQ's Möbius interactions (``explain.py``) into FSII of every order with
``shapiq``'s ``MoebiusConverter``, and scores each order as ``../orders/compute/faithfulness.py``
does, over all coalitions. The molecules are listed in ``[orders_demo]`` of ``config.toml``.
