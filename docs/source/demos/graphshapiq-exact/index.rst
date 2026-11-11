GraphSHAP-IQ: Exact Interactions, and Still Feasible
====================================================

.. rst-class:: shapiq-examples-lead

Exact Shapley interactions of :math:`n` players take :math:`2^n` evaluations of the model: beyond a
few dozen players, out of reach. For a graph neural network (GNN), GraphSHAP-IQ
:cite:p:`Muschalik.2025` computes them exactly from a small fraction of them. Point at an atom to
see why.

.. shapiq-demo:: graphshapiq-exact
   :fallback: images/hero_graphshapiq_exact_demo.png
   :alt: Cholesterol with the receptive field of one of its atoms highlighted in violet; beside it, 2^28 = 268,435,456 maskings against the 129,664 that GraphSHAP-IQ evaluates for a GNN with three layers.

The idea
--------

A GNN with :math:`\ell` layers passes messages along bonds :math:`\ell` times, so what it computes
for an atom depends only on the atoms within :math:`\ell` bonds, the atom's **receptive field**. If
the GNN sums its atoms up and reads the sum out linearly, as most GNNs for molecules do, two atoms
that never share a receptive field cannot influence each other: their interaction, and that of
every group that holds both, is exactly zero :cite:p:`Muschalik.2025`. GraphSHAP-IQ evaluates the
GNN only on the maskings within receptive fields, computes their Möbius interactions, and from
them every index of every order, exactly.

What it saves
-------------

The numbers are those of :class:`~shapiq.graph.GraphSHAPIQ`, which plans its maskings before it
evaluates any. For cholesterol, 28 atoms, all maskings number :math:`2^{28}`, more than a quarter of
a billion. A GNN with three layers needs 129,664 of them, 2,070 times fewer; a GNN with two layers,
5,509; one with four, 1,384,448. The fewer the layers and the larger the molecule, the larger the share of interactions that
are exactly zero. For benzene, a three-layer GNN's receptive fields cover the whole ring, and
GraphSHAP-IQ evaluates all 64 maskings: exact, but no cheaper.

Run it again
------------

``python pack.py --demos exact`` in
`docs/source/demos/graphshapiq-common/compute <https://github.com/mmschlk/shapiq/tree/main/docs/source/demos/graphshapiq-common/compute>`__
writes this demo's ``data.js``; its molecules are the ``[[exact_demo]]`` entries of ``config.toml``.
No GNN is evaluated: the maskings depend only on the molecule's graph and the number of layers.
