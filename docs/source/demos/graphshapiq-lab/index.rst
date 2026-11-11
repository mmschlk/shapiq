GraphSHAP-IQ Lab: Build a Molecule, Explain It Exactly
======================================================

.. rst-class:: shapiq-examples-lead

Put groups on a benzene ring and a graph neural network (GNN) tells whether the molecule is a PAINS
or a mutagen. Your browser then computes the **exact** Shapley interactions of its atoms with
GraphSHAP-IQ :cite:p:`Muschalik.2025`, the algorithm of ``shapiq``, while you watch: nothing is
precomputed, and nothing is estimated.

.. shapiq-demo:: graphshapiq-lab
   :fallback: images/hero_graphshapiq_lab_demo.png
   :alt: Catechol, a benzene ring with two neighbouring OH groups, explained by a GNN for the question whether it is a PAINS: the oxygens and the carbons between them are joined by red bands.

Things to try
-------------

- **Make a catechol, then break it.** The example opens on catechol, a benzene ring with two
  neighbouring OH groups: the *PAINS?* GNN is sure (more than 99%). Move one OH a position further,
  to resorcinol, and the answer drops to 11%; to the far side of the ring, hydroquinone, 25%. The
  same atoms, in another arrangement: the GNN learned a pair, not two groups, and the interactions
  show it.
- **Watch an atom change sides.** Under *Atoms alone*, every atom of the catechol gets a Shapley
  value; under *Atoms together*, the ring around an atom shows what it adds on its own, which for the
  catechol's atoms is negative, and the bands show what they add in pairs, which is positive.
- **Add a nitro group.** Under *Mutagenic?*, a single NO\ :sub:`2` puts strong red bands between its
  nitrogen and each of its oxygens: the structural alert the GNN learned from the Ames test
  :cite:p:`Kazius.2005`.

How it works
------------

Each edit builds the molecule's graph and hands it to a Web Worker, which runs the GNN and
GraphSHAP-IQ. A GNN with three layers mixes an atom only with the atoms within three bonds, so atoms
that never share such a neighbourhood cannot interact, and GraphSHAP-IQ needs the GNN only on the
maskings within neighbourhoods (:doc:`../graphshapiq-exact/index` shows how much that saves for
large molecules). The lab's molecules are small, their neighbourhoods cover them, and GraphSHAP-IQ
evaluates the GNN on all :math:`2^n` maskings: up to 32,768, a few seconds at most, shown by the bar
under the molecule. From these values, the worker computes the Möbius interactions, the Shapley
values, and the 2-SII :cite:p:`Bordt.2023`, exactly as ``shapiq``'s ``MoebiusConverter`` does; the
browser's numbers agree with the Python package's to :math:`10^{-5}` (``check_lab.py`` checks it).
The lab allows at most 15 heavy atoms, so that the computation stays interactive.

The GNNs are those of :doc:`../graphshapiq-explorer/index`: B-XAIC's GIN :cite:p:`Xu.2019`, trained
on B-XAIC :cite:p:`Proszewska.2025` for *PAINS?* and on the Mutagenicity dataset
:cite:p:`Morris.2020` for *Mutagenic?*, written into the page as chains of affine maps. A masked atom
keeps its bonds, and takes the features of the average atom of the GNN's training molecules.

Run it again
------------

The GNNs and the lab's data come from the scripts in
`docs/source/demos/graphshapiq-common/compute <https://github.com/mmschlk/shapiq/tree/main/docs/source/demos/graphshapiq-common/compute>`__
(``python pack.py --demos lab`` writes the lab's ``data.js``). The lab itself, the groups it
offers and the examples, is ``assets/demo.js`` of this page's folder.
