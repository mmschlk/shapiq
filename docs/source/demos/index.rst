Demos
=====

.. rst-class:: shapiq-examples-lead

What ``shapiq`` explains on real models and data. Every demo was computed once with ``shapiq``,
and its page shows the result, interactive where that helps. The scripts that computed it are in
the repository, next to its page, so that you can run it again or adapt it to your own models and
data.

Demos are not :doc:`examples <../examples>`: the examples run with every build of this
documentation and show how to use the API, step by step. A demo is a snapshot of a computation that
takes too long or needs too large a model to run with the docs, and it shows what comes out of it.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: FIxLIP: which words go with which parts of a photo?
      :link: fixlip/index
      :link-type: doc
      :img-top: /_static/images/hero_fixlip_demo.png
      :img-alt: The FIxLIP demo: a giraffe drinking at a pond and its caption, with the regions and words that together raise CLIP's similarity the most.

      How well do CLIP and SigLIP 2 match a photo and its caption, and why? The interactions of
      image regions and words of five COCO photos, explained as in FIxLIP and computed with
      ProxySHAP.

   .. grid-item-card:: From Shapley values to the full decomposition
      :link: ../introduction/how-much-explained
      :link-type: doc
      :img-top: /_static/images/hero_orders_demo.png
      :img-alt: The orders demo on a wide screen: a molecule's GCN prediction explained at every order, as layers from the Shapley values to the full decomposition above the molecule and the model's mechanism; the layer of order 2 is pointed at, its explanation shown on the card beside the layers, its fit to the model (R² = 0.88) on the plot beside them.

      How much of a model does an explanation explain? One prediction, explained at every order of
      interaction, from the Shapley values to the full decomposition, which adds up to the model:
      a molecule, houses, TabPFN and SigLIP 2, the figure built up step by step.

   .. grid-item-card:: GraphSHAP-IQ: which atoms act together?
      :link: graphshapiq-explorer/index
      :link-type: doc
      :img-top: /_static/images/hero_graphshapiq_explorer_demo.png
      :img-alt: The GraphSHAP-IQ explorer: 1,3-dinitrobenzene, each nitrogen joined to its oxygens by red bands, beside the strongest pairs of atoms.

      Three graph neural networks, eight molecules, every molecule explained for every question:
      Shapley values, interactions of up to four atoms, and the Möbius interactions of all orders,
      exactly, with GraphSHAP-IQ.

   .. grid-item-card:: GraphSHAP-IQ Lab: build a molecule, explain it exactly
      :link: graphshapiq-lab/index
      :link-type: doc
      :img-top: /_static/images/hero_graphshapiq_lab_demo.png
      :img-alt: The GraphSHAP-IQ lab: catechol, a benzene ring with two neighbouring OH groups, judged a PAINS by a GNN, its atoms joined by red bands.

      Put groups on a benzene ring, and your browser computes the exact Shapley interactions of the
      molecule's atoms with GraphSHAP-IQ while you watch: two neighbouring OH groups make a PAINS,
      two apart do not.

   .. grid-item-card:: GraphSHAP-IQ: exact interactions, and still feasible
      :link: graphshapiq-exact/index
      :link-type: doc
      :img-top: /_static/images/hero_graphshapiq_exact_demo.png
      :img-alt: Cholesterol with the receptive field of one atom highlighted in violet, beside 2^28 = 268,435,456 maskings against the 129,664 that GraphSHAP-IQ evaluates.

      Why exact Shapley interactions of a graph neural network do not take 2ⁿ evaluations: point
      at an atom to see its receptive field, and how many maskings GraphSHAP-IQ needs instead.

   .. grid-item-card:: GraphSHAP-IQ: one prediction at every order
      :link: graphshapiq-orders/index
      :link-type: doc
      :img-top: /_static/images/hero_graphshapiq_orders_demo.png
      :img-alt: 1,3-dinitrobenzene explained by pairs of atoms, with the curve of faithfulness climbing from below zero for the Shapley values to one for the full decomposition.

      A molecule explained at every order, from the Shapley values to the full decomposition, and
      how faithful each order is: pairs suffice for a nitro group, a catechol needs groups of three.

.. toctree::
   :hidden:

   fixlip/index
   graphshapiq-explorer/index
   graphshapiq-lab/index
   graphshapiq-exact/index
   graphshapiq-orders/index
