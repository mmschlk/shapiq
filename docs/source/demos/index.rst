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
      exactly, with GraphSHAP-IQ, and why that does not take 2ⁿ evaluations.

   .. grid-item-card:: GraphSHAP-IQ Lab: build a molecule, explain it exactly
      :link: graphshapiq-lab/index
      :link-type: doc
      :img-top: /_static/images/hero_graphshapiq_lab_demo.png
      :img-alt: The GraphSHAP-IQ lab: catechol, a benzene ring with two neighbouring OH groups, judged a PAINS by a GNN, its atoms joined by red bands.

      Put groups on a benzene ring, and your browser computes the exact Shapley interactions of the
      molecule's atoms with GraphSHAP-IQ while you watch: two neighbouring OH groups make a PAINS,
      two apart do not.

.. toctree::
   :hidden:

   fixlip/index
   graphshapiq-explorer/index
   graphshapiq-lab/index
