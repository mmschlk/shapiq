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

   .. grid-item-card:: GraphSHAP-IQ: which atoms does a graph neural network see together?
      :link: graphshapiq/index
      :link-type: doc
      :img-top: /_static/images/hero_graphshapiq_demo.png
      :img-alt: The GraphSHAP-IQ demo: TNT, with the interactions of its atoms that make a graph neural network call it mutagenic, the nitrogen and oxygens of its nitro groups.

      Is it an indole? Is it mutagenic? The exact Shapley interactions of the atoms of eight
      molecules, from serotonin to TNT, for the GNNs that answer, computed with GraphSHAP-IQ.

.. toctree::
   :hidden:

   fixlip/index
   graphshapiq/index
