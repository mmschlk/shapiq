.. The talk "One Explanation Does Not Fit All", as slides (_ext/shapiq_talks.py, talks/README.md).
   Read on the page, it is a handout with notes. "Present" or the key P shows one slide at a time.
   Its own figures are drawn by assets/figures.js. The demos are those of demos/.

One Explanation Does Not Fit All
================================

.. rst-class:: shapiq-examples-lead

A talk on post-hoc, local explanations from a game-theoretic perspective. It looks at what we gain
by going beyond SHAP to higher-order explanations such as Shapley interactions, and at the common
foundation of many explanation methods. Every slide is part of this page and runs live in the
browser, with the demos of these docs.

.. shapiq-slide:: One Explanation Does Not Fit All
   :layout: title

   .. rst-class:: shapiq-talk-subtitle

   Going Beyond SHAP to Higher-Order Explanations

   .. rst-class:: shapiq-talk-speaker

   | **Maximilian Muschalik**
   | LMU Munich and Munich Center for Machine Learning (MCML)

   .. shapiq-notes::

      Introduce yourself in one sentence. Postdoc at the chair of Eyke Hüllermeier. Shapley values
      and interactions, and the maintainer of ``shapiq``, the library behind every demo of this
      talk.

.. shapiq-slide:: SHAP Gives One Score per Feature
   :layout: wide

   .. shapiq-demo:: orders
      :step: shapley
      :examples: house

   .. shapiq-notes::

      SHAP has become the standard tool to explain a single prediction. A house in California
      and the price that XGBoost predicts for it. SHAP splits the
      prediction among the features. Red raises the price, blue lowers it. Latitude and
      longitude get two separate scores, although only together they locate the house.

.. shapiq-slide:: One Score per Feature Can Hide the Model
   :layout: wide
   :steps: 3

   .. shapiq-talk-figure:: same-shap
      :alt: Three models of two features. All three give both features the Shapley value 1. The first adds both features, the second needs both together, and the third needs only one of them. Their Shapley interactions are 0, +2 and −2.

   .. shapiq-notes::

      Step 1. Three models, two features, one prediction each. SHAP gives the same explanation
      for all three. Both features get a score of 1.

      Step 2. The models behave very differently. The first adds the features. The second
      predicts something only when both are present. For the third, either feature is enough.

      Step 3. Shapley interactions tell them apart. The joint effect of both features is 0, +2 or
      −2. SHAP has spread it evenly across the two features. Similar explanations can stem from
      very different model behaviour.

.. shapiq-slide:: Shapley Interactions
   :steps: 2

   The Shapley value of a feature :math:`i` averages what :math:`i` adds to every coalition
   :math:`T` of the other features.

   .. math::

      \phi(i) = \sum_{T \subseteq N \setminus \{i\}} \frac{1}{n \binom{n-1}{t}} \big[ \nu(T \cup \{i\}) - \nu(T) \big]

   .. container:: shapiq-from-1

      The Shapley interaction of a group :math:`S` averages what the features of :math:`S` add
      *together*, beyond what they add alone :cite:p:`Grabisch.1999`.

      .. math::

         \phi(S) = \sum_{T \subseteq N \setminus S} \frac{1}{(n - s + 1) \binom{n-s}{t}} \, \Delta_S \nu(T),
         \qquad \Delta_S \nu(T) = \sum_{L \subseteq S} (-1)^{s - l} \, \nu(T \cup L)

   .. shapiq-notes::

      Keep it short. The game :math:`\nu` is the model's prediction with only some features
      present. The discrete derivative :math:`\Delta_S` is what the group adds together. For a
      pair, it is :math:`\nu(T \cup ij) - \nu(T \cup i) - \nu(T \cup j) + \nu(T)`.
      Draft slide. Decide whether to show the formulas or only the words.

.. shapiq-slide:: How Much Does an Explanation Explain?
   :layout: wide

   .. shapiq-demo:: orders
      :examples: house, molecule, fixlip

   .. shapiq-notes::

      One prediction, explained at every order of interaction. The plot on the side is the fit
      to the model, :math:`R^2`. The pairs alone recover most of what the Shapley values miss.
      Going higher is worth it where the fit still rises. This is the question of when going
      beyond SHAP pays off.

.. shapiq-slide:: Higher-Order Explanations in Practice
   :layout: section

   .. rst-class:: shapiq-talk-statement

   Vision–language encoders and graph neural networks, explained by interactions.

   .. shapiq-notes::

      Two models where single features say little. Words and image regions only match together.
      Atoms only form a functional group together.

.. shapiq-slide:: Which Words Go with Which Parts of a Photo?
   :layout: wide

   .. shapiq-demo:: fixlip

   .. shapiq-notes::

      FIxLIP :cite:p:`Baniecki.2025`. The similarity of an image–text pair is explained by
      interactions between image patches and words. Pick another example, and show the clique
      slider.

.. shapiq-slide:: Which Atoms Act Together?
   :layout: wide

   .. shapiq-demo:: graphshapiq-explorer

   .. shapiq-notes::

      GraphSHAP-IQ :cite:p:`Muschalik.2025`. Exact Shapley interactions of the atoms of a
      molecule, for a graph neural network. Build a molecule live and let the browser explain it.

.. shapiq-slide:: A Common Foundation
   :layout: section

   .. rst-class:: shapiq-talk-statement

   Many popular explanation methods differ in only **two choices**.

.. shapiq-slide:: A Unified Framework for Game-Theoretic XAI
   :layout: wide
   :steps: 3

   .. shapiq-talk-figure:: framework
      :alt: A grid of local feature-based explanations. Its columns are how absent features are treated, by a baseline, the marginal or the conditional distribution. Its rows are how higher-order interactions enter, as pure, partial or full effects. Each cell lists the methods that fall into it, for individual, joint and interaction explanations.

   .. shapiq-notes::

      Every explanation fixes two things :cite:p:`Fumagalli.2025`. First, how absent features
      are treated. This is the fANOVA decomposition, with a baseline, the marginal or the
      conditional distribution. Second, how much of the higher-order interactions it takes in.
      Pure effects take none, full effects all of them, and partial effects a share, as the
      Shapley value does. SHAP and its variants are one row of this grid.
      Draft slide. Check the placements against Figure 1 of the paper.

.. shapiq-slide:: Which Explanation Answers Which Question?

   .. container:: shapiq-talk-answers

      .. list-table::
         :header-rows: 1

         * - Question
           - Explanation
         * - How much does each feature contribute?
           - Shapley values
         * - Which features act together?
           - Shapley interactions
         * - How exactly does the model combine them?
           - Möbius interactions
         * - How do absent features behave?
           - The choice of the game

   .. rst-class:: shapiq-talk-draft

   Draft

   .. shapiq-notes::

      The closing argument. One explanation does not fit all, because each one answers a
      different question. Draft slide. The rows are a first proposal.

.. shapiq-slide:: Thank You
   :layout: title

   .. rst-class:: shapiq-talk-subtitle

   Every demo of this talk is computed with ``shapiq``.

   .. code-block:: bash

      pip install shapiq

   .. rst-class:: shapiq-talk-speaker

   | `github.com/mmschlk/shapiq <https://github.com/mmschlk/shapiq>`_
   | `shapiq.readthedocs.io <https://shapiq.readthedocs.io>`_
