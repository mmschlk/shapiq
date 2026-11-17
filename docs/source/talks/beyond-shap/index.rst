.. The talk "One Explanation Does Not Fit All", as slides (_ext/shapiq_talks.py, talks/README.md).
   Read on the page, it is a handout: every slide with its notes in prose, code to try and the
   references it cites. "Present" or the key P shows one slide at a time, and a phone scrolls.
   Its own figures are drawn by assets/figures.js. The demos are those of demos/.
   assets/qr-deck.svg opens this page as a deck (?present); README.md says how to make it again.

One Explanation Does Not Fit All
================================

.. rst-class:: shapiq-examples-lead

A talk on post-hoc, local explanations from a game-theoretic perspective. It looks at what we gain
by going beyond SHAP to higher-order explanations such as Shapley interactions, and at the common
foundation of many explanation methods. Every slide is part of this page and runs live in the
browser, with the demos of these docs. Below each slide, its notes say in words what the talk
says, and code reproduces what the slide shows.

.. Once the talk is given, name the recording (a URL that takes t=<seconds>) and the event here, and
   give the slides a :recording: time each (m:ss).

.. shapiq-talk::

.. shapiq-slide:: One Explanation Does Not Fit All
   :layout: title

   .. rst-class:: shapiq-talk-subtitle

   Going Beyond SHAP to Higher-Order Explanations

   .. rst-class:: shapiq-talk-speaker

   | **Maximilian Muschalik**
   | LMU Munich and Munich Center for Machine Learning (MCML)

   .. container:: shapiq-talk-qr

      .. image:: assets/qr-deck.svg
         :alt: A QR code that opens this talk on shapiq.readthedocs.io.

      Follow along on your phone

   .. shapiq-notes::

      SHAP has become the standard tool to explain individual predictions of machine learning
      models. This talk asks what one score per feature hides, what higher-order explanations such
      as Shapley interactions reveal, and when they are worth it. Every demo of the talk is
      computed with ``shapiq``, an open-source Python library for Shapley-based explanations.

.. shapiq-slide:: SHAP Gives One Score per Feature
   :layout: wide
   :cite: Lundberg.2017

   .. shapiq-demo:: orders
      :step: shapley
      :examples: house

   .. shapiq-notes::

      A model predicts the value of a house in California from eight features. SHAP splits this
      prediction among the features. Each feature gets one score, red where it raises the
      prediction and blue where it lowers it. Latitude and longitude get a score each, although
      only together they locate the house.

   .. shapiq-try::

      .. code-block:: python

         from sklearn.datasets import fetch_california_housing
         from xgboost import XGBRegressor

         import shapiq

         X, y = fetch_california_housing(return_X_y=True, as_frame=True)
         model = XGBRegressor(n_estimators=100, max_depth=4).fit(X, y)
         x = X.iloc[0].to_numpy()

         # the exact Shapley values of one house
         explainer = shapiq.TreeExplainer(model, index="SV", max_order=1)
         shapley_values = explainer.explain(x)
         shapley_values.plot_force(feature_names=list(X.columns))

.. shapiq-slide:: One Score per Feature Can Hide the Model
   :layout: wide
   :steps: 3

   .. shapiq-talk-figure:: same-shap
      :alt: Three models of two features. All three give both features the Shapley value 1. The first adds both features, the second needs both together, and the third needs only one of them. Their Shapley interactions are 0, +2 and −2.

   .. shapiq-notes::

      Three models of two features explain the same prediction. SHAP gives all three the same
      explanation, a score of 1 for each feature. Yet the models behave very differently. The
      first adds up the two features. The second predicts something only when both features are
      present. For the third, either feature is enough.

      Shapley interactions tell the three models apart. The joint effect of both features is 0
      for the first model, +2 for the second and −2 for the third. SHAP spreads this joint effect
      evenly across the two features. Similar explanations can thus stem from very different
      model behaviour.

   .. shapiq-try::

      .. code-block:: python

         import numpy as np

         import shapiq


         # the three models as games of two features: the prediction when only the features in a
         # coalition keep their value 1 and the others take the baseline 0
         def additive(coalitions):
             return coalitions[:, 0] + coalitions[:, 1]


         def synergy(coalitions):
             return 2 * coalitions[:, 0] * coalitions[:, 1]


         def redundancy(coalitions):
             return 2 * np.maximum(coalitions[:, 0], coalitions[:, 1])


         for model in (additive, synergy, redundancy):
             computer = shapiq.ExactComputer(lambda c, f=model: f(c.astype(float)), n_players=2)
             print(model.__name__)
             print("  Shapley values:      ", computer("SV").dict_values)
             print("  Shapley interactions:", computer("k-SII", order=2).dict_values)

.. shapiq-slide:: Shapley Interactions
   :steps: 2
   :cite: Shapley.1951 Grabisch.1999 Bordt.2023

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

      An explanation game :math:`\nu` assigns a value to every coalition of features, the
      prediction of the model when only these features are present. The Shapley value of a
      feature averages its marginal contributions over all coalitions of the other features. The
      Shapley interaction index does the same for a group of features. It replaces the marginal
      contribution by the discrete derivative, which is what the group adds together beyond what
      its features add alone. For a pair, this is
      :math:`\nu(T \cup ij) - \nu(T \cup i) - \nu(T \cup j) + \nu(T)`. Explanations of order
      :math:`k`, such as the k-SII, then split the prediction among all groups of up to
      :math:`k` features :cite:p:`Bordt.2023`.

   .. shapiq-try::

      .. code-block:: python

         from sklearn.datasets import fetch_california_housing
         from xgboost import XGBRegressor

         import shapiq

         X, y = fetch_california_housing(return_X_y=True, as_frame=True)
         model = XGBRegressor(n_estimators=100, max_depth=4).fit(X, y)
         x = X.iloc[0].to_numpy()

         # the exact Shapley interactions of one house, up to pairs
         explainer = shapiq.TreeExplainer(model, index="k-SII", max_order=2)
         interactions = explainer.explain(x)
         print(interactions.get_top_k(5))
         interactions.plot_si_graph(feature_names=list(X.columns))

.. Draft slide. Decide whether to show the formulas or only the words.

.. shapiq-slide:: How Much Does an Explanation Explain?
   :layout: wide
   :cite: Tsai.2022 Grabisch.1999

   .. shapiq-demo:: orders
      :examples: house, molecule, fixlip

   .. shapiq-notes::

      One prediction can be explained at every order of interaction. The Shapley values sit at
      the top. Below them follow the Shapley interactions of order 2, 3 and so on, down to the
      full decomposition, which is the model itself. Faithfulness measures how well each order
      predicts what the model does when only some features are present. The pairs alone recover
      most of what the Shapley values miss. Going to higher orders is worth it as long as the fit
      still rises.

   .. shapiq-try::

      The guide :doc:`../../introduction/how-much-explained` builds this figure up step by step
      and ends with the code that computed it.

.. shapiq-slide:: Higher-Order Explanations in Practice
   :layout: section

   .. rst-class:: shapiq-talk-statement

   Vision–language encoders and graph neural networks, explained by interactions.

   .. shapiq-notes::

      Two kinds of models show where single features say little. In a vision–language encoder,
      words and image regions match only together. In a graph neural network on molecules, atoms
      form a functional group only together.

.. shapiq-slide:: Which Words Go with Which Parts of a Photo?
   :layout: wide
   :cite: Baniecki.2025 Thies.2026

   .. shapiq-demo:: fixlip

   .. shapiq-notes::

      FIxLIP explains the similarity that a vision–language encoder such as CLIP or SigLIP 2
      predicts for an image–text pair :cite:p:`Baniecki.2025`. Its explanation consists of
      interactions between image patches and text tokens. The strongest of them link a region of
      the photo to the words that describe it. The demo shows five examples from MS COCO.

   .. shapiq-try::

      :doc:`../../demos/fixlip/index` explains FIxLIP step by step and ends with the code of the
      vision–language game and its explanation with ``shapiq``.

.. shapiq-slide:: Which Atoms Act Together?
   :layout: wide
   :cite: Muschalik.2025

   .. shapiq-demo:: graphshapiq-explorer

   .. shapiq-notes::

      GraphSHAP-IQ computes the Shapley interactions of the atoms of a molecule exactly, for a
      graph neural network :cite:p:`Muschalik.2025`. A node only sees its neighbourhood, so the
      explanation needs far fewer evaluations than all :math:`2^n` coalitions. The explorer shows
      eight molecules and three questions. A molecule of one's own is explained in the browser.

   .. shapiq-try::

      :class:`~shapiq.graph.GraphExplainer` explains ``torch_geometric`` models with a
      graph-level output, and ``shapiq.Explainer`` picks it for models with message passing
      layers. See :doc:`../../demos/graphshapiq-explorer/index`.

.. shapiq-slide:: A Common Foundation
   :layout: section

   .. rst-class:: shapiq-talk-statement

   Many popular explanation methods differ in only **two choices**.

.. shapiq-slide:: A Unified Framework for Game-Theoretic XAI
   :layout: wide
   :steps: 3
   :cite: Fumagalli.2025

   .. shapiq-talk-figure:: framework
      :alt: A grid of local feature-based explanations. Its columns are how absent features are treated, by a baseline, the marginal or the conditional distribution. Its rows are how higher-order interactions enter, as pure, partial or full effects. Each cell lists the methods that fall into it, for individual, joint and interaction explanations.

   .. shapiq-notes::

      Many feature-based explanations rest on a common foundation :cite:p:`Fumagalli.2025`. Each
      of them makes two choices. The first is how absent features are treated, by a fixed
      baseline, by their marginal distribution or by their distribution given the present
      features. Each choice corresponds to a functional ANOVA decomposition of the model. The
      second choice is how much of the higher-order interactions an explanation takes in. Pure
      effects take in none of them, full effects all of them, and partial effects a share, as the
      Shapley value does. SHAP and its variants form one row of this grid.

   .. shapiq-try::

      .. code-block:: python

         from sklearn.datasets import fetch_california_housing
         from xgboost import XGBRegressor

         import shapiq

         X, y = fetch_california_housing(return_X_y=True, as_frame=True)
         model = XGBRegressor(n_estimators=100, max_depth=4).fit(X, y)
         data, x = X.to_numpy()[:100], X.iloc[0].to_numpy()

         # the columns: how absent features are treated (baseline, marginal or conditional)
         for imputer in ("baseline", "marginal", "conditional"):
             explainer = shapiq.TabularExplainer(model, data=data, imputer=imputer, index="SV")
             print(imputer, explainer.explain(x, budget=256).dict_values)

         # the rows: how much of the interactions an explanation takes in (pure, partial or full)
         game = shapiq.TabularExplainer(model, data=data, imputer="marginal").imputer
         game.fit(x)
         computer = shapiq.ExactComputer(game, n_players=X.shape[1])
         for index in ("Moebius", "SII", "Co-Moebius"):
             print(index, computer(index, order=2).get_top_k(3).dict_values)

.. Draft slide. Check the placements against Figure 1 of the paper.

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

      One explanation does not fit all, because each explanation answers a different question.
      Shapley values say how much each feature contributes. Shapley interactions say which
      features act together. The Möbius interactions say exactly how the model combines them.
      And the choice of the game says what happens to the features that are absent.

.. Draft slide. The rows are a first proposal.

.. shapiq-slide:: Thank You
   :layout: title
   :cite: Muschalik.2024

   .. rst-class:: shapiq-talk-subtitle

   Every demo of this talk is computed with ``shapiq``.

   .. code-block:: bash

      pip install shapiq

   .. rst-class:: shapiq-talk-speaker

   | `github.com/mmschlk/shapiq <https://github.com/mmschlk/shapiq>`_
   | `shapiq.readthedocs.io <https://shapiq.readthedocs.io>`_
