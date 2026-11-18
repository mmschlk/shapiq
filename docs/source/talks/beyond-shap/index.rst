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

      An XGBoost model predicts the value of a house in Los Angeles from eight features. SHAP
      splits this prediction among the features. Each feature gets one score, red where it raises
      the prediction and blue where it lowers it. Longitude lowers the value and latitude raises
      it, although neither of them locates the house. Only both together do.

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

.. shapiq-slide:: The Same Scores Can Come from Different Models
   :layout: wide
   :steps: 3
   :cite: Shapley.1951 Grabisch.1999

   .. shapiq-talk-figure:: same-shap
      :alt: Three models of two features. All three give both features the Shapley value 1. The first adds both features, the second needs both together, and the third needs only one of them. Their Shapley interactions are 0, +2 and −2.

   .. shapiq-notes::

      Three models of two features explain the same prediction. SHAP gives all three the same
      explanation, a score of 1 for each feature. Yet the models behave very differently. The
      first adds up the two features. The second predicts something only when both features are
      present. For the third, either feature is enough.

      SHAP shares the joint effect of the two features equally between them, so it disappears
      into their scores. A Shapley interaction is what two features add together beyond what each
      adds alone. It is 0 for the first model, +2 for the second and −2 for the third.

      Formally, an explanation game :math:`\nu` assigns a value to every coalition of features,
      the prediction of the model when only these features are present. The Shapley value of a
      feature :math:`i` averages what :math:`i` adds to every coalition :math:`T` of the other
      features :cite:p:`Shapley.1951`.

      .. math::

         \phi(i) = \sum_{T \subseteq N \setminus \{i\}} \frac{1}{n \binom{n-1}{t}} \big[ \nu(T \cup \{i\}) - \nu(T) \big]

      The Shapley interaction index of a group :math:`S` replaces this marginal contribution by
      the discrete derivative, what the group adds together beyond what its features add alone
      :cite:p:`Grabisch.1999`. For a pair, it is
      :math:`\nu(T \cup ij) - \nu(T \cup i) - \nu(T \cup j) + \nu(T)`.

      .. math::

         \phi(S) = \sum_{T \subseteq N \setminus S} \frac{1}{(n - s + 1) \binom{n-s}{t}} \, \Delta_S \nu(T),
         \qquad \Delta_S \nu(T) = \sum_{L \subseteq S} (-1)^{s - l} \, \nu(T \cup L)

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

.. shapiq-slide:: Every Order Explains More of the Model
   :layout: wide
   :cite: Tsai.2022 Bordt.2023

   .. shapiq-demo:: orders
      :examples: house, applicant, tabpfn

   .. shapiq-notes::

      One prediction can be explained at every order of interaction. The Shapley values sit at
      the top. Below them follow the interactions of order 2, 3 and so on, down to the full
      decomposition, which is the model itself :cite:p:`Bordt.2023`. Faithfulness measures how
      well each order predicts what the model does when only some features are present
      :cite:p:`Tsai.2022`. For the house, the Shapley values reach an R² of 0.36. With the pair of
      latitude and longitude, the explanation fits almost exactly. An XGBoost model of credit
      default shows the same for an applicant who defaulted. The Shapley values reach 0.24 there, and
      the pairs of the credit limit with the bill, and of the bill with the last payment, lift the
      fit to 0.98. TabPFN, a tabular foundation model, values the house of before, and its Shapley
      values explain close to nothing.

   .. shapiq-try::

      The guide :doc:`../../introduction/how-much-explained` builds this figure up step by step
      and ends with the code that computed it.

.. shapiq-slide:: Often the Shapley Values Are Enough
   :layout: wide

   .. shapiq-demo:: orders
      :examples: houses, applicants

   .. shapiq-notes::

      Across 2,000 houses, the Shapley values fit with an R² of 0.79 on average, and across 2,000
      applicants with 0.78. For many predictions, SHAP is a good explanation. The house of the previous slide fits worse than most of its
      neighbours. Whether interactions are worth it thus depends on the prediction, and it can be
      measured instead of guessed.

   .. shapiq-try::

      The guide :doc:`../../introduction/how-much-explained` explains how the fit is measured and
      how the houses were computed.

.. shapiq-slide:: For Tree Ensembles, Interactions Are Exact
   :layout: wide
   :steps: 2
   :cite: Muschalik.2024a Lundberg.2020

   .. container:: shapiq-talk-columns

      .. container:: shapiq-talk-column

         .. code-block:: python

            import shapiq

            explainer = shapiq.TreeExplainer(
                model, index="k-SII", max_order=2
            )
            # one prediction, exactly
            explanation = explainer.explain(x)

         .. container:: shapiq-from-1

            .. code-block:: python

               # every prediction of a test set
               explanations = explainer.explain_X(X_test)

               # what "absent" means is an argument
               shapiq.TreeExplainer(
                   model, mode="interventional",
                   reference_dataset=X_ref,
               )

         .. rst-class:: shapiq-talk-libraries

         XGBoost · LightGBM · CatBoost · scikit-learn

      .. container:: shapiq-talk-column

         .. shapiq-demo:: orders
            :step: full
            :examples: house

   .. shapiq-notes::

      Every explanation of the house so far was computed exactly, with no sampling and no
      approximation. A path in a tree of depth 5 splits on at most five features, so no more than
      five features interact. The layers of the house thus end at order 5, where the explanation
      fits the model exactly. TreeSHAP computes the Shapley values of tree ensembles in
      polynomial time :cite:p:`Lundberg.2020`. TreeSHAP-IQ uses the same structure for Shapley
      interactions of any order :cite:p:`Muschalik.2024a`. In ``shapiq``, it takes three lines for
      XGBoost, LightGBM, CatBoost and scikit-learn models. What it means for a feature to be absent
      is an argument of the explainer, ``mode``.

   .. shapiq-try::

      .. code-block:: python

         from sklearn.datasets import fetch_california_housing
         from xgboost import XGBRegressor

         import shapiq

         X, y = fetch_california_housing(return_X_y=True, as_frame=True)
         model = XGBRegressor(n_estimators=100, max_depth=5).fit(X, y)
         X_test = X.to_numpy()[:200]

         # the exact Shapley interactions of 200 houses, up to pairs
         explainer = shapiq.TreeExplainer(model, index="k-SII", max_order=2)
         portfolio = explainer.explain_X(X_test)
         print(portfolio[0].get_top_k(5))

         # absent features drawn from 100 houses instead of the trees' own distribution
         explainer = shapiq.TreeExplainer(
             model, index="k-SII", max_order=2, mode="interventional", reference_dataset=X_test[:100]
         )
         print(explainer.explain(X_test[0]).get_top_k(5))

.. shapiq-slide:: Similarity Lives Between Image and Text
   :layout: wide
   :cite: Baniecki.2025 Thies.2026

   .. shapiq-demo:: fixlip

   .. shapiq-notes::

      A vision–language encoder such as CLIP or SigLIP 2 scores how well a photo matches a
      caption. The photo alone has no similarity, and neither has the caption. One score per patch
      thus says little. FIxLIP explains the similarity by interactions between image patches and
      text tokens :cite:p:`Baniecki.2025`. The strongest of them link a region of the photo to the
      words that describe it, and they show what a heat map cannot. The word "giraffe" matches a
      dead tree as well.

   .. shapiq-try::

      :doc:`../../demos/fixlip/index` explains FIxLIP step by step and ends with the code of the
      vision–language game and its explanation with ``shapiq``.

.. shapiq-slide:: A Functional Group Is an Interaction
   :layout: wide
   :cite: Muschalik.2025

   .. shapiq-demo:: graphshapiq-explorer

   .. shapiq-notes::

      A graph neural network flags molecules that disturb lab tests. Catechol, a ring with two
      neighbouring OH groups, is flagged. Moving one OH group by one position gives resorcinol,
      with the same atoms in another arrangement, and the answer drops. No score per atom can
      express that. GraphSHAP-IQ computes the Shapley interactions of the atoms exactly
      :cite:p:`Muschalik.2025`. A node only sees its neighbourhood, so the explanation needs far
      fewer evaluations than all :math:`2^n` coalitions, and a molecule of one's own is explained
      in the browser.

   .. shapiq-try::

      :class:`~shapiq.graph.GraphExplainer` explains ``torch_geometric`` models with a
      graph-level output, and ``shapiq.Explainer`` picks it for models with message passing
      layers. See :doc:`../../demos/graphshapiq-explorer/index`.

.. shapiq-slide:: SHAP Shares Interactions Equally. Other Methods Do Not.
   :layout: wide
   :steps: 2
   :cite: Fumagalli.2025

   .. shapiq-talk-figure:: effects
      :alt: The three models of two features again, with three scores for each feature. The pure effect is the feature on its own, 1, 0 and 2. The partial effect is the Shapley value, 1 for all three. The full effect is what the prediction loses when the feature is removed, 1, 2 and 0.

   .. shapiq-notes::

      An explanation with one score per feature has to decide where the joint effects go
      :cite:p:`Fumagalli.2025`. A pure effect ignores them and keeps what a feature does on its
      own. A partial effect shares them, as the Shapley value does. A full effect gives each
      feature all of its joint effects, which is what removing the feature alone measures, as
      occlusion does. For the redundant model, removing one feature changes nothing, so its full
      effect is 0. All three answers are correct. They answer different questions.

   .. shapiq-try::

      .. code-block:: python

         import numpy as np

         import shapiq


         def synergy(coalitions):
             return 2 * coalitions[:, 0] * coalitions[:, 1]


         computer = shapiq.ExactComputer(lambda c: synergy(c.astype(float)), n_players=2)
         print("pure:   ", computer("Moebius", order=2).dict_values)  # main effects (0, 0)
         print("partial:", computer("SV").dict_values)  # Shapley values (1, 1)
         print("full:   ", computer("Co-Moebius", order=2).dict_values)  # full effects (2, 2)

.. shapiq-slide:: Many Methods Differ in Two Choices
   :layout: wide
   :steps: 3
   :cite: Fumagalli.2025

   .. shapiq-talk-figure:: framework
      :alt: A grid of local feature-based explanations. Its columns are how absent features are treated, by a baseline, the marginal or the conditional distribution. Its rows are how higher-order interactions enter, as pure, partial or full effects. Each cell lists the methods that fall into it, for individual, joint and interaction explanations.

   .. shapiq-notes::

      Many feature-based explanations rest on a common foundation :cite:p:`Fumagalli.2025`. Each
      of them makes two choices. The first is how absent features are treated, by a fixed
      baseline, by their marginal distribution or by their distribution given the present
      features. With correlated features, such as income and loan amount, these choices give
      different answers. The second choice is how much of the interactions an explanation takes
      in, as pure, partial or full effects. The variants of SHAP form one row of this grid.
      Occlusion, PDP and ICE sit in other cells. These methods do not compete. Each cell answers
      its own question.

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

.. Check the placements against Figure 1 of the paper.

.. shapiq-slide:: Each Explanation Answers a Different Question

   .. container:: shapiq-talk-answers

      .. list-table::
         :header-rows: 1

         * - Question
           - Explanation
         * - How much does each feature contribute?
           - Shapley values
         * - Which features act together?
           - Shapley interactions
         * - Is that enough for this prediction?
           - Faithfulness of the order
         * - What does a feature do alone, and what if it is removed?
           - Pure and full effects
         * - What does it mean that a feature is absent?
           - Baseline, marginal or conditional

   .. shapiq-notes::

      One explanation does not fit all, because each explanation answers a different question.
      SHAP is a good answer to the first one and no answer to the others. The question thus comes
      first, and the explanation follows. Where features act together, interactions show how, and
      faithfulness says how far to go. For tree ensembles, all of this is exact.

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
