.. The talk "One Explanation Does Not Fit All", as slides (_ext/shapiq_talks.py, talks/README.md).
   Read on the page, it is a handout: every slide with its notes in prose, code to try and the
   references it cites. "Present" or the key P shows one slide at a time, and a phone scrolls.
   Its own figures are drawn by assets/figures.js. The demos are those of demos/.
   assets/qr-deck.svg opens this page as a deck (?present, off the title slide until the page is
   online), assets/qr-shapiq.svg the repository of
   shapiq (on "Thank You"); README.md says how to make them again. The Appendix after
   "Thank You" holds slides for questions. appendix-parked.txt holds more, set aside for now.
   assets/probly/ holds the logo of probly (MIT, github.com/pwhofman/probly) and a QR code of its
   repository, made like qr-deck.svg (README.md).

One Explanation Does Not Fit All
================================

.. rst-class:: shapiq-examples-lead

A talk on post-hoc, local explanations from a game-theoretic perspective. It looks at what
higher-order explanations such as Shapley interactions add beyond SHAP, and at the common
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

   .. container:: shapiq-talk-logo

      .. image:: /_static/logo/logo_shapiq_light.svg
         :class: shapiq-hero-logo only-light
         :alt: The shapiq logo: a network of players, linked by their interactions.

      .. image:: /_static/logo/logo_shapiq_dark.svg
         :class: shapiq-hero-logo only-dark
         :alt: The shapiq logo: a network of players, linked by their interactions.

   .. shapiq-notes::

      SHAP has become the standard tool to explain individual predictions of machine learning
      models. This talk asks what one score per feature hides, what higher-order explanations such
      as Shapley interactions reveal, and when they are worth it. Every demo of the talk is
      computed with ``shapiq``, an open-source Python library for Shapley-based explanations.

.. shapiq-slide:: SHAP Explains Any Model
   :layout: wide
   :steps: 2
   :cite: Lundberg.2017 Baniecki.2025

   .. container:: shapiq-talk-columns shapiq-talk-teaser

      .. container:: shapiq-talk-column

         .. shapiq-demo:: orders
            :step: teaser
            :examples: houses

      .. container:: shapiq-talk-column

         .. shapiq-demo:: orders
            :step: teaser
            :examples: fixlip
            :pick: giraffe

      .. The "i" above the examples opens this panel on top of them (talk.js, talk.css). The picture
         is the figure "encoders" of the FIxLIP demo, for this pair and SigLIP 2, light and dark.

      .. container:: shapiq-talk-about

         .. rubric:: How SigLIP 2 Works

         .. image:: assets/siglip-encoders-light.webp
            :class: only-light
            :alt: The photo of the giraffe in 64 patches goes into a vision encoder, and its caption in 19 tokens into a language encoder. Each encoder returns an embedding of 768 values. The cosine similarity of the two embeddings is 0.170.

         .. image:: assets/siglip-encoders-dark.webp
            :class: only-dark
            :alt: The photo of the giraffe in 64 patches goes into a vision encoder, and its caption in 19 tokens into a language encoder. Each encoder returns an embedding of 768 values. The cosine similarity of the two embeddings is 0.170.

         SigLIP 2 is a vision–language encoder :cite:p:`Baniecki.2025`. A vision encoder maps the
         patches of the photo to an embedding. A language encoder maps the tokens of the caption to
         another. The model predicts how well photo and caption match as the cosine similarity of
         the two embeddings.

   .. container:: shapiq-talk-definition shapiq-talk-attribution shapiq-from-1

      .. rubric:: Feature Attribution.

      A feature attribution summarizes :math:`f` at :math:`x` by one score :math:`\phi_i` for each
      feature :math:`i \in D`, so that

      .. math::

         f(x) \;\approx\; \phi_\emptyset + \sum_{i \in D} \phi_i .

   .. shapiq-notes::

      Two models, two black boxes. An XGBoost model values 2,000 houses of California. SigLIP 2
      matches a photo of a giraffe to its caption. SHAP explains both in the same way
      :cite:p:`Lundberg.2017`. Every feature gets one score, a house on the map, a patch of the
      photo or a word of the caption. Red raises the output, and blue lowers it. What SHAP gives is a
      feature attribution, one score for each feature that adds up to the prediction.

.. shapiq-slide:: The Shapley Value
   :layout: wide
   :steps: 3
   :cite: Shapley.1951 Lundberg.2017

   .. container:: shapiq-talk-columns shapiq-talk-columns-theory

      .. container:: shapiq-talk-column

         .. container:: shapiq-talk-definition

            .. rubric:: Definition 1 (Cooperative Game).

            A **cooperative game** is a pair :math:`(D, \nu)` of a **player set**
            :math:`D = \{1, \dots, d\}` and a **value function**
            :math:`\nu: \mathcal{P}(D) \to \mathbb{R}` that assigns a **payoff** to each
            **coalition** :math:`T \subseteq D`.

         .. container:: shapiq-talk-definition shapiq-from-1

            .. rubric:: Definition 2 (Shapley Value).

            The Shapley value of a player :math:`i \in D` is a weighted average of its **marginal
            contributions** :math:`\Delta_i(T) = \nu(T \cup \{i\}) - \nu(T)`,

            .. math::

               \phi^{\text{SV}}_i(\nu) = \sum_{T \subseteq D \setminus \{i\}}
               \frac{1}{d \binom{d-1}{|T|}}\, \Delta_i(T).

      .. container:: shapiq-talk-column

         .. shapiq-talk-figure:: lattice-sv
            :alt: The lattice of all coalitions of four players of the giraffe example, two patches of the photo and two words of the caption. The empty coalition lies at the bottom, the full one at the top, and every edge adds one player, labelled with its marginal contribution. The Shapley value of a player averages the marginal contributions on its edges.

   .. shapiq-notes::

      Cooperative game theory asks how to share the payoff of a team among its players. A game is
      a set of players and a value function, which gives each coalition of players its payoff
      :cite:p:`Shapley.1951`. For the giraffe, the players are patches of the photo and words of
      the caption, and the payoff is how well the photo matches the caption with only these
      players present. All coalitions form a lattice, from the empty one at the bottom to the full
      one at the top. Every edge adds one player, and the change of the payoff along it is the
      player's marginal contribution.

      The Shapley value of a player averages its marginal contributions over all edges that add
      it, weighted so that every size of coalition counts the same. It is the only way to share
      that is efficient, symmetric, linear and gives nothing to a player that never adds anything.
      SHAP computes it for the features of a model :cite:p:`Lundberg.2017`.

.. shapiq-slide:: The Hidden Mechanism of the Model
   :layout: wide
   :cite: Lundberg.2017 Baniecki.2025

   .. shapiq-demo:: orders
      :step: shapley
      :examples: houses, fixlip
      :pick: giraffe

   .. shapiq-notes::

      The Shapley values float above the input as a sheet. Below the input lies the mechanism they
      explain, the model itself. For the houses, it is a tree ensemble. For the giraffe, it is
      SigLIP 2, a vision–language encoder. The plot beside the sheet says how much of the model
      the Shapley values explain. For the houses, that is most of it. For the giraffe, it is much
      less.

.. shapiq-slide:: Faithfulness of Shapley Value Explanations
   :layout: wide
   :class: shapiq-slide--top
   :cite: Tsai.2022 Hedstrom.2023 Agarwal.2022

   .. shapiq-talk-figure:: faithfulness
      :class: talk-faith-first
      :alt: How faithfulness is computed, for a credit applicant. Left, some features are kept and the others removed: the model predicts −1.10 in log-odds with only the credit limit and the bill of September, while the Shapley values add up to −0.57. Right, the same for all 254 coalitions: the points scatter widely around the diagonal, with an R² of 0.24.

   .. container:: shapiq-talk-also

      Other properties measured by Quantus and OpenXAI

      .. rst-class:: shapiq-talk-properties

      | Robustness
      | Complexity
      | Localization
      | Randomization
      | Axiomatic properties

   .. shapiq-notes::

      An explanation is a claim about the model, and a good one predicts what the model does.
      Faithfulness measures this in three steps. First, keep some features and remove the others,
      a coalition :math:`S`. Second, ask the model, :math:`\nu(S)`, and ask the explanation,
      which adds up its values over all groups of kept features, :math:`\hat{\nu}_k(S)`. With
      only the credit limit and the bill of September, the model predicts −1.10 in log-odds. The
      Shapley values add up to −0.57. The value of the bill also holds its shares of interactions
      with features that were removed, such as the payment of August. Third, repeat this
      for all coalitions and compare. Weighted as the Shapley value weighs coalitions, this gives
      an :math:`R^2` :cite:p:`Tsai.2022`. Here it is measured beyond what the number of kept
      features alone predicts. For the applicant, the Shapley values reach only 0.24.

      Faithfulness is one property among several. An explanation can also be robust to small
      changes of the input, simple, localized on the relevant features, and sensitive to the model
      it explains. Toolkits such as Quantus :cite:p:`Hedstrom.2023` and OpenXAI
      :cite:p:`Agarwal.2022` measure these properties.

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

         \phi_i = \sum_{T \subseteq D \setminus \{i\}} \frac{1}{d \binom{d-1}{t}} \big[ \nu(T \cup \{i\}) - \nu(T) \big]

      The Shapley interaction index of a group :math:`S` replaces this marginal contribution by
      the discrete derivative, what the group adds together beyond what its features add alone
      :cite:p:`Grabisch.1999`. For a pair, it is
      :math:`\nu(T \cup ij) - \nu(T \cup i) - \nu(T \cup j) + \nu(T)`.

      .. math::

         \phi_S = \sum_{T \subseteq D \setminus S} \frac{1}{(d - s + 1) \binom{d-s}{t}} \, \Delta_S \nu(T),
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

.. shapiq-slide:: Shapley Interactions Extend the Shapley Value
   :layout: wide
   :steps: 3
   :cite: Grabisch.1999 Bordt.2023 Tsai.2022

   .. container:: shapiq-talk-columns shapiq-talk-columns-theory

      .. container:: shapiq-talk-column

         .. container:: shapiq-talk-definition

            .. rubric:: Definition 3 (Shapley Interactions).

            Shapley interactions average **discrete derivatives**
            :math:`\Delta_S(T) = \sum_{L \subseteq S} (-1)^{|S| - |L|}\, \nu(T \cup L)` instead of
            marginal contributions. With :math:`s = |S|` and :math:`t = |T|`, the index is

            .. math::

               \phi^{\text{SII}}_S(\nu) = \sum_{T \subseteq D \setminus S}
               \frac{(d - s - t)!\; t!}{(d - s + 1)!}\, \Delta_S(T).

      .. container:: shapiq-talk-column

         .. shapiq-talk-figure:: lattice-sii
            :alt: The lattice of the four players of the giraffe example again. A pair of players spans a square in it at every coalition of the other players. The discrete derivative of the pair is the sum around that square, with alternating signs. The Shapley interaction of the pair averages the discrete derivatives of all its squares.

   .. shapiq-notes::

      In the lattice, a pair of players spans a square at every coalition of the
      others. Going around the square gives the discrete derivative, what the pair adds together
      beyond what each adds alone. For the patch of the giraffe and the word "giraffe", it is
      large. The Shapley interaction index averages these squares :cite:p:`Grabisch.1999`, as the
      Shapley value averages edges. Its variants k-SII :cite:p:`Bordt.2023`, Shapley-Taylor and
      Faith-SHAP :cite:p:`Tsai.2022` make the values of each order add up to the prediction.

.. shapiq-slide:: From Feature Attributions to Interactions
   :layout: wide
   :class: shapiq-talk-attributions

   .. container:: shapiq-talk-definition shapiq-talk-attribution

      .. rubric:: Feature Attribution.

      A feature attribution summarizes :math:`f` at :math:`x` by one score :math:`\phi_i` for each
      feature :math:`i \in D`, so that

      .. math::

         f(x) \;\approx\; \phi_\emptyset + \sum_{i \in D} \phi_i .

   .. container:: shapiq-talk-definition shapiq-talk-attribution

      .. rubric:: Feature Attribution with Interactions.

      An explanation of order :math:`k` summarizes :math:`f` at :math:`x` by values for single
      features, pairs, and groups of up to :math:`k` features, so that

      .. math::

         f(x) \;\approx\; \phi_\emptyset + \sum_{i} \phi_i + \sum_{i < j} \phi_{ij}
         + \dots + \sum_{|S| = k} \phi_S .

   .. shapiq-notes::

      A feature attribution gives one score to each feature, and the scores add up to the
      prediction. An explanation of order :math:`k` adds a value for every pair, every triple,
      and so on up to groups of :math:`k` features. Together they approximate the prediction.
      Order 1 is the feature attribution again. Higher orders show what features do only
      together.

.. shapiq-slide:: Similarity Lives Between Image and Text
   :layout: wide
   :cite: Baniecki.2025 Thies.2026

   .. shapiq-demo:: fixlip
      :pick: giraffe-tree
      :select: full

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

.. shapiq-slide:: The Full Decomposition Opens the Black Box
   :layout: wide
   :cite: Muschalik.2024a Fujimoto.2006 Baniecki.2025

   .. shapiq-demo:: orders
      :select: full
      :reveal: open
      :examples: houses, fixlip
      :pick: giraffe

   .. shapiq-notes::

      At the highest order, the explanation is the model. For the tree ensemble, every order is
      computed exactly, and the full decomposition reproduces the model with an :math:`R^2` of 1
      :cite:p:`Muschalik.2024a`. The black box opens all the way. For SigLIP 2, the game has too
      many players to evaluate every coalition. Its explanations are estimated, and what the full
      decomposition falls short of 1 is the estimation gap. The mechanism stays blurry.

.. shapiq-slide:: Faithfulness of Shapley Interaction Explanations
   :layout: wide
   :steps: 5
   :cite: Tsai.2022 Bordt.2023

   .. shapiq-talk-figure:: faithfulness
      :class: talk-faith-climb
      :alt: Faithfulness order by order, for a credit applicant. The explanation of each order predicts what the model does with only some features present. The R² rises from 0.24 for the Shapley values to 0.98 for order 2 and to 1 at order 5.

   .. shapiq-notes::

      Faithfulness again, now order by order. The Shapley values reach an :math:`R^2` of 0.24 for
      the applicant. The pairs lift it to 0.98. Take the four features kept here, the credit limit,
      the status of July, and the bill and the payment of September. The single features miss the
      model by far. The pairs add most of what is missing, and groups of three and four close the
      rest. Each further order adds what the previous ones missed, and at order 5, the depth of the
      trees, the explanation is exact. The order thus moves an explanation from
      compact to complete, and faithfulness says how far to go :cite:p:`Tsai.2022`.

.. shapiq-slide:: Every Explanation Makes Three Choices
   :layout: wide
   :steps: 3
   :cite: Fumagalli.2025 Herbinger.2026 Enouen.2025

   .. container:: shapiq-talk-recipe

      .. math::

         \phi \;=\; \mathcal{I} \circ \mathcal{B} \circ \mathcal{M}\, f

      .. container:: shapiq-talk-callout

         .. rubric:: AISTATS 2025

         | **Unifying Feature-Based Explanations** with Functional ANOVA and Cooperative
           Game Theory
         | Fumagalli, Muschalik, Hüllermeier, Hammer and Herbinger

   .. container:: shapiq-talk-cards

      .. container:: shapiq-talk-card

         .. rubric:: Masking :math:`\mathcal{M}`

         How absent features are removed

         .. math::

            f_S(x) = \mathbb{E}_{X_{-S} \sim P}\big[f(x_S, X_{-S})\big]

         | **Baseline**: :math:`X_{-S} = b_{-S}`
         | **Marginal**: :math:`P(X_{-S})`
         | **Conditional**: :math:`P(X_{-S} \mid x_S)`

      .. container:: shapiq-talk-card shapiq-from-1

         .. rubric:: Behavior :math:`\mathcal{B}`

         What is explained

         .. math::

            \nu(S) = \mathcal{B}\big(f_S\big)

         | **Local**: the prediction :math:`f_S(x)`
         | **Sensitivity**: :math:`\mathbb{V}\big[f_S(X)\big]`
         | **Risk**: :math:`-\mathbb{E}\big[\ell(f_S(X), Y)\big]`

      .. container:: shapiq-talk-card shapiq-from-2

         .. rubric:: Interaction :math:`\mathcal{I}`

         How effects of groups are shared

         .. math::

            \phi(i) = \sum_{T \ni i} w_T\, m(T)

         | **Pure**: :math:`w_T = 1` for :math:`T = \{i\}`, else 0
         | **Partial**: :math:`w_T = 1 / |T|`
         | **Full**: :math:`w_T = 1`

   .. shapiq-talk-figure:: choices
      :alt: A picture of each option of the three choices. Masking: a cloud of two correlated features, in which the removed feature takes one baseline value, every value of the data, or the values of data with a similar kept feature. Behavior: one prediction, the spread of the predictions, or their errors against the labels. Interaction: two features as overlapping circles, of which feature 1 gets its own part only, its own part and half the overlap, or its whole circle.

   .. shapiq-notes::

      Many explanation methods are one recipe with three choices :cite:p:`Fumagalli.2025`. The
      notation as operators follows :cite:t:`Herbinger.2026`.
      Masking decides how absent features are removed: replaced by a baseline, averaged over
      their distribution, or averaged given the present features. Behavior decides what is
      explained: one prediction, how much the prediction varies, or how well the model performs.
      Interaction decides how the effects of groups of features count for a single feature, in
      terms of the Möbius transform :math:`m` :cite:p:`Fumagalli.2025`. A pure effect counts only
      the feature alone, a full effect counts every group that holds it, and a partial effect
      shares each group equally among its members, as the Shapley value does. SHAP is the partial
      effect of a local prediction under marginal masking,
      :math:`\mathcal{I}_{\text{partial}} \circ \mathcal{B}_{\text{local}} \circ \mathcal{M}_{\text{marginal}}`. Shapley interactions keep masking and
      behavior, and change only how groups are counted :cite:p:`Enouen.2025`.

.. role:: shapiq-accent

.. shapiq-slide:: SHAP Shares Interactions Equally. :shapiq-accent:`Others Do Not.`
   :layout: wide
   :class: shapiq-slide--top
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

.. shapiq-slide:: For One Prediction, Methods Differ in Two Choices
   :layout: wide
   :class: shapiq-slide--top
   :steps: 3
   :cite: Fumagalli.2025 Herbinger.2026 Tsang.2020 Friedman.2001 Apley.2020 Sundararajan.2020a
          Janzing.2020 Lundberg.2017 Lundberg.2020 Zeiler.2014 Goldstein.2015 Zintgraf.2017
          Blucher.2022 Harris.2022 Bordt.2023 Tsai.2022 Sundararajan.2020

   .. shapiq-talk-figure:: framework
      :alt: A grid of local feature-based explanations. Its columns are how absent features are treated, by a baseline, the marginal or the conditional distribution. Its rows are how higher-order interactions enter, as pure, partial or full effects. Each cell lists the methods that fall into it, for individual, joint and interaction explanations.

   .. The chips of the grid link to these citations, as every citation of the docs links to the
      bibliography (figures.js finds each by its class).

   .. container:: shapiq-talk-cites

      .. container:: shapiq-cite-tsang-2020

         :cite:p:`Tsang.2020`

      .. container:: shapiq-cite-friedman-2001

         :cite:p:`Friedman.2001`

      .. container:: shapiq-cite-apley-2020

         :cite:p:`Apley.2020`

      .. container:: shapiq-cite-sundararajan-2020a

         :cite:p:`Sundararajan.2020a`

      .. container:: shapiq-cite-janzing-2020

         :cite:p:`Janzing.2020`

      .. container:: shapiq-cite-lundberg-2017

         :cite:p:`Lundberg.2017`

      .. container:: shapiq-cite-lundberg-2020

         :cite:p:`Lundberg.2020`

      .. container:: shapiq-cite-zeiler-2014

         :cite:p:`Zeiler.2014`

      .. container:: shapiq-cite-goldstein-2015

         :cite:p:`Goldstein.2015`

      .. container:: shapiq-cite-zintgraf-2017

         :cite:p:`Zintgraf.2017`

      .. container:: shapiq-cite-blucher-2022

         :cite:p:`Blucher.2022`

      .. container:: shapiq-cite-harris-2022

         :cite:p:`Harris.2022`

      .. container:: shapiq-cite-bordt-2023

         :cite:p:`Bordt.2023`

      .. container:: shapiq-cite-tsai-2022

         :cite:p:`Tsai.2022`

      .. container:: shapiq-cite-sundararajan-2020

         :cite:p:`Sundararajan.2020`

   .. shapiq-notes::

      Many feature-based explanations rest on a common foundation :cite:p:`Fumagalli.2025`. Each
      of them makes two choices. The first is how absent features are treated, by a fixed
      baseline, by their marginal distribution or by their distribution given the present
      features. With correlated features, such as income and loan amount, these choices give
      different answers. The second choice is how much of the interactions an explanation takes
      in, as pure, partial or full effects :cite:p:`Herbinger.2026`. The variants of SHAP form one
      row of this grid. Path-dependent TreeSHAP, the default for trees, lets absent features follow
      the splits of the trees, which approximates the conditional distribution. Methods defined on
      any game, such as the k-SVs, Faith-SHAP and the Shapley-Taylor interactions, have a variant
      for each of the three distributions.
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

.. shapiq-slide:: shapiq-v2
   :layout: title
   :class: shapiq-talk-v2-slide
   :cite: Fumagalli.2025 Herbinger.2026

   .. container:: shapiq-talk-v2-hero

      .. container:: shapiq-talk-v2-logo

         .. image:: /_static/logo/logo_shapiq_light.svg
            :class: shapiq-hero-logo only-light
            :alt: The shapiq logo: a network of players, linked by their interactions.

         .. image:: /_static/logo/logo_shapiq_dark.svg
            :class: shapiq-hero-logo only-dark
            :alt: The shapiq logo: a network of players, linked by their interactions.

         .. rst-class:: shapiq-talk-v2-mark

         v2

      .. container:: shapiq-talk-v2-blocks

         .. math::

            \phi \;=\; \mathcal{I} \circ \mathcal{B} \circ \mathcal{M}\, f

   .. shapiq-notes::

      This framework is the core of shapiq-v2. A large team builds it. Masking,
      behavior and interaction become building blocks that combine freely, so that every cell of
      the grid, and every explanation of the three choices, is one call away.

.. shapiq-slide:: probly: Uncertainty Quantification for Machine Learning
   :layout: wide

   .. container:: shapiq-talk-probly-ad

      .. container:: shapiq-talk-probly

         .. image:: assets/probly/logo-light.svg
            :class: only-light shapiq-talk-probly-logo
            :alt: The probly logo.

         .. image:: assets/probly/logo-dark.svg
            :class: only-dark shapiq-talk-probly-logo
            :alt: The probly logo.

         *Turn any model into one that knows what it doesn't know.*

         .. rst-class:: shapiq-talk-oral

         Oral at NeurIPS 2026

      .. container:: shapiq-talk-qr shapiq-talk-probly-qr

         .. image:: assets/probly/qr-probly.svg
            :alt: A QR code that opens github.com/pwhofman/probly.

         github.com/pwhofman/probly

   .. shapiq-notes::

      shapiq has a sibling. probly, developed by Paul Hofman and colleagues, quantifies
      the uncertainty of any model. More than 40 methods, from Bayesian networks and deep
      ensembles to credal and conformal prediction, share one pipeline. probly was accepted as an
      oral at NeurIPS 2026. The QR code opens its repository.

   .. shapiq-try::

      .. code-block:: bash

         pip install probly

      The documentation of probly is at `pwhofman.github.io/probly
      <https://pwhofman.github.io/probly>`_, its code at `github.com/pwhofman/probly
      <https://github.com/pwhofman/probly>`_.

.. The word shapiq as its logo writes it (talk.css: the last slide).

.. role:: shapiq-wordmark
.. role:: shapiq-wordmark-iq

.. shapiq-slide:: Thank You
   :layout: title
   :class: shapiq-talk-thanks
   :cite: Muschalik.2024

   .. rst-class:: shapiq-talk-subtitle

   Every demo of this talk is computed with :shapiq-wordmark:`shap`\ :shapiq-wordmark-iq:`iq`.

   .. container:: shapiq-talk-install

      .. code-block:: bash

         pip install shapiq

   .. The links carry an icon each, by their address (talk.css).

   .. container:: shapiq-talk-socials

      .. rst-class:: shapiq-talk-speaker

      | **Maximilian Muschalik**
      | `maxmuschalik.com <https://maxmuschalik.com>`_
      | `in/maximilian-muschalik <https://www.linkedin.com/in/maximilian-muschalik>`_
      | `mmschlk <https://github.com/mmschlk>`_
      | `Google Scholar <https://scholar.google.com/citations?user=jJBCW74AAAAJ>`_

   .. container:: shapiq-talk-thanks-card

      .. container:: shapiq-talk-logo

         .. image:: /_static/logo/logo_shapiq_light.svg
            :class: shapiq-hero-logo only-light
            :alt: The shapiq logo: a network of players, linked by their interactions.

         .. image:: /_static/logo/logo_shapiq_dark.svg
            :class: shapiq-hero-logo only-dark
            :alt: The shapiq logo: a network of players, linked by their interactions.

      .. container:: shapiq-talk-qr

         .. image:: assets/qr-shapiq.svg
            :alt: A QR code that opens github.com/mmschlk/shapiq.

         github.com/mmschlk/shapiq

.. shapiq-slide:: Appendix
   :layout: section

   .. shapiq-notes::

      Slides for questions after the talk.

.. shapiq-slide:: Feature-Based Explainable AI
   :layout: wide
   :cite: Lundberg.2017 Friedman.2001 Breiman.2001 Covert.2021

   .. container:: shapiq-talk-definition

      .. rubric:: Definition (Feature-Based Explanation).

      Let :math:`f: \mathcal{X} \to \mathcal{Y}` be a trained model with the features
      :math:`D = \{1, \dots, d\}`. A feature-based explanation is a mapping

      .. math::

         e: \mathcal{S} \to \mathcal{E}, \qquad S \mapsto e(S),

      where :math:`\mathcal{S} \subseteq 2^D` is a **chosen collection of feature subsets** and
      :math:`\mathcal{E}` an explanation space. For each :math:`S \in \mathcal{S}`, the
      **explanation** :math:`e(S)` describes how the features in :math:`S` **influence** the output
      of :math:`f`.

   .. shapiq-talk-figure:: feature-xai
      :alt: Above, features of five kinds of data: the columns of a California housing table, the 64 patches of a photo of a giraffe, the 19 tokens of its caption, the windows of a time series and the atoms of a molecule. Below, four shapes of explanations: the Shapley values of one house (SHAP) and the Shapley interactions of the giraffe's patches and words for one prediction, the partial dependence curve of MedInc and the permutation importance of the features for the whole model.

   .. shapiq-notes::

      Feature-based explanations describe a model through its input features: the columns of a
      table, the patches of an image, the tokens of a text, the windows of a time series, or the
      nodes of a graph. They are post-hoc,
      applied to a model that is already trained, and they apply to any model, because features
      are the language that every model speaks. Formally, a feature-based explanation maps a
      chosen collection of feature subsets to an explanation space. SHAP gives one score to each
      feature of one prediction :cite:p:`Lundberg.2017`. Partial dependence gives each feature a
      curve over the whole model :cite:p:`Friedman.2001`, and permutation importance a score of
      how much the model relies on it :cite:p:`Breiman.2001`. Shapley interactions give a score
      to every group of features up to a size :math:`k`. Many of these methods remove features
      and watch what the model does :cite:p:`Covert.2021`. This talk stays with explanations of
      one prediction.
