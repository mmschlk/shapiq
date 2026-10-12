Quickstart
==========

From a trained model to Shapley values and Shapley interactions, their plots, and the
game-theoretic view behind them. Every snippet on this page runs as it is, in order, after
:doc:`installing shapiq <installation>`.


Explain a Model
---------------

:class:`~shapiq.explainer.Explainer` takes a model and background data, and picks the explainer
that fits the model. For a random forest, that is the
:class:`~shapiq.explainer.TreeExplainer`, which computes the values exactly:

.. code-block:: python

   from sklearn.datasets import fetch_california_housing
   from sklearn.ensemble import RandomForestRegressor

   import shapiq

   data = fetch_california_housing()
   X, y, feature_names = data.data, data.target, data.feature_names
   model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=0).fit(X, y)

   # pairwise Shapley interactions (k-SII up to order 2) of the first house
   explainer = shapiq.Explainer(model, data=X, index="k-SII", max_order=2)
   interaction_values = explainer.explain(X[0])
   print(interaction_values)

.. code-block:: text
   :class: shapiq-output

   InteractionValues(
       index=k-SII, max_order=2, min_order=0, estimated=False, estimation_budget=None,
       n_players=8, baseline_value=2.0674158944573646,
       Top 10 interactions:
           (0,): 2.266025054354417
           (): 2.0674158944573646
           (7,): 0.2783641039520189
           (0, 1): 0.2572118576963052
           (1,): 0.13368765294448415
           (1, 7): -0.06216726789294137
           (1, 6): -0.06654762058114373
           (6,): -0.0890745350881206
           (0, 6): -0.1877658520839556
           (0, 7): -0.20161040935064278
   )

The result is an :class:`~shapiq.interaction_values.InteractionValues` object, indexed by tuples
of players, here the features: ``(0,)`` is the effect of the median income alone, ``(0, 1)`` the
interaction of the median income and the house age, and ``()`` the baseline, the average
prediction. Together, the values add up to the prediction:

.. code-block:: python

   interaction_values[(0, 1)]       # 0.257: median income and house age together
   interaction_values.values.sum()  # 4.379: the prediction model.predict(X[:1])

.. dropdown:: Which index and order?
   :icon: question

   ``index`` picks the game-theoretic concept and ``max_order`` the largest group of features
   that gets its own value:

   .. list-table::
      :header-rows: 1
      :widths: 15 85
      :class: shapiq-table

      * - Index
        - Explanation
      * - ``"SV"``
        - Shapley values :cite:p:`Shapley.1951` (``max_order=1``), as in ``shap``.
      * - ``"k-SII"``
        - Shapley interactions up to order ``k`` that add up to the prediction
          :cite:p:`Bordt.2023`; order 1 gives the Shapley values.
      * - ``"STII"``
        - The Shapley-Taylor interaction index :cite:p:`Sundararajan.2020`, whose highest order
          collects all interactions of that order and above.
      * - ``"FSII"``, ``"FBII"``
        - The faithful Shapley and Banzhaf interaction indices :cite:p:`Tsai.2022`, the best
          approximations of the model by interactions up to order ``k``.
      * - ``"SII"``
        - The Shapley interaction index :cite:p:`Grabisch.1999`.

   Higher orders reveal more, but give more values to read: order 2 (pairs) is a good start.


Explain Any Model
-----------------

For models without an exact explainer, :class:`~shapiq.explainer.TabularExplainer` removes
features by replacing them with values of the background data, and estimates the values from a
``budget`` of such model evaluations. With 8 features, there are 2\ :sup:`8` = 256 coalitions of
features, so a budget of 256 evaluates all of them and the values are exact:

.. code-block:: python

   from sklearn.neural_network import MLPRegressor
   from sklearn.pipeline import make_pipeline
   from sklearn.preprocessing import StandardScaler

   mlp = make_pipeline(StandardScaler(), MLPRegressor(early_stopping=True, random_state=0))
   mlp.fit(X, y)

   explainer = shapiq.TabularExplainer(
       mlp, data=X, index="k-SII", max_order=2, sample_size=100, random_state=0
   )
   interaction_values_mlp = explainer.explain(X[0], budget=256)

For more features, the budget sets the trade-off between accuracy and run time. The
:doc:`approximators </auto_examples/approximators/index>` behind it can also be chosen with
``approximator=``, and the imputer that removes features with ``imputer=``.


Shapley Values, as in SHAP
--------------------------

With ``index="SV"`` and ``max_order=1`` (the defaults), the explainers compute Shapley values:

.. code-block:: python

   explainer = shapiq.Explainer(model, data=X, index="SV", max_order=1)
   shapley_values = explainer.explain(X[0])
   shapley_values.plot_waterfall(feature_names=feature_names)


Plot the Values
---------------

:class:`~shapiq.interaction_values.InteractionValues` plot themselves, and the functions of
:mod:`shapiq.plot` summarize many explanations at once:

.. tab-set::

   .. tab-item:: Force plot

      .. code-block:: python

         interaction_values.plot_force(feature_names=feature_names)

      .. image:: ../_static/images/fsii_tabpfn_force_plot_example.png
         :class: shapiq-framed
         :alt: A force plot of Shapley interactions: features and pairs of features push the
               prediction up (red) or down (blue) from the baseline.

   .. tab-item:: Network plot

      .. code-block:: python

         interaction_values.plot_network(feature_names=feature_names)

      .. image:: ../_static/network_example2.png
         :class: shapiq-framed
         :width: 480
         :alt: A network plot of the interactions between image patches: the nodes are the
               patches, the edges their interactions.

   .. tab-item:: UpSet plot

      .. code-block:: python

         interaction_values.plot_upset(feature_names=feature_names)

      .. image:: ../_static/images/upset_plot.png
         :class: shapiq-framed
         :width: 560
         :alt: An UpSet plot: bars of the largest values, with the features of each value
               marked below.

   .. tab-item:: Many explanations

      .. code-block:: python

         explanations = list(explainer.explain_X(X[:100]))
         shapiq.beeswarm_plot(explanations, X[:100], feature_names=feature_names)
         shapiq.bar_plot(explanations, feature_names=feature_names)

The plots above come from other models and data; the :doc:`visualization gallery
</auto_examples/visualization/index>` shows all of them.


Games and Approximators
-----------------------

Behind every explanation is a cooperative game: a value function that maps each coalition of
players to a number. ``shapiq`` computes the values of any game, given as a callable or as a
:class:`~shapiq.game.Game`, exactly with the :class:`~shapiq.game_theory.ExactComputer` (for few
players) or approximately with an approximator and a budget of evaluations:

.. code-block:: python

   import numpy as np

   weights = np.array([0.4, 0.3, 0.2, 0.1, 0.05, -0.1, -0.2, -0.3])


   def game(coalitions: np.ndarray) -> np.ndarray:
       """The value of each coalition (one row of booleans per coalition)."""
       return coalitions @ weights + 0.5 * coalitions[:, 0] * coalitions[:, 1]


   exact = shapiq.ExactComputer(game, n_players=8)(index="k-SII", order=2)
   approximator = shapiq.KernelSHAPIQ(n=8, index="k-SII", max_order=2, random_state=0)
   estimate = approximator.approximate(budget=100, game=game)
   exact[(0, 1)], estimate[(0, 1)]  # (0.5, 0.491)

The games of :mod:`shapiq_games` cover synthetic games with known values and many machine
learning tasks; :mod:`shapiq_benchmark` compares approximators on them against exact values:

.. code-block:: python

   from shapiq_benchmark import Benchmark, run
   from shapiq_games import SOUM

   benchmark = Benchmark(SOUM(n=12, n_basis_games=30, random_state=0))
   results = run(
       benchmark,
       [shapiq.KernelSHAPIQ, shapiq.SVARMIQ],
       budgets=[200, 1000],
       index="k-SII",
       order=2,
       seeds=[0, 1],
   )
   results.groupby(["approximator", "budget"])["mse"].mean()


Next Steps
----------

.. grid:: 1 1 3 3
   :gutter: 3
   :class-container: shapiq-cards

   .. grid-item-card:: :octicon:`light-bulb` Shapley Interactions
      :link: index
      :link-type: doc
      :class-card: shapiq-card

      What Shapley interactions are, and when they tell more than Shapley values.

   .. grid-item-card:: :octicon:`image` Examples
      :link: /examples
      :link-type: doc
      :class-card: shapiq-card

      Tutorials for every part of ``shapiq``, from trees to vision transformers.

   .. grid-item-card:: :octicon:`code` API Reference
      :link: /api_reference
      :link-type: doc
      :class-card: shapiq-card

      Every explainer, approximator, game, and plot, with its parameters.
