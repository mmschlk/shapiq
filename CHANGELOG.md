# Changelog

## v1.3.1 (2025-07-11)

### New Features
- adds the `shapiq.plot.beesvarm_plot()` function to shapiq. The beeswarm plot was extended to also support interactions of features. Beeswarm plots are useful in visualizing dependencies between feature values. The beeswarm plot was adapted from the SHAP library by sub-dividing the y-axis for each interaction term. [#399](https://github.com/mmschlk/shapiq/issues/399)
- adds JSON support to `InteractionValues` and `Game` objects, allowing for easy serialization and deserialization of interaction values and game objects [#412](https://github.com/mmschlk/shapiq/pull/412) usage of `pickle` is now deprecated. This change allows us to revamp the data structures in the future and offers more flexibility.

### Testing, Code-Quality and Documentation
- adds a testing suite for testing deprecations in `tests/tests_deprecations/` which allows for easier deprecation managment and tracking of deprecated features [#412](https://github.com/mmschlk/shapiq/pull/412)

## Deprecated
- The `Game(path_to_values=...)` constructor is now deprecated and will be removed in version 1.4.0. Use `Game.load(...)` or `Game().load_values(...)` instead.
- Saving and loading `InteactionValues` via `InteractionValues.save(..., as_pickle=True)` and `InteractionValues.save(..., as_npz=True)` is now deprecated and will be removed in version 1.4.0. Use `InteractionValues.save(...)` to save as json.

## v1.3.0 (2025-06-17)

### Highlights
- `shapiq.SPEX` (Sparse Exact) approximator for efficient computation of sparse interaction values for really large models and games. Paper: [SPEX: Scaling Feature Interaction Explanations for LLMs](https://arxiv.org/abs/2502.13870)
- `shapiq.AgnosticExplainer` a generic explainer that works for any value function or `shapiq.Game` object, allowing for more flexibility in explainers.
- prettier graph-based plots via `shapiq.si_graph_plot()` and `shapiq.network_plot()`, which now use the same backend for more flexibility and easier maintenance.

### New Features
- adds the SPEX (Sparse Exact) module in `approximator.sparse` for efficient computation of sparse interaction values [#379](https://github.com/mmschlk/shapiq/pull/379)
- adds `shapiq.AgnosticExplainer` which is a generic explainer that can be used for any value function or `shapiq.Game` object. This allows for more flexibility in the explainers. [#100](https://github.com/mmschlk/shapiq/issues/100), [#395](https://github.com/mmschlk/shapiq/pull/395)
- changes `budget` to be a mandatory parameter given to the `TabularExplainer.explain()` method [#355](https://github.com/mmschlk/shapiq/pull/356)
- changes logic of `InteractionValues.get_n_order()` function to be callable with **either** the `order: int` parameter and optional assignment of `min_order: int` and `max_order: int` parameters **or** with the min/max order parameters [#372](https://github.com/mmschlk/shapiq/pull/372)
- renamed `min_percentage` parameter in the force plot to `contribution_threshold` to better reflect its purpose [#391](https://github.com/mmschlk/shapiq/pull/391)
- adds ``verbose`` parameter to the ``Explainer``'s ``explain_X()`` method to control weather a progress bar is shown or not which is defaulted to ``False``. [#391](https://github.com/mmschlk/shapiq/pull/391)
- made `InteractionValues.get_n_order()` and `InteractionValues.get_n_order_values()` function more efficient by iterating over the stored interactions and not over the powerset of all potential interactions, which made the function not usable for higher player counts (models with many features, and results obtained from `TreeExplainer`). Note, this change does not really help `get_n_order_values()` as it still needs to create a numpy array of shape `n_players` times `order` [#372](https://github.com/mmschlk/shapiq/pull/372)
- streamlined the ``network_plot()`` plot function to use the ``si_graph_plot()`` as its backend function. This allows for more flexibility in the plot function and makes it easier to use the same code for different purposes. In addition, the ``si_graph_plot`` was modified to make plotting more easy and allow for more flexibility with new parameters. [#349](https://github.com/mmschlk/shapiq/pull/349)
- adds `Game.compute()` method to the `shapiq.Game` class to compute game values without changing the state of the game object. The compute method also introduces a `shapiq.utils.sets.generate_interaction_lookup_from_coalitions()` utility method which creates an interaction lookup dict from an array of coalitions. [#397](https://github.com/mmschlk/shapiq/pull/397)
- streamlines the creation of network plots and graph plots which now uses the same backend. The network plot via `shapiq.network_plot()` or `InteractionValues.plot_network()` is now a special case of the `shapiq.si_graph_plot()` and `InteractionValues.plot_si_graph()`. This allows to create more beautiful plots and easier maintenance in the future. [#349](https://github.com/mmschlk/shapiq/pull/349)

### Testing, Code-Quality and Documentation
- activates ``"ALL"`` rules in ``ruff-format`` configuration to enforce stricter code quality checks and addressed around 500 (not automatically solvable) issues in the code base. [#391](https://github.com/mmschlk/shapiq/pull/391)
- improved the testing environment by adding a new fixture module containing mock `InteractionValues` objects to be used in the tests. This allows for more efficient and cleaner tests, as well as easier debugging of the tests  [#372](https://github.com/mmschlk/shapiq/pull/372)
- removed check and error message if the ``index`` parameter is not in the list of available indices in the ``TabularExplainer`` since the type hints were replaced by Literals [#391](https://github.com/mmschlk/shapiq/pull/391)
- removed multiple instances where ``shapiq`` tests if some approximators/explainers can be instantiated with certain indices or not in favor of using Literals in the ``__init__`` method of the approximator classes. This allows for better type hinting and IDE support, as well as cleaner code. [#391](https://github.com/mmschlk/shapiq/pull/391)
- Added documentation for all public modules, classes, and functions in the code base to improve the documentation quality and make it easier to understand how to use the package. [#391](https://github.com/mmschlk/shapiq/pull/391)
- suppress a ``RuntimeWarning`` in ``Regression`` approximators ``solve_regression()``method when the solver is not able to find good interim solutions for the regression problem.
- refactors the tests into ``tests_unit/`` and ``tests_integration/`` to better separate unit tests from integration tests. [#395](https://github.com/mmschlk/shapiq/pull/395)
- adds new integration tests in ``tests/tests_integration/test_explainer_california_housing`` which compares the different explainers against ground-truth interaction values computed by ``shapiq.ExactComputer`` and interaction values stored on [disk](https://github.com/mmschlk/shapiq/tree/main/tests/data/interaction_values/california_housing) as a form of regression test. This test should help finding bugs in the future when the approximators, explainers, or exact computation are changed. [#395](https://github.com/mmschlk/shapiq/pull/395)

### Bug Fixes
- fixed a bug in the `shapiq.waterfall_plot` function that caused the plot to not display correctly resulting in cutoff y_ticks. Additionally, the file was renamed from `watefall.py` to `waterfall.py` to match the function name [#377](https://github.com/mmschlk/shapiq/pull/377)
- fixes a bug with `TabPFNExplainer`, where the model was not able to be used for predictions after it was explained. This was due to the model being fitted on a subset of features, which caused inconsistencies in the model's predictions after explanation. The fix includes that after each call to the `TabPFNImputer.value_function`, the tabpfn model is fitted on the whole dataset (without omitting features). This means that the original model can be used for predictions after it has been explained. [#396](https://github.com/mmschlk/shapiq/issues/396).
- fixed a bug in computing `BII` or `BV` indices with `shapiq.approximator.MonteCarlo` approximators (affecting `SHAP-IQ`, `SVARM` and `SVARM-IQ`). All orders of BII should now be computed correctly. [#395](https://github.com/mmschlk/shapiq/pull/395)

## v1.2.3 (2025-03-24)
- substantially improves the runtime of all `Regression` approximators by a) a faster pre-computation of the regression matrices and b) a faster computation of the weighted least squares regression [#340](https://github.com/mmschlk/shapiq/issues/340)
- removes `sample_replacements` parameter from `MarginalImputer` and removes the DeprecationWarning for it
- adds a trivial computation to `TreeSHAP-IQ` for trees that use only one feature in the tree (this works for decision stumps or trees splitting on only one feature multiple times). In such trees, the computation is trivial as the whole effect of $\nu(N) - \nu(\emptyset)$ is all on the main effect of the single feature and there is no interaction effect. This expands on the fix in v1.2.1 [#286](https://github.com/mmschlk/shapiq/issues/286).
- fixes a bug with xgboost where feature names where trees that did not contain all features would lead `TreeExplainer` to fail
- fixes a bug with `stacked_bar_plot` where the higher order interactions were inflated by the lower order interactions, thus wrongly showing the higher order interactions as higher than they are
- fixes a bug where `InteractionValues.get_subset()` returns a faulty `coalition_lookup` dictionary pointing to indices outside the subset of players [#336](https://github.com/mmschlk/shapiq/issues/336)
- updates default value of `TreeExplainer`'s `min_order` parameter from 1 to 0 to include the baseline value in the interaction values as per default
- adds the `RegressionFBII` approximator to estimate Faithful Banzhaf interactions via least squares regression [#333](https://github.com/mmschlk/shapiq/pull/333). Additionally, FBII support was introduced in TabularExplainer and MonteCarlo-Approximator.
- adds a `RandomGame` class as part of `shapiq.games.benchmark` which always returns a random vector of integers between 0 and 100.

## v1.2.2 (2025-03-11)
- changes python support to 3.10-3.13 [#318](https://github.com/mmschlk/shapiq/pull/318)
- fixes a bug that prohibited importing shapiq in environments without write access [#326](https://github.com/mmschlk/shapiq/issues/326)
- adds `ExtraTreeRegressors` to supported models [#309](https://github.com/mmschlk/shapiq/pull/309)

## v1.2.1 (2025-02-17)
- fixes bugs regarding plotting [#315](https://github.com/mmschlk/shapiq/issues/315) and [#316](https://github.com/mmschlk/shapiq/issues/316)
- fixes a bug with TreeExplainer and Trees that consist of only one feature [#286](https://github.com/mmschlk/shapiq/issues/286)
- fixes SV init with explainer for permutation, svarm, kernelshap, and unbiased kernelshap [#319](https://github.com/mmschlk/shapiq/issues/319)
- adds a progress bar to `explain_X()` [#324](https://github.com/mmschlk/shapiq/issues/324)

## v1.2.0 (2025-01-15)
- adds ``shapiq.TabPFNExplainer`` as a specialized version of the ``shapiq.TabularExplainer`` which offers a streamlined variant of the explainer for the TabPFN model [#301](https://github.com/mmschlk/shapiq/issues/301)
- handles ``explainer.explain()`` now through a common interface for all explainer classes which now need to implement a ``explain_function()`` method
- adds the baseline_value into the InteractionValues object's value storage for the ``()`` interaction if ``min_order=0`` (default usually) for all indices that are not ``SII```(SII has another baseline value) such that the values are efficient (sum up to the model prediction) without the awkward handling of the baseline_value attribute
- renames ``game_fun`` parameter in ``shapiq.ExactComputer`` to ``game`` [#297](https://github.com/mmschlk/shapiq/issues/297)
- adds a TabPFN example notebook to the documentation
- removes warning when class_index is not provided in explainers [#298](https://github.com/mmschlk/shapiq/issues/298)
- adds the `sentence_plot` function to the `plot` module to visualize the contributions of words to a language model prediction in a sentence-like format
- makes abbreviations in the `plot` module optional [#281](https://github.com/mmschlk/shapiq/issues/281)
- adds the `upset_plot` function to the `plot` module to visualize the interactions of higher-order [#290](https://github.com/mmschlk/shapiq/issues/290)
- adds support for IsoForest models to explainer and tree explainer [#278](https://github.com/mmschlk/shapiq/issues/278)
- adds support for sub-selection of players in the interaction values data class [#276](https://github.com/mmschlk/shapiq/issues/276) which allows retrieving interaction values for a subset of players
- refactors game theory computations like `ExactComputer`, `MoebiusConverter`, `core`, among others to be more modular and flexible into the `game_theory` module [#258](https://github.com/mmschlk/shapiq/issues/258)
- improves quality of the tests by adding many more semantic tests to the different interaction indices and computations [#285](https://github.com/mmschlk/shapiq/pull/285)

## v1.1.1 (2024-11-13)

### Improvements and Ease of Use
- adds a `class_index` parameter to `TabularExplainer` and `Explainer` to specify the class index to be explained for classification models [#271](https://github.com/mmschlk/shapiq/issues/271) (renames `class_label` parameter in TreeExplainer to `class_index`)
- adds support for `PyTorch` models to `Explainer` [#272](https://github.com/mmschlk/shapiq/issues/272)
- adds new tests comparing `shapiq` outputs for SVs with alues computed with `shap`
- adds new tests for checking `shapiq` explainers with different types of models

### Bug Fixes
- fixes a bug that `RandomForestClassifier` models were not working with the `TreeExplainer` [#273](https://github.com/mmschlk/shapiq/issues/273)

## v1.1.0 (2024-11-07)

### New Features and Improvements
- adds computation of the Egalitarian Core (`EC`) and Egalitarian Least-Core (`ELC`) to the `ExactComputer` [#182](https://github.com/mmschlk/shapiq/issues/182)
- adds `waterfall_plot` [#34](https://github.com/mmschlk/shapiq/issues/34) that visualizes the contributions of features to the model prediction
- adds `BaselineImputer` [#107](https://github.com/mmschlk/shapiq/issues/107) which is now responsible for handling the `sample_replacements` parameter. Added a DeprecationWarning for the parameter in `MarginalImputer`, which will be removed in the next release.
- adds `joint_marginal_distribution` parameter to `MarginalImputer` with default value `True` [#261](https://github.com/mmschlk/shapiq/issues/261)
- renames explanation graph to `si_graph`
- `get_n_order` now has optional lower/upper limits for the order
- computing metrics for benchmarking now tries to resolve not-matching interaction indices and will throw a warning instead of a ValueError [#179](https://github.com/mmschlk/shapiq/issues/179)
- add a legend to benchmark plots [#170](https://github.com/mmschlk/shapiq/issues/170)
- refactored the `shapiq.games.benchmark` module into a separate `shapiq.benchmark` module by moving all but the benchmark games into the new module. This closes [#169](https://github.com/mmschlk/shapiq/issues/169) and makes benchmarking more flexible and convenient.
- a `shapiq.Game` can now be called more intuitively with coalitions data types (tuples of int or str) and also allows to add `player_names` to the game at initialization [#183](https://github.com/mmschlk/shapiq/issues/183)
- improve tests across the package

### Documentation
- adds a notebook showing how to use custom tree models with the `TreeExplainer` [#66](https://github.com/mmschlk/shapiq/issues/66)
- adds a notebook show how to use the `shapiq.Game` API to create custom games [#184](https://github.com/mmschlk/shapiq/issues/184)
- adds a notebook showing hot to visualize interactions [#252](https://github.com/mmschlk/shapiq/issues/252)
- adds a notebook showing how to compute Shapley values with `shapiq` [#193](https://github.com/mmschlk/shapiq/issues/197)
- adds a notebook for conducting data valuation [#190](https://github.com/mmschlk/shapiq/issues/190)
- adds a notebook showcasing introducing the Core and how to compute it with `shapiq` [#191](https://github.com/mmschlk/shapiq/issues/191)

### Bug Fixes
- fixes a bug with SIs not adding up to the model prediction because of wrong values in the empty set [#264](https://github.com/mmschlk/shapiq/issues/264)
- fixes a bug that `TreeExplainer` did not have the correct baseline_value when using XGBoost models [#250](https://github.com/mmschlk/shapiq/issues/250)
- fixes the force plot not showing and its baseline value

## v1.0.1 (2024-06-05)

- add `max_order=1` to `TabularExplainer` and `TreeExplainer`
- fix `TreeExplainer.explain_X(..., n_jobs=2, random_state=0)`

## v1.0.0 (2024-06-04)

Major release of the `shapiq` Python package including (among others):

- `approximator` module implements over 10 approximators of Shapley values and interaction indices.
- `exact` module implements a computer for over 10 game theoretic concepts like interaction indices or generalized values.
- `games` module implements over 10 application benchmarks for the approximators.
- `explainer` module includes a `TabularExplainer` and `TreeExplainer` for any-order feature interactions of machine learning model predictions.
- `interaction_values` module implements a data class to store and analyze interaction values.
- `plot` module allows visualizing interaction values.
- `datasets` module loads datasets for testing and examples.

Documentation of `shapiq` with tutorials and API reference is available at https://shapiq.readthedocs.io
