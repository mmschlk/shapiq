## Changelog

### Development

### v1.2.2 (2025-03-11)
- changes python support to 3.10-3.13 [#318](https://github.com/mmschlk/shapiq/pull/318)
- fixes a bug that prohibited importing shapiq in environments without write access [#326](https://github.com/mmschlk/shapiq/issues/326)
- adds `ExtraTreeRegressors` to supported models [#309](https://github.com/mmschlk/shapiq/pull/309)

### v1.2.1 (2025-02-17)
- fixes bugs regarding plotting [#315](https://github.com/mmschlk/shapiq/issues/315) and [#316](https://github.com/mmschlk/shapiq/issues/316)
- fixes a bug with TreeExplainer and Trees that consist of only one feature [#286](https://github.com/mmschlk/shapiq/issues/286)
- fixes SV init with explainer for permutation, svarm, kernelshap, and unbiased kernelshap [#319](https://github.com/mmschlk/shapiq/issues/319)
- adds a progress bar to `explain_X()` [#324](https://github.com/mmschlk/shapiq/issues/324)

### v1.2.0 (2025-01-15)
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

### v1.1.1 (2024-11-13)

#### Improvements and Ease of Use
- adds a `class_index` parameter to `TabularExplainer` and `Explainer` to specify the class index to be explained for classification models [#271](https://github.com/mmschlk/shapiq/issues/271) (renames `class_label` parameter in TreeExplainer to `class_index`)
- adds support for `PyTorch` models to `Explainer` [#272](https://github.com/mmschlk/shapiq/issues/272)
- adds new tests comparing `shapiq` outputs for SVs with alues computed with `shap`
- adds new tests for checking `shapiq` explainers with different types of models

#### Bug Fixes
- fixes a bug that `RandomForestClassifier` models were not working with the `TreeExplainer` [#273](https://github.com/mmschlk/shapiq/issues/273)

### v1.1.0 (2024-11-07)

#### New Features and Improvements
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

#### Documentation
- adds a notebook showing how to use custom tree models with the `TreeExplainer` [#66](https://github.com/mmschlk/shapiq/issues/66)
- adds a notebook show how to use the `shapiq.Game` API to create custom games [#184](https://github.com/mmschlk/shapiq/issues/184)
- adds a notebook showing hot to visualize interactions [#252](https://github.com/mmschlk/shapiq/issues/252)
- adds a notebook showing how to compute Shapley values with `shapiq` [#193](https://github.com/mmschlk/shapiq/issues/197)
- adds a notebook for conducting data valuation [#190](https://github.com/mmschlk/shapiq/issues/190)
- adds a notebook showcasing introducing the Core and how to compute it with `shapiq` [#191](https://github.com/mmschlk/shapiq/issues/191)

#### Bug Fixes
- fixes a bug with SIs not adding up to the model prediction because of wrong values in the empty set [#264](https://github.com/mmschlk/shapiq/issues/264)
- fixes a bug that `TreeExplainer` did not have the correct baseline_value when using XGBoost models [#250](https://github.com/mmschlk/shapiq/issues/250)
- fixes the force plot not showing and its baseline value

### v1.0.1 (2024-06-05)

- add `max_order=1` to `TabularExplainer` and `TreeExplainer`
- fix `TreeExplainer.explain_X(..., n_jobs=2, random_state=0)`

### v1.0.0 (2024-06-04)

Major release of the `shapiq` Python package including (among others):

- `approximator` module implements over 10 approximators of Shapley values and interaction indices.
- `exact` module implements a computer for over 10 game theoretic concepts like interaction indices or generalized values.
- `games` module implements over 10 application benchmarks for the approximators.
- `explainer` module includes a `TabularExplainer` and `TreeExplainer` for any-order feature interactions of machine learning model predictions.
- `interaction_values` module implements a data class to store and analyze interaction values.
- `plot` module allows visualizing interaction values.
- `datasets` module loads datasets for testing and examples.

Documentation of `shapiq` with tutorials and API reference is available at https://shapiq.readthedocs.io
