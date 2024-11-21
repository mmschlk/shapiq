## Changelog

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
