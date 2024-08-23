## Changelog

### development

- [added](https://github.com/mmschlk/shapiq/pull/185) Exact Core computation closing [#182](https://github.com/mmschlk/shapiq/issues/182)
- refactored the `shapiq.games.benchmark` module into a separate `shapiq.benchmark` module by moving all but the benchmark games into the new modul. This closes [#169](https://github.com/mmschlk/shapiq/issues/169) and makes benchmarking more flexible and convenient.
- add waterfall plot as described in [#34](https://github.com/mmschlk/shapiq/issues/34)
- add a legend to benchmark plots [#170](https://github.com/mmschlk/shapiq/issues/170)
- fix the force plot not showing and its baseline value
- improve tests for plots and benchmarks
- renames explanation graph to si_graph
- bugfixes with plotting and benchmarks
- `get_n_order` now has optional lower/upper limits for the order
- computing metrics now tries to resolve not-matching interaction indices and will throw a warning instead of a ValueError [#179](https://github.com/mmschlk/shapiq/issues/179)
- ...

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
