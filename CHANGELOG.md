## Changelog

### v1.0.1 (2024-06-05)

- add `max_order=1` to `TabularExplainer` and `TreeExplainer`
- fix `TreeExplainer.explain_X(..., njobs=2, random_state=0)`

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
