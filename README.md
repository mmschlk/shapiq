
---
**NOTE:**

`shapiq` is still in an alpha stage and under active development. The [initial release](https://github.com/mmschlk/shapiq/milestone/1) is scheduled to be May 31st.

---

<p align="center">
  <img height="250px" src="docs/source/_static/logo_shapiq_light.svg" alt="shapiq_logo"/>
</p>

[![License](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)
[![Coverage Status](https://coveralls.io/repos/github/mmschlk/shapiq/badge.svg?branch=main)](https://coveralls.io/github/mmschlk/shapiq?branch=main)
[![Tests](https://github.com/mmschlk/shapiq/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/mmschlk/shapiq/actions/workflows/unit-tests.yml)
[![Read the Docs](https://readthedocs.org/projects/shapiq/badge/?version=latest)](https://shapiq.readthedocs.io/en/latest/?badge=latest)

[![PyPI Version](https://img.shields.io/pypi/pyversions/shapiq.svg)](https://pypi.org/project/shapiq)
[![PyPI status](https://img.shields.io/pypi/status/shapiq.svg?color=blue)](https://pypi.org/project/shapiq)
[![PePy](https://static.pepy.tech/badge/shapiq?style=flat-square)](https://pepy.tech/project/shapiq)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# shapiq: Shapley Interactions for Machine Learning
> An interaction may speak more than a thousand main effects.

Shapley Interaction Quantification (`shapiq`) is a Python package for (1) approximating any-order Shapley interactions, (2) benchmarking game-theoretical algorithms for machine learning, (3) explaining feature interactions of model predictions. `shapiq` extends the well-known [shap](https://github.com/shap/shap) package for both researchers working on game theory in machine learning, as well as the end-users explaining models. SHAP-IQ extends indivdual Shapley values by quantifying the **synergy** effect between entities (aka **players** in the jargon of game theory) like explanatory features, data points, or weak learners in ensemble models. Synergies between players give a more comprehensive view of machine learning models.

# 🛠️ Install
`shapiq` is intended to work with **Python 3.9 and above**. Installation can be done via `pip`:

```sh
pip install shapiq
```

# ⭐ Quickstart
You can use `shapiq` in different ways. If you have a trained model you can rely on the `shapiq.explainer` classes.
If you are interested in the underlying game theoretic algorithms, then check out the `shapiq.approximator` modules.
You can also plot and visualize your interaction scores with `shapiq.plot`.

## 📈 Compute k-SII values

Explain your models with Shapley interaction values like the k-SII values:

```python
import shapiq
# load data
X, y = shapiq.load_california_housing(to_numpy=True)
# train a model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X, y)
# explain with k-SII interaction scores
explainer = shapiq.TabularExplainer(
    model=model,
    data=X,
    index="k-SII",
    max_order=2
)
interaction_values = explainer.explain(X[0], budget=256)

print(interaction_values)
>> InteractionValues(
>>    index=k-SII, max_order=2, min_order=0, estimated=False,
>>    estimation_budget=256, n_players=8, baseline_value=0.86628,
>>    Top 10 interactions:
>>        (0,): 3.58948354047   # main effect for feature 0
>>        (7,): 1.61175123142
>>        (0, 1): 0.208496403   # interaction for features 0 & 1
>>        (5,): 0.20069311333
>>        (2,): 0.17536356571
>>        (0, 5): -0.09740194
>>        (0, 3): -0.12671954
>>        (0, 6): -0.21245009
>>        (6, 7): -0.34294075
>>        (0, 7): -1.15889485
>> )
```

## 📊 Visualize your Interactions

One handy way of visualizing interaction scores (up to order 2) are network plots.
You can see an example of such a plot below.
The nodes represent **attribution** scores and the edges represent the **interactions**.
The strength and size of the nodes and edges are proportional to the absolute value of the
attribution scores and interaction scores, respectively.

```python
shapiq.network_plot(
    first_order_values=interaction_values.get_n_order_values(1),
    second_order_values=interaction_values.get_n_order_values(2)
)
```

The pseudo-code above can produce the following plot (here also an image is added):

<p align="center">
  <img width="400px" src="docs/source/_static/network_example.png" alt="network_plot_example">
</p>

## 📖 Documentation
The documentation for ``shapiq`` can be found [here](https://shapiq.readthedocs.io/en/latest/).

## 💬 Citation

If you **enjoy** `shapiq` consider starring ⭐ the repository. If you **really enjoy** the package or it has been useful to you, and you would like to cite it in a scientific publication, please refer to our [paper](https://openreview.net/forum?id=IEMLNF4gK4):


```bibtex
@inproceedings{shapiq,
  title        = {{SHAP-IQ}: Unified approximation of any-order Shapley interactions},
  author       = {Fabian Fumagalli and
                  Maximilian Muschalik and
                  Patrick Kolpaczki and
                  Eyke H{\"{u}}llermeier and
                  Barbara Hammer},
  booktitle    = {NeurIPS},
  year         = {2023}
}
```
