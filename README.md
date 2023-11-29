<p align="center">
  <img height="250px" src="docs/source/_static/logo_shapiq_light.svg" alt="shapiq_logo">
</p>

<p align="center">
  <!-- Tests -->
  <a href="https://github.com/mmschlk/shapiq/actions/workflows/unit-tests.yml">
    <img src="https://github.com/mmschlk/shapiq/actions/workflows/unit-tests.yml/badge.svg" alt="unit-tests">
  </a>
  
  <!-- Read the Docs -->
  <a href='https://shapiq.readthedocs.io/en/latest/?badge=latest'>
      <img src='https://readthedocs.org/projects/shapiq/badge/?version=latest' alt='Documentation Status' />
  </a>
  
  <!-- PyPI Version -->
  <a href="https://pypi.org/project/shapiq">
    <img src="https://img.shields.io/pypi/v/shapiq.svg?color=blue" alt="PyPi">
  </a>
  
  <!-- PyPI status -->
  <a href="https://pypi.org/project/shapiq">
    <img src="https://img.shields.io/pypi/status/shapiq.svg?color=blue" alt="PyPi_status
  </a>
      
  <!-- License -->
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-brightgreen.svg" alt="mit_license">
  </a>
</p>


# SHAP-IQ: SHAP Interaction Quantification
> An interaction may speak more than a thousand main effects.

SHAP Interaction Quantification (short SHAP-IQ) is an **XAI framework** extending on the well-known `shap` explanations by introducing interactions to the equation.
Shapley interactions extend on indivdual Shapley values by quantifying the **synergy** effect between machine learning entities such as features, data points, or weak learners in ensemble models.
Synergies between these entities (also called players in game theory jargon) allows for a more intricate evaluation of your **black-box** models!

# 🛠️ Install
**shapiq** is intended to work with **Python 3.9 and above**. Installation can be done via `pip`:

```sh
pip install shapiq
```

# ⭐ Quickstart

## 📈 Compute n-SII values

## 📊 Visualize your Interactions

One handy way of visualizing interaction scores (up to order 2) are network plots.
You can see an example of such a plot below.
The nodes represent **attribution** scores and the edges represent the **interactions**.
The strength and size of the nodes and edges are proportional to the absolute value of the
attribution scores and interaction scores, respectively.

```python
from shapiq.plot import network_plot

network_plot(
    first_order_values=n_sii_first_order,  # first order n-SII values
    second_order_values=n_sii_second_order # second order n-SII values
)
```

The pseudo-code above can produce the following plot (here also an image is added):

<p align="center">
  <img width="400px" src="docs/source/_static/network_example.png" alt="network_plot_example">
</p>

## 📖 Documentation
The documentation for ``shapiq`` can be found [here](https://shapiq.readthedocs.io/en/latest/).
