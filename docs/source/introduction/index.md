# What Are Shapley Interactions?

Shapley values are the go-to method of explainable AI: they are easy to interpret and
theoretically well-founded. They do not show, however, how features work *together*. Shapley
interactions do. This page explains what interactions are, what Shapley values miss, and how
Shapley interactions fill the gap.

## What Are Interactions?

Imagine predicting house prices with the California housing dataset. Features such as the
location, the median income, and the age of a house all play a role. The location is described by
two features: the longitude (west to east) and the latitude (north to south).

Neither of them alone tells the whole story: it is their *combination*, the exact location, that
drives the price. Take two otherwise similar houses near the ocean:

- close to San Francisco: \$454,000,
- far from the cities: \$210,000.

The difference is explained neither by the longitude nor by the latitude alone, but by their
**interaction**: the effect of one feature (the longitude) depends on the value of another (the
latitude).

## What Shapley Values Miss

Shapley values {cite:p}`Shapley.1951` distribute a prediction fairly among the features. Explain
the house in San Francisco, with a predicted price of \$454,000, and both the longitude and the
latitude contribute positively:

```{figure} ../_static/images/motivation_sv.png
---
figclass: shapiq-figure
alt: >-
  A force plot of the Shapley values of a house in San Francisco: the longitude and the latitude
  both raise the predicted price.
---
The Shapley values of the house in San Francisco.
```

But a Shapley value merges the feature's own effect and its interactions with other features into
one number. So we cannot tell

- how much of a feature's influence is its own and how much it shares with other features, and
- which features it interacts with.

## Shapley Interactions

Shapley interactions {cite:p}`Grabisch.1999,Bordt.2023` split the effects into the
**contributions of single features** and the **interactions of groups of features**: instead of
one value per feature, every group of features can get a value. Back to the house: its
explanation with Shapley interactions up to order 2 shows that

- the longitude has a large effect of its own: the house is close to the ocean,
- the latitude has little effect of its own: its contribution comes from its interaction with the
  longitude. In the figure below, the latitude vanishes and the interaction of the latitude and
  the longitude appears; together, they locate the expensive area around San Francisco.

```{figure} ../_static/images/motivation_sv_and_si.png
---
figclass: shapiq-figure
alt: >-
  Force plots of the Shapley values and of the Shapley interactions of order 2 of the same house:
  the interactions credit the latitude and the longitude jointly.
---
The same house explained with Shapley values (top) and with Shapley interactions of order 2
(bottom).
```

Shapley interactions read like the Shapley values you know from SHAP, but they tell more: they
**uncover the synergies and redundancies** that Shapley values average away.

## Balancing Insight and Complexity

The **maximum interaction order** sets how detailed an explanation is. Order 2 explains pairs of
features (like the longitude and the latitude) besides the single features; higher orders explain
larger groups. Each order adds insight, but also many more values to read, so the right order
depends on your question.

```{tip}
Start with pairs (`max_order=2`), and look at higher orders where pairs do not explain enough.
```

## Shapley Interactions with shapiq

`shapiq` computes Shapley values and interactions for any model: exactly for tree ensembles,
nearest neighbors, and kernel methods, and with approximators for all other models. The
{doc}`start` shows the first steps, and the {doc}`examples </examples>` show `shapiq` at work,
from scikit-learn models to vision transformers.

```{admonition} How shapiq started
:class: note

A few years ago, we attended a talk that introduced Shapley interactions, and we quickly realized
that they were exactly what we needed. The presenter also mentioned that computing them is
extremely hard, if not impossible. We started experimenting and soon had a first approach to
compute Shapley interactions for any machine learning model {cite:p}`Fumagalli.2023`. The interest
of the community showed us that this research gap was a chance to pursue our own interests and to
create impact for others: `shapiq` was born {cite:p}`Muschalik.2024`.
```
