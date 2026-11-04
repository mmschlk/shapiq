# Showcase

% A showcase of the styles of the user guide (definitions, equations, citations, examples, tabs,
% dropdowns, footnotes), with the perspective-neutral definition of an explanation as its content.

```{admonition} Showcase
:class: seealso

This page tries out the styles of the user guide: a definition, equations and references to them,
citations, a table, an example, tabs, a dropdown, and a footnote. Its content is the general
definition of a feature-based explanation. The symbols of the {doc}`notation`, such as
$\players$ or $\mask$, link to their entry there.
```

A trained model maps inputs to outputs. A *feature-based explanation* relates the output back to
the input features: which features matter, how much, and together with which others
{cite:p}`Covert.2021,Fumagalli.2025`. Explanations of this kind range from Shapley values
{cite:p}`Lundberg.2017` to partial dependence plots {cite:p}`Friedman.2001`. Before we look at any
of them, we fix what they have in common, following {cite:t}`Muschalik.2026{Def. 2.1}`.

## The Definition

```{admonition} Definition 1.1 (Feature-based explanation)
:class: note

Let $\model : \mathcal{X} \to \R$ be a trained model on the features
$\players = \{1, \dots, \nplayers\}$.[^notation] A **feature-based explanation** is a map

$$
e : \mathcal{S} \to E, \qquad S \mapsto e(S),
$$ (eq-explanation)

where $\mathcal{S} \subseteq 2^{\players}$ is a collection of feature subsets, the **units**, and
$E$ is the **explanation space**. For each unit $S \in \mathcal{S}$, the explanation $e(S)$
describes how the features in $S$ influence the output of $\model$.
```

The map {eq}`eq-explanation` makes two choices and leaves a third open:

- The collection $\mathcal{S}$ fixes the **unit**: single features give attributions, larger
  subsets give groups or interactions (see {ref}`ug-units`).
- The space $E$ fixes the **form**: a number or a curve (see {ref}`ug-forms`).
- The definition says nothing about **how** $e(S)$ is computed (see {ref}`ug-open`).

(ug-units)=
## Units: What Is Explained

| Units $\mathcal{S}$ | The explanation assigns a value to | Example |
| --- | --- | --- |
| $\{\{i\} : i \in \players\}$ | each feature | Shapley values {cite:p}`Lundberg.2017` |
| a partition of $\players$ into groups | each group of features, taken together | grouped importance {cite:p}`Au.2022` |
| $\{S \subseteq \players : \lvert S \rvert \leq k\}$ | each feature and each interaction of up to $k$ features | Shapley interactions {cite:p}`Bordt.2023` |

```{admonition} Example
:class: tip

Take a house in the California housing data. An attribution assigns one number to its latitude
and one to its longitude. Interactions of order $k = 2$ also assign a number to the pair
$\{\text{latitude}, \text{longitude}\}$: their joint effect, which is the location of the house.
```

(ug-forms)=
## Forms: What an Explanation Looks Like

The explanation space $E$ decides what the reader sees. Most explanations are numbers, $E = \R$:
the attribution of a prediction to a feature, or the importance of a feature for the model.
Feature effects are curves. The partial dependence plot {cite:p}`Friedman.2001` explains a feature
$i$ by the function

$$
e(\{i\}) = \Big( z \mapsto \E\big[\model(z, X_{\players \setminus \{i\}})\big] \Big),
$$ (eq-pdp)

the average prediction when feature $i$ is set to $z$. Its explanation space is a space of
functions of a single feature.

(ug-open)=
## What the Definition Leaves Open

Definition 1.1 does not say how $e(S)$ is obtained. Each of the three lenses of
{doc}`perspectives/index` gives its own answer:

````{tab-set}

```{tab-item} Functional decomposition

The model is a sum of components, one per subset of features {cite:p}`Stone.1994,Hooker.2004`,

$$
\model(x) = \sum_{S \subseteq \players} f_S(x_S),
$$ (eq-fanova)

and the explanation of a unit is its component, $e(S) = f_S$.
```

```{tab-item} Taylor expansion

The model is expanded around a baseline $b$ {cite:p}`Deng.2024`,

$$
\model(x) = \model(b) + \sum_{i \in \players} \psi(i)
  + \sum_{S \subseteq \players,\, \lvert S \rvert > 1} J(S),
$$ (eq-taylor)

and the explanation of a unit collects the terms that involve exactly its features:
$e(\{i\}) = \psi(i)$, and $e(S) = J(S)$ for $\lvert S \rvert > 1$.
```

```{tab-item} Game theory

The model defines a cooperative game $\game : 2^{\players} \to \R$, a value for every coalition of
features {cite:p}`Shapley.1951,Grabisch.1999`, and the explanation of a unit is its value or
interaction index in this game, $e(S) = \phi(S)$.
```

````

```{important}
The three answers describe the same object. Let $\game_x(S) = \model_S(x)$ be the local game of an
instance $x$, where $\model_S$ is the model restricted to the features in $S$. Its Möbius transform
is the fANOVA component at $x$ {cite:p}`Fumagalli.2025{Cor. 1}`:

\begin{align}
m_x(S) &= \sum_{T \subseteq S} (-1)^{\lvert S \rvert - \lvert T \rvert} \, \game_x(T) \\
       &= f_S(x).
\end{align}

With a baseline in place of the data distribution, the components are also the Taylor terms of
{eq}`eq-taylor` {cite:p}`Fumagalli.2025{Thm. 1}`. {doc}`perspectives/three_views` shows this in detail.
```

```{dropdown} The fANOVA components in detail

The component of $S$ is the restricted model minus the components of all strict subsets of $S$
{cite:p}`Fumagalli.2025`,

$$
f_S(x) = \model_S(x) - \sum_{T \subsetneq S} f_T(x),
\qquad
\model_S(x) = \int \model(x) \, \mathrm{d}P(x_{\players \setminus S}),
$$ (eq-fanova-recursion)

where $P$ is a distribution of the features outside $S$. Choosing it, as a baseline, the marginal,
or the conditional distribution, is the topic of {doc}`framework/index`.
```

```{seealso}
{cite:t}`Covert.2021` on removing features as the principle behind most explanations, and
{cite:t}`Fumagalli.2025` on the unified framework of {doc}`framework/index`.
```

[^notation]: {cite:t}`Muschalik.2026` writes $D = \{1, \dots, d\}$ for the features; this guide
    writes $\players$ and $\nplayers$, as for the players of a game.
