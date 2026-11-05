# Feature-Based Explanations

% Story points of this chapter, to be written out. The illustrations and the definition sit after
% the story points they belong to.

- **Features are the universal language of models.** Whatever its architecture or data, every model
  receives its input as features, so explanations in terms of features apply to any model
  {cite:p}`Muschalik.2026{Sec. 2.2},Covert.2021,Fumagalli.2025`.
- **Features take many forms.** The columns of a table, the pixels or patches of an image, the tokens
  of a text, the nodes of a graph.

```{shapiq-illustration} vocabulary
:anchor: illustration-vocabulary
:alt: Four inputs of machine learning models, a row of a table, an image, a sentence, and a molecule, each split into its features: the columns, the patches, the tokens, and the atoms. One model F reads the features and predicts.

Features are the vocabulary of every model. Whatever the input, the model reads its features, and a
feature-based explanation speaks about them. Point at a feature to see where it comes from.
```

- **A running example.** A house in the California housing data: which features drive its predicted
  price? Latitude and longitude matter only together, as the location of the house.

```{admonition} Definition 2.1 (Feature-Based Explanation)
:class: note

Let $\model : \featurespace \to \outputspace$ be a trained model on the features
$\players = \{1, \dots, \nplayers\}$. A **feature-based explanation** is a map

$$
\expl : \units \to \explspace, \qquad S \mapsto \expl(S),
$$ (eq-feature-based-explanation)

where $\units \subseteq 2^{\players}$ is a collection of sets of features, the **units**, and
$\explspace$ is the **explanation space**. For each unit $S \in \units$, the explanation $\expl(S)$
describes how the information in the features of $S$ influences the output of the model
{cite:p}`Muschalik.2026{Def. 2.1}`.
```

- **The units decide the question.** Single features give attributions, groups of features their
  joint effect, and interactions their synergy or redundancy.
- **The explanation space decides the form.** A number, such as an attribution or an importance, or
  a curve, such as a feature effect in a partial dependence plot.

```{shapiq-illustration} units-and-forms
:anchor: illustration-units-and-forms
:alt: Explanations of one house of the California housing data, for each choice of the units (single features, a group of features, or pairs of features) and of the explanation space (numbers or curves): bars of Shapley values, curves of the prediction as a feature varies, the value of a group, surfaces of two features, and a network of pairwise interactions.

Definition 2.1 at work on one house. The units decide what is explained, the explanation space what
the explanation looks like. Pick the units and the space, then the features.
```
- **The scope.** Local explanations explain one prediction, global ones the whole model, and regional
  ones a region of the input space in between {cite:p}`Herbinger.2026`.
- **Model-agnostic or model-specific.** An explanation probes the model with modified inputs, or it
  exploits the model's structure, such as its gradients or the paths of its trees.
- **One principle behind most explanations.** Remove the information of some features and observe
  how the output of the model changes {cite:p}`Covert.2021`.
- **What is not covered.** Counterfactual, example-based, and concept-based explanations do not
  explain through subsets of features {cite:p}`Muschalik.2026{Sec. 2.3}`.
- **What the definition leaves open.** Definition 2.1 says nothing about how $\expl(S)$ is computed:
  each of the three lenses of this part answers this in its own language.
