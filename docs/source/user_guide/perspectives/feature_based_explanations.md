# Feature-Based Explanations

% Story points of this chapter, to be written out.

- **Features are the universal language of models.** Whatever its architecture or data, every model
  receives its input as features, so explanations in terms of features apply to any model
  {cite:p}`Muschalik.2026{Sec. 2.2},Covert.2021,Fumagalli.2025`.
- **Features take many forms.** The columns of a table, the pixels or patches of an image, the tokens
  of a text, the nodes of a graph.
- **A running example.** A house in the California housing data: which features drive its predicted
  price? Latitude and longitude matter only together, as the location of the house.
- **The definition.** A feature-based explanation $e : \mathcal{S} \to E$ assigns each unit $S$ of a
  collection $\mathcal{S} \subseteq 2^{\players}$ an object $e(S)$ in an explanation space $E$
  {cite:p}`Muschalik.2026{Def. 2.1}`.
- **The units decide the question.** Single features give attributions, groups of features their
  joint effect, and interactions their synergy or redundancy.
- **The explanation space decides the form.** A number, such as an attribution or an importance, or
  a curve, such as a feature effect in a partial dependence plot.
- **The scope.** Local explanations explain one prediction, global ones the whole model, and regional
  ones a region of the input space in between {cite:p}`Herbinger.2026`.
- **Model-agnostic or model-specific.** An explanation probes the model with modified inputs, or it
  exploits the model's structure, such as its gradients or the paths of its trees.
- **One principle behind most explanations.** Remove the information of some features and observe
  how the output of the model changes {cite:p}`Covert.2021`.
- **What is not covered.** Counterfactual, example-based, and concept-based explanations do not
  explain through subsets of features {cite:p}`Muschalik.2026{Sec. 2.3}`.
- **What the definition leaves open.** It says nothing about how $e(S)$ is computed: each of the
  three lenses of this part answers this in its own language.
