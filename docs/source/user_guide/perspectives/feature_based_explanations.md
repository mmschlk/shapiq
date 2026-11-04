# Feature-Based Explanations

% Story points of this chapter, to be written out. The EU AI Act is not in references.bib yet.

- **Opaque models make decisions.** The models that perform best are the hardest to inspect, and
  regulation such as the EU AI Act asks that their outputs can be interpreted
  {cite:p}`Muschalik.2026{Sec. 2.2}`.
- **Explaining after the fact.** Instead of trading performance for a simpler model, post-hoc
  explanations explain a trained model without changing it.
- **Features as common ground.** Every model takes features as input, so explanations in terms of
  features work across model classes and types of data {cite:p}`Covert.2021,Fumagalli.2025`.
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
