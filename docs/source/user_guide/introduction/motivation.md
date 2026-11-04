# Motivation

% Story points of this chapter, to be written out. The EU AI Act is not in references.bib yet.

- **Models make decisions.** Decisions that people used to make are delegated to models: in
  medicine, in finance, in hiring {cite:p}`Muschalik.2026{Sec. 1.1}`.
- **Learned, not programmed.** A model's decision logic is learned from data, not written down, so it
  cannot be read from its code {cite:p}`Muschalik.2026{Sec. 1.1}`.
- **The best models are the hardest to inspect.** Linear models and shallow trees are transparent but
  limited; ensembles, deep networks, and foundation models perform best and are opaque
  {cite:p}`Muschalik.2026{Sec. 2.2}`.
- **Equal performance, different reasons.** Models with the same accuracy can rely on different
  mechanisms, so accuracy alone does not say how a model decides {cite:p}`Muschalik.2026{Sec. 2.1}`.
- **Regulation asks for it.** The EU AI Act requires that the outputs of high-risk systems can be
  interpreted by those who deploy them {cite:p}`Muschalik.2026{Sec. 2.2}`.
- **Explainable AI.** Methods that make the behavior of models understandable to people.
- **Explaining after the fact.** Instead of trading performance for a simpler model, post-hoc
  explanations explain a trained model without changing it. This guide is about them.
