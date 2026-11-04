# What is a feature-based explanation?

% intent: the perspective-neutral definition every later chapter refers to. Kept short: the
% operational definition follows in Part II ("From a model to an explanation").
% sources: thesis §2.2-2.3 (Def. 2.1, Fig. 2.2); Covert.2021; Fumagalli.2025 §1

## Post-hoc explanations

% intent: why explain a trained model after the fact; the trade-off between interpretability and
% performance; post-hoc explanations vs. inherently interpretable models.
% sources: thesis §2.2, Fig. 2.2a

## Explaining a model by its features

% intent: the definition. A feature-based explanation picks a collection of feature subsets (the units)
% and assigns each unit S an object e(S) in an explanation space E. It says nothing about how e(S) is
% computed; the three perspectives of this part each answer "what is e(S)?".
% sources: thesis Def. 2.1
% TODO eq: e : \mathcal{S} \to E, S \mapsto e(S), with \mathcal{S} \subseteq 2^{\players}

## Units and forms of explanations

% intent: the unit (single features, groups, interactions) is fixed by the collection of subsets, the
% form (a number, a curve) by the explanation space. Preview of the families of Part III: attributions
% and importances are numbers, effects are curves.
% sources: thesis §2.3 (introductions of 2.3.1-2.3.3); Fumagalli.2025 §3.3

## Local, regional, and global explanations

% intent: the scope of an explanation: one prediction, a region of the input space, the whole model.
% sources: thesis §2.2, Fig. 2.2b; Herbinger.2022; Laberge.2024; Herbinger.2026

## Model-agnostic and model-specific explanations

% intent: probing a black box with perturbed inputs vs. exploiting its structure (gradients, trees).
% Foreshadows Part IV: the same explanation can often be computed either way.
% sources: thesis §2.2

## Removing features

% intent: the principle behind most feature-based explanations: remove the information of some features
% and observe how the output or the performance of the model changes. Sets up masking in Part II.
% sources: Covert.2021; thesis §2.3 (paragraph on feature removal)
