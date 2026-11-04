# Taylor expansion

% intent: the analytical perspective, home of gradient-based attributions. Expanding a model around a
% baseline splits it into terms per feature and per group of features (Taylor interactions); gradient
% methods differ in how they distribute these terms. "One decomposition, three views" shows that these
% terms are the components of the baseline fANOVA.
% sources: Fumagalli.2025 §2.2 and App. C.1; Deng.2024
% open: shapiq does not compute gradient-based attributions. Keep this perspective short and conceptual,
%   or also place the gradient methods in Part III, marked as not computed by shapiq?

## Expanding a model around a baseline

% intent: the multivariate Taylor expansion of F around a baseline b; generic main effects psi(i) and
% generic interaction effects J(S), collecting the terms by the features they differentiate.
% sources: Fumagalli.2025 §2.2; Deng.2024
% TODO eq: F(x) = F(b) + sum_i psi(i) + sum_{|S| > 1} J(S)

## Gradient-based attributions

% intent: Grad×Input as a first-order approximation; Integrated Gradients as a path integral of
% gradients (the Aumann-Shapley value); DeepLIFT, LRP, Expected Gradients, SmoothGrad. All of them
% distribute Taylor interactions among the features.
% sources: Fumagalli.2025 §4.1 (influence via gradients); Sundararajan.2017; Shrikumar.2017; Bach.2015;
%   Erion.2021; Smilkov.2017; Ancona.2018; Aumann.1974

## Taylor interactions

% intent: the interaction terms of the expansion; fourteen attribution methods unified by how they
% allocate Taylor interactions; Integrated Hessians for pairwise interactions.
% sources: Deng.2024; Janizek.2021

## The choice of baseline

% intent: the baseline b decides what "absent" means, and it strongly affects gradient attributions.
% Foreshadows baseline masking in Part II.
% sources: Sturmfels.2020; Fumagalli.2025 §4.1
