# How the operators interact

% intent: for some models and data the choices of the operators do not matter; when explanations
% disagree, the difference can be traced to a single operator. Leads to regional explanations, which
% restrict the explanation to regions where the operators agree.
% sources: Fumagalli.2025 Thm. 2; Herbinger.2026 §3 (Def. 4, Thms. 1-3, Def. 5); Krishna.2024

## When the choices do not matter

% intent: for linear models marginal and baseline masking agree (b = E[X]); for independent features
% conditional and marginal masking agree; without interactions pure, partial, and full effects agree.
% sources: Fumagalli.2025 Thm. 2; Herbinger.2026 §1

## Why explanations disagree

% intent: the disagreement problem: with the behavior and the unit fixed, two explanations can only
% differ in their masking or in their interaction operator.
% sources: Herbinger.2026 Def. 4; Krishna.2024

## Disagreement from interactions

% intent: with the same masking, the difference between two interaction operators is a weighted sum of
% pure interactions.
% sources: Herbinger.2026 Thms. 1 and 2

## Disagreement from feature dependencies

% intent: with the same interaction operator, the difference between conditional and marginal masking is
% driven by the ratio of the conditional to the marginal density.
% sources: Herbinger.2026 Thm. 3

## Regional explanations

% intent: partition the input space into regions where the disagreement is small; regional explanations
% restrict masking and behavior to a region; the trade-off between agreement and the number of regions.
% sources: Herbinger.2026 §3.2-3.5 (Def. 5, Eqs. (2) and (3))
