# One decomposition, three views

% intent: the hinge between Part I and Part II. The three perspectives describe the same object, a
% decomposition of the model into pure effects of feature subsets, and differ in what they assume and
% need. This is why one framework can hold methods from all three traditions.
% sources: Fumagalli.2025 Thm. 1, Cor. 1, Remark 1; Herbinger.2026 §2.2

## fANOVA components are Möbius coefficients

% intent: the Möbius transform of the local explanation game at x is the fANOVA component evaluated at
% x; both are instances of Möbius inversion.
% sources: Fumagalli.2025 Cor. 1 and Remark 1

## Baseline fANOVA components are Taylor interactions

% intent: with the distribution concentrated at a baseline b, the fANOVA components at x are the generic
% main and interaction effects of the Taylor expansion around b.
% sources: Fumagalli.2025 Thm. 1; Deng.2024

## What each view needs and offers

% intent: fANOVA needs a distribution and offers functions and shares of variance; Taylor needs gradients
% and offers cheap local attributions; game theory needs only queries to the model and offers axioms and
% a common language for all aggregations.
% open: a comparison table or prose?

## From three views to one framework

% intent: the framework of Part II takes the distribution from fANOVA (masking), the quantity of interest
% from the explanation goal (behavior), and the aggregation from game theory (interaction).
% sources: Fumagalli.2025 §3; Herbinger.2026 §2.2
