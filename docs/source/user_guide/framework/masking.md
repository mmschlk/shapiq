# Masking: how features are removed

% intent: the first operator. What it means for a feature to be absent, the three classical choices,
% and the other ways shapiq removes features.
% sources: Fumagalli.2025 §3.1 (Defs. 1-3, Thm. 2, Table 1); Herbinger.2026 Def. 1, Tables 1 and 3;
%   thesis Def. 2.2 and Eqs. (2.10)-(2.12); Covert.2021

## Masked models

% intent: the masking operator maps F to the masked models F_S, the model restricted to the features in S.
% sources: Herbinger.2026 Def. 1; thesis Def. 2.2
% TODO eq: F_S(x) = (\mask \model)(x, S)

## Baseline masking

% intent: replace absent features by a fixed baseline b; the baseline fANOVA; cheap, but the result
% depends on b.
% sources: Fumagalli.2025 Def. 1; thesis Eq. (2.10); Sturmfels.2020

## Marginal masking

% intent: integrate absent features over their joint marginal distribution (interventional); the marginal
% fANOVA; breaks dependencies and can evaluate the model where there is no data (extrapolation).
% sources: Fumagalli.2025 Def. 2; thesis Eq. (2.11); Janzing.2020; Hooker.2007

## Conditional masking

% intent: integrate absent features over their distribution given the present ones (observational); the
% conditional fANOVA; respects dependencies, but needs a model of the conditional distributions.
% sources: Fumagalli.2025 Def. 3; thesis Eq. (2.12); Aas.2021; Frye.2021

## True to the model or true to the data

% intent: the trade-off between the three choices: the influence of the feature distribution grows from
% baseline over marginal to conditional masking.
% sources: Chen.2020; Janzing.2020; Fumagalli.2025 §3.1

## How masking changes an explanation

% intent: a worked example: a linear model and a model with one interaction, on two correlated
% features; main and interaction effects under the three maskings.
% sources: Fumagalli.2025 Thm. 2, Table 1, App. D.1

## Other ways to remove features

% intent: masking along the decision paths of trees (path-dependent); retraining without the features;
% recontextualizing in-context learners such as TabPFN; masking tokens, image patches and superpixels,
% and the nodes of a graph.
% sources: Lundberg.2020 (path-dependent); Covert.2021 (retraining); Rundel.2024 and Hollmann.2025
%   (TabPFN); Achanta.2012 (superpixels); Muschalik.2025 (graph nodes)

## In shapiq

% v1: imputers (BaselineImputer, MarginalImputer, GaussianImputer, GaussianCopulaImputer,
%   GenerativeConditionalImputer, TabPFNImputer); TreeExplainer(mode="pathdependent" or "interventional").
% v2: maskers are games whose values are masked inputs; MaskedPredictorGame(masker, model) applies the
%   model (shapiq/game/masked_predictor.py, a sketch so far).
