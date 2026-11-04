# Computing the interaction operator

% intent: pure and full effects need a few game evaluations per term; partial effects (Shapley values and
% interactions) need all 2^n coalitions. They are computed exactly for some model classes and
% approximated otherwise.
% sources: Fumagalli.2025 §4.3 (explanation cost); thesis §2.4.2, Table 2.1; Muschalik.2024 Table 2, §4.2

## The cost of each effect

% intent: pure and full individual and joint effects need two game evaluations, pure and full
% interactions of a set of size s need 2^s, partial effects need 2^n or an approximation.
% sources: Fumagalli.2025 §4.3

## Exact computation by brute force

% intent: evaluate all coalitions; feasible up to about 20 players.

## Exact computation for model classes

% intent: linear models, tree ensembles (TreeSHAP, TreeSHAP-IQ, interventional TreeSHAP), graph neural
% networks (GraphSHAP-IQ), nearest-neighbor models, product-kernel models and Gaussian processes.
% sources: thesis §2.4.2 (model-specific computation); Lundberg.2020; Yu.2022; Zern.2023;
%   Muschalik.2024a; Muschalik.2025; Jia.2019a; Wang.2023; Wang.2024; Mohammadi.2025; Mohammadi.2026;
%   Wettenstein.2026; Wettenstein.2026a; Nadel.2026

## Approximation by sampling permutations

% sources: Castro.2009; Mitchell.2022; Sundararajan.2020; Tsai.2022; Fumagalli.2023

## Approximation by Monte Carlo with sample reuse

% intent: maximum sample reuse; Unbiased KernelSHAP, SHAP-IQ, SVARM, SVARM-IQ; stratification.
% sources: thesis Eq. (2.35); Covert.2021a; Fumagalli.2023; Kolpaczki.2024; Kolpaczki.2024a; Wang.2023a

## Approximation by regression

% intent: the Shapley value as a weighted least squares problem; KernelSHAP, LeverageSHAP, kADD-SHAP,
% PolySHAP; KernelSHAP-IQ and regression for FSII and FBII.
% sources: thesis Eq. (2.37); Charnes.1988; Lundberg.2017; Covert.2021a; Musco.2025; Pelegrina.2023;
%   Fumagalli.2026; Fumagalli.2024; Tsai.2022

## Approximation with surrogates and sparsity

% intent: fit a surrogate of the game whose values are exact (trees, sparse Möbius or Fourier
% representations): RegressionMSR, ProxySHAP, SPEX, ProxySPEX.
% sources: Witter.2025; Thies.2026; Kang.2024; Kang.2025; Butler.2025

## Amortized explanations

% intent: learn a model that predicts explanations (FastSHAP, InstaSHAP).
% sources: Jethani.2022; Covert.2024; Enouen.2025
% open: not in shapiq; mention or leave out?

## Choosing an algorithm

% intent: which algorithm for which index, number of players, budget, and domain; benchmark results.
% sources: Muschalik.2024 §4.2, Fig. 5

## In shapiq

% v1: ExactComputer, TreeExplainer, GraphExplainer, the approximators (shapiq.approximator).
% v2: algorithms behind the transformations, with an explicit budget, returning the same result game.
