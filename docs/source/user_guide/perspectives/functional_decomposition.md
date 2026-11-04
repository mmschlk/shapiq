# Functional decomposition

% intent: the statistical perspective. A model is a sum of components, one per subset of features; the
% components need a distribution to be well defined, and the choice of this distribution is what
% masking formalizes in Part II.
% sources: Fumagalli.2025 §2.1; Herbinger.2026 §2.1; Hoeffding.1948; Stone.1994; Hooker.2004;
%   Hooker.2007; Owen.2013; Chastaing.2012

## The functional ANOVA decomposition

% intent: F(x) as the sum of components f_S(x_S) over all subsets S: a constant, main effects, and pure
% interaction effects; the recursive definition f_S = F_S - sum of f_T over T ⊂ S, where F_S integrates
% out the other features.
% sources: Fumagalli.2025 Eq. (1); Herbinger.2026 Eq. (1); Stone.1994; Hooker.2004
% TODO eq: the decomposition and the recursive definition

## The role of the feature distribution

% intent: the components depend on the distribution P that integrates out the other features. For
% independent features they are zero-centered, orthogonal and unique; for dependent features they are
% not orthogonal. Preview: baseline, marginal, and conditional choices of P.
% sources: Fumagalli.2025 §2.1 and §3.1; Hooker.2007; Chastaing.2012

## Variance decomposition and Sobol' indices

% intent: for independent features the variance of F splits into the variances of the components;
% closed and total Sobol' indices and the superset measure summarize them. They return as global
% feature importance in Part III.
% sources: Fumagalli.2025 §2.1; Owen.2013; Sobol.2001; Hooker.2004
% TODO eq: V[F(X)] = sum over S of sigma_S^2; closed, total, and superset indices

## What the decomposition explains

% intent: main effects and interactions as functions; feature effect plots as estimates of components;
% additive models (GAMs) as models that are their own decomposition.
% sources: thesis §2.3.1; Hiabu.2023; Bordt.2023; Tan.2023
