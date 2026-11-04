# Feature effects

% intent: local behavior evaluated along the range of a feature or a pair and shown as a curve or
% surface: how the prediction changes with the value of the feature. In the framework, a batch of local
% explanations, one per value on the grid.
% sources: Fumagalli.2025 §4.1 (Cor. 2), Fig. 1; thesis §2.3.1; Herbinger.2026 §5
% open: shapiq has no feature effect plots yet (v2: batches of games or partial games over feature values).

## Effects of single features

### Individual conditional expectation (ICE)

% card: masking baseline | behavior local | interaction full | unit individual (Fumagalli.2025 Fig. 1)
% neighbors: PDP (the average of the ICE curves)
% sources: Goldstein.2015

### Partial dependence plot (PDP)

% card: masking marginal | behavior local, averaged over the data | interaction pure | unit feature or
%   group (Fumagalli.2025 Fig. 1, Cor. 2)
% neighbors: M-plot (conditional masking); ICE (one curve per instance)
% sources: Friedman.2001; thesis Eqs. (2.16)-(2.19)

### Marginal plot (M-plot)

% card: masking conditional | behavior local, averaged over the data | interaction pure | unit feature or
%   group (Fumagalli.2025 Fig. 1, Cor. 2)
% neighbors: PDP (marginal masking)
% sources: Apley.2020; Friedman.2001

### Accumulated local effects (ALE)

% card: masking conditional | behavior local | interaction pure, via local gradients | unit individual
%   (Fumagalli.2025 Fig. 1, §4.1)
% sources: Apley.2020
% open: not computed by shapiq

## Effects of feature pairs

### Two-dimensional partial dependence

% card: masking marginal | behavior local, averaged over the data | interaction pure | unit joint
% sources: Friedman.2001

### Pure interaction effects

% card: masking marginal | behavior local | interaction pure | unit interaction
% note: the second-order fANOVA component as a surface, as in Herbinger.2026 Fig. 2 and Fig. 3b
% sources: Herbinger.2026; Hooker.2004
