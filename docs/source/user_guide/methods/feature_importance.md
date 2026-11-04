# Feature importance

% intent: global explanations that assign one number to each feature, group, or interaction for the
% whole model. Sections by behavior: sensitivity (variance explained) and risk (loss explained); the
% unit is given in each card.
% sources: Fumagalli.2025 §4.2, Fig. 1, App. E.2; thesis §2.3.2
% open: sections by behavior here, by unit in "Feature attribution": keep this asymmetry?

## Sensitivity: variance explained

### Sobol' indices

% card: masking marginal | behavior sensitivity | interaction pure (closed index) or full (total index) |
%   unit individual or joint (Fumagalli.2025 §4.2, Fig. 1)
% sources: Sobol.2001; Owen.2013

### Shapley effects

% card: masking marginal (Owen.2014) or conditional (Owen.2017) | behavior sensitivity | interaction
%   partial | unit individual (Fumagalli.2025 §4.2)
% sources: Owen.2014; Owen.2017

### H-statistic

% card: masking marginal | behavior sensitivity | interaction pure | unit interaction (Fumagalli.2025 Fig. 1)
% sources: Friedman.2008; Inglis.2022

### Superset importance

% card: masking marginal | behavior sensitivity | interaction full | unit interaction (Fumagalli.2025 Fig. 1)
% sources: Hooker.2004

## Risk: loss explained

### Permutation feature importance (PFI)

% card: masking marginal | behavior risk | interaction full | unit individual (Fumagalli.2025 Fig. 1)
% neighbors: CFI (conditional masking); SAGE (partial effect)
% sources: Breiman.2001; Fisher.2019

### Conditional feature importance (CFI)

% card: masking conditional | behavior risk | interaction full | unit individual (Fumagalli.2025 Fig. 1)
% neighbors: PFI (marginal masking)
% sources: Strobl.2008; Molnar.2024

### SAGE

% card: masking conditional in theory, marginal in practice | behavior risk | interaction partial | unit
%   individual (Fumagalli.2025 §4.2)
% neighbors: PFI (full effect); interventional SHAP (local behavior)
% sources: Covert.2020; Muschalik.2023

### Grouped feature importance

% card: masking marginal | behavior risk | interaction full | unit joint (not in Fumagalli.2025 Fig. 1)
% sources: Au.2022

### Interactions in the risk

% card: masking marginal | behavior risk | interaction pure or partial | unit interaction
% note: SAGE-like interactions of order two, as in Herbinger.2026 Fig. 4
% sources: Herbinger.2026; Covert.2020
