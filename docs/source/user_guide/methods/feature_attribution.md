# Feature attribution

% intent: local explanations that assign a number to each feature, group, or interaction for one
% prediction. Sections by unit: individual features, groups, interactions.
% sources: Fumagalli.2025 §4.1, Fig. 1, App. E.1; thesis §2.3.3

## Attributions to individual features

### Occlusion

% card: masking baseline | behavior local | interaction full | unit individual (Fumagalli.2025 §4.1)
% neighbors: baseline SHAP (partial instead of full effect)
% sources: Zeiler.2014

### Baseline SHAP

% card: masking baseline | behavior local | interaction partial | unit individual (Fumagalli.2025 Fig. 1)
% neighbors: interventional SHAP (marginal masking); occlusion (full effect)
% sources: Sundararajan.2020a; Lundberg.2017

### Interventional SHAP

% card: masking marginal | behavior local | interaction partial | unit individual (Fumagalli.2025 Fig. 1)
% neighbors: baseline and observational SHAP (masking); SAGE (behavior)
% sources: Lundberg.2017; Janzing.2020

### Observational SHAP

% card: masking conditional | behavior local | interaction partial | unit individual (Fumagalli.2025 §4.1)
% neighbors: interventional SHAP (marginal masking)
% sources: Lundberg.2017; Aas.2021; Frye.2021

### TreeSHAP

% card: masking path-dependent (tree-specific) | behavior local | interaction partial | unit individual
%   (not in Fumagalli.2025; the masking is none of the three classical ones, see "Other ways to remove
%   features" in Part II)
% sources: Lundberg.2020

### Gradient-based attributions

% card: masking baseline | behavior local | interaction: distributes Taylor interactions | unit individual
%   (Fumagalli.2025 §4.1; LRP-epsilon is a pure effect with baseline masking in Fig. 1)
% covers: Grad×Input, Integrated Gradients, DeepLIFT, LRP, Expected Gradients, SmoothGrad
% sources: Sundararajan.2017; Shrikumar.2017; Bach.2015; Erion.2021; Smilkov.2017; Deng.2024
% open: not computed by shapiq

### LIME

% note: related to SHAP, but not covered by the framework (Fumagalli.2025 §7)
% open: a short note or leave out? Ribeiro et al. (2016) is missing in references.bib.

## Attributions to groups of features

### ArchAttribute

% card: masking baseline | behavior local | interaction pure | unit joint (Fumagalli.2025 Fig. 1)
% sources: Tsang.2020

### Joint Shapley values

% card: masking baseline | behavior local | interaction partial | unit joint (Fumagalli.2025 Fig. 1)
% sources: Harris.2022; Marichal.2007

### Occlusion of patches

% card: masking baseline | behavior local | interaction full | unit joint (Fumagalli.2025 Fig. 1)
% sources: Zeiler.2014

### PredDiff

% card: masking conditional | behavior local | interaction full | unit joint (Fumagalli.2025 Fig. 1)
% sources: Zintgraf.2017; Blucher.2022; Robnik-Sikonja.2008

## Attributions to interactions

### Möbius interactions

% card: masking any | behavior local | interaction pure | unit interaction (Fumagalli.2025 Table 2)
% note: the pure local interaction is the fANOVA component at x; all orders give the full decomposition
% sources: Grabisch.2000; Fumagalli.2025 Cor. 1; Kang.2024

### Shapley interactions

% card: masking any | behavior local | interaction partial | unit interaction
% covers: SII, k-SII (marginal masking in Fumagalli.2025 Fig. 1), STII, FSII (Faith-SHAP: baseline
%   masking in Fig. 1)
% sources: Grabisch.1999; Lundberg.2019; Bordt.2023; Sundararajan.2020; Tsai.2022; Muschalik.2024

### Banzhaf interactions

% card: masking any | behavior local | interaction partial, with Banzhaf weights (no efficiency) | unit
%   interaction (not in Fumagalli.2025 Fig. 1)
% covers: BII, FBII, weighted Banzhaf interactions (as in FIxLIP)
% sources: Grabisch.1999; Tsai.2022; Marichal.2011; Baniecki.2025

### Shapley-GAM

% card: masking conditional | behavior local | interaction pure | unit interaction (Fumagalli.2025 Fig. 1)
% sources: Bordt.2023

### Taylor interactions

% card: masking baseline | behavior local | interaction: Taylor allocation | unit interaction
% sources: Deng.2024; Janizek.2021
% open: not computed by shapiq
