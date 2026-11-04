# Regional explanations

% intent: methods that split the input space into regions and explain each region, reducing the
% disagreement caused by interactions or dependencies. The mechanism is in Part II ("How the operators
% interact"); here the methods, by the disagreement they reduce.
% sources: Herbinger.2026 §4, Table 4, App. D; thesis §2.3.1 (regional feature effects)
% card for regional methods: the disagreement a method reduces (full to pure, partial to pure,
%   conditional to marginal), and the maskings and behaviors it supports (Herbinger.2026 Table 4)

## Regions from clustering

### VINE

% card: reduces full to pure | masking marginal | behavior local (Herbinger.2026 Table 4)
% sources: Britton.2019

### CohEx

% card: reduces partial to pure | masking marginal | behavior local (Herbinger.2026 Table 4)
% sources: Meng.2025

## Regions from recursive partitioning

### REPID

% card: reduces full to pure | masking marginal | behavior local (Herbinger.2026 Table 4)
% sources: Herbinger.2022

### GADGET

% card: reduces full to pure and partial to pure | masking marginal or conditional | behavior local
%   (Herbinger.2026 Table 4)
% sources: Herbinger.2024

### FD-Trees

% card: reduces full to pure and partial to pure | masking marginal | behavior local (Herbinger.2026 Table 4)
% sources: Laberge.2024

### Transformation trees

% card: reduces conditional to marginal | behavior local and risk | interaction full or pure
%   (Herbinger.2026 Table 4)
% sources: Molnar.2024

### GRANITE

% card: reduces all three | any masking, behavior, and interaction; also groups and interactions
%   (Herbinger.2026 Table 4)
% sources: Herbinger.2026
% open: not computed by shapiq (code in github.com/gablabc/GRANITE)
