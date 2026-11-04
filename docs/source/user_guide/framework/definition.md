# From a model to an explanation

% intent: the operational definition the whole guide uses, phi = I(B(M F)), and the four questions it
% answers. Every method card of Part III points back here: its four columns are the arguments.
% sources: Herbinger.2026 Defs. 1-3, Table 1; Fumagalli.2025 §3 and §4.3; thesis Defs. 2.2 and 2.3
% open: running example for Part II. Suggestion: the toy model of Herbinger.2026 §3.1 (Table 2),
%   F(X) = 3 X1 X2 + X3 + 2 X4 with X4 | X3 ~ N(X3, 1): one interaction (X1 X2) and one dependency
%   (X4 on X3), so that every operator changes something visible.

## Three operators

% intent: masking maps F to masked models F_S; behavior maps the masked models to an explanation game nu;
% interaction maps nu to an explanation phi on the units.
% TODO eq: \phi = \inter(\behav(\mask \model))
% TODO fig: model -> masked models -> explanation game -> explanation

## Explanations are set functions

% intent: phi assigns a value to sets of features, so an explanation is itself a set function, i.e. a
% game. Feature effects fit too: one explanation per value of the feature, i.e. a batch of games.
% v2: matches "Explanation: an interpretation of a transformed game" (CONTEXT.md); effects as game batches
%   or as partial games over feature values.

## Four questions behind every explanation

% intent: (1) which behavior: local or global? (2) which unit: individual, joint, or interaction?
% (3) which masking: baseline, marginal, or conditional? (4) how much higher-order influence: pure,
% partial, or full? Returns as a flowchart in Part VI.
% sources: Fumagalli.2025 §4.3; Herbinger.2026 §2.2
% open: the unit (question 2) as a fourth operator, or part of "Interaction"? Herbinger.2026 counts four
%   dimensions; this scaffold treats the unit inside "Interaction".
