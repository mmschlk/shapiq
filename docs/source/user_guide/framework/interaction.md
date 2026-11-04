# Interaction: how higher-order effects enter

% intent: the third operator. The explanation game contains effects of all orders; the interaction
% operator decides how much of the higher-order effects ends up in the explanation of a feature, a
% group, or an interaction. Shapley values and interactions are one choice here among several.
% sources: Fumagalli.2025 §3.3-3.4 (Thm. 4, Tables 2 and 3); Herbinger.2026 Def. 3, Table 1, App. B;
%   thesis §2.4.1

## Pure, partial, and full effects

% intent: pure: the influence of a feature in the absence of all others (Möbius coefficient); full: in the
% presence of all others (co-Möbius coefficient); partial: a weighted share of everything in between
% (Shapley value). The influence of higher-order interactions grows from pure to full.
% sources: Fumagalli.2025 §3.4; Herbinger.2026 Def. 3, Table 1
% TODO eq: pure nu(i) - nu(∅); partial: the Shapley value; full nu(N) - nu(N \ i)

## Individual, joint, and interaction influence

% intent: the unit: a single feature, a group of features taken together, or the synergy of a group;
% the 3x3 table of pure, partial, full x individual, joint, interaction in Möbius coefficients.
% sources: Fumagalli.2025 §3.3, Table 2; Herbinger.2026 App. B
% TODO table: Fumagalli.2025 Table 2

## A worked example

% intent: three independent features and F = x1 + x2 + x3 + x1 x2 + x1 x2 x3: the share of each fANOVA
% term in each of the nine explanations.
% sources: Fumagalli.2025 Table 3

## Partial effects of features

% intent: the Shapley value as the partial effect; the Banzhaf value and the semivalues with other
% weights; why efficiency matters for explanations.
% sources: thesis §2.4.1; Fumagalli.2025 §3.4; Banzhaf.1964; Weber.1988; Marichal.2011
% v2: Beta(alpha, beta) and weighted Banzhaf weights (shapiq/game/weights.py)

## Partial effects of groups and interactions

% intent: generalized values and Joint Shapley values for groups; interaction indices for interactions:
% SII, k-SII, STII, FSII, FBII, and Banzhaf interactions; generalized efficiency.
% sources: Marichal.2007; Harris.2022; Grabisch.1999; Bordt.2023; Sundararajan.2020; Tsai.2022;
%   Fumagalli.2023; Muschalik.2024 Table 1

## Choosing the order

% intent: the order k trades complexity for faithfulness, from the Shapley value (k = 1) to the Möbius
% transform (k = n); pairwise interactions already gain the most; Shapley-weighted faithfulness.
% sources: thesis Fig. 2.3 and Eqs. (2.31)-(2.33); Muschalik.2024 §4.1, Fig. 4

## In shapiq

% v1: index="SV", "k-SII", "STII", "FSII", "SII", "BV", "FBII", "Moebius" (pure), "Co-Moebius" (full),
%   "SGV", "JointSV", ... (shapiq/game_theory/indices.py); ExactComputer computes all of them.
% v2: interaction operators are transformations of games (shapley_game, proposed); Möbius and Fourier are
%   casts to another basis; discrete-derivative games in shapiq/game/discretederivative.py.
