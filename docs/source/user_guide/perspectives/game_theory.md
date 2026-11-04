# Cooperative game theory

% intent: the combinatorial perspective. The model is seen only through a set function that assigns a
% value to every coalition of features; the Möbius transform decomposes it into pure contributions, and
% values and interaction indices summarize it. Kept general: the explanation games follow in Part II
% ("Behavior").
% sources: thesis §2.4 (Defs. 2.4-2.7, Example 2.1, Table 2.1); Fumagalli.2025 §2.3 and App. C.2;
%   Muschalik.2024 §2; Grabisch.1999; Grabisch.2000; Fujimoto.2006
% open: running example. The thesis uses four founders of a startup (Example 2.1), the landing page the
%   latitude and longitude of a house in the California housing data. One of them throughout?

## Cooperative games

% intent: players, coalitions, and the value function nu: 2^N -> R; centering nu(∅) = 0; the grand
% coalition. A player need not be a feature.
% sources: thesis Def. 2.4; Muschalik.2024 §2

## Marginal contributions and discrete derivatives

% intent: the marginal contribution Delta_i(T); the discrete derivative Delta_S(T) generalizes it to
% groups and removes lower-order effects; positive means synergy, negative redundancy, zero additive
% independence.
% sources: thesis Eq. (2.29); Fumagalli.2025 §2.3; Grabisch.1999
% TODO eq: Delta_S(T) = sum over L ⊆ S of (-1)^{|S|-|L|} nu(T ∪ L)

## The Möbius transform

% intent: the unique decomposition of a game into pure contributions of coalitions (Harsanyi dividends);
% every game value is the sum of the Möbius coefficients of its subsets; the co-Möbius transform as the
% counterpart in the presence of all other players.
% sources: thesis Def. 2.5; Rota.1964; Harsanyi.1963; Grabisch.2000; Fumagalli.2025 §2.3
% TODO eq: m(S) = sum over T ⊆ S of (-1)^{|S|-|T|} nu(T), and nu(T) = sum over S ⊆ T of m(S)

## Values

% intent: the attribution problem; the Shapley value and its axioms (efficiency, dummy, symmetry,
% linearity); the Banzhaf value without efficiency; semivalues (weighted Banzhaf, Beta Shapley) as the
% family that contains both.
% sources: thesis §2.4.1 (Def. 2.6); Shapley.1951; Banzhaf.1964; Weber.1988; Dubey.1981; Marichal.2011
% v2: the weights of shapiq v2 are exactly these families (Beta(alpha, beta), weighted Banzhaf).

## Interaction indices

% intent: aggregating discrete derivatives: the Shapley interaction index (SII), the Banzhaf interaction
% index, and indices up to an order k (k-SII, STII, FSII) with generalized efficiency. Only the
% definitions here; their role in explanations is "Interaction" in Part II.
% sources: thesis Def. 2.7 and Eqs. (2.31)-(2.33); Grabisch.1999; Fujimoto.2006; Sundararajan.2020;
%   Tsai.2022; Bordt.2023

## Games in machine learning

% intent: many questions in machine learning define a game: explaining a prediction, the performance of
% a model, the value of training data. Points to Part II for the explanation games and to Part VII for
% the others.
% sources: thesis §2.4 (paragraph on cooperative game theory); Muschalik.2024 §2, Table 3; Covert.2021
