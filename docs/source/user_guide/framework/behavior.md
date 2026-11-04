# Behavior: what is explained

% intent: the second operator. It turns the masked models into the explanation game, a value for every
% set of features. The choice of behavior decides between local and global explanations.
% sources: Fumagalli.2025 §3.2 (Defs. 4-6, Thm. 3); Herbinger.2026 Def. 2, Table 1; thesis Def. 2.3 and
%   Eqs. (2.13)-(2.15)

## Explanation games

% intent: the behavior operator maps the masked models to a set function nu.
% sources: Herbinger.2026 Def. 2; thesis Def. 2.3
% TODO eq: \game(S) = (\behav(\mask \model))(S)

## Local behavior: one prediction

% intent: nu(S) = F_S(x) for the explained instance x; its Möbius transform is the fANOVA component at x.
% sources: Fumagalli.2025 Def. 4, Cor. 1; thesis Eq. (2.13)

## Global behavior: sensitivity

% intent: nu(S) = V[F_S(X)], the variance of the masked model: how much of the variation of the
% prediction the features in S explain; for independent features its Möbius transform is the variance
% of the fANOVA component.
% sources: Fumagalli.2025 Def. 5, Thm. 3; thesis Eq. (2.14); Owen.2013

## Global behavior: risk

% intent: nu(S) = -E[loss(F_S(X), Y)], the performance of the masked model; needs labels; its Möbius
% transform is the improvement in risk obtained by S.
% sources: Fumagalli.2025 Def. 6; thesis Eq. (2.15); Covert.2020

## Which output to explain

% intent: the explained output must be a number: the prediction for regression; for classification a
% class probability or a logit, which changes the explanation; multiclass outputs.
% sources: thesis §2.3 (note after Def. 2.2)
% open: Noé et al. (2024) on multiclass probabilities is cited in the thesis but missing in references.bib.

## Regional behavior

% intent: restricting the behavior (and the distribution of the masking) to a region of the input space;
% defined here, used in "How the operators interact" and in "Regional explanations" (Part III).
% sources: Herbinger.2026 §3.2, Eq. (2)

## In shapiq

% v1: games of shapiq_games (TabularLocalExplanation, TabularGlobalExplanation); the explainers build the
%   local game. TabularGlobalExplanation compares masked with full predictions, without labels: there is
%   no game with labels (risk) and no variance game (sensitivity) yet.
% v2: games built from maskers and models; global behavior perhaps as a reduction over a batch of games
%   (open design question).
