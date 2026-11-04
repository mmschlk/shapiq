# Estimating explanation games

% intent: evaluating the explanation game for many coalitions: local games need the masked models at one
% instance, global games an average over data; batching and storing evaluations.
% sources: Fumagalli.2025 §4.3; Muschalik.2024 §4 (pre-computed games)

## Local and global games

% intent: one instance vs. a sample of the data; how many instances a global game needs.
% sources: Covert.2020; Fumagalli.2025 §4.3

## Evaluating many coalitions at once

% intent: batching coalitions into few model calls; vectorized value functions.
% v2: batches of games and coalitions, broadcasting, JAX (jit, vmap)

## Precomputing and storing games

% intent: storing all game values once for small games; reproducibility and reuse.
% sources: Muschalik.2024 §4

## In shapiq
