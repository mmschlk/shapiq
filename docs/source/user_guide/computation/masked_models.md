# Estimating masked models

% intent: baseline masking needs one model call per masked model; marginal and conditional masking need
% expectations, estimated from data or from a model of the data.
% sources: Fumagalli.2025 §4.3; thesis Eq. (2.18); Olsen.2024

## Background data

% intent: Monte Carlo estimates of marginal masking from a background sample; how many rows; compressing
% the background data.
% sources: thesis Eq. (2.18); Baniecki.2025a

## Conditional samplers

% intent: estimating conditional masking: Gaussian and Gaussian copula models, generative models,
% in-context learners; their assumptions and cost.
% sources: Aas.2021; Olsen.2022; Olsen.2024; Frye.2021; Rundel.2024

## Masking along the paths of trees

% intent: path-dependent masking estimates the conditional distribution from the training data that
% reached each node.
% sources: Lundberg.2020

## In shapiq

% v1: imputers and their sample_size; TreeExplainer modes.
% v2: maskers.
