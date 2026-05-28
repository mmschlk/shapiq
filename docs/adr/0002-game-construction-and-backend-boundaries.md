# Game Construction And Backend Boundaries

`Game` is a real base abstraction with `n_players`, `target_shape`, and a coalition-to-value call contract. Existing callables that already satisfy that contract are adapted through `CallableGame`, while model-backed games are composed from `MaskedPredictor` and `LinkFunction` through `MaskedGame`.

This separates three concerns that would otherwise be tangled: masking coalitions into model-native inputs, producing model-native predictions, and linking those predictions into shapiq values. `MaskedPredictor` carries game-like metadata and may be a custom coalition-aware model, while `LinkFunction` remains a lightweight callable protocol that receives only model outputs and performs backend normalization into the value representation.

Backend-specific conversion lives at adapter boundaries instead of inside `CoalitionArray`. For torch, `TorchCallableGame` is isolated under `shapiq.games.torch`, uses best-effort DLPack conversion, defaults to inference-oriented behavior (`no_grad` and detached outputs), and is loaded only when users explicitly import the torch submodule.
