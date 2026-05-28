# Core Interfaces

This note summarizes the current interface sketch. The glossary in `CONTEXT.md` defines the domain language; ADRs in `docs/adr/` record why the major trade-offs were chosen.

## Shapes

`CoalitionArray.shape` is the logical array shape of coalition elements and excludes the player dimension. Dense coalition storage has physical shape `coalitions.shape + (coalitions.n_players,)`.

`Game.target_shape` is the shape of explanation targets. When a game evaluates coalitions, the logical value shape is based on broadcasting the target shape with the coalition shape. For sampled approximation, samplers own the coalition shape policy and append the final sample axis.

`ExplanationArray.shape` is the logical shape of explanation elements and equals the game's explanation target shape. It excludes interaction and value dimensions.

## Games

`Game[ValueT]` is the base abstraction for coalition-to-value evaluation. It validates player-count compatibility at the boundary where it receives coalitions.

`CallableGame` adapts callables that already behave like games. It can convert `CoalitionArray` inputs into a callable-native representation and convert raw outputs into `ValueT`.

`MaskedGame` composes a `MaskedPredictor[PredictionT]` with a `LinkFunction[PredictionT, ValueT]`. `ModelMaskedPredictor` is the default composition of `Masker` and a callable model.

## Explanations

`ExplanationArray[ValueT]` is callable: `explanations(interaction)` aliases `explanations.attribution(interaction)`. `__getitem__` indexes explanation target axes only.

Interaction lookup accepts one tuple interaction or an array-api-compatible integer array for fixed-size multi-interaction access. Tuple lookup returns attributions for all selected explanation targets. Array lookup broadcasts explanation shape with the interaction array shape excluding the final interaction-members axis.

`has(interaction)` reports where attribution lookup is available. `attribution(interaction)` is strict: invalid interactions raise `ValueError`, and missing sparse attributions without a default raise `KeyError`.

## Sampling And Approximation

`Sampler.sample(state, budget)` returns `(coalitions, next_sampler)`. `budget` is a non-negative integer number of new samples; `budget=0` returns an empty coalition array and the same sampler.

Samplers own sample sharing. `sample_sharing=None` preserves `target_shape`; `True` shares across all target axes; an integer or tuple of integers shares across selected axes by replacing those target dimensions with `1`; `False` is rejected.

`Approximator.sample(budget)` validates public budget input, asks the sampler for coalitions, evaluates the game, updates its approximation state through the concrete approximator, and returns a new approximator. It does not revalidate sampler output shape on every call.

`ApproximationState.history(reverse=False, include_self=True)` and `rollback(steps=1)` are available when history is enabled. `Approximator.history()` combines state history with aligned sampler snapshots.

## Public Method Names

Explainers expose `explain()` as the canonical materialization method and are callable as an alias. The older `compute()` name is not part of the planned base interface.

Approximators additionally expose `sample(budget)` and `approximate(budget=0)`.
