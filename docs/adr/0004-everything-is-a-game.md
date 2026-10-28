# Everything Is a Game

**Status: Accepted.** Replaces the explainer-to-explanation pipeline, supersedes ADR 0003, and updates the result/storage choices in ADRs 0001 and 0002.

**Revision in progress:** the [working game algebra](../design/game-algebra.md) admits nonzero-cost game-preserving casts and values interpreted as zero-player games. Those requirements supersede the constant-time-only cast restriction below. The partiality/reduction model is being refined there before changes to the runtime skeleton.

## Decision and Trade-Off

The central computational interface is a game transformation. A model can be embedded in a game and then transformed into other games; an explanation is an interpretation of a resulting game, not a separate universal result container. This makes results composable through the same player/coalition/value interface, at the cost of requiring explicit mathematical semantics for each result rather than relying on a common attribution-array schema.

The Shapley transformation produces a zero-empty additive game with Shapley values as singleton values. Its grand-coalition value is the source's grand-minus-empty value. Exact and approximate algorithms implement this common result form; approximation policy must be explicit. Mathematical capability protocols, such as k-additivity, remain separate from concrete storage types built on ndlike's protected-axis support.

## Casts Preserve a Game; Transformations May Compute

`flextype.cast` registrations (`@caster` / `@castmethod`) are restricted to storage-sharing, constant-time game views. They preserve players, coalition admissibility, batch shape, and all coalition values. Costly conversions are named transformations even when they preserve the mathematical game. Player-changing or value-changing operations are transformations even when they happen to be cheap. This intentionally restricts flextype's more general casting machinery so routing through a view cannot hide numerical work.

Concrete transformation handlers can use `flextype.flexdispatch`, including model-type routing inside model-backed game families. Cast lookup and transformation dispatch are distinct mechanisms. Mathematical protocol conformance is an implementer's guarantee, not an inferred runtime proof; no additional registry metaclass is required by this design.

## Partial Games Preserve Missing Inputs

**Superseded by [ADR 0005](0005-family-specific-partial-stages.md).** Partial games remain callable families rather than games or arbitrary callables treated as games. The current interface uses an abstract `Partial[T]` with family-specific `__call__` and `map` implementations, rather than result-dependent facades or a universal parameter signature.

Explicit nested mapping preserves the order and concrete types of pending-input stages. A callback can introduce another partial family, retained intact. Shape and dtype tracking are outside the minimal partial interface for now; concrete games and players retain their batch semantics.

The former automatic recursive lifting, partial-shape propagation, and full-graph deferral requirements are not promises of the current functional core. Each stage executes when called; scheduling and inspection are separate future decisions.
