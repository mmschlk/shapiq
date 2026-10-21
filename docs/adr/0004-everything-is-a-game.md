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

A nominal `PGame` is a callable family, not a `Game` and not an arbitrary callable treated as one. It binds a declared positional/keyword parameter stage and returns a game or a further partial game. Families of families nest without a fixed depth limit; for example, player removal can leave a further feature-value parameter to bind. Public transformations lift through every level, preserving parameter stages and applying their semantics to eventual concrete games. A transformation may itself introduce another partial family.

Games are always batches, including shape `()`, and partiality must not discard batch information. Every `PGame` carries a `PartialShape` for its eventual game batch: known dimensions are retained, `None` dimensions are unresolved, and a wholly `None` shape has unresolved rank. Binding refines that shape at every level. Transformations expose metadata-only `output_shape` rules, so shape-preserving or shape-changing behavior flows through the graph without game evaluation. This deliberately small representation sacrifices symbolic relationships between dimensions rather than introducing a shape-expression language now.

Graph composition and intermediate parameter binding defer game computation until all required inputs are supplied. The equation `T(p)(args) = T(p(args))` describes results, not eager execution: naive postcomposition is insufficient when transformations introduce further missing inputs. The shared lifting layer will need to retain dependencies and expose remaining stages without computing upstream games. That implementation, including its node representation, is deferred; transparent lifting is not claimed to be implemented by the API protocols.

Preserving nested stages avoids argument-name collisions and ambiguous automatic currying. `ParamSpec` retains an immediate stage's signature, while the recursive `LiftedGame` alias erases deeper signatures as a pragmatic typing limit. This buys a small public API without committing to a general-purpose monad framework, global scheduler, automatic caching, or arbitrary callable introspection.
