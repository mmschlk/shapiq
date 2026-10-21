# shapiq Context

## Language

**Game**:
A JAX-like batch of cooperative games, mapping admissible **Coalitions** to **Values**. Games are always batched objects, including the zero-dimensional batch shape `()`. Their logical shape describes the game batch. The **Players** shape broadcasts to that shape without enlarging it. Indexing selects games; calling evaluates coalitions with logical batch broadcasting. Explanation results are also games.
_Avoid_: explanation container, model wrapper as the general abstraction

**Player**:
An explainable unit whose participation is represented in a **Coalition**.
_Avoid_: feature when the player need not be a model feature

**Players**:
A JAX-like batch of player sets. Its logical shape excludes each set's internal structure. `Players[C]` declares compatible coalition representations and checks whole-batch admissibility as one boolean, normally using type, dtype, internal shape, and logical batch broadcasting metadata. A **Game** delegates this check with its own batch shape.
_Avoid_: player axis when referring to the player-set batch

**Coalition**:
A selection of players from a player set. Coalition representations and their internal axes are defined by the compatible **Players** type. Representations may be batched; for example, a boolean JAX array can use its final axis for player membership and its leading axes for the coalition batch.
_Avoid_: mandatory CoalitionArray container, sample when referring to arbitrary coalitions

**Value**:
The output assigned by a **Game** to a **Coalition**, represented by an ordinary supported array or structured value. A value has a **Zero-Player Game** interpretation without needing to be stored as a game object.
_Avoid_: prediction when the game's value space differs from model output

**Zero-Player Game**:
A **Game** whose sole coalition is the empty coalition, assigned a specified **Value**. The value need not be zero.
_Avoid_: zero-valued game

**Game Transformation**:
A function producing a **Game** or **Partial Game**, potentially changing player sets or coalition values. Its inputs can be adapted through game-preserving **Casts**, and it applies to partial games through **Lifting**.
_Avoid_: explainer as a separate core object

**Game Family**:
A semantic or representation-based category of games. Mathematical capability protocols describe semantic guarantees; concrete game types specify storage and available operations. A parameterized family with missing inputs is a **Partial Game**.
_Avoid_: assuming that mathematical capability determines memory layout

**K-Additivity**:
For a finite game with suitable additive values, the guarantee that all Möbius coefficients above order `k` vanish. The `KAdditive` capability declares a nonnegative upper bound through `additivity`; the bound need not be tight. An empty-coalition coefficient is allowed.
_Avoid_: maximum coalition size, storage layout

**Additive Game**:
A zero-empty game whose value for any coalition is the sum of its singleton values. The `Additive` capability is a normalized special case of **K-Additivity** with bound one.
_Avoid_: assuming every additive game is a Shapley game of a particular source

**Shapley Game**:
The **Additive Game** produced by a Shapley **Game Transformation**: each singleton value is that player's Shapley value in the source game. It preserves the source player set and has empty-coalition value zero. Its grand-coalition value equals the source's grand-coalition value minus its empty-coalition value.
_Avoid_: Shapley value array as the core result type

**Game Cast**:
A change of representation or interpretation that preserves the mathematical **Game**, including its players and coalition values. A basis conversion may require substantial computation; a value can be interpreted as a **Zero-Player Game**.
_Avoid_: cast for player removal, estimation, or value-changing transformations

**Game View**:
A constant-time, storage-sharing **Game Cast** exposing another interface to the same mathematical game.
_Avoid_: assuming every game cast is a view

**Game Basis**:
A collection of linearly independent coalition functions in which a finite numeric **Game** can be represented by coefficients. Coalition, Möbius, and Fourier bases represent the same game with different coefficients.
_Avoid_: confusing a basis element with its coefficient

**Reduction**:
Supplying the missing input appropriate to an object: family parameters for a **Partial Game**, or a **Coalition** for a **Game**, to obtain the next result. Calling an object denotes its default reduction.
_Avoid_: assuming all reductions take the same argument types

**Partial Game / PGame**:
A parameterized or deferred family of related game batches, supporting game operations through **Lifting**. Binding executes one declared **Reduction** stage and removes its pending-input requirement, possibly returning another partial with its own signature; a zero-parameter partial defers work until explicitly reduced.
_Avoid_: game with missing coalition evaluations, explanation state

**Partial Shape**:
Metadata describing an eventual game batch before all inputs are bound. `PartialShape` is a tuple of known sizes and `None` for unresolved dimensions, or `None` for unresolved rank. Shape `()` is a known zero-dimensional batch. Binding inputs refines this metadata; transformations propagate it through metadata-only output-shape rules.
_Avoid_: treating unknown shape as unbatched, counting family nesting as batch dimensions

**Lifting**:
Extending a **Game Transformation** through surrounding **Partial Game** stages until its declared input type is reached, which can itself be a partial. Surrounding stages are preserved and the transformation's result is inserted intact, including any new partial stages.
_Avoid_: forcing or flattening the partial stack, guessing arguments shared by different stages

**Protected Axis**:
A representation-internal axis hidden from a game's or player set's logical batch shape by ndlike. Concrete representations may protect player, interaction, and value axes while exposing array operations over batch axes.
_Avoid_: treating representation axes as game batch axes

**Explanation**:
An interpretation of a transformed **Game**, such as reading singleton values from a **Shapley Game**. Convenience accessors or visualization may expose those values without defining a separate core result type.
_Avoid_: ExplanationArray as the universal transformation result

**Interaction Index**:
A mathematical rule for attributing effects to interactions among players. In the game-centered API, the rule determines a **Game Transformation** and the interpretation of its resulting game; algorithm choice determines how that result is computed or estimated.
_Avoid_: coupling an index to one algorithm or storage representation
