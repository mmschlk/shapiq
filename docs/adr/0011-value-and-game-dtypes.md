# Value and Game Dtypes

**Status: Accepted.** Refines the value, game, and explanation-backend contracts of ADRs 0001, 0004, 0009, and 0010.

## Decision

Dtypes are hardware and performance controls owned by the user, following JAX's promotion and casting semantics rather than hidden precision babysitting.

### Ownership Across Values and Games

Every `Value` and every `Game` exposes `.dtype`, `.weak_type` (via a shared internal spec hook), and `.astype(dtype)`. Keeping value dtypes on games aligns with `ndlike.JaxLike` and prepares for game-valued games (`Game[C1, Game[C2, V]]`), where dtype inspection and casting recurse uniformly.

Each class owns its value dtype and casting contract. `ndlike.JaxAxisProtected`'s default behavior (reading the first protected field's dtype and casting every protected field) is only a convenience for pure value containers such as `VectorValue`; games that protect non-value fields (`players`, `interactions`, `explain_index`, etc.) must override `.dtype` and `.astype(dtype)` so non-value metadata is never leaked or cast. Composite `Value` types own how `.dtype` and `.astype(dtype)` map onto their internal leaves; games delegate to values and never inspect a value's leaves directly.

Shared dtype primitives—extracting `(dtype, weak_type)` specs, resolving JAX op output specs via `jax.eval_shape`, resolving index weight dtypes, and validating numeric dtypes—live in a single module (`shapiq.value._dtype`) rather than being re-implemented across game submodules.

### No Silent Upcasting

`shapiq` does not silently upcast low-precision values (`float16`, `bfloat16`, narrow integers) to prevent potential overflow:

- Basis-to-table casts (`MoebiusGame`, `AdditiveMoebiusGame`, `FourierGame` to `TableGame`) scatter and transform directly in the game's value dtype. Users who want wider accumulation cast explicitly beforehand (e.g. `game.astype(jnp.float32)`).
- Pointwise game arithmetic (`+`, `-`, `*`, `/`, `**`) and batch reductions (`game.sum(axis=..., dtype=...)`) follow JAX's exact promotion and `dtype=` rules.
- Structural and algorithmic bookkeeping (player IDs, bit-packed indices, coalition counts, sampler capacities, importance `log_weights`) is not game value arithmetic and uses whatever precision the algorithm requires.

### Explicit Casting (`astype` and `CastGame`)

For an explicit dtype, `game.astype(dtype)` returns a game whose evaluated coalition values have the JAX-canonicalized target dtype. Casting before an operation can produce a different result from casting its output.

- **Storage-backed games** (`TableGame`, `BasisGame` subclasses) cast their stored `values` natively while keeping `players`, `interactions`, and other metadata unchanged (moving all dynamic fields together when `device=` is given).
- **Lazy and arbitrary games** inherit `FinitePlayerGame.astype(dtype)`, which wraps the game in a `CastGame` that applies `value.astype(dtype)` to evaluated outputs and reports strong `dtype` (`weak_type=False`). Chaining `.astype(...)` on a `CastGame` nests wrappers to preserve intermediate rounding or truncation. The same target dtype can reuse the wrapper when `copy=False` and neither cast specifies a device.
- **`CastGame` capability participation**: `CastGame` participates in `flextype` zero-cost `KAdditiveGame` / `AdditiveGame` refinement when its child admits a zero-cost view and the target dtype is `None` (placement-only) or numeric without non-linear inexact-to-integer truncation. Floating-point rounding differences do not block refinement, consistent with `AffineGame` and `AdditiveMoebiusGame.astype`.

Unlike `flextype.cast` **Game Casts** (ADR 0004), `.astype(dtype)` may round or truncate coalition values and is a value-dtype operation rather than a strictly game-preserving representation cast.

### Index and Transform Weights

Scalar index parameters (`p`, `alpha`, `beta`) remain weakly typed instead of forcing `dtype=float`, while explicit array parameters retain their strong dtypes. Because JAX drops `weak_type` through functions such as `scipy.special.comb` and `(-1) ** int32`:

- Mathematical weights (e.g. discrete-derivative weights in `ExactComputer` and `SHAPIQ`) are computed safely in default floating precision—avoiding intermediate combinatorial overflow in `float16`—and then explicitly cast to the resolved computation dtype `result_type(game_values, *index_params, 0.0)` before combining with game values.
- Sign arrays in discrete derivatives, Fourier bases, and Möbius inversion use weakly typed `jnp.where(parity, 1, -1)` so narrow numeric games are not inadvertently promoted.

### Opaque Model Dtype Inference (`XAIGame`)

Model output dtypes are inferred through a `@flexdispatch` function (`infer_model_spec`) wired with centralized lazy type constants in `shapiq.lazy_types` and `flextype`'s `delayed_register` pattern (mirroring `probly`):

- The default handler traces one abstract JAX input row through `predict` via `jax.eval_shape`. Masked games also trace masking and aggregation to include their dtype and weak typing effects. Opaque callables fall back to weakly typed default float (`weak_type=True`).
- Optional backends (`torch`, `sklearn`, etc.) register lightweight metadata estimates in their own modules. Estimates use the active `jax_enable_x64` configuration. Use `output_dtype=` or `.astype(...)` to enforce a dtype.
- `XAIGame` also accepts an optional explicit strong `output_dtype` override (consistent with `MappedGame` and `SumReductionGame`) for custom or opaque models.
- `XAIGame.astype(..., device=...)` places evaluated values on the requested device and keeps the inputs and their indices in place for evaluation.

## Trade-Off

Users get predictable, JAX-aligned hardware control, lower memory and compute cost on low-precision games (including exponential `TableGame` materializations and `ExactComputer`), and a single place to inspect and maintain dtype rules. In exchange, users must opt into wider dtypes explicitly when intermediate overflow is possible, casting a basis game's coefficients can differ slightly in rounding from casting evaluated outputs, and unregistered opaque models report a weak default float unless `output_dtype=` is provided.

Revisiting algebraic-rewrite dtype guards (in `AffineGame` flattening and `SumReductionGame` k-additive refinement) and generalizing `TableGame` / `BasisGame` storage beyond floating arrays are deferred until game-valued games are designed.
