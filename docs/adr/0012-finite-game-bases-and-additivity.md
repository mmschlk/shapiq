# Finite-Game Bases, Uniform Additivity, and Symmetric Materialized Games

**Status: Accepted.** Refines ADR 0004 and ADR 0011 by moving discrete bases and additivity onto `FinitePlayerGame`, removing `KAdditiveGame`/`AdditiveGame`/`flextype.autocast`, and unifying dense and sparse materialized games.

## Decision

### 1. Keep `Game` Basis-Agnostic; Place Discrete Bases on `FinitePlayerGame`

`Game[C, V]` (`shapiq.game.base`) stays minimal (`players`, `is_valid`, `eval` / `__call__`, dtype/shape) so future continuous-coalition games (such as Aumann–Shapley over $[0, 1]^n$) can inherit from `Game` without discrete set-function assumptions.

`FinitePlayerGame[P, V]` (`shapiq.game.finite`) owns the fixed set of discrete bases via `Basis` (`COALITION`, `MOEBIUS`, `FOURIER`; Co-Möbius is deferred):

- `eval(coalition)` / `__call__(coalition)` remains the privileged Coalition-basis entry point with the exact `Game.eval` signature.
- `coefficients(interaction | order=..., *, basis=Basis.MOEBIUS)` routes across all supported bases, defaulting to Möbius.
- `moebius_coefficients(...)` and `fourier_coefficients(...)` provide explicit shorthands.
- `contributions()` returns singleton Möbius coefficients (`moebius_coefficients(order=1)`).

### 2. Uniform Bool-Abstract + Optional-Packed Hooks and Native-Basis Base Classes

To avoid a `2 × 3` explosion of packed/unpacked × basis base classes and prevent circular default implementations:

- `FinitePlayerGame` defines three `@abstractmethod` boolean hooks (`_coalition_bool`, `_moebius_bool`, `_fourier_bool`) and three concrete packed hooks (`_coalition_packed`, `_moebius_packed`, `_fourier_packed`) that default to unpacking and calling the matching boolean hook. `BoolFinitePlayerGame` and `PackedFinitePlayerGame` are removed; classes with native packed paths (`DenseGame`, `ArithmeticGame`, `ReductionGame`, `CastGame`) override the relevant `_*_packed` hooks directly.
- Three small native-basis base classes — `CoalitionDefinedGame`, `MoebiusDefinedGame`, and `FourierDefinedGame` — each leave their own `_*_bool` hook abstract and supply concrete conversion defaults for the other two bases.

### 3. Flatten Additivity onto `FinitePlayerGame` and Drop `autocast`

Every finite game has an additivity order; it only differs in whether that bound is statically known:

- `FinitePlayerGame.additivity()` is a method (signaling possible computation). When a static bound is not known, it computes the exact additivity from Möbius coefficients.
- `KnownAdditivity` stores `_additivity: int` on direct subclasses (`SparseMoebiusGame`, `SparseFourierGame`) and overrides `additivity()` to return it without coalition evaluation. Its `__instancecheck__` checks `_known_additivity() is not None` on `FinitePlayerGame` instances without invoking coalition code (`O(1)` on direct subclasses, walking only wrapper metadata on lazy compositions), so `isinstance(game, KnownAdditivity)` is the single evaluation-free check across both direct subclasses and lazy wrappers.
- Lazy wrappers (`AffineGame`, `ProductGame`, `SumReductionGame`, `CastGame`, `_ScalarDivGame`, `_IntPowerGame`) propagate known bounds through `_known_additivity()` and forward linear Möbius/Fourier queries directly without creating shadow subclasses or using `flextype.autocast`. For `CastGame`, floating/complex casts preserve structural additivity under ADR 0011's floating-point rounding policy, while integer targets require a lossless widening/same-range integer conversion so modular wraparound cannot hide interactions.
- `KAdditiveGame`, `AdditiveGame`, `AdditiveMoebiusGame`, and all private `_KAdditive*` / `_Additive*` refinement subclasses are removed.

### 4. Symmetric `Dense*` / `Sparse*` Materialized Games and Cast Graph

Materialized finite games follow a symmetric naming and storage structure:

- **Dense (`shapiq.game.dense`)**: `DenseGame` base (`values: (*batch, 2**n)` + JIT packed lookup) with `DenseCoalitionGame` (renamed from `TableGame`), `DenseMoebiusGame`, and `DenseFourierGame`. Every pairwise dense basis conversion is a single $O(n 2^n)$ $2 \times 2$ Kronecker pass registered at `cost=3.0`.
- **Sparse (`shapiq.game.sparse`)**: `SparseGame` base (`values`, `interactions`, `players`) with `SparseCoalitionGame` (not `KnownAdditivity`), `SparseMoebiusGame`, and `SparseFourierGame` (both `KnownAdditivity`). Layout casts (`Sparse<B> <-> Dense<B>`) are registered at `cost=2.0`; 0/1-additive `SparseMoebiusGame -> DenseCoalitionGame` keeps its $O(2^n)$ shortcut at `cost=2.0`.
- `ExactComputer` moves out of `shapiq.game` to `shapiq.index.exact`.

## Trade-Off

Every finite game exposes a uniform basis, coefficient, contribution, and additivity API, removing ten shadow wrapper classes and all `autocast` metaclass machinery. The trade-off is that static type annotations no longer distinguish 1-additive or $k$-additive games as separate nominal types (such as `AdditiveGame` vs `KAdditiveGame`), and calling `additivity()` on a non-`KnownAdditivity` game may perform an expensive Möbius transform unless guarded by `isinstance(game, KnownAdditivity)`.
