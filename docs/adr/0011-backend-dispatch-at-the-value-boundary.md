# 11. Backend dispatch at the value boundary

Date: 2026-07-10

## Status

Accepted

## Context

shapiq v2 keeps its dependency surface small: the core computes on JAX, and torch support
lives isolated in `shapiq.games.torch`, imported only when users ask for it. Two frictions
remained. First, core components could not *accept* backend-native objects without knowing
the backend: `MaskedGame` required a `link_function` even when it was pure ceremony
(`link_function=to_jax`), because turning a torch prediction into a JAX value needs torch
knowledge somewhere. Second, the codebase had no principled place for "handle this type if
its library happens to be installed" logic; ad-hoc `isinstance` towers or eager imports were
the alternatives. Separately, `matplotlib` sat in the core dependencies without a single use
in the v2 tree.

`flextype` (a zero-dependency library sharing this project's toolchain) provides
`flexdispatch`: `functools.singledispatch` with lazy string registrations — a handler can be
registered against a fully-qualified type name like `"torch.Tensor"` and is resolved only
when a matching value is dispatched, and `delayed_register` can defer to an integration
module that performs the real registrations on first contact.

## Decision

- `flextype` becomes a core dependency; `matplotlib` leaves the core dependencies.
- Backend-native values enter the core through one dispatched conversion:
  `shapiq.games.to_values(predictions)`. Its fallback is `jnp.asarray` (JAX, NumPy, Python
  numbers and sequences); backend handlers register lazily against qualified type names
  collected in `shapiq/_lazy_types.py`. The torch handler (`DLPack` import with host-memory
  fallback, autograd detach) lives in `shapiq.games.torch._convert` and is materialized by a
  `delayed_register("torch.Tensor")` callback — importing shapiq, and converting JAX or NumPy
  values, never imports torch.
- `MaskedGame.link_function` defaults to `None`, meaning `to_values`: composed games work
  without a ceremonial link, and custom links remain the way to transform predictions
  (probabilities, log-odds, class selection) before they become values.
- **Index-family registries may be built on `singledispatch` as a registration mechanism,
  but exact-type semantics are non-negotiable.** *(Amended 2026-07-21: the exact-type
  entry guards are removed — MRO inheritance is a feature; see the amendment below. The
  atomicity rule stands unchanged.)* Each sampling method keeps ONE atomic registry whose
  entries bundle everything the method needs per index (a family: sampler builder plus
  estimator for permutation walks; sampler, pairing rule, intercept convention, and solve
  for kernel regression), and every entry point verifies exact-type membership in that
  registry before dispatching — so MRO resolution never selects an index handler silently,
  and subclasses of shipped indices are rejected with a teaching error unless they
  register their own family. Splitting a method across several parallel dispatchers is
  forbidden: a half-registered index would inherit the missing half via the MRO (reviewed
  and demonstrated 2026-07-10). The registries are an internal clean-code mechanism, not
  public extension API. flextype/MRO dispatch remains reserved for *backend value and
  model types*, which are open by nature; `ExactExplainer` stays hand-dispatched because
  its capability arms are typing Protocols, whose structural `isinstance` hooks
  mis-dispatch under `singledispatch` for indices carrying ``None``-valued members.

## Consequences

- The dependency story is: core = numpy + array-api-compat + jax + flextype; everything
  torch-shaped materializes on first contact with a torch object and is otherwise invisible.
- New backends (or array libraries) integrate by registering against `to_values` from their
  own integration module, without core changes and without import-time cost.
- String registrations match a type's real `module.qualname` along its MRO; the qualified
  names in `shapiq/_lazy_types.py` must track upstream renames (e.g. `pathlib.Path` became
  `pathlib._local.Path` in Python 3.13 — `torch.Tensor` is stable).
- The laziness is contractual: a subprocess test pins that `import shapiq` plus NumPy
  conversion leaves `torch` out of `sys.modules`.

## Amendment (2026-07-21)

The exact-type entry guards are removed; MRO inheritance through the family registries is
a feature, not a hazard. A subclass of a shipped index inherits its parent's *complete*
family — deliberately: experimenters subclass to explore, and an index riding a shipped
estimator answers for its own semantics. The hazard demonstrated 2026-07-10 was a
*half*-registered subclass inheriting the missing half of a split method; the defense
against that is the one-atomic-registry-per-method rule, which stands unchanged — the
guards only added a blanket ban on inheritance, and that ban also blocked legitimate
experimentation. `ExactExplainer`'s hand-dispatched dedicated solvers follow the same
rule through `isinstance` arms, as does `TreeExplainer`'s game axis (a subclassed tree
game inherits the closest registered ancestor's closed form; its unregistered-game error
is the domain error `UnsupportedGameError`). What remains rejected are genuine mistakes:
`reject_common_index_mistakes` still catches name strings and index classes.
