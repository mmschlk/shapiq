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
- **Interaction indices stay exact-type dispatched and never move to `singledispatch`
  semantics.** Dispatch by MRO would hand a subclass lookalike (`class MyFBII(FBII)`) the
  shipped handler silently — precisely what the entry-point gates reject with teaching
  errors (ADR 0007). Index sets are closed and explicit; flextype dispatch is reserved for
  *backend value and model types*, which are open by nature.

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
