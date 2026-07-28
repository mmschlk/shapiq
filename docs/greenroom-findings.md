# Green-room findings: what two clean rooms taught us

Two independent agent teams rebuilt the library from the pillared seed
(`docs/greenroom-seed.md`), isolated from the incumbent. Both finished:
p1 (Opus) — 81/81 tests, all eight scenarios verified live; p2 (Fable) —
64/64 tests, nine scenarios verified live, 1.71M evals/s. Both ran their
own adversarial review agents. Full analyst reports live in the session
record; branches `greenroom/*` share the empty root `b838698c` for
diffing. Five learnings.

## 1. The spine was forced — the domain even names itself

Every question the seed left open converged three ways, often to the same
function and constant: the output is a readable Möbius-basis game with
one frozen provenance record and one type for every route (sampled,
exact, structural); evolving state is an append-only record the user
holds, beside the game, never inside it; split invariance comes from draw
identity as a pure function of (seed, position), atoms bought whole,
remainders surfaced; dedup charges novel rows only and stops
deterministically; extension is structural protocols and typed values,
no strings. Both rooms independently renamed their library **harsanyi**
— after the basis. The spine is a discovery, not a preference: stop
re-litigating it.

## 2. The one 2–0 vote against us: provenance must survive arithmetic

We drop provenance on `game + game`. Both rooms independently built
provenance as a **tree** (`Receipt.parts` / `Provenance.parents`):
per-parent spends never summed, totals derived, uncertainty refusing to
combine across dependent parents (with the reason stamped). Their shared
argument: combination is the currency's central operation, and a flat
record cannot represent it. Adopt the recursive-record shape.

## 3. Our bit-identity claim has a measured hole for model backends

Both rooms' reviewers independently found that fixed-shape batching is
not enough on BLAS/AMX: a torch row's bits depend on co-batched row
*content* (8-ULP drift). Our split-invariance claim for torch games is
exposed exactly there. p1 closed it unconditionally (blocks bought whole
with fixed content across any split, budget carried); p2 tiered the
claim honestly (`check_row_purity`, batch=1 bit-exact mode). Action:
adversarial-split tests against our torch games, then adopt whole-block
content invariance or tier the claim.

## 4. Honesty features are the gap, and one of our theorems needs a word

Both rooms shipped what we lack: the SII game-plane caveat (an exact SII
estimate is a *bad surrogate of its own game* — R² −1 to −10, worsening
with budget; p2 stamps `mobius_faithful` on the index, p1 reports
measured border/efficiency gaps beside R²), fidelity computed on the
already-paid record instead of a 2^n sweep, spend-zero proven by call
counter, and numeric validators for third-party extensions. And p1's
reviewer falsified the unrestricted cannot-help theorem: a *biased*
zero-cost shrinkage always beats the direct estimate, so our claim (and
S5's) must say **unbiased** correction.

## 5. The method works — keep it

Scenario-driven seeds with falsifiable pillars produced scars instead of
slogans: rooms hired their own critics, who caught a rigged assertion,
broke a premature bit-identity design at awkward splits, deflated a
load-bearing-mechanism overclaim by instrumenting and counting, and
scoped a theorem correctly. Where the rooms split — sampling family
(coverage-complete without-replacement sweep vs with-replacement
streams), dedup's home (engine mode vs metered wrapper), record vs
accumulator, whether evaluation deserves `__call__` — is the honest map
of genuine design freedom, and exactly where human attention should go
next.
