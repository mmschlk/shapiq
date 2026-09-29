# Independent phase audits

Each phase is reviewed by an agent who did not implement it. This log records
concrete checks and limitations, not a claim of comprehensive scientific coverage.

## Phase 1 — local benchmark

Reviewer: `audit_phase1`. All identified blockers resolved.

- Confirmed exhaustive ground truth and the estimator use the same coalition game.
- Checked coordinate alignment, false-positive sparse coefficients, zero-energy
  nMSE, duplicate/end-point query counting, sticky budget limits, and tamper checks.
- Fixed revision serialization, overflow of derived error metrics, and local
  dataclass adapter imports; added regression coverage.
- Seven targeted tests pass. The real California pilot completed 27/27 runs.
- Runtime is diagnostic table-oracle time, not a qualified hardware leaderboard.

## Phase 2 — static public and private reports

Reviewer: `audit_phase1` (independent of the implementation). All blockers resolved.

- Confirmed strict snapshot/method joins, conflicting duplicate rejection, planned
  coverage, target separation, stripped truth/coefficients, and safe DOM rendering.
- Fixed public export defaults/canonical-site protection, deployment permissions,
  explicit asset staging, and separate timing identities for separate result files.
- Eighteen runner/report tests pass. Headless Chromium verified the real public
  table, budget filtering, private candidate comparison, and 390px mobile layout.
- Fixed the mobile filter-grid overflow found by browser testing.
- Deployed the measured baseline preview to the user's fork via the pinned Pages
  workflow; HTTPS returned 200 at https://www.rtealwitter.com/shapiq/.

## Phase 3 — existing structured games

Reviewer: `audit_phase1`. All blockers resolved.

- Checked live tree probability/SV/pairwise k-SII and fixed-denominator KNN
  correct-label utility against exact eight-player counterparts.
- Fixed unsafe/duplicate game IDs, KNN label selection, frozen reconstruction
  parameters/neighbor ordering, and unsupported active-player claims.
- Twenty-six combined tests pass. The 30-feature tree / 128-row KNN campaign
  completed with 60 successful, 12 unsupported, and zero failed records.
- Independently prepared 64-feature Digits tree interactions and 1,024-row KNN
  truth without large powerset tables. Small-counterpart maximum errors were
  below 4e-16 for all qualified routes. No estimator campaign at 1,024 was run.
