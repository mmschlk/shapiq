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
