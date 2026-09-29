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

## Phase 4 — bounded campaigns and Hopper

Reviewers: `audit_phase1`, with the final optional-backend diagnostic checked by
`games_benchmarks`. All identified blockers resolved.

- Checked subprocess isolation, process-group cleanup (including a child left by
  a successful worker), time/memory limits, checkpoint locking, strict resume,
  per-worker source verification, and actual thread/placement metadata.
- Fixed full-node qualification, changed-source detection, worker failure handling,
  and reservation of a full cell allowance before starting another cell.
- Eleven campaign tests pass. A real exclusive EPYC 9754 run completed 72 cells:
  60 successful and 12 unsupported, resumed across Slurm jobs 360541 and 360544.
  The final reviewer independently inspected that checkpoint and worker metadata.
- A separate 22-method catalog probe completed: 16 successful, three unsupported,
  and three failed (two missing optional backends; one ProxySPEX worker failure).
  Optional-backend errors now report the missing dependency directly.
- Resource qualification is verified; runtime remains diagnostic. No estimator
  algorithm was changed to improve its benchmark score.

## Phase 5 — comparisons, chronology, and reproduction

Reviewer: `games_benchmarks`, independent of all implementation authors.

- Checked hierarchical means/lower weighted medians, complete-panel eligibility,
  ties, order-independent regularized Bradley–Terry ratings, clustered uncertainty,
  and unknown-date exclusion. Fifteen chronology dates were checked against the
  linked primary arXiv submission histories by the implementation researcher.
- Fixed browser/Python disagreement for a pending zero-energy game and preserved
  original panel IDs for preset matching. The reviewer verified parity across
  15 actual and nine imbalanced synthetic presets.
- The reviewer extracted an actual reproduction archive, executed a private
  candidate, generated a seven-method local report, and verified public rejection.
- Public archives now remove raw exception text as well as private candidates;
  the audit found that website sanitization alone did not protect the archive.
- Headless Chromium verified target switching, paired tables/history, budget caps,
  missing-budget coverage, and a 390px layout without overflow or script errors.
- Confidence intervals are intentionally unavailable for the current one-model
  strata. More seeds do not remedy missing independent model replication.
- Final review approved the archive sanitization: all 13 bundle tests passed, and
  the reviewer's original private-path reproducer confirmed no leak in any ZIP
  member and no mutation of the source results. No audit blockers remain.
- Final verification: `uv run pytest tests/shapiq_benchmark -q` passed all 76 tests,
  including the existing setup and optional-dependency checks. Required all-file
  pre-commit passes every formatting/lint hook; its type-check hook still reports
  30 pre-existing optional-import/unused-ignore diagnostics. Focused type checks
  pass for all seven new modules.
- Published preview data was regenerated at clean source commit `4fd1eeb6`.
  Slurm job `360547` completed in 4m35s with 60 valid and 12 unsupported cells;
  every accuracy result exactly matched the previously audited campaign.
  The release archive checksums and website snapshot identity were verified.

## Expanded families and website redesign

Reviewer: `audit_phase1`, independent of the implementation authors.

- All 115 benchmark tests passed after adding family preparation, per-game
  relative budgets, shard selection, and NumPy coefficient serialization.
- The reviewer separately passed 54 focused backend tests and browser fixtures
  for 8- and 128-player relative budgets, zero-energy exclusion, real/diagnostic
  separation, Elo ordering, partial curves, separate timing profiles, linked
  hover highlighting, and a 390px layout without overflow.
- Every family recipe uses existing shipped games. Frozen stochastic realizations
  are explicitly distinguished from population expectations. Original zero-energy
  cases are retained; a prespecified forest companion supplies a nonconstant
  baseline-imputation example without selecting explanation points by their scores.
- The existing image wrapper includes an unused segment index and has small
  batch-dependent inference roundoff. Qualification removes only the verified null
  player and fixes inference batch size to one before saving exact table truth.
- Required pre-commit passes all formatting/lint hooks. Its type-check hook still
  reports eight pre-existing optional Woodelf import / unused-ignore diagnostics.
- The reproduction bundle accepts multiple raw shards while retaining each run's
  hardware metadata. All 19 bundle tests passed independently, including later
  shard privacy/provenance conflicts, sanitization, checksums, and round-trip merge.
- The full clean-source campaign at `15ea510f` completed all 24,948 cells across
  Slurm jobs 360594, 360602, and 360608: 9,357 successful, 14,814 unsupported,
  and 777 failed, with none pending. All 34 preparation entries qualified.
- Actual worker metadata verifies EPYC 9754, distinct pinned cores, and one native
  thread. An independent reviewer recomputed the first 703 successful scores.
- The expanded export exposed numerical line-search failures on seven Elo panels.
  A five-method strict-order fixture reproduces the failure. After measurement
  finished, gradient tolerance was changed to 1e-8 and line-search allowance to
  100; the rating objective and convergence guard remain unchanged. All 876
  expanded presets and all 120 fixture permutations pass with the adjustment.
  An independent Newton solution agrees within 0.000018 Elo points.
- Independent browser checks on 24,898 records passed all six targets, 22 methods,
  Python/browser summary parity, Elo sorting, B/d and absolute axes, game changes,
  coordinated hover/focus, and 390px layout without overflow or script errors.
- Final verification passed all 122 benchmark tests. Formatting/lint hooks pass;
  only the eight documented pre-existing type diagnostics remain.
- The final independent audit recomputed every one of the 9,357 successful scores,
  verified query limits, core/thread settings, all 100 archive checksum entries,
  authenticated the snapshot, and remerged all 64 extracted shards. No private
  methods, raw exception messages, or local paths were found in the archive.
- Final browser SV rankings match Python at all budgets and 2d/8d/64d; Elo,
  history, and the 128-player KNN partial curves render correctly. No script
  errors or mobile overflow remain. No audit blockers remain.

## Relative budgets, branding, and estimator investigation

- The public suite now specifies nine queries-per-player points from 0.5d to
  128d. The pilot and auxiliary suites also specify relative budgets.
- Independent reviewer `investigate_leverage` passed all 38 runner, materialization,
  report, and campaign tests, including worker timeout/recovery. The full suite
  passed its other 122 tests; that timeout fixture initially encountered the
  intentional source-change guard during a concurrent edit, then passed after
  source was frozen.
- A candidate regression covers Python and NumPy randomness during import,
  construction, and execution. An actual SPEX run with seeds 0, 1, 0 reproduces
  seed-zero estimates exactly while seed one differs. The corrected RNG protocol
  is recorded in provenance; old SPEX results are not reused.
- LeverageSHAP's independent constrained solve agrees within 1.8e-11; all 325
  existing estimator tests pass. No estimator algorithms or defaults changed.
- Both static exports include the official logo and favicon. The sweep requests
  all 128 physical cores on himem02 with one native thread per worker; timings
  remain diagnostic because workers run concurrently.
- Required pre-commit passes formatting and lint hooks, with only the eight
  previously documented Woodelf import / unused-ignore type diagnostics.
