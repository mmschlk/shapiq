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

## Four construction instances and larger structured games

Reviewers: `brand_relative_ui`, `investigate_optional`, and `investigate_leverage`,
reviewing modules they did not implement.

- Separated four game-construction seeds from one estimator seed. Verified stable
  recipe/stratum identity, distinct instance IDs and artifacts, and shared model
  clusters for four text/image inputs to one pretrained model.
- Qualified 64 larger instances: 30/64-feature forests across all six targets,
  30/64-feature RBF product kernels for SV, and 128/256-player KNN for SV.
  Maximum discrepancy against exhaustive eight-player truth was 3.34e-11.
- Fixed float32 tree-routing disagreement in the benchmark adapter. Checked
  output class/scale, saved-array reconstruction, and nonempty SII/FBII truth
  without imposing an invalid efficiency constraint.
- Checked available-result nMSE, matched-cell Elo, connected comparison graphs,
  and complete-panel history. Medians now average neighboring values at an exact
  half-weight boundary, replacing the earlier lower-median convention.
- Browser fixtures verify four cells rather than sixteen, median 4 for errors
  1/3/5/7, stable setting counts, compact worker-reference hydration, legacy
  reports, and mobile layout. Worker deduplication preserves per-run hardware.
- Cluster bootstrap intervals remain descriptive: they do not model dependence
  between distinct masking recipes that share a fitted model. Overall panels
  containing single pretrained-model strata withhold these intervals.
- All 165 benchmark tests pass. Repository pre-commit formatting/lint checks pass;
  the type-check hook still reports eight existing diagnostics in optional
  Woodelf imports and tree-conversion ignore comments.
- Distinct constructions need not produce distinct payoff functions: two small
  baseline instances are zero-energy games, and some small KNN tables coincide
  despite different data rows. Zero-energy cases are excluded uniformly, and
  coverage remains explicit; no instance is replaced after inspecting scores.
- A second export audit round-tripped all 24,948 prior published records through
  compact JSON and verified identical browser tables, filters, curves, history,
  and hardware details. Removed unused per-game presets after removing that UI;
  overall/family preset views remain identical. All 36 summary/report tests pass.

## Plot controls, estimator sources and behavior-preserving cleanup

Independent reviewers: `investigate_leverage` and `investigate_optional`.

- Verified independent plot-family selection, full budget curves, weighted
  midpoint medians, log-axis handling of zero/roundoff, and unchanged table and
  history selection. Raw values remain available on hover below the display floor.
- Verified all 22 estimator descriptions and pinned implementation locations.
  Primary-paper links include the user-supplied ProxySHAP paper; its verified
  first-publication date is May 21, 2026.
- Checked keyboard opening/closing and focus restoration for the estimator
  dialog, mobile layout, safe literal rendering, and private-method fallback.
  Report export and Pages deployment copy the same six assets. The official SVG
  keeps its geometry/colors with its white background removed.
- Refactored the 424-line page renderer into an 80-line coordinator and named
  section renderers. An independent source reconstruction confirmed every
  original statement/formula and execution order. No build system was added.
- Reduced CSS from 1,204 to 1,141 lines, grouped component rules and consolidated
  responsive blocks. All computed styles and element rectangles match across
  30 viewport/state combinations; independent desktop/mobile screenshots are
  pixel-identical. Numeric and interaction browser fixtures still pass.
- Backend cleanup removes one redundant directory creation and updates stale
  preparation docstrings. All 165 benchmark tests pass. Required formatting and
  lint checks pass; the type hook retains the eight previously recorded,
  unrelated diagnostics.
- Refreshed the previous public report after cleanup: all 24,948 measurement
  rows, worker details and provenance remain unchanged. Independently recomputed
  2,401 eligible summaries across 288 family/overall presets, checked Elo
  objectives and ProxySHAP chronology, and exercised six targets and 60 family
  views in the browser. The larger four-instance campaign remains separate.

## Generated data outside the source-code PR

Reviewer: `investigate_optional`.

- The results JSON is an ignored local export and a public release asset. A small
  URL/SHA-256 manifest pins the bytes used by Pages; the browser still requests
  `data.json` from the same site URL.
- Independently downloaded 12,344,319 bytes / 24,948 records and verified exact
  identity with the audited report. Executed the deployment script in a fresh
  checkout without data and confirmed that it stages the same six site assets.
- Tampered bytes fail the checksum before writing; private methods, exact truth,
  artifact references and raw estimates also fail the publication checks.
  All 14 report tests pass, including local private-candidate exports.

## Inline estimator details and visual polish

Independent reviewers: `investigate_leverage` and `investigate_optional`.

- Replaced the estimator dialog with inline table and legend disclosures using
  shared rendering/toggle helpers. Verified native keyboard interaction, focus,
  unique disclosure IDs, all 22 source links and safe private-method fallback.
- Unified dropdown chevrons, tightened history legends, added subtle shading,
  and softened the best-dated-result line. History now draws across its container
  with readable text; resize preserves expanded details. Mobile year labels do
  not overlap, and pages do not overflow at 320 pixels.
- Independent browser fixtures confirm unchanged scores, independent family
  selection, full budget curves and logarithmic display floors. All 24,948
  published measurements and 288 presets remain unchanged. JavaScript syntax
  and formatting checks pass; repository pre-commit retains only the eight
  previously documented type diagnostics.

## History labels at curve endpoints

Independent reviewer: `investigate_leverage`.

- Right-side labels follow endpoint error order, with faint connectors and
  spacing for ties. Narrow screens use an ordered list with nMSE values. Method
  details expand below the figure and remain open across responsive changes.
- Verified exact connector endpoints and label alignment, nonoverlapping labels,
  one/all 22 methods, ties, zeros, sub-floor values, empty results and widths from
  320 to 1,280 pixels. Numeric aggregation, filters and log-scale fixtures remain
  unchanged. JavaScript syntax/format checks pass; no new type diagnostics.

## Simplified arrows, history and logo

Independent reviewer: `investigate_optional`.

- Replaced remaining font-based sorting arrows with the shared vector chevron;
  show direction only for the active sort. Enlarged/aligned disclosure arrows
  and explicitly disabled native WebKit select arrows. Chromium did not reproduce
  duplicate native arrows, so that was not established as the original cause.
- Compared all seven headers in both directions against the previous renderer:
  every score and row order matches. Verified arrow states, mobile layouts and
  removal of the best-dated-result curve.
- XML comparison confirms the SVG keeps its colored geometry/styles while
  removing only the lower wordmark and invisible sizing rectangle, with a tighter
  viewport. No framework or dependency was added.

## Separate LeverageSHAP restoration

[PR #603](https://github.com/mmschlk/shapiq/pull/603) is based on upstream `main`
in an isolated worktree; no estimator code changed in this benchmark branch.
Independent reviewer: `investigate_optional`.

- Traced the historical `0.001` ridge safeguard, its August removal and the
  sampler-only alignment PR. Verified original/current weight normalization in
  36 cases. Same-sample diagnostics isolate the missing low-budget stabilization.
- All 325 existing LeverageSHAP tests pass unchanged, plus 16 new checks and 12
  neighboring tests (353 total). Independent review verified 160 exact zero-ridge
  comparisons, 108 exact unaffected-path comparisons and 52 constrained ridge
  references. It caught and resolved a tiny-penalty numerical-nullspace issue;
  extreme-penalty probes preserve finite results and efficiency.
- The fix keeps queries unchanged and documents bias. Public measurements and
  the running four-instance campaign retain their original frozen source.

## Corrected full rerun and final history-control cleanup

Independent reviewers: `investigate_optional` (execution preflight) and
`investigate_leverage` (browser changes).

- Job `360683` reruns every cell using clean execution commit `d4ac18e6`, with
  LeverageSHAP source identical to PR #603. All 188 original frozen artifacts,
  808 game definitions and 159,984 planned cells authenticate unchanged. Seven
  compiled extensions match the prior environment; preparation and execution
  provenance remain distinct. Initial LeverageSHAP/KernelSHAP smoke checks pass.
- Corrected Median nMSE control spacing from 12 to 30 pixels after identifying
  a CSS-specificity override, and removed only history-label connector lines.
  Across six viewport widths and both metrics, score rows, all 13 real history
  curve coordinates, label ordering/positions and chart geometry are identical.
  Hover, keyboard, empty views, and one/all-method tie/zero fixtures also pass.

## Provisional corrected-run publication

Independent reviewers: `investigate_optional` (checkpoint/export) and
`investigate_leverage` (browser).

- Froze all 128 checkpoints before exporting, leaving the active run untouched.
  Verified 11,586 unique cells: 2,215 successful, 9,306 unsupported and 65 failed;
  148,398 of 159,984 planned cells remain pending. Every successful score was
  independently recomputed. Failures are ProxySPEX minimum-sample errors.
- Preparation remains `1472a003`; every execution record uses clean corrected
  source `d4ac18e6`. All 2,280 observed workers meet the pinned-core, CPU model,
  single-thread and resource-limit protocol. No LeverageSHAP scores had completed
  at capture, and no earlier measurements are substituted.
- A generic provisional notice reports missing cells without changing scoring.
  The report and reproduction archive are separate checksummed release assets;
  generated measurements remain outside Git. Public JSON exactly matches the
  frozen records; the ZIP passes all 318 internal checksums. Both exports omit
  raw exception text and local paths.
- Browser comparison preserves scores for all six targets, retains null missing
  scores and reports the exact pending count. Desktop and 390/320-pixel layouts
  pass after resize rendering settles. Empty history explains its complete-panel
  requirement. Formatting/lint pass; the same eight existing type diagnostics
  remain in optional Woodelf imports and tree-conversion ignore comments.

## Completed corrected-run publication

Independent reviewers: `investigate_optional` (campaign and exports) and
`investigate_leverage` (browser).

- Job `360683` finished normally after 3:41:53. All 128 shards are complete,
  covering exactly 159,984 unique planned cells: 58,517 successful, 94,284
  unsupported and 7,183 failed. No cells remain pending. Every successful score
  was independently recomputed against the unchanged frozen truth.
- All 65,397 recorded workers match the fixed CPU/affinity/thread protocol and
  execution source `d4ac18e6`; preparation provenance remains `1472a003`.
  The failure total comprises 6,880 minimum-budget/sample errors and 303 timeouts.
- Publication replaces the early provisional dataset with the full nine-budget,
  four-construction campaign. The previous release remains available separately.
  Failed results stay unscored and continue to reduce reported coverage.
- Final public JSON matches every frozen record after lossless compaction. The
  reproduction ZIP passes all 318 internal checksums and preserves exact baseline
  fields except intentional exception-text sanitization. No private methods or
  local paths enter either public asset.
- The completed dataset renders all six targets, all nine relative budgets, and
  65,700 evaluated attempts without a provisional notice. All 1,332 LeverageSHAP
  SV cells succeeded. Desktop and mobile layouts pass.

## Simpler details and game descriptions

- Removed only estimator-name chevrons; names still open inline details by mouse
  or keyboard. Ordinary select controls retain their arrows. Elo and Hardware
  use the same bold labels as the other score explanations.
- Failed-run summaries now include only `failed` records. Known unsupported
  explanation types remain compatibility metadata rather than apparent failures;
  selections with no failed attempts say so explicitly.
- Coverage paragraphs describe each frozen game kind, its dataset and player
  counts, and link to the preparation-version implementation and data sources.
  Larger structured games join their matching kind instead of duplicating
  descriptions across instances and explanation indices.
- Independent source checks verified all 28 distinct implementation paths at
  the frozen preparation commit and the dataset mappings for all 31 game kinds.
  Formatting/lint and JavaScript syntax pass; repository-wide type checking
  retains the eight previously recorded optional-import/ignore diagnostics.

## Under-budget reporting and unchanged OddSHAP behavior

- Public exports recognize only exact known budget/sample guards for OddSHAP,
  ShaplEIG, SPEX and ProxySPEX. Safe reason/minimum metadata survives reproduction
  ZIP round trips; generic validation errors and exception paths are not exposed
  or reclassified. Status, score, timing and coverage fields remain unchanged.
- Separate drawers distinguish under-budget outcomes from other failures. Known
  thresholds display relative to player count; coverage tooltips show both counts.
- A scratch-only guard removal tested 88 OddSHAP configurations. All 48 newly
  accepted calls respected the query cap and returned finite efficient values;
  24 previously accepted cases were bit-identical. Tiny-budget constant proxies
  selected one tied singleton, assigning all credit to player zero. The existing
  OddSHAP implementation and public scores remain unchanged.
- Independent backend review passed all 51 report/bundle tests, checked exact
  known-error matching and sanitized round trips, and verified unchanged public
  fields against archived measurements. Browser review found identical scores,
  sorting, curves and tooltips; the default panel's 204 OddSHAP budget rejections
  appear only under insufficient budgets. Mobile layouts pass.
- Full export comparison verified all 159,984 records and 720 presets unchanged
  after removing only the new reason/minimum fields. There are 6,880 recognized
  under-budget cases and 303 remaining timeouts. The ZIP passes all 318 internal
  checksums and preserves every other baseline field and frozen artifact.
  Formatting/lint pass; the same eight existing type diagnostics remain.

## Player counts and dataset diversity

- Expanded the declared suite from 37 to 74 settings, retaining four construction
  seeds and one estimator run per cell. Added real feature, row, group and model
  player counts rather than padding games with dummy players. New settings remain
  separate from the published dataset until preparation, execution and export.
- An independent audit passed 141 family/materialization/structured/summary tests.
  All 52 legacy constructor/seed comparisons retained identical payoffs and prior
  metadata; four legacy TabPFN data splits were unchanged. All eight old large-KNN
  artifacts and truths also match the published snapshot.
- All 36 structured KNN settings qualified and reloaded. A missing-class failure
  in the new Digits 16-player recipe was caught and fixed with explicit stratified
  training-row selection; legacy selection is unchanged.
- Sixteen new SV/SII definitions matched an independent factorial-weight reference.
  New threshold-neighbor radii matched an independent training-distance calculation;
  all eight Wine payoff tables were finite. Seed two remains constant at both
  player counts and is retained under the existing zero-energy exclusion rule.
- Coverage paragraphs still group the 31 game kinds, link their data sources,
  and fit mobile screens. The six preparation-cost probes passed on Hopper;
  the 12-player TabPFN probe suggests roughly 32 minutes per full payoff table,
  before exact-coefficient work. This is a scheduling estimate, not a timing score.
- All 231 benchmark tests pass. The frozen execution checkout also passes 456
  combined LeverageSHAP/OddSHAP tests; both estimator sources match their separate
  PRs byte for byte, and compiled extensions are unchanged. Independent review of
  parallel preparation reproduced the serial snapshot identity exactly and checked
  resource limits, complete-coverage checks and atomic snapshot publication.

## OddSHAP guard-only pull request

- Separate PR #605 removes the interaction-factor budget guard and is assigned to
  Fabian Fumagalli. It retains shapiq's existing proxy, singleton selection and
  regression; the sampler still rejects budgets below two before game calls.
- All 115 OddSHAP tests pass. Independent comparison covered 244 previously
  accepted configurations with identical outputs and queries, 64 newly accepted
  configurations with finite budget-respecting outputs, and 72 invalid calls.
  Full-census small-game checks remain exact.
- Nondefault interaction-factor checks also exposed an existing efficiency
  discrepancy with identical old/new results, recorded in AGENTS.md for separate
  investigation. Tiny-budget outputs are not guaranteed accurate. Current public
  results still refer to the old guard until a versioned rerun is published.

## Minimum eleven players and SV display grouping

- The replacement suite declares a minimum of eleven players. Invalid minima,
  explicit undersized recipes, actual undersized constructions and undersized
  authenticated snapshots are rejected. At the smallest admitted dimension,
  `128d = 1,408 < 2^d = 2,048`; larger budgets are not added. This prevents full
  enumeration within the grid, not exact recovery of easy games.
- All 192 non-media constructions passed actual player-count and finite-payoff
  checks. Gaussian imputers reject Bike Sharing's binary columns, so those
  recipes use Wine's continuous features and a recorded classifier probability.
  Independent reconstruction matched that probability for all sixteen cases.
- All eight authored text inputs have the requested 11/12 tokenizer players;
  all eight image configurations have the requested nonempty segments. Existing
  model, masking, null-slot removal and legacy defaults are preserved.
- All sixteen causal cost probes passed on Hopper. Extrapolated preparation for
  the four causal tables totals 260–285 minutes per construction seed, before
  other recipes. These estimates inform resource limits, not benchmark scores.
- The default SV display groups only KernelSHAPIQ under KernelSHAP and SVARMIQ
  under SVARM. Each pair shares the same base configuration and produced identical
  values and query sequences in 36 bounded comparisons. ProxySHAP and RegressionMSR
  differed in all 36 cases; FSII, kADD and inconsistent regression paths also
  differed and are not asserted equivalent.
- Independent browser comparison found unchanged scores, Elo, curve coordinates
  and underlying selections with All variants enabled. Interaction views and
  local reports without the canonical method still expose the original classes.
  Grouping does not pool measurements or move publication dates. Mobile checks pass.
- All 250 benchmark tests pass. Formatting/lint pass; type checking retains the
  same eight existing optional-import/ignore diagnostics. The preparation wrapper
  retains its audited four-seed merge, with an eight-hour limit per child and an
  eight-hour-thirty-minute Slurm allocation for the larger exact tables.

## Compatible dataset × recipe matrix

- An independent audit reconstructed all 1,363 candidate decisions: 196 selected
  tabular settings and 1,167 explicit exclusions. Generic exact tables are bounded
  to 11/12 players; larger structured settings keep their original payoff and
  target restrictions. Selection and runtime qualification remain distinct.
- All 760 new core constructions passed finite-payoff probes. All 296 legacy
  constructions retained bit-identical payoffs and metadata. Classifier recipes
  use the appropriate probabilities, accuracy, votes or decision margin; Gaussian
  Digits recipes choose nonbinary columns using training data only.
- All 275 benchmark tests pass; the final structured-dimension guard also passes
  all 12 matrix tests. Formatting/lint pass, with the same eight pre-existing type
  diagnostics. Public export preserves the safe matrix inventory.
- Four Breast Cancer/Digits TabPFN cost pilots passed on Hopper (job 360869).
  Estimated full-table time is about 18 minutes at 11 players and 37 minutes at
  12, with peak RSS below 1.2 GiB. These are preparation estimates, not measured
  estimator performance. Together with causal probes they support a 12-hour
  per-seed preparation limit; full preparation remains the qualification gate.
- Frozen source `f0199253` passes all 456 OddSHAP/LeverageSHAP tests. Independent
  review verified source hashes, estimator fixes, all seven compiled extensions,
  exact regeneration of the expanded suite and the preparation wrapper changes.
  Jobs 360871/360872 follow the selected-pairing campaign with separate completion
  watchers; the sweep starts only after successful preparation.

## Twenty-player enumeration, cached costs and signal qualification

- The matrix now selects 356 of 1,524 tabular candidates through 20 players,
  including native Wine at 13. All 620 new setting/seed dimensionality checks
  and 112 bounded construction/payoff probes pass. The full plan has 381
  settings, 8,924 game/target definitions and 1,766,952 planned cells.
- Exact interactions use first/second differences of the saved table, avoiding
  the existing FII solver's enormous diagonal matrix. Independent direct and
  constrained-regression references agree within 2.7e-15 relative error on
  small random, shifted and near-null games. Analytic 20-player constant,
  unanimity and parity cases pass for all six targets. Exactly zero SV and real
  1e-12 perturbations remain distinct; no coefficient threshold is applied.
- Larger tables use authenticated 4,096-coalition checkpoints. Tests cover
  interrupted preparation, reuse without oracle calls, missing chunks, tampered
  payloads, mismatched recipes/source and canonical mixed-game assembly. The
  parallel driver binds its own source hash and locks the staging directory;
  snapshot and plan manifests are written atomically.
- Query cost accounting independently passed duplicate/endpoint/overflow/failure
  checks. Measured runtime is retained; estimated uncached runtime substitutes
  recorded batch-amortized evaluation costs for cache lookup time. No sleeping
  or inferred historical costs are introduced. Desktop/mobile browser checks
  confirm both modes, unchanged accuracy and uniform signal exclusions.
- The new matrix declares a scale-independent 1e-6 minimum signal ratio. Ranking
  exclusions use the full target coordinate count, including sparse zeros.
  Independent checks match physically removing excluded games for accuracy,
  Elo and history while retaining original panel identities and raw results.
- All 305 benchmark tests passed before the final cache-validation guards;
  focused checks cover those guards separately. Formatting/lint pass; the same
  eight pre-existing optional-import/ignore type diagnostics remain.
- Hopper pilot 360874 passed four TabPFN 16/20-player probes. At 20 players,
  estimated work is 7.7–7.9 CPU-days per table; measured peak RSS stays below
  1.2 GiB. This is a small-batch extrapolation, not a runtime guarantee. The
  earlier pending 12-player-cap matrix jobs 360871/360872 were cancelled.
- Clean frozen source `e497a15a` passes all 456 OddSHAP/LeverageSHAP tests.
  Hopper job 360893 tested the actual CLI deadline, resumption, mixed table/KNN
  preparation and repeated assembly: all 13 definitions qualified, and the
  cached artifacts and snapshot were byte-identical on resume. Replacement
  preparation 360900 and evaluation 360901 are submitted with a separate watcher.

- Final boundary checks exposed constant Digits border pixels with undefined
  singleton clustering scores. The recipe now selects nonconstant columns using
  its actual training rows, recording original IDs and the rule. All 676 selected
  feature-game instances pass singleton/endpoint/pair probes; all 95 family/matrix
  tests pass. Frozen source `0762a2e5` passed Hopper test 360899 with 19 definitions,
  including an exhaustively prepared Digits clustering table. Deadline/resume
  again preserved every artifact and snapshot byte. The earlier pending
  matrix20 jobs 360894/360895 were cancelled before execution.

## Modular cleanup (September 30)

- Exact table calculations and authenticated payoff checkpoints now have their
  own modules. Independent before/after checks produced identical metadata for
  18 target definitions and byte-identical small, chunked and classifier tables.
  The numerical function bodies are unchanged.
- The preparation driver separates planning, worker execution and assembly.
  All 1,564 case configurations and 81,100 planned tasks match the previous
  version. Independent checks preserve CPU assignments, commands, timeouts,
  failures and deadline handling.
- Chart rendering moves into one ordinary JavaScript file, loaded by both local
  exports and Pages. The controller retains filters, tables and page state;
  there is still no frontend build or additional dependency. Desktop and
  720/390/320-pixel browser comparisons preserve DOM, geometry and interactions;
  chart function bodies are unchanged. All 35 report tests pass, including the
  new asset-export check.
- All 314 benchmark tests pass. Formatting and lint pass; type checking retains
  the same eight pre-existing optional-import/ignore diagnostics. Frozen
  campaign sources, submitted jobs and completion watchers are unchanged.

## Selected-pairing preparation recovery (September 30)

- Preparation 360853 saved every payoff table but reached its 12-GiB
  address-space limit during exact FII qualification for causal-local 12-player
  seeds 0 and 2. Its dependent sweep 360873 was cancelled without running.
- Recovery 361008 completed in 27 seconds using the same frozen implementation
  and saved payoffs in fresh processes. Successful seeds 1 and 3 served as
  controls; their complete definitions matched exactly. Original preparation
  files remain intact, with a separate recovery input checksum manifest.
- An independent audit authenticated all 1,484 definitions and 324 artifacts,
  verified all 1,472 previously qualified definitions and original file hashes
  were unchanged, and recomputed all six exact targets for the four recovered
  causal-local tables. The final snapshot is
  `0d205e4015d5a81e8b3538578180de58dfd9b97c0a778d3596f2d91c7433e212`.
- Replacement sweep 361009 is queued with the same exclusive-node resources and
  estimator limits. Matrix preparation 360900 follows it; evaluation 360901 is
  unchanged. Both completion watchers remain armed. No new estimator results
  have been published from this recovery.
