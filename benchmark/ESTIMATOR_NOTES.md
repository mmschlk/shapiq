# Understanding surprising estimator results

The historical comparisons below concern the **September 29, 2026 sweep with budgets
2d, 8d, and 64d**, two estimator seeds, and the library defaults at source commit
`15ea510f`. Observations from the new nine-budget grid are labeled separately. The frozen
inputs and raw measurements are in the
[original reproduction release](https://github.com/rtealwitter/shapiq/releases/tag/benchmark-families-2026-09-29).

The current website instead shows the completed nine-budget, four-instance rerun
at execution commit `d4ac18e6`, with the restored LeverageSHAP safeguard.
LeverageSHAP succeeded on all 1,332 SV cells (148 games × nine budgets), with
no failed SV evaluations. Its unsupported interaction targets are separate.
The historical results below are retained to explain the investigation, not as
current leaderboard measurements.

## LeverageSHAP: an instability near 2d

LeverageSHAP completed all 198 SV cells successfully. Its poor aggregate mean
comes primarily from the lowest budget, especially the 128-player KNN game.
The published **real-game, family-balanced SV panel** gives:

| Budget | LeverageSHAP mean nMSE | KernelSHAP mean nMSE |
| --- | ---: | ---: |
| 2d | 16.4211 | 7.00552 |
| 8d | 0.0813057 | 0.165174 |
| 64d | 0.00097276 | 0.00208605 |
| All three | 5.50112 | 2.39093 |

The pooled median favors LeverageSHAP: 0.02853 versus 0.08146. Its Elo also
exceeds KernelSHAP's, 1126.21 versus 1086.02. Mean error is sensitive to the
magnitude of a few very poor results; Elo reflects how often a method wins.

The cause is the regression's effective sample size. LeverageSHAP defaults to
complementary coalition pairs. After enforcing efficiency, a pair supplies one
independent design direction. At 2d, the two endpoint queries leave exactly
d−1 pairs for d−1 free coefficients. A poorly conditioned sample can therefore
amplify the game's nonadditive residual dramatically. More budget usually
stabilizes the regression. KernelSHAP's shapiq default is unpaired, so its
instability occurs in a different budget regime.

A separate diagnostic with 20 seeds on the same frozen KNN128 game reproduced
this pattern:

| Budget | Mean nMSE | Median nMSE |
| --- | ---: | ---: |
| 1d | 2.5510 | 2.4548 |
| 2d | 21588.3 | 877.502 |
| 4d | 2.02024 | 1.83708 |
| 8d | 0.70614 | 0.69878 |
| 16d | 0.29124 | 0.28963 |

These extra runs diagnose the instability; they are **not published leaderboard
measurements or timing results**. For seed zero, the weighted design condition
number drops from 553 at 2d to 5.25 at 4d. An independent constrained solve using
an orthonormal basis reproduces the original problematic estimates within
`1.8e-11`; the existing 325 LeverageSHAP tests also pass. We found no adapter,
baseline, sampling-weight, or solver error explaining the spike.

Disabling pairing moves the instability toward 1d and worsens the KNN results at
4d and above. Published runs retain their frozen defaults; any corrected
estimator needs a separately identified evaluation.

### Historical regularization was removed

The author's implementation previously included a low-budget ridge safeguard.
[Commit f3c0427](https://github.com/rtealwitter/leverageshap/commit/f3c0427e23ee627e64beb8f51a78f6a9352b0d18)
added it in November 2025: for budgets at most `3d`, a condition-number check
could add `0.001 I` to the projected regression's Gram matrix. The warning
incorrectly called the penalty `0.000001`; the executed penalty was `0.001`.

[Audit commit 04cc121](https://github.com/rtealwitter/leverageshap/commit/04cc121295d508d0e45aaf5bd224724aad122cba)
removed that safeguard in August 2026, reasoning that minimum-norm least squares
already handled singularity. That addresses solving a singular system, but does
not prevent statistical amplification along small nonzero singular directions.
[Shapiq PR #583](https://github.com/mmschlk/shapiq/pull/583), merged August 25,
aligned the sampler without adding regularization to shapiq's existing solver.
That PR's comparison covered `5d–160d`, above the safeguard's `3d` threshold,
so it could not detect the missing low-budget behavior.

The old and current fixed-count implementations use the same weight scale.
Applying the historical `0.001` Gram penalty to the **same** 20 sampled designs at
`2d` changes diagnostic mean nMSE from 21,588 to 2.129 for KNN128, and from 32.148
to 0.1766 for the 30-player tree. These are private diagnostic comparisons, not
replacement leaderboard scores. Ridge introduces bias and need not improve every
game. [Restoration PR #603](https://github.com/mmschlk/shapiq/pull/603) uses the
historical penalty and requested-budget threshold, with `ridge=0` as an opt-out.
It bypasses exhaustive samples and removes the old condition-number gate, which
is unreliable because efficiency already makes the Gram matrix singular. The
original measurements remain unchanged in their archived release; the current
website uses the separately identified corrected rerun.

The [LeverageSHAP paper, Section 5](https://arxiv.org/html/2410.01917v2#S5)
starts its experiments at 5d and reports medians and quartiles over 100 runs.
Its Kernel SHAP baselines also differ from the shapiq implementation. The
paper's comparison therefore does not directly predict this sweep's 2d mean
error. More seeds and the intermediate 4d point make the behavior easier to
interpret.

## OddSHAP: low-budget support selection

The original sweep recorded 190 successful SV cells and eight failures. All
eight failures are four-player cases at 2d: eight queries are below the default
minimum of ten. Interaction targets are unsupported, not failed.

At low budgets, the current implementation also deliberately differs from the
[OddSHAP paper](https://arxiv.org/abs/2602.01399), which falls back to a tree proxy
in this regime. Below 10d, unless the budget already covers every coalition,
shapiq limits the active singleton support to `ceil(B / 10)`; omitted players
receive zero. Its default LightGBM proxy can remain constant with very small
training sets. A diagnostic
additive eight-player game with true values `[1, 2, 3, 4, 5, 6, 7, 8]` returned
approximately `[12.439, 23.561, 0, 0, 0, 0, 0, 0]` at 2d, omitted one player at
8d, and recovered the values at 16d. This identifies a low-budget implementation
limitation, not a coefficient-ordering error in the benchmark. Keep it visible
and investigate changes to the estimator separately.

## SPEX: minimum query blocks and random seeds

All 754 failed SPEX cells in the original sweep were below the sparse
transform's minimum query requirement; 380 cells succeeded. These were not
missing-dependency errors. Under the measured default configuration, the minimum
query counts are 216 for four players, 264 for eight, 312 for thirty, and 408 for
128. Thus many small-game relative budgets are infeasible even though exhaustive
enumeration would be cheap. The first feasible points on the new grid are 64d,
64d, 16d, and 4d, respectively.

SPEX can request repeated coalitions: an eight-player diagnostic at 64d charged
504 queries but visited 226 distinct coalitions. Every request remains charged,
consistently with the other estimators. The
[SPEX paper](https://arxiv.org/abs/2502.13870) targets sparse interaction recovery;
the small-game and tiny-budget settings need not favor that design.

The investigation also found a reproducibility issue: the optional sparse
transform dependency uses global NumPy and Python random generators, beyond the
estimator's `random_state`. Repeating the same estimator seed could therefore
produce different queries. The benchmark correction seeds both generators
before importing a candidate or constructing an estimator, and records the
protocol in provenance. A real SPEX run on the frozen product-kernel game
repeated its output exactly with seeds `[0, 1, 0]`, while seed one differed.
New measurements use that protocol; the original release remains unchanged and
should not be treated as exactly reproducible from `random_state` alone.

## ShaplEIG: computation per selected query

The original sweep recorded 183 successful SV cells and 15 timeouts at the
120-second worker limit: all six KNN128 cells, four tree30 cells, and five
eight-player cells. Unsupported interaction targets are counted separately.

ShaplEIG fits a Gaussian process and optimizes information gain to select
queries. Defaults refit the model every iteration and consider 1,024 candidates.
A bounded diagnostic on KNN128 completed the 129 initialization queries and only
two adaptive queries in 40 seconds. That login-host probe identifies expensive
steps; it is not a standardized runtime measurement. The implementation also
requires at least `d + 2` queries, making 0.5d and 1d infeasible.

The [ShaplEIG paper](https://arxiv.org/abs/2606.02247) motivates spending more
computation to choose valuable evaluations. A saved-table oracle makes the game
queries cheap and exposes that computational overhead. Changing refit frequency,
warm starts, or candidate counts would be a separate configuration; do not tune
these silently to remove timeouts.

## New-grid observation: ProxySPEX at the smallest budgets

On the nine-budget grid, ProxySPEX fails at 0.5d for eight-player games and at
0.5d and 1d for four-player games. Its default LightGBM proxy uses hyperparameter
search with five-fold cross-validation. Two or four sampled coalitions cannot
form five folds, so fitting raises `n_splits=5 > n_samples`. This is a limitation
of the installed default configuration, not the SPEX transform's query-block
minimum or a constructor-level budget check. At 1d, eight-player cases already
complete successfully, although some validation folds then have only one row,
so their R² search scores are undefined. Successful output does not establish
useful hyperparameter selection at such small sample sizes.

ProxySHAP and RegressionMSR default to a bare XGBoost proxy without that search,
and `k_folds=1`; they do not inherit this five-fold requirement. Their shared
coalition sampler requires at least two queries for the endpoints. Custom proxy
models, hyperparameter search, or extra folds can introduce other constraints.
Disabling ProxySPEX's search would be a separate configuration, so the benchmark
retains and reports these failures.

## Why some chart errors are almost zero

The initial `local_baseline_forest` example declares eight feature players, but
its frozen payoff table depends on only player zero: it has two distinct payoffs
and zero additive residual. Several estimators therefore recover its values at
2d (16 queries); errors around 1e-30 are numerical roundoff, not a budget mismatch.
Charts now aggregate the selected game families, while retaining
this easy game in the dataset.

The horizontal budget coordinate is the requested query cap divided by the
player count. Estimators may use fewer queries; hover details show actual usage
in the same relative units. For an eight-player game, the entire coalition space
fits in 32d, so exact recovery at larger budgets can also be legitimate. The
30-player tree and 128-player KNN cases remain useful for studying scaling.

## Limits of this comparison

Most family representatives have only four or eight players. Once a budget
covers all coalitions, estimators that enumerate the game can achieve numerical
precision. That regime says little about their scaling on larger problems.
The expanded suite adds four instances of larger tree, product-kernel and KNN
cases; these structured families still do not represent every large-player game. The original sweep's two seeds are especially
limited for heavy-tailed errors such as LeverageSHAP's 2d results.

Small-game timings include estimator work against **saved payoff tables**;
they exclude training, image inference, and other costs already paid during
preparation. Large structured-game timings include live coalition evaluations.
All workers use the standardized Hopper configuration, but concurrent runs are
still diagnostic timings rather than an isolated runtime study. Compare time
within a game and oracle type, and do not interpret cached-payoff timing as the
end-to-end cost of explaining an application model.
