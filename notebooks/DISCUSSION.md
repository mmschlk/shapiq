# Project Report & Discussion (WIP)

## Task 3: Reproduction Study

- **Setup:** We reproduced the paper's experiments using XGBoost models trained on real datasets (IRIS, California, Diabetes, Breast Cancer, and Communities).
- **Ground Truth:** For small models ($n \le 10$), we computed the exact Shapley values using brute-force Weighted Least Squares on all coalitions. For large models ($n=30$ and $n=99$), we used the `InterventionalTreeExplainer`.

#### Successful Reproductions:

- **Sampling Probabilities (Figure 2):** We confirmed that LeverageSHAP places equal total probability on every subset size, whereas KernelSHAP uses a U-shaped distribution.
- **Identity-Line Tightness (Figure 1):** We visually reproduced the tight clustering of predicted versus true Shapley values at a budget of $m=5n$.
- **Large-$n$ Superiority (Figure 3):** At $n=30$ and $n=99$, LeverageSHAP clearly beats optimized KernelSHAP (with pairing) at every tested budget.
- **Runtime Efficiency:** We confirmed the algorithm runs in an $O(n \log n)$ evaluation regime, with the $n=99$ dataset finishing in 8.1 seconds.
- **Documented Discrepancy (Table 1):** The original paper claims LeverageSHAP outperforms optimized KernelSHAP even at small feature sizes like $n=8$ and $n=10$. In our notebook, the two methods were tied within noise at these sizes.
- **Analysis of Discrepancy:** For our small-$n$ tests, we used a mean-substitution game. This produces a much smoother and lower-variance game compared to the paper's method, meaning any capable solver will find the same answer. When we switched to the high-variance interventional game (averaging over 20 background rows) for large $n$, LeverageSHAP's theoretical advantage fully reappeared.

## Task 4: Benchmark against existing approximators

- **Setup:** We utilized an automated benchmarking suite (`benchmark.performance`) to evaluate LeverageSHAP against all registered `shapiq` approximators (including KernelSHAP, StratifiedSampling, and OwenSampling) on synthetic Sum of Unanimity Models (SOUM) for $n \in \{6, 8, 10\}$.
- **Metrics & Budgets:** The suite evaluated the algorithms across fractional budgets ($5\%$, $25\%$, $50\%$, and $100\%$ of $2^n$) over multiple random seeds, tracking 7 distinct metrics including MSE, MAE, Precision@k, and Kendall's Tau.
- **Ground Truth:** The exact Shapley values for the SOUM games were computed analytically via the Möbius transform, requiring no exponential evaluations.
- **Benchmark Results (Where it wins/plateaus):** The results show a nuanced picture. LeverageSHAP consistently dominates sampling methods like OwenSampling and StratifiedSampling in the mid-budget regime ($25\%$ to $75\%$ of $2^n$), rapidly reaching an MSE of $\approx 10^{-3}$. However, at full budget allocations ($m = 2^n$), LeverageSHAP plateaus at this error level, while exact methods (and KernelSHAP with pairing) hit the machine precision floor.
- **Theoretical Reasoning:** SOUM is a structureless, synthetic worst-case benchmark. LeverageSHAP's error bound scales with a measure ($\gamma$) of how well Shapley values fit the regression. For real ML models (like our Task 3 XGBoost models), this measure is tiny, but for random SOUM games it is $\approx 1$, making the theoretical bound vacuous. Furthermore, Algorithm 1 solves an unconstrained least squares problem, which explains the plateau at full budget since it does not strictly force boundary matching like exact formulas do.
- **KernelSHAP's Accidental Win:** KernelSHAP heuristically over-samples small and large subsets. This heuristic accidentally aligns with the specific mathematical structure of SOUM unanimity games, giving it a coincidental advantage that becomes a liability on real ML models.

# TODO (work-in-progress)

*Please remove this TODO section once the paper is finished.*

- [ ] Task 3 (Reproduction): Outline the experimental setup, matched results, and honestly document discrepancies.
- [ ] Task 4 (Benchmark): Discuss where the new method wins/loses against existing approximators on benchmark games and explain why based on theory.
