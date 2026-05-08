**### 1. The Estimator**
- Leverage SHAP: modification of the widely used Kernel SHAP algorithm
- Normal Shapley values have a strict rule: all values combined must exactly equal the total output of the model (efficiency axiom)
- => This constraint makes the math incredibly hard and slow for the computer
- Leverage SHAP: reformulates the Shapley value estimation from a linearly constrained regression problem into an unconstrained least squares regression problem using a projection matrix
- Leverage SHAP uses a clever trick:
- uses matrix projection to simply ignore this rule during the calculation
- take Matrix Z and "filter" it: A = ZP; P = projection matrix
- To satisfy the efficiency axiom: it simply adds a fixed constant to every result b: b = y - Z1
- estimator: subsample of model evaluations
- estimator finds the approximate Shapley values by using the least squares method (minimizing the sum of the squared residuals)

**### 2. The Sampling Scheme**
- Kernel SHAP picks feature combinations based on a heuristic rule
- Leverage SHAP: uses leverage scores instead
- Leverage scores: statistical metric that tells you exactly which combinations have the biggest "leverage" or impact on the final result
- Normally, calculating this leverage takes forever
- the authors prove (in Lemma 3.2) that the leverage only depends on the coalition size
- Every 3-feature coalition has the exact same leverage as any other 3-feature team
- how does it work in practice? => algorithm just picks a random group size, and then randomly draws feature teams of that exact size
- It never draws the same team twice (= sampling without replacement)
- whenever it draws a team, it immediately draws its exact opposite as well (Paired Sampling)

**### 3. Claimed Convergence**
- Kernel SHAP: could never strictly prove mathematically that it always works well with small samples
- Leverage SHAP provides this exact proof (non-asymptotic guarantees)
- Result: It needs extremely few model evaluations
- If a model has $n$ features, Leverage SHAP only needs a budget that grows by $O(n \log n)$ (nearly linear, not exponential)

**### 4. The Experiments**
- The authors tested the method on 8 real-world datasets
- ground truth: they used Tree SHAP, because it can calculate the exact ground truth Shapley values perfectly
- benchmarked Leverage SHAP against the standard Kernel SHAP and a highly optimized version of Kernel SHAP
- result: Leverage SHAP won in almost all tests (measured by the $\ell_2$-norm error)
- advantage was especially massive when the AI model had a lot of features (large $n$), but the algorithm only had a very small test budget available (small $m$).


**### 5. Implementation Notes**

**Sampling (`_sample`)**
- in regression problems, leverage scores tell me which data points are most influential for the solution -> sampling coalitions proportional to their leverage scores, I get lower-variance estimates
- Problem: $\ell(S) = \frac{1}{\binom{n}{s}}$
- always includes empty coalition `{}` and grand coalition `N`
- for remaining budget, samples coalitions using leverage score sampling (uniform distribution over sizes + Bernoulli without replacement)
- sampling P for each coalition == leverage score

**Regression Solver (`_solve`)**
- efficiency constraint: code computes the "fair share" each player would get if we distributed this uniformly -> constrained regression that needs to be converted to an unconstrained regression by projecting it onto the orthogonal complement of the all-ones vector with projection matrix PM = I - some fractions
- row centering trick instead of computing dot product with Z@PM because then mathematically/numerically cheaper
- coalitions with certain criteria enter regression
- projection trick thing (Lemma 3.1)
- compute IS-corrected weights (binomial terms cancel!)
- solve weighted least squares > minimization problem
- shift solution back to restore efficiency
- corrected weight is just $\frac{1}{s(n-s)}$ -> no huge binomial coefficients anywhere!

**#### Remarks on LeverageSHAP approximation**
- For non-additive games shaply values increasingly off -> because for additive games there is no resudual 
- For complex interactions there is might not be a fixed indivisual contribution for each feature -> because it depends on what other features are present. 
- When run a few hundert times then Shapley values should converge to exect values


**### 6. Numerical Considerations in the _solve Implementation**

**Why `lstsq` instead of `solve`**
- `A·1 = 0` always (every row sums to zero after centering) -> the Gram matrix $A^\top W A$ is always rank $n-1$
- `np.linalg.solve` does not detect near-singularity for float matrices -> silently returns one arbitrary solution from the whole solution family, shifted by an unknown multiple of $\mathbf{1}$ along the null space -> breaks the efficiency correction
- `lstsq` (SVD) detects the null direction and returns the unique **minimum-norm** solution, which lies in $(\text{span}\,\mathbf{1})^\perp$ -> exactly what we need

**Why explicit $W^{1/2}$ row-scaling instead of forming normal equations $A^\top W A$**
- forming $A^\top W A$ explicitly squares the condition number: $\kappa(A^\top W A) = \kappa(A)^2$
- for ill-conditioned design matrices (correlated features, extreme weight ratios) this quadratic amplification can erase half the digits of float64 precision
- by passing $W^{1/2} A$ directly to `lstsq`, the effective condition number stays at $\kappa(W^{1/2} A)$, not its square

**Why the binomial cancellation matters numerically**
- $\binom{100}{50} \approx 10^{29}$ -> even though float64 can represent that, dividing a large binom weight by an equally large binom probability loses significant digits in the cancellation
- the IS cancellation means we never compute $\binom{n}{s}$ at all -> $w_\text{is} = \frac{1}{s(n-s)}$ is always in the range $[\frac{4}{n^2}, 1]$, no precision loss regardless of $n$

**Why the minimum-norm solution makes efficiency correction exact**
- because `lstsq` returns $\phi^\perp \in (\text{span}\,\mathbf{1})^\perp$, we know $\langle \phi^\perp, \mathbf{1} \rangle = 0$ exactly (up to machine epsilon)
- adding `efficiency_shift`$\cdot \mathbf{1}$ therefore shifts the sum from $0$ to $v(N) - v(\emptyset)$ exactly -> efficiency axiom satisfied without any further correction

(I hope i payed enough attention in my Nummeric I lecture)