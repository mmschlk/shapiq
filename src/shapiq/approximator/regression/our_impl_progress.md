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

---

**### 7. The Math Behind `_solve` — Step by Step**

**What `_solve` receives**

After sampling and game evaluation, `_solve` is handed three things:

- $Z \in \{0,1\}^{m \times n}$ — the coalition matrix. Each row is a binary vector: $Z_{ji} = 1$ if feature $i$ is present in coalition $j$, $0$ otherwise. $m$ is the number of sampled coalitions, $n$ is the number of features.
- $\mathbf{v} \in \mathbb{R}^m$ — the game values. Entry $v_j = v(z_j)$ is the model's output when only the features indicated by row $j$ are active.
- $v_0 = v(\emptyset)$ — the baseline: model output with no features active.

The goal is to find $\phi \in \mathbb{R}^n$ — one Shapley value per feature — such that:

1. $\phi$ minimises the weighted squared error between the game values and what the linear model $Z\phi + v_0$ predicts, and
2. the **efficiency axiom** holds: $\displaystyle\sum_{i=1}^n \phi_i = v(N) - v(\emptyset)$, i.e. the Shapley values sum to the total payout of the game.

---

**Why the efficiency axiom creates a problem — and why we need the shift**

A naive unconstrained regression would just minimise $\|Z\phi - (\mathbf{v} - v_0)\|^2$. The solution is not guaranteed to satisfy $\sum_i \phi_i = v(N) - v(\emptyset)$, so the efficiency axiom would be violated.

One fix is constrained regression (add $\mathbf{1}^\top \phi = v(N) - v(\emptyset)$ as a hard constraint). That works but requires a constrained solver — slower and more complex.

LeverageSHAP uses a smarter approach: **project the problem into the subspace where the constraint is automatically satisfied**. The idea is to decompose $\phi$ into two parts:

$$\phi = \underbrace{\phi^\perp}_{\text{regression solves for this}} + \underbrace{\delta \cdot \mathbf{1}}_{\text{we add this back at the end}}$$

where $\delta = \frac{v(N) - v(\emptyset)}{n}$ is the uniform "fair share" per feature. Because $\phi^\perp \perp \mathbf{1}$ (its entries sum to zero by construction), adding $\delta$ to every entry shifts the sum from $0$ to exactly $v(N) - v(\emptyset)$ — satisfying efficiency axiom without a constrained solver.

This is why the efficiency shift $\delta$ must be computed first: it is used both to **transform the regression problem** (remove the uniform component from the target vector) and to **restore the solution** afterwards.

**What is $\phi$ and what does $\phi^\perp$ mean geometrically?**

$\phi \in \mathbb{R}^n$ is just a vector of $n$ numbers — one Shapley value per feature. Think of it as a point in $n$-dimensional space.

The efficiency axiom says this point must lie on a specific hyperplane:

$$H = \{\phi \in \mathbb{R}^n : \phi_1 + \phi_2 + \cdots + \phi_n = v(N) - v(\emptyset)\}$$

This hyperplane is perpendicular to the all-ones vector $\mathbf{1} = (1,1,\ldots,1)^\top$. Every point on $H$ can be written as:

$$\phi = \underbrace{\delta \cdot \mathbf{1}}_{\text{one fixed point on } H} + \underbrace{\phi^\perp}_{\text{displacement within } H}$$

where $\delta \cdot \mathbf{1}$ is the centre of $H$ (the uniform solution) and $\phi^\perp$ is the displacement away from that centre, constrained to stay inside $H$, i.e. $\mathbf{1}^\top \phi^\perp = 0$.

So $\phi^\perp$ is **what makes the Shapley values different from each other** — it encodes all the relative importance information. The regression only needs to find $\phi^\perp$; the shift $\delta \cdot \mathbf{1}$ is already known.

**Visual intuition with $n = 2$ features**

Say there are 2 features and $v(N) - v(\emptyset) = 1.0$. The efficiency axiom says:

$$\phi_1 + \phi_2 = 1.0$$

In 2D, this is just a diagonal line. Every valid Shapley solution must sit on it:

```
φ₂
 |        * all valid (φ₁, φ₂) pairs live on this line
 1  \
    \
0.5  · ← δ·1 = (0.5, 0.5)  [uniform centre, both features equal]
    \
 0    \
 +----\----→ φ₁
 0   0.5   1
```

The uniform centre $\delta \cdot \mathbf{1} = (0.5, 0.5)$ is the midpoint of the line. $\phi^\perp$ is how far we slide along the line from that centre:

```
φ₂
 |
 1  \
    \   ← slide up-left: feature 2 more important
0.5  ·  centre (δ·1)
    \   ← slide down-right: feature 1 more important
 0    \
 +----\----→ φ₁
```

Notice: sliding along the line always keeps $\phi_1 + \phi_2 = 1.0$ because the line direction is $(-1, +1)$ — which is perpendicular to $\mathbf{1} = (1,1)$. That is exactly $\phi^\perp$: a vector with entries that sum to zero (moving one feature up forces the other down by the same amount).

So the decomposition is:

$$\phi = \underbrace{(0.5,\ 0.5)}_{\delta\cdot\mathbf{1},\ \text{the centre}} + \underbrace{(d,\ -d)}_{\phi^\perp,\ \text{slide along the line}}$$

The regression finds $d$ (and its generalisation to $n$ features). $\delta$ was never in question.

**Why replacing $Z$ with $A = ZP$ enforces $\mathbf{1}^\top \phi^\perp = 0$ automatically**

Substituting $\phi = \phi^\perp + \delta \cdot \mathbf{1}$ into the regression $Z\phi \approx \mathbf{v} - v_0$:

$$Z(\phi^\perp + \delta\mathbf{1}) \approx \mathbf{v} - v_0$$
$$Z\phi^\perp + \delta \underbrace{Z\mathbf{1}}_{=\, \mathbf{s}} \approx \mathbf{v} - v_0$$
$$Z\phi^\perp \approx \underbrace{(\mathbf{v} - v_0) - \delta\mathbf{s}}_{=\, \mathbf{b}}$$

This is already an unconstrained regression in $\phi^\perp$, but $Z$ still has columns that are not orthogonal to $\mathbf{1}$, so an unconstrained solver could return a solution with $\mathbf{1}^\top\phi^\perp \neq 0$.

Think of it this way: the solver does not know it is supposed to stay on the diagonal line. Without the projection it could wander off the line and return a $\phi^\perp$ that is not actually perpendicular to $\mathbf{1}$.

The projection matrix $P = I - \frac{1}{n}\mathbf{1}\mathbf{1}^\top$ zeroes out the $\mathbf{1}$ direction entirely:

$$P\mathbf{1} = \mathbf{1} - \frac{1}{n}\mathbf{1}(\mathbf{1}^\top\mathbf{1}) = \mathbf{1} - \mathbf{1} = \mathbf{0}$$

For $n = 2$: $P = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix} - \frac{1}{2}\begin{pmatrix} 1 \\ 1 \end{pmatrix}\begin{pmatrix} 1 & 1 \end{pmatrix} = \begin{pmatrix} 0.5 & -0.5 \\ -0.5 & 0.5 \end{pmatrix}$

Multiplying any vector by $P$ removes its component along $\mathbf{1}$ and keeps only the part along $(-1, +1)$ — i.e. it keeps only the part that lives on the diagonal line. So $A = ZP$ has all its information compressed into the "slide along the line" direction. `lstsq` then returns the minimum-norm solution in that direction, which by construction sums to zero — no constrained solver needed.

---

**Goal:** Find Shapley values $\phi \in \mathbb{R}^n$ such that the weighted least squares problem is solved and the efficiency axiom $\sum_i \phi_i = v(N) - v(\emptyset)$ is satisfied.

---

**Step 1 — Efficiency shift**

$$\delta = \frac{v(N) - v(\emptyset)}{n}$$

This is the "fair share" each feature would receive if the total payout were distributed uniformly. It will be used both to build the regression target and to restore efficiency at the end.

---

**Step 2 — Projection trick (Lemma 3.1): $A = Z \cdot P$**

Directly solving the efficiency-constrained regression is hard. Instead, project $Z$ onto the orthogonal complement of $\mathbf{1}$ via the projection matrix:

$$P = I - \frac{1}{n}\mathbf{1}\mathbf{1}^\top$$

Rather than materialising $P$ (which would cost $O(n^2)$), we exploit:

$$A_{j} = z_j - \frac{s_j}{n} \cdot \mathbf{1}^\top$$

i.e., subtract the row mean $s_j/n$ from every entry of coalition row $z_j$. This is the **row-centering trick**. Result: $A \cdot \mathbf{1} = 0$ for every row, so the null space of $A$ is $\text{span}\{\mathbf{1}\}$.

---

**Step 3 — Centered target vector $b$**

To match the transformed design matrix $A$, the target must be adjusted accordingly:

$$b_j = \bigl(v(z_j) - v(\emptyset)\bigr) - \delta \cdot s_j$$

This removes the component of the game values that is already explained by the uniform efficiency shift.

---

**Step 4 — IS-corrected weights $w$**

We cannot evaluate all $2^n - 2$ coalitions, so we sample $m$ of them. Because we sample proportionally to leverage scores (not uniformly), each sample is over- or under-represented relative to what the regression needs. **Importance-sampling (IS) correction** fixes this: divide the required regression weight by the probability we used to draw the sample.

$$w_{\text{IS}}(S) = \frac{w_{\text{required}}(S)}{p(S)}$$

**Required weight — the Shapley kernel.** The unique weight that makes WLS reproduce exact Shapley values is:

$$w_{\text{KSHAP}}(s) = \frac{n-1}{\binom{n}{s} \cdot s \cdot (n-s)}$$

This is not a free parameter — it is derived from the Shapley axioms. The $\binom{n}{s}$ in the denominator can reach $\approx 10^{29}$ for $n = 100$.

**Sampling probability — leverage scores.** LeverageSHAP draws coalitions in two steps: pick size $s$ uniformly from $\{1,\ldots,n-1\}$, then pick a specific coalition uniformly from all $\binom{n}{s}$ of that size:

$$p(S) = \frac{1}{n-1} \cdot \frac{1}{\binom{n}{s}}$$

**The cancellation.** Plugging both into $w_{\text{IS}} = w_{\text{KSHAP}} / p$:

$$w_{\text{IS}}(s) = \frac{\dfrac{n-1}{\binom{n}{s} \cdot s \cdot (n-s)}}{\dfrac{1}{(n-1)\,\binom{n}{s}}} = \frac{n-1}{\binom{n}{s} \cdot s(n-s)} \cdot (n-1)\binom{n}{s} = \frac{(n-1)^2}{s(n-s)}$$

The $\binom{n}{s}$ appears in both numerator and denominator and **cancels exactly**. $(n-1)^2$ is a global constant that scales all weights equally and leaves the minimiser $\phi$ unchanged, so:

$$\boxed{w_{\text{IS}}(s) = \frac{1}{s(n-s)}}$$

Values stay in $\left[\frac{4}{n^2},\ 1\right]$ for all $n$ — no overflow, no catastrophic cancellation. The cancellation is **by design**: the leverage-score probability was chosen specifically to share the $\binom{n}{s}$ factor with the Shapley kernel, so they cancel analytically before any number is computed.

---

**Step 5 — Weighted least squares**

Solve:

$$\min_{\phi^\perp} \left\| W^{1/2} A\, \phi^\perp - W^{1/2} b \right\|_2^2$$

where $W = \text{diag}(w_{\text{IS}})$. We pass $W^{1/2} A$ and $W^{1/2} b$ directly to `np.linalg.lstsq` (SVD-based). This keeps the effective condition number at $\kappa(W^{1/2}A)$ instead of $\kappa(A^\top W A) = \kappa(W^{1/2}A)^2$.

Because $A \cdot \mathbf{1} = 0$, the Gram matrix $A^\top W A$ is rank $n-1$. `lstsq` detects the null direction and returns the unique **minimum-norm solution** $\phi^\perp \in (\text{span}\,\mathbf{1})^\perp$, i.e. $\sum_i \phi^\perp_i = 0$.

---

**Step 6 — Efficiency correction**

$$\phi = \phi^\perp + \delta \cdot \mathbf{1}$$

Because $\phi^\perp \perp \mathbf{1}$, we have $\sum_i \phi_i = 0 + n\delta = v(N) - v(\emptyset)$ exactly. The efficiency axiom is satisfied without any further solver constraint.
