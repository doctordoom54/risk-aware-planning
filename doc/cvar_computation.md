# CVaR Computation (`src/risk_planner.py`)

## Definition

$$
\mathrm{CVaR}_\alpha(Z) = \min_t \; t + \frac{1}{\alpha}\,\mathbb{E}\big[(Z-t)_+\big]
$$

Average of the worst $\alpha$-fraction of outcomes. $\alpha=1\Rightarrow$ mean (risk-neutral); $\alpha\to 0\Rightarrow$ worst-case.

**Sample estimator** (`cvar()` / `cvar_weights()`): sort $n$ samples descending, put weight $1/(\alpha n)$ on the worst $\lfloor \alpha n\rfloor$, the remainder on the boundary sample.

## Edge risk = two terms added together

For each AO-RRT edge, `_edge_tracking_cvar` computes:

$$
\text{cost}_{\text{edge}} = \underbrace{\mathrm{CVaR}_\alpha(L)}_{\text{tracking deviation}} \;+\; \underbrace{s_k \cdot \mathrm{CVaR}_\alpha(\mathcal N(0,1))}_{\text{obstacle boundary uncertainty}}
$$

**Term 1 — tracking-induced penetration $L$:**
- Roll out the full closed-loop tracked ensemble ($M = \texttt{dist\_grid\_n}^2$ members) under the edge's control.
- Per member: take the **min** body-surface clearance (signed distance − `disc_radius`) over the whole edge → worst point of approach.
- Hinge: $L^{(m)} = \max(0, -\text{clearance}^{(m)})$ — zero if that member stayed clear, positive if its disc actually overlapped an obstacle.
- $\mathrm{CVaR}_\alpha(L)$ over the $M$ members — always $\geq 0$ since the hinge is applied *before* the CVaR.

**Term 2 — obstacle boundary's own Gaussian uncertainty** (mean 0, std $s_k$): each
obstacle instance $k$ (rock/crater/wall) draws its own std uniform in
`(obstacle_sigma_min, obstacle_sigma_max)` (`EnvParams`, see `map.sdf.ObstacleSigmaField`)
instead of sharing one constant. $s_k$ is read off (`env.obstacle_sigma_at` /
the fused kernel's `obstacle_sigma_device` lookup) at the ensemble's single worst
point of approach for this edge — i.e. the boundary of whichever obstacle the edge
actually got closest to. Closed form via the CVaR decomposition (subadditivity, then
positive homogeneity):

$$
\mathrm{CVaR}_\alpha(L+\epsilon) \le \mathrm{CVaR}_\alpha(L) + \mathrm{CVaR}_\alpha(\epsilon) = \mathrm{CVaR}_\alpha(L) + s_k\,\frac{\phi(\Phi^{-1}(1-\alpha))}{\alpha}
$$

— a cheap constant (`cvar_gaussian(alpha)`), no extra sampling.

## Where it lives
- `cvar()`, `cvar_gaussian()`, `edge_tracking_cvar()` — reference implementations, `src/risk_planner.py`.
- `RiskSensitiveAORRT._edge_tracking_cvar` — same computation, fused into one JIT kernel (rollout + SDF lookup + reduction, one host round-trip) for use inside `_extend_cost`.
- Toggle: `RiskParams.use_edge_cvar` / weight: `RiskParams.edge_cvar_weight`.
