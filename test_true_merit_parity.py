"""
Parity check for RiskAwareSCP._true_cost against the QP's own objective
(_build_qp), verifying the true-merit fix added to fix the rho ratio test's
blind spot (see scp_vel.py's solve()/module docstring).

Uses a real AO-RRT chain as a genuinely dynamically-feasible nominal -- every
knot IS f_edge(prev knot, prev control) by construction (VelPoseDynamics.
batch_propagate is the exact rollout EdgeJacobianEvaluator.batch_jacobians
differentiates), so:

  1. the true dynamics-defect residual at this nominal should be ~0
  2. with the QP's trust region collapsed to a tiny box around this exact
     nominal, _true_cost(S_bar, U_bar, lin.R_bar, lin.f_val, s_goal) should
     match prob.value to a small relative tolerance -- this is the "Delta -> 0
     implies J_pred ~= J_true" check (a Taylor expansion is exact at its own
     expansion point, so risk and defect, the only two approximated terms,
     should introduce zero error there)
  3. the CVaR term specifically must match risk_planner.cvar's exact
     Rockafellar-Uryasev estimator, not a naive top-K-mean shortcut -- checked
     directly against the QP's own optimal tau/eta for the same fixed R.

    python test_true_merit_parity.py [seed] [iters]
"""
import os
import sys
import numpy as np
import cvxpy as cp

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.config import PlannerConfig
from src.environment import make_map_env
from src.ao_rrt import AORRT
from src.scp_vel import RiskAwareSCP, edges_from_chain, _wrap_defect
from src.risk_planner import cvar


def main():
    args = sys.argv[1:]
    nums = [int(a) for a in args if a.isdigit()]
    seed = nums[0] if len(nums) > 0 else 6545
    iters = nums[1] if len(nums) > 1 else 4000

    cfg = PlannerConfig()
    cfg.aorrt.max_iterations = iters
    cfg.risk.dist_grid_n = 5     # M=25 scenarios -- small, fast sanity run

    start = (0.5, 0.5)
    goal = (5.0, 5.0)
    grid, env = make_map_env(cfg, seed, num_rocks=2, start=start, goal=goal,
                              with_risk=True, undulation=1,
                              risk_weights=(0.3, 0.3, 0.4),
                              obstacle_sigma_min=0.01, obstacle_sigma_max=0.05)

    pl = AORRT(env, cfg, start, goal)
    pl.plan(verbose=False)
    if not pl.goal_reached():
        print("AO-RRT did not reach the goal -- try more iterations or a different seed.")
        return
    chain = pl._chain()
    scp = RiskAwareSCP(env, cfg, pl.model)

    # the AO-RRT chain's OWN nsteps_k -- reusing it exactly matters: a different
    # substep count gives a different f_edge and a spurious nonzero defect that
    # has nothing to do with the parity being tested.
    S_bar, U_bar, nsteps_k = edges_from_chain(chain)
    K = U_bar.shape[0]
    M = scp.risk_eval.D.shape[0]
    print(f"AO-RRT chain: K={K} edges, M={M} disturbance scenarios")

    # ---- 1. defect ~= 0 at this genuinely-feasible nominal ---------------------
    lin = scp._linearize(S_bar, U_bar, nsteps_k)
    resid = _wrap_defect(lin.f_val - S_bar[1:])
    max_defect = float(np.max(np.abs(resid)))
    print(f"\n[1] true dynamics-defect at AO-RRT nominal: max |resid| = {max_defect:.3e}")
    assert max_defect < 1e-6, (
        f"AO-RRT chain isn't self-consistent under f_edge (max defect {max_defect:.3e}) "
        f"-- check nsteps_k matches generation-time value")

    # ---- 2. build the QP, pin it to the nominal with a tiny trust region -------
    scp._build_qp(K, M)
    s_start = S_bar[0].copy()
    s_goal = S_bar[-1].copy()
    s_goal[0:2] = np.asarray(goal, dtype=np.float64)
    tiny = 1e-8
    trust_s = np.full(scp.NX, tiny)
    trust_u = np.full(scp.NU, tiny)
    scp._update_params(S_bar, U_bar, lin, trust_s, trust_u, s_start, s_goal)
    scp.prob.solve(solver=cfg.scp_vel.solver)
    print(f"\n[2] QP solve status: {scp.prob.status}")
    assert scp.prob.status == cp.OPTIMAL, (
        f"QP not OPTIMAL at the nominal ({scp.prob.status}) -- likely a control-box "
        f"clip-at-generation-vs-solve mismatch; parity check is vacuous without this")

    # ---- 3. J_pred ~= J_true at Delta -> 0 --------------------------------------
    J_true = scp._true_cost(S_bar, U_bar, lin.R_bar, lin.f_val, lin.h_sdf, s_goal)
    J_pred = float(scp.prob.value)
    rel_err = abs(J_true - J_pred) / max(abs(J_pred), 1e-9)
    print(f"\n[3] J_true = {J_true:.8f}   J_pred (prob.value) = {J_pred:.8f}   "
          f"rel_err = {rel_err:.3e}")
    tol = 1e-6 if cfg.scp_vel.solver == "CLARABEL" else 1e-4
    assert rel_err < tol, f"J_true/J_pred parity failed: rel_err={rel_err:.3e} >= tol={tol:.1e}"

    # ---- 4. CVaR term specifically matches the QP's own tau/eta optimum --------
    alpha = cfg.risk.alpha
    tau_opt = np.asarray(scp.tau.value)
    eta_opt = np.asarray(scp.eta.value)
    max_cvar_err = 0.0
    for k in range(K):
        cvar_true = cvar(lin.R_bar[k], alpha)
        cvar_qp = tau_opt[k] + eta_opt[k, :].sum() / (alpha * M)
        max_cvar_err = max(max_cvar_err, abs(cvar_true - cvar_qp))
    print(f"\n[4] max |cvar(R_bar[k]) - (tau[k] + sum(eta[k,:])/(alpha*M))| over "
          f"{K} edges = {max_cvar_err:.3e}")
    assert max_cvar_err < 1e-5, (
        f"_true_cost's CVaR term diverges from the QP's own RU-epigraph optimum "
        f"(max err {max_cvar_err:.3e}) -- check _true_cost still calls "
        f"risk_planner.cvar (the exact discrete RU estimator), not a naive "
        f"top-K-mean shortcut")

    print("\nAll parity checks passed.")


if __name__ == "__main__":
    main()
