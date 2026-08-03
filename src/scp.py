"""
SCP smoothing for the skid-steer kinodynamic plan.

Based on the SCP template of SCPpy (github.com/ynakka/SCPpy; Nakka & Chung,
"Trajectory Optimization of Chance-Constrained Nonlinear Stochastic Systems",
IEEE T-RO 2022): a sequential-convex-programming loop that linearises the nonlinear
dynamics each iteration and solves a convex sub-problem in a trust region.

Improvements over the reference template:
  * SDF-based collision avoidance.  A HARD linearised signed-distance keep-out at
    every knot and inter-knot midpoint --  sdf(p_k) - margin + grad.(p - p_k) >= 0 --
    instead of an explicit list of obstacle half-spaces. Affine in the variables,
    so the sub-problem stays a QP.
  * QP form (box trust regions + quadratic costs), solved by OSQP / PIQP, rather
    than the SOCP/ECOS formulation.
  * Exact terminal reach via an L1 slack penalty (lands on the goal when feasible).

The loop follows SCPpy: solve the convex sub-problem, accept the solution, shrink
the trust region each accepted iterate (grow it on an infeasible solve), and ITERATE
UNTIL the trajectory change  max_t ||x_sol[t] - x_prev[t]|| < tol -- i.e. until it
converges to a smooth fixed point. A larger initial trust lets the first iterates
unwind loops; the shrinking trust then converges the solution.

It runs on the planner's per-dt path + controls (the seed is exactly the rollout of
its controls, so it satisfies the dynamics equality and is collision-free -> the
first QP is feasible). Cost per iteration:

    min  sum_k w_control*||u_k||^2*dt + term_penalty*||s_T||_1 + w_goal*||p_N-goal||^2
    s.t. s_0 = start;  s_{k+1} = f(s_k,u_k) linearised;  |u_k| <= torque_max
         sdf(p_k) - margin + grad.(p_k - p_k^prev) >= 0   (knots + midpoints, hard)
         |s_k - s_k^prev| <= trust_x,  |u_k - u_k^prev| <= trust_u   (box trust)

Requires: numpy, cvxpy (+ a QP solver such as OSQP or PIQP).
"""
import time
import numpy as np
import cvxpy as cp

from .dynamics import SkidSteer4W, NX, NU


class SCPRefiner:
    def __init__(self, env, cfg):
        d = cfg.dyn; s = cfg.scp
        self.env = env
        self.model = SkidSteer4W(d.wheel_radius, d.track_width, d.wheel_inertia,
                                 d.wheel_damping, d.torque_max)
        self.tmax = d.torque_max
        self.w_control = s.w_control; self.w_goal = s.w_goal
        self.term_penalty = s.terminal_slack_penalty
        self.goal_tol = s.goal_tol
        self.margin = env.disc_radius + s.risk_margin + s.clearance_buffer
        self.w_arc = s.w_arc
        self.free_time = s.free_time
        self.dt_min = s.dt_min; self.dt_max = s.dt_max; self.trust_h = s.trust_h
        self.trust_x0 = s.trust_x; self.trust_u0 = s.trust_u
        self.trust_grow = s.trust_grow; self.trust_shrink = s.trust_shrink
        self.trust_x_max = s.trust_x_max; self.trust_u_max = s.trust_u_max
        self.trust_min = s.trust_min
        self.iter_max = s.iter_max; self.tol = s.tol; self.solver = s.solver

    def _clearance(self, p):
        """Signed clearance of point p (m): sdf - margin (>=0 means disc-clear).
        Scalar convenience; the SCP keep-out uses env.sdf_and_grad (batched,
        autodiff) for the value AND exact gradient at every knot/midpoint."""
        return float(self.env.sdf_at(p[0], p[1]) - self.margin)

    def _rollout(self, x0, U, dt):
        """True nonlinear rollout of a control sequence (for honest evaluation)."""
        X = np.empty((len(U) + 1, len(x0))); X[0] = x0; x = np.asarray(x0, float).copy()
        for k in range(len(U)):
            x = self.model.step(x, U[k], dt); X[k + 1] = x
        return X

    def _merit(self, X, U, dt, goal):
        """Control effort + heavy terminal-reach penalty (the cost SCP minimises)."""
        eff = self.control_cost(U, dt)
        term = self.term_penalty * float(np.hypot(X[-1, 0] - goal[0], X[-1, 1] - goal[1]))
        return eff + term

    def control_cost(self, U, dt):
        return float(self.w_control * np.sum(U * U) * dt)

    def _risk_terms(self, cp, Sv, Uv, Sprev, Uprev, hbar):
        """Hook: extra convex objective term + constraints to add to the SCP QP.
        The base refiner adds nothing; the risk-sensitive refiner overrides this to
        add a CVaR penalty on cumulative terrain risk (Rockafellar-Uryasev epigraph).
        Returns (obj_add, cons_add) where obj_add is a cvxpy expression (or 0)."""
        return 0, []

    def refine(self, path, U_seed, dt, goal_xy, verbose=True, solver=None):
        """Smooth the AO-RRT plan. `path` (M,7) and `U_seed` (M-1,4) are the
        per-dt state/control sequence straight from the planner: since `path` is
        exactly the rollout of `U_seed` at step `dt`, the seed satisfies the
        dynamics equality and is collision-free, so the HARD linearised-SDF
        keep-out QP is feasible from the first iterate."""
        
        solver = solver or self.solver
        tok = getattr(cp, solver, None)
        Sprev = np.asarray(path, float).copy()
        Uprev = np.asarray(U_seed, float).copy()
        N = len(Sprev)
        if len(Uprev) != N - 1:                      # guard against length mismatch
            N = min(N, len(Uprev) + 1)
            Sprev = Sprev[:N]; Uprev = Uprev[:N - 1]
        x0 = Sprev[0].copy()
        goal = np.asarray(goal_xy, float)
        # SCPpy-style loop: solve the convex sub-problem, shrink the trust each
        # accepted iterate (grow on an infeasible solve), and ITERATE UNTIL the
        # trajectory stops changing -- max_t ||x_sol[t] - x_prev[t]|| < tol -- i.e.
        # until it converges to a smooth fixed point.
        trust = {"x": self.trust_x0, "u": self.trust_u0}
        hbar = float(dt)                                  # current (global) timestep
        cost0 = self.control_cost(Uprev, dt); hist = [cost0]
        n_solves = 0; solve_t = 0.0; iters = 0; error = np.inf
        t_start = time.perf_counter()

        while iters < self.iter_max and error >= self.tol:
            # Linearise the dynamics about the current iterate (state, control, and --
            # for free final time -- the timestep h via dstep/d(dt)). EXACT autodiff
            # over the whole horizon in one vmapped, jit-compiled JAX call (replaces
            # the per-knot finite-difference loop).
            A, B, f = self.model.batch_jacobians(Sprev[:N - 1], Uprev[:N - 1], hbar)
            C = (self.model.batch_dstep_ddt(Sprev[:N - 1], Uprev[:N - 1], hbar)
                 if self.free_time else np.zeros((N - 1, NX)))
            # EXACT SDF value + spatial gradient at every knot and inter-knot
            # midpoint, batched through the autodiff bilinear lookup.
            hk, gk = self.env.sdf_and_grad(Sprev[:, 0:2]); hk = hk - self.margin
            mids = 0.5 * (Sprev[:N - 1, 0:2] + Sprev[1:N, 0:2])
            hm, gm = self.env.sdf_and_grad(mids); hm = hm - self.margin
            Sv = cp.Variable((N, NX)); Uv = cp.Variable((N - 1, NU))
            sT = cp.Variable(2, nonneg=True)                  # terminal-reach slack
            cons = [Sv[0] == x0]
            cons += [Sv[N - 1, 0:2] - goal <= sT, goal - Sv[N - 1, 0:2] <= sT]
            obj = self.term_penalty * cp.sum(sT) + self.w_goal * cp.sum_squares(Sv[N - 1, 0:2] - goal)
            # free final time: one global timestep variable (uniform dt), bounded and
            # trust-limited; the trajectory can shrink its duration instead of filling
            # a fixed horizon with a loop.
            if self.free_time:
                dtv = cp.Variable(nonneg=True)
                cons += [dtv >= self.dt_min, dtv <= self.dt_max,
                         cp.abs(dtv - hbar) <= self.trust_h]
                dh = dtv - hbar
            for k in range(N - 1):
                dyn = f[k] + A[k] @ (Sv[k] - Sprev[k]) + B[k] @ (Uv[k] - Uprev[k])
                if self.free_time:
                    dyn = dyn + C[k] * dh
                cons += [Sv[k + 1] == dyn]
                cons += [cp.abs(Uv[k]) <= self.tmax]
                cons += [cp.abs(Uv[k] - Uprev[k]) <= trust["u"]]
                obj += self.w_control * dt * cp.sum_squares(Uv[k])
                # arc-length penalty: shortest evenly-spaced path -> unwinds loops
                obj += self.w_arc * cp.sum_squares(Sv[k + 1, 0:2] - Sv[k, 0:2])
            # HARD linearised-SDF collision keep-out (the signed distance encodes all
            # obstacles): at every knot and inter-knot midpoint,
            #     sdf(p^prev) - margin + grad_sdf . (p - p^prev) >= 0.
            # With a TRUE signed SDF (negative inside, outward gradient) this pushes
            # any knot out of an obstacle and keeps the converged knots clear by the
            # rover disc radius.
            for k in range(N):
                cons += [hk[k] + gk[k] @ (Sv[k, 0:2] - Sprev[k, 0:2]) >= 0]
                cons += [cp.abs(Sv[k] - Sprev[k]) <= trust["x"]]
            for k in range(N - 1):
                mp = 0.5 * (Sprev[k, 0:2] + Sprev[k + 1, 0:2])
                cons += [hm[k] + gm[k] @ (0.5 * (Sv[k, 0:2] + Sv[k + 1, 0:2]) - mp) >= 0]
            # risk-sensitive hook: a convex CVaR penalty on cumulative terrain risk
            # (no-op in the base refiner; see RiskSensitiveSCP).
            obj_risk, cons_risk = self._risk_terms(cp, Sv, Uv, Sprev, Uprev, hbar)
            obj = obj + obj_risk
            if cons_risk:
                cons += cons_risk
            prob = cp.Problem(cp.Minimize(obj), cons)
            _t = time.perf_counter()
            try:
                prob.solve(solver=tok) if tok is not None else prob.solve()
            except Exception:
                prob.solve()
            solve_t += time.perf_counter() - _t; n_solves += 1
            if Sv.value is None:                              # INCREASE_TRUST and retry
                trust["x"] = min(trust["x"] * self.trust_grow, self.trust_x_max)
                trust["u"] = min(trust["u"] * self.trust_grow, self.trust_u_max)
                if trust["x"] >= self.trust_x_max:
                    break
                continue
            # accept the convex solution; convergence on trajectory change
            Xs = np.array(Sv.value); Us = np.clip(np.array(Uv.value), -self.tmax, self.tmax)
            error = float(np.max(np.linalg.norm(Xs - Sprev, axis=1)))
            Sprev, Uprev = Xs, Us
            if self.free_time and dtv.value is not None:
                hbar = float(np.clip(dtv.value, self.dt_min, self.dt_max))
            trust["x"] *= self.trust_shrink; trust["u"] *= self.trust_shrink
            iters += 1; hist.append(self.control_cost(Uprev, hbar))
            if verbose:
                clr = min(self.env.sdf_at(p[0], p[1]) - self.env.disc_radius for p in Sprev)
                print(f"  SCP it {iters:2d}: max_move={error:.4f}  dt={hbar:.3f}  "
                      f"trust_x={trust['x']:.2f}  ctrl={hist[-1]:.3f}  min_clear={clr:.3f}")
        # honest collision check on a DENSE interpolation of the knot path
        # (true disc clearance = sdf - disc_radius, ignoring the planning buffer)
        dense = []
        for k in range(len(Sprev) - 1):
            for a in (0.0, 0.2, 0.4, 0.6, 0.8):
                dense.append((1 - a) * Sprev[k, 0:2] + a * Sprev[k + 1, 0:2])
        dense.append(Sprev[-1, 0:2])
        min_clear = float(min(self.env.sdf_at(p[0], p[1]) - self.env.disc_radius for p in dense))
        term_err = float(np.hypot(Sprev[-1, 0] - goal[0], Sprev[-1, 1] - goal[1]))
        info = {"iters": iters, "cost0": cost0, "cost_final": hist[-1], "dt": hbar,
                "dt_seed": dt, "total_time": (N - 1) * hbar,
                "cost_history": hist, "min_clearance": min_clear,
                "feasible": bool(min_clear >= -1e-6), "n_solves": n_solves,
                "terminal_err": term_err, "reached": bool(term_err <= self.goal_tol),
                "solve_time_s": solve_t, "wall_time_s": time.perf_counter() - t_start,
                "converged": bool(error < self.tol),
                "solver": solver if tok is not None else "default"}
        if verbose:
            print(f"  SCP done: {iters} iters, ctrl {cost0:.3f} -> {hist[-1]:.3f}, "
                  f"goal_err={term_err:.3f} (reached={info['reached']}), "
                  f"min_clear={min_clear:.3f}, solve={solve_t:.3f}s")
        return Sprev, Uprev, info