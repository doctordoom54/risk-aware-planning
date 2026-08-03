import math
import time
import numpy as np

from .dynamics_vel import VelPoseDynamics
from .nn_index import NeighborIndex

class _Node:
    __slots__ = ("x", "cost", "t", "parent", "u", "nsteps", "edgeX", "children")

    def __init__(self, x, cost, t, parent, u, nsteps, edgeX):
        self.x = x
        self.cost = cost
        self.t = t
        self.parent = parent
        self.u = u
        self.nsteps = nsteps
        self.edgeX = edgeX
        self.children = []

class AORRT:
    def __init__(self, env, cfg, start_xy, goal_xy, start_heading=None):
        """
        Asymptotically-Optimal RRT for a two-wheel skid-steer rover.

        Parameters
        ----------
        env       : Environment — collision checker (path_free / sdf_and_grad) and map bounds.
        cfg       : PlannerConfig — groups DynParams (d) and AORRTParams (p) used below.
        start_xy  : (x, y) start position in meters.
        goal_xy   : (x, y) goal  position in meters.
        start_heading : initial heading theta0 (radians); defaults to None, which
            points the rover at the goal (atan2(goal-start)) as before. Pass an
            explicit value (e.g. math.pi/2 for +y) to override that default.

        Attributes set
        --------------
        model / dt / v_max / w_max
            JAX-JIT velocity-pose dynamics (RK4, nominal linear model + learned residual),
            integration timestep, and forward-speed / yaw-rate command bounds.
        max_steps
            Max RK4 steps per edge (~max_steps * dt seconds of simulated motion per branch).
        goal_tol / goal_bias / max_iterations
            Goal-acceptance radius, goal-directed sampling probability, and iteration budget.
        cost_mode / w_u / cost_weight
            Objective: "time", "control", or "both"; control-effort weight; cost-axis scaling
            in the nearest-neighbour metric (makes the search asymptotically optimal).
        candidate_k / branch_k / turn_frac / reverse_prob
            ANN candidates evaluated per sample; controls propagated in parallel per iteration
            (JAX vmap batch); turning/reversing fractions for control sampling.
        nodes
            List of _Node objects forming the tree.  Root node x0 has heading aimed at goal,
            zero wheel speeds, zero cost.
        P / np_ / nn
            Pre-allocated (cap, 3) array of [x, y, cost] for every node and a matching
            NeighborIndex (cost axis scaled by sqrt(cost_weight)) for O(log n) approximate Nearest Neighbors queries.
            cap : Calculates the absolute maximum number of nodes the tree could possibly generate, plus a buffer of 16.
        best_cost / best_goal / t_first
            Lowest goal-reaching cost found so far, the corresponding node, and wall-clock
            time of the first solution (used to measure planning latency separately from
            the subsequent re-wiring / optimisation phase).
        clearance_margin
            Extra obstacle keep-out inflated on top of the rover disc radius. Defaults to
            enough to make env.disc_radius + clearance_margin meet cfg.scp_vel.d_safe --
            the RiskAwareSCP refiner (scp_vel.py) requires every knot to clear obstacles
            by d_safe, and AO-RRT's own path_free() check used to allow anything down to
            bare disc_radius, so a seed could (and did) come back with a knot just inside
            d_safe but still outside disc_radius -- collision-free by AO-RRT's own
            standard, but violating the SCP's stricter one, which the SCP's honest
            dynamics-defect check then correctly refuses to silently patch over (see
            scp_vel.py's solve() module docstring / defect-gate discussion). Tightening
            the margin here fixes it at the source instead. (RiskSensitiveAORRT does NOT
            override this -- despite this being an old belief, the CVaR margin against
            stochastic obstacle boundaries is baked into env.sdf itself now via
            map.risk_sdf.build_risk_sdf, not added here as a uniform scalar.)
        timers
            Per-phase wall-clock profiling: nn / prop / collision / total.
        """
        self.env = env
        d = cfg.dyn; p = cfg.aorrt
        
        # velocity-pose dynamics: nominal A_n/B_n stacked with the learned residual
        self.model = VelPoseDynamics(a_diag=d.a_diag, b_diag=d.b_diag)
        self.dt = d.dt

        self.v_max = p.v_max
        self.w_max = p.w_max

        self.max_steps = p.max_prop_steps
        self.goal_tol = p.goal_tol; self.goal_bias = p.goal_bias
        self.max_iterations = p.max_iterations
        self.cost_mode = p.cost_mode; self.w_u = p.w_control
        self.cost_weight = p.cost_weight; self.candidate_k = p.candidate_k
        self.branch_k = p.branch_k
        self.turn_frac = p.turn_frac; self.reverse_prob = p.reverse_prob

        # x0 -> [px, py, theta, v_b, w_b] (5 States)
        theta0 = (math.atan2(goal_xy[1] - start_xy[1], goal_xy[0] - start_xy[0])
                  if start_heading is None else start_heading)
        x0 = np.array([start_xy[0], start_xy[1], theta0, 0.0, 0.0])
                       
        self.goal = np.array(goal_xy, float)
        self.nodes = [_Node(x0, 0.0, 0.0, None, None, 0, None)]
        
        cap = self.max_iterations * max(1, self.branch_k) + 16
        self.P = np.empty((cap, 3)); self.P[0] = (x0[0], x0[1], 0.0); self.np_ = 1
        self.nn = NeighborIndex(dim=3, weight=[1.0, 1.0, math.sqrt(self.cost_weight)],
                                max_elements=cap)
        self.nn.add(self.P[0])
        self.best_cost = math.inf; self.best_goal = None
        self.t_first = None
        self.clearance_margin = max(0.0, cfg.scp_vel.d_safe - env.disc_radius)
        self.timers = {"nn": 0.0, "prop": 0.0, "collision": 0.0, "total": 0.0}

    # ---- [v_cmd, omega_cmd] control sampling -------------------- #
    def _sample_controls(self, K):
        """Vectorised batch of K [v_cmd, omega_cmd] controls -> (K, 2).
        v_cmd magnitude uniform in [0, v_max], sign governed by reverse_prob;
        omega_cmd uniform in [-w_max, w_max], scaled by turn_frac. Guarantees
        one pure-forward sample per batch."""
        rev = np.random.random(K) < self.reverse_prob
        v_mag = np.random.uniform(0.0, self.v_max, K)
        v = np.where(rev, -v_mag, v_mag)

        w = np.random.uniform(-1.0, 1.0, K) * self.turn_frac * self.w_max

        #v[0], w[0] = self.v_max, 0.0   # guarantee one pure forward sample per batch
        return np.stack([v, w], axis=1)

    def _edge_cost(self, u, nsteps):
        duration = nsteps * self.dt
        if self.cost_mode == "time":
            return duration
        # normalize each command by its own bound BEFORE squaring/summing: v_cmd (m/s)
        # and omega_cmd (rad/s) are different physical units on different natural
        # scales, so a raw u@u privileges whichever axis has the larger bound. Each
        # ratio is in [-1, 1] by construction (_sample_controls never exceeds
        # v_max/w_max), so uu is a dimensionless mean-of-squares in [0, 1].
        v_n = u[0] / self.v_max
        w_n = u[1] / self.w_max
        uu = 0.5 * (v_n * v_n + w_n * w_n)
        if self.cost_mode == "control":
            return self.w_u * uu * duration
        return (1.0 + self.w_u * uu) * duration

    def _extend_cost(self, node, X, u, nsteps):
        return node.cost + self._edge_cost(u, nsteps)

    def _sample(self):
        if np.random.random() < self.goal_bias:
            tx, ty = self.goal
        else:
            tx = np.random.uniform(0, self.env.width); ty = np.random.uniform(0, self.env.height)
        cmax = (self.best_cost if math.isfinite(self.best_cost)
                else float(self.P[:self.np_, 2].max() + 1.0))
        return np.array([tx, ty, np.random.uniform(0, cmax)])

    def _add_node(self, child, parent_idx):
        self.nodes.append(child)
        self.nodes[parent_idx].children.append(len(self.nodes) - 1)
        if self.np_ >= len(self.P):                       
            self.P = np.vstack([self.P, np.empty_like(self.P)])
        self.P[self.np_] = (child.x[0], child.x[1], child.cost)
        self.nn.add(self.P[self.np_])
        self.np_ += 1

    # ---- main loop ----------------------------------------------------- #
    def plan(self, verbose=False):
        t_start = time.perf_counter()
        w = self.nn.w
        for it in range(self.max_iterations):
            if verbose and it % 1000 == 0:
                bc = f"{self.best_cost:.2f}" if math.isfinite(self.best_cost) else "inf"
                print(f"iter {it}/{self.max_iterations} nodes={len(self.nodes)} best={bc}")
            q = self._sample()
            t = time.perf_counter()
            cand = self.nn.knn(q, self.candidate_k)
            self.timers["nn"] += time.perf_counter() - t
            d = ((self.P[cand] - q) * w) ** 2
            near = int(cand[int(np.argmin(d.sum(axis=1)))])
            node = self.nodes[near]

            # batched extension: propagate branch_k sampled controls from `node` in
            # one vmapped JAX call
            nsteps = int(np.random.randint(5, self.max_steps + 1))
            U = self._sample_controls(self.branch_k)              # (K, 2)
            t = time.perf_counter()
            Xb = self.model.batch_propagate(node.x, U, nsteps, self.dt)   # (K, nsteps+1, 5)
            self.timers["prop"] += time.perf_counter() - t
            for k in range(self.branch_k):
                X = Xb[k]
                t = time.perf_counter()
                free = self.env.path_free(X, self.clearance_margin)
                self.timers["collision"] += time.perf_counter() - t
                if not free:
                    continue
                #if the edge is not free, we just skip further computations and move to the next branch.
                xn = X[-1]
                c_new = self._extend_cost(node, X, U[k], nsteps)
                if math.isfinite(self.best_cost) and c_new > self.best_cost:
                    continue                                
                child = _Node(xn, c_new, node.t + nsteps * self.dt, near,
                              U[k].copy(), nsteps, X)
                self._add_node(child, near)
                if math.hypot(xn[0] - self.goal[0], xn[1] - self.goal[1]) <= self.goal_tol \
                        and c_new < self.best_cost:
                    self.best_cost = c_new
                    self.best_goal = child
                    if self.t_first is None:
                        self.t_first = time.perf_counter() - t_start
        self.timers["total"] = time.perf_counter() - t_start
        return self.extract_path()

    # ---- output -------------------------------------------------------- #
    def goal_reached(self):
        return self.best_goal is not None

    def _nearest_to_goal(self):
        P = self.P[:self.np_]
        d = (P[:, 0] - self.goal[0]) ** 2 + (P[:, 1] - self.goal[1]) ** 2
        return self.nodes[int(np.argmin(d))]

    def _chain(self):
        node = self.best_goal if self.best_goal is not None else self._nearest_to_goal()
        chain = []
        while node is not None:
            chain.append(node); node = None if node.parent is None else self.nodes[node.parent]
        chain.reverse()
        return chain

    def extract_path(self):
        chain = self._chain()
        if len(chain) == 1:
            return chain[0].x[None, :]
        segs = [chain[0].x[None, :]]
        for nd in chain[1:]:
            segs.append(nd.edgeX[1:])
        return np.vstack(segs)

    def extract_controls(self):
        chain = self._chain()
        # Returns an (N, 2) array of [v_cmd, omega_cmd] commands
        U = [np.tile(nd.u, (nd.nsteps, 1)) for nd in chain[1:] if nd.u is not None]
        return np.vstack(U) if U else np.zeros((0, 2))

    def total_time(self):
        return self._chain()[-1].t

    def tree_edges(self):
        return [nd.edgeX for nd in self.nodes if nd.edgeX is not None]