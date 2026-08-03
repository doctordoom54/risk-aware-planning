"""
AO-RRT on the point-cloud-derived map in pcd/data/processed_pcd (real rover
arena, not the synthetic LunarMapGenerator map main.py uses).

    python hardwarexps/test_exp.py [cost_mode] [seed] [iters] [savecsv]
        cost_mode = time | control | both     (default: control)
        savecsv   = optional flag; also settable via SAVE_TRAJECTORY_CSV below.
                    Writes the solved trajectory (x, y, theta, v_b, omega, t)
                    to results/<CSV_NAME>.

Loads riskmap1_sdf.npz (the TRUE unclamped signed-distance field baked offline
by pcd/src/sdf.py, in the exact same convention as src/environment.py's
_true_signed_sdf -- positive outside obstacles, negative inside, metres) and
riskmap1_polygons.npz (map boundary + rock outlines, for plotting only; already
stored in the same mirrored frame as the SDF, so it's used as-is, unlike
boundary_polygons.npz which needs _mirror_x), wraps them in PCDEnvironment
below, and runs the SAME AORRT planner main.py uses -- AORRT's dynamics model
is the NN-residual VelPoseDynamics (src/dynamics_vel.py) by default; this
script never touches the risk-aware SCP/CVaR path (src/scp_vel.py,
src/risk_planner.py).

No existing method in src/ or map/ is modified: PCDEnvironment only reuses the
pure bilinear-lookup helpers from src.environment (_bilinear/_bilinear_batch)
and re-implements Environment's read-only surface AORRT actually calls
(width/height/disc_radius/path_free) directly against the precomputed SDF --
map.sdf / map.risk_sdf / environment.Environment's own SDF computation are
never invoked, since the field is already baked.

Start/goal are fixed at (0.75, 0.5) -> (3.0, 3.4) m (both verified clear of
every obstacle in the baked SDF).
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root (this file lives in hardwarexps/)
sys.path.append(ROOT_DIR)
from src.config import PlannerConfig
from src.ao_rrt import AORRT
from src.environment import _bilinear, _bilinear_batch


PCD_DIR = r"C:\Users\skelkar37\GT\Research F26\pcd\data\processed_pcd"
SDF_NAME = "riskmap5_sdf.npz"
POLY_NAME = "riskmap5_polygons.npz"
CONTROLS = ("v_cmd", "omega_cmd"); CCOL = ["tab:blue", "tab:orange"]

START = (2.75, 0.35)
GOAL = (0.75, 3.4)

# Set True (or pass "savecsv" on the command line) to write the solved
# trajectory to results/. Off by default so exploratory runs don't pile up files.
SAVE_TRAJECTORY_CSV = False
RESULTS_DIR = os.path.join(ROOT_DIR, "results")


class PCDEnvironment:
    """Minimal read-only adapter exposing exactly what AORRT (src/ao_rrt.py)
    calls on `env`: .width, .height, .disc_radius, .path_free(X, extra_margin).
    Backed directly by a precomputed SDF grid -- no map/environment SDF-build
    code runs here, so nothing there needs to be touched or duplicated."""

    def __init__(self, sdf, extent, resolution_m, origin, disc_radius, clearance=0.0):
        self.sdf = sdf
        x_min, x_max, y_min, y_max = extent
        self.origin = np.asarray(origin, float)
        self.res_m = resolution_m
        self.resolution = 1.0 / resolution_m       # cells per metre
        self.width = x_max - x_min
        self.height = y_max - y_min
        self.disc_radius = disc_radius
        self.clearance = clearance
        self.H, self.W = sdf.shape

    def _cell(self, x, y):
        return (x - self.origin[0]) * self.resolution, (y - self.origin[1]) * self.resolution
    
    def sdf_at(self, x, y):
        gx, gy = self._cell(x, y)
        return _bilinear(self.sdf, gx, gy)

    def collision_free(self, x, y):
        return self.sdf_at(x, y) > self.clearance + self.disc_radius

    def in_bounds(self, x, y):
        r = self.disc_radius
        lx, ly = x - self.origin[0], y - self.origin[1]
        return (r <= lx <= self.width - r) and (r <= ly <= self.height - r)

    def valid(self, x, y):
        return self.in_bounds(x, y) and self.collision_free(x, y)

    def path_free(self, X, extra_margin=0.0):
        """Same semantics as environment.Environment.path_free: one batched
        bilinear SDF lookup over the whole edge, in/out-of-bounds included."""
        px = X[:, 0]; py = X[:, 1]; r = self.disc_radius + extra_margin
        lx0, ly0 = px.min() - self.origin[0], py.min() - self.origin[1]
        lx1, ly1 = px.max() - self.origin[0], py.max() - self.origin[1]
        if lx0 < r or lx1 > self.width - r or ly0 < r or ly1 > self.height - r:
            return False
        gx = (px - self.origin[0]) * self.resolution
        gy = (py - self.origin[1]) * self.resolution
        vals = _bilinear_batch(self.sdf, gx, gy)
        return bool(np.all(vals > self.clearance + r))


def _mirror_x(polygons, x_min, x_max):
    """Match pcd/src/sdf.py's compute_map_sdf: it mirrors every polygon
    (x -> x_min + x_max - x) via plotter.mirror_x BEFORE rasterizing +
    building boundary_sdf.npz, so the baked SDF is in the MIRRORED frame while
    boundary_polygons.npz stores the raw (unmirrored) polygons. Plotting the
    raw polygons directly over the SDF-planned path makes a collision-free
    path look like it drives through rocks -- this reproduces the same flip
    purely for the overlay so it lines up with the frame the SDF (and hence
    the planner) actually used."""
    flipped = []
    for poly in polygons:
        p = np.asarray(poly, float).copy()
        p[..., 0] = x_min + x_max - p[..., 0]
        flipped.append(p)
    return flipped


def load_pcd_env(cfg, pcd_dir=PCD_DIR, sdf_name=SDF_NAME, poly_name=POLY_NAME):
    sdf_npz = np.load(os.path.join(pcd_dir, sdf_name))
    sdf = sdf_npz["sdf"]
    resolution_m = float(sdf_npz["resolution_m"])
    origin = sdf_npz["origin"]
    extent = tuple(sdf_npz["extent"])   # (x_min, x_max, y_min, y_max)

    poly_npz = np.load(os.path.join(pcd_dir, poly_name), allow_pickle=True)
    kinds = poly_npz["kinds"]
    # riskmap1_polygons.npz is already stored in the same (mirrored) frame as
    # riskmap1_sdf.npz -- unlike boundary_polygons.npz/boundary_sdf.npz, no
    # extra _mirror_x flip is needed here.
    polygons = poly_npz["polygons"]

    env = PCDEnvironment(sdf, extent, resolution_m, origin,
                          disc_radius=cfg.env.disc_radius, clearance=cfg.env.clearance)
    return env, kinds, polygons


def save_trajectory_csv(path, dt, out_path):
    """path: (N,5) AORRT state trajectory [x, y, theta, v_b, omega] at one
    dt-spaced RK4 substep per row (see ao_rrt.py:97's x0 layout and
    dynamics_vel.py's state convention -- global x, y, heading, body-frame
    forward speed, body-frame yaw rate). Adds a 6th column, timestamp = row
    index * dt, and writes x,y,theta,v_b,omega,t to out_path."""
    t = np.arange(len(path)) * dt
    data = np.column_stack([path, t])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savetxt(out_path, data, delimiter=",",
               header="x,y,theta,v_b,omega,t", comments="")
    print(f"saved trajectory csv -> {out_path}")


def main():
    args = [a for a in sys.argv[1:]]
    cost_mode = next((a for a in args if a in ("time", "control", "both")), "control")
    nums = [int(a) for a in args if a.isdigit()]
    seed = nums[0] if len(nums) > 0 else 60
    iters = nums[1] if len(nums) > 1 else 15000
    save_csv = SAVE_TRAJECTORY_CSV or ("savecsv" in args)

    np.random.seed(seed)
    cfg = PlannerConfig()
    cfg.aorrt.cost_mode = cost_mode
    cfg.aorrt.max_iterations = iters

    env, kinds, polygons = load_pcd_env(cfg)
    print(f"pcd map {env.width:.2f}x{env.height:.2f} m (from {PCD_DIR}), seed {seed} | "
          f"AO-RRT cost_mode='{cost_mode}', {iters} iters")
    assert env.valid(*START), f"start {START} not collision-free (sdf={env.sdf_at(*START):.3f})"
    assert env.valid(*GOAL), f"goal {GOAL} not collision-free (sdf={env.sdf_at(*GOAL):.3f})"

    pl = AORRT(env, cfg, START, GOAL)
    path = pl.plan(verbose=True)
    print(f"AO-RRT: goal={pl.goal_reached()} cost={pl.best_cost:.3f} "
          f"nodes={len(pl.nodes)} time={pl.timers['total']:.2f}s "
          f"t_first={pl.t_first if pl.t_first else float('nan'):.2f}s")
    U = pl.extract_controls(); dt = cfg.dyn.dt

    if save_csv:
        if pl.goal_reached():
            out_path = os.path.join(RESULTS_DIR, f"test_exp_trajectory_seed{seed}.csv")
            save_trajectory_csv(path, dt, out_path)
        else:
            print("savecsv requested but goal not reached -- skipping csv write")

    # ---- plot -------------------------------------------------------------
    ext = [0, env.width, 0, env.height]
    fig, (axm, axu) = plt.subplots(1, 2, figsize=(15, 7))
    for kind, poly in zip(kinds, polygons):
        poly = np.asarray(poly)
        closed = np.vstack([poly, poly[0]])
        if kind == "map":
            axm.plot(closed[:, 0], closed[:, 1], "-", color="k", lw=1.2, zorder=2)
        else:
            axm.fill(closed[:, 0], closed[:, 1], color="0.6", alpha=0.85, zorder=1)
    for e in pl.tree_edges():
        axm.plot(e[:, 0], e[:, 1], "-", color="0.8", lw=0.4, alpha=0.5, zorder=1)
    axm.plot(path[:, 0], path[:, 1], "-", color="tab:cyan", lw=2.0, label="AO-RRT", zorder=3)
    s = max(1, len(path) // 25)
    axm.quiver(path[::s, 0], path[::s, 1], np.cos(path[::s, 2]), np.sin(path[::s, 2]),
               color="tab:blue", scale=25, width=0.004, zorder=4)
    axm.plot(*START, "o", color="lime", ms=12, mec="k"); axm.plot(*GOAL, "*", color="magenta", ms=20, mec="k")
    axm.set_xlim(0, env.width); axm.set_ylim(0, env.height); axm.set_aspect("equal")
    axm.set_xlabel("x (m)"); axm.set_ylabel("y (m)"); axm.legend(loc="upper left", fontsize=9)
    axm.set_title(f"AO-RRT on pcd map ('{cost_mode}' cost = {pl.best_cost:.1f})")

    tu = np.arange(len(U)) * dt
    v_max, w_max = cfg.aorrt.v_max, cfg.aorrt.w_max
    axu2 = axu.twinx()
    axu.step(tu, U[:, 0], where="post", lw=1.0, color=CCOL[0], alpha=0.75, label="AO $v_{cmd}$")
    axu2.step(tu, U[:, 1], where="post", lw=1.0, color=CCOL[1], alpha=0.75, label="AO $\\omega_{cmd}$")

    axu.axhline(v_max, color=CCOL[0], ls=":", lw=0.8); axu.axhline(-v_max, color=CCOL[0], ls=":", lw=0.8)
    axu2.axhline(w_max, color=CCOL[1], ls=":", lw=0.8); axu2.axhline(-w_max, color=CCOL[1], ls=":", lw=0.8)

    axu.set_xlabel("time (s)")
    axu.set_ylabel("v_cmd (m/s)", color=CCOL[0]); axu.tick_params(axis="y", labelcolor=CCOL[0])
    axu2.set_ylabel("omega_cmd (rad/s)", color=CCOL[1]); axu2.tick_params(axis="y", labelcolor=CCOL[1])
    axu.set_title("Control inputs (v_cmd, omega_cmd)")
    lines1, labels1 = axu.get_legend_handles_labels(); lines2, labels2 = axu2.get_legend_handles_labels()
    axu.legend(lines1 + lines2, labels1 + labels2, fontsize=7, ncol=2)
    axu.grid(alpha=0.3)

    fig.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
