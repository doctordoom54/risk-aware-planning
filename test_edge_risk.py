"""
Sanity test for scp_vel.EdgeRiskEvaluator: build a small map (2 rocks, 1 gentle
hill, low obstacle-boundary uncertainty -- just enough terrain structure to
exercise the risk map, not a stress test), plan a real AO-RRT path through it,
then evaluate the batched R_{k,m} LogSumExp edge-risk grid over the path's real
tree edges and print a summary.

    python test_edge_risk.py [seed] [iters]
"""
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.config import PlannerConfig
from src.environment import make_map_env
from src.ao_rrt import AORRT
from src.scp_vel import EdgeRiskEvaluator, edges_from_chain


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
    pl.plan(verbose=True)
    if not pl.goal_reached():
        print("AO-RRT did not reach the goal -- try more iterations or a different seed.")
        return
    chain = pl._chain()
    print(f"\nAO-RRT path: {len(chain) - 1} edges, cost={pl.best_cost:.3f}")

    evaluator = EdgeRiskEvaluator(env, cfg, pl.model)
    S_bar, U_bar, nsteps_k = edges_from_chain(chain)     # iteration-1 anchor = AO-RRT plan
    R = evaluator.edge_risk(S_bar[:-1], U_bar, nsteps_k)  # (K, M) -- edge-start states only
    print(f"R shape: {R.shape}  (K edges x M={cfg.risk.dist_grid_n**2} disturbance scenarios)\n")
    for k in range(R.shape[0]):
        print(f"  edge {k:2d}: min={R[k].min():.4f}  mean={R[k].mean():.4f}  max={R[k].max():.4f}")
    worst_k = int(R.max(axis=1).argmax())
    print(f"\nOverall R: min={R.min():.4f}  mean={R.mean():.4f}  max={R.max():.4f}  "
          f"(worst edge = {worst_k})")


if __name__ == "__main__":
    main()
