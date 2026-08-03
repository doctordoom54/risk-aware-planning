"""
Interactive tool: click matching points between a top-down reference plot of
riskmap5's rock centroids (each labelled with its polygon index) and a phone
photo of the physical arena, to build the world<->pixel point correspondences
a planar homography needs (see homography.py). Saves (world_pts, pixel_pts,
rock_idx, photo_path) to correspondences.npz for overlay_risk_on_photo.py.

Purely mouse-driven (no console input()) -- some run configurations (e.g. an
IDE "Run" button without an attached terminal) don't forward stdin, which
would make input() raise EOFError.

Reads riskmap5_polygons.npz/riskmap5_sdf.npz directly from THIS repo's own
data/ folder (not the separate pcd/ project's data dir that other hardwarexps
scripts point at) -- nothing outside hardwarexps/ or data/ is touched.

    python hardwarexps/photo_overlay/pick_correspondences.py [photo_path]
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
HARDWAREXPS_DIR = os.path.dirname(HERE)
ROOT_DIR = os.path.dirname(HARDWAREXPS_DIR)

DATA_DIR = os.path.join(ROOT_DIR, "data")
SDF_NAME = "riskmap6_polygons_cleaned_sdf.npz"
POLY_NAME = "riskmap6_polygons_cleaned.npz"
DEFAULT_PHOTO = os.path.join(DATA_DIR, "hardware pics", "test2.png")
OUT_PATH = os.path.join(HERE, "correspondences.npz")


def load_polygons():
    poly_npz = np.load(os.path.join(DATA_DIR, POLY_NAME), allow_pickle=True)
    return poly_npz["kinds"], poly_npz["polygons"]


def rock_landmarks(kinds, polygons):
    """(idx, cx, cy) for every 'rock' polygon, idx = its position in the full
    kinds/polygons list (matches riskmap1_boundary_risk.rasterize_obstacle_id's
    own obstacle-id numbering)."""
    out = []
    for idx, (kind, poly) in enumerate(zip(kinds, polygons)):
        if kind != "rock":
            continue
        poly = np.asarray(poly, float)
        out.append((idx, poly[:, 0].mean(), poly[:, 1].mean()))
    return out


def _nearest_landmark(click_xy, landmarks):
    cx, cy = click_xy
    dists = [(idx, x, y, (x - cx) ** 2 + (y - cy) ** 2) for idx, x, y in landmarks]
    return min(dists, key=lambda t: t[3])[:3]   # (idx, x, y) of the closest rock


def main():
    photo_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PHOTO
    if not os.path.exists(photo_path):
        raise FileNotFoundError(f"{photo_path} does not exist -- pass the phone photo's path as an "
                                 f"argument, or place it at the default path above.")

    kinds, polygons = load_polygons()
    landmarks = rock_landmarks(kinds, polygons)

    # Step 1: click rocks on the top-down reference plot, in whatever order you like.
    # Left-click = add point, right-click = undo last, Enter/middle-click = done.
    fig_ref, ax_ref = plt.subplots(figsize=(6, 6))
    for kind, poly in zip(kinds, polygons):
        poly = np.asarray(poly, float)
        closed = np.vstack([poly, poly[0]])
        if kind == "map":
            ax_ref.plot(closed[:, 0], closed[:, 1], "-", color="k", lw=1.2)
        else:
            ax_ref.fill(closed[:, 0], closed[:, 1], color="0.6", alpha=0.7)
    for idx, cx, cy in landmarks:
        ax_ref.text(cx, cy, str(idx), color="red", fontsize=11, fontweight="bold", ha="center", va="center")
    ax_ref.set_aspect("equal")
    ax_ref.set_xlabel("x (m)"); ax_ref.set_ylabel("y (m)")
    ax_ref.set_title("STEP 1: click >=4 rocks you can identify in the photo\n"
                      "(order doesn't matter here) -- Enter/middle-click when done")
    print("\nclick >=4 identifiable rocks on the top-down reference window "
          "(right-click undoes, Enter/middle-click finishes)...")
    ref_clicks = plt.ginput(n=-1, timeout=0)
    plt.close(fig_ref)

    if len(ref_clicks) < 4:
        raise ValueError(f"need >=4 points for a homography fit, got {len(ref_clicks)}")

    chosen = []
    world_pts = []
    for click in ref_clicks:
        idx, wx, wy = _nearest_landmark(click, landmarks)
        chosen.append(idx)
        world_pts.append((wx, wy))
    world_pts = np.array(world_pts)
    print("snapped to rocks (in click order):", chosen)

    # Step 2: click the SAME rocks, in the SAME order, on the photo.
    img = plt.imread(photo_path)
    fig_photo, ax_photo = plt.subplots(figsize=(10, 6))
    ax_photo.imshow(img)
    ax_photo.set_title(f"STEP 2: click rocks {chosen} IN THAT ORDER (one click each)")
    print(f"\nclick rocks {chosen} in that exact order on the photo window...")
    pixel_pts = plt.ginput(n=len(chosen), timeout=0)
    plt.close(fig_photo)

    pixel_pts = np.array(pixel_pts)

    np.savez(OUT_PATH, world_pts=world_pts, pixel_pts=pixel_pts,
             rock_idx=np.array(chosen), photo_path=photo_path)
    print(f"\nsaved {len(chosen)} correspondences -> {OUT_PATH}")
    for idx, (wx, wy), (px, py) in zip(chosen, world_pts, pixel_pts):
        print(f"  rock {idx}: world=({wx:.3f},{wy:.3f})  pixel=({px:.1f},{py:.1f})")


if __name__ == "__main__":
    main()
