"""
Warp riskmap5's 2D Gaussian-boundary risk field onto a phone photo of the
physical arena, as a light translucent wash, and display it in a matplotlib
window. Nothing is written to disk unless you click a button: "Save Adj"
writes the small numeric per-rock adjustment file, "Save Image" writes the
displayed figure to IMAGE_OUT_PATH.

Each rock gets its OWN risk patch, warped independently through the SAME
fitted homography -- so residual misalignment (from imperfect click
correspondences) can be nudged PER ROCK rather than as one rigid group.
Per-obstacle halo formula duplicated from riskmap1_boundary_risk.
fused_obstacle_risk's loop body (kept separate here instead of OR-fused, so
each rock's patch is independent); rasterize_obstacle_id/assign_sigma still
imported read-only from that module, unmodified.

The world<->pixel point correspondences (rock centroids matched to photo
pixels) come from pick_correspondences.py's correspondences.npz -- run that
first. A planar homography (homography.py) is fit from those points and used
to warp each rock's risk raster into the photo's pixel space.

GROUND-PLANE-ONLY approximation: correct on the flat sand, does not wrap over
the rocks' 3D surfaces (see homography.py's docstring).

    python hardwarexps/photo_overlay/overlay_risk_on_photo.py [photo_path]
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from scipy.ndimage import distance_transform_edt
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
HARDWAREXPS_DIR = os.path.dirname(HERE)
ROOT_DIR = os.path.dirname(HARDWAREXPS_DIR)
sys.path.append(HARDWAREXPS_DIR)
sys.path.append(HERE)

from riskmap1_boundary_risk import (          # noqa: E402  (read-only reuse, not modified)
    rasterize_obstacle_id, assign_sigma, MANUAL_SIGMA, RANDOM_SIGMA_RANGE, DEFAULT_SEED,
)
from homography import (                       # noqa: E402
    fit_homography, raster_to_photo_homography, warp_rgba_raster_to_photo, adjust_overlay, warp_points,
)

DATA_DIR = os.path.join(ROOT_DIR, "data")
POLY_NAME = "riskmap6_polygons_cleaned.npz"
SDF_NAME = "riskmap6_polygons_cleaned_sdf.npz"
CORR_PATH = os.path.join(HERE, "correspondences.npz")
ADJ_PATH = os.path.join(HERE, "overlay_adjustment.npz")   # per-rock dx/dy/scale/angle (numbers only, no image)
IMAGE_OUT_PATH = os.path.join(HERE, "risk_overlay_result.png")   # written only when "Save Image" is clicked

PLOT_DATA_PATH = os.path.join(DATA_DIR, "plot_data.csv")
# style per pose trace, keyed by whichever of these substrings is in the row's "topic"
POSE_TRACE_STYLE = {
    "current": dict(color="tab:red", ls="-", label="current (tracked) pose"),
    "desired": dict(color="tab:cyan", ls="--", label="desired (reference) pose"),
}

MAX_ALPHA = 130   # 0-255; caps overlay opacity for a LIGHT wash rather than a solid heatmap


def load_map():
    poly_npz = np.load(os.path.join(DATA_DIR, POLY_NAME), allow_pickle=True)
    kinds, polygons = poly_npz["kinds"], poly_npz["polygons"]
    sdf_npz = np.load(os.path.join(DATA_DIR, SDF_NAME))
    resolution_m = float(sdf_npz["resolution_m"])
    origin = np.asarray(sdf_npz["origin"], float)
    return kinds, polygons, sdf_npz["sdf"].shape, resolution_m, origin


def build_per_obstacle_rasters(seed=DEFAULT_SEED):
    """{rock idx -> (H,W) risk raster in [0,1]}, one per rock, kept SEPARATE
    (not OR-fused like fused_obstacle_risk) so each can be warped/nudged on
    its own. Same Gaussian-boundary-halo formula as fused_obstacle_risk's
    per-obstacle loop body."""
    kinds, polygons, sdf_shape, resolution_m, origin = load_map()
    obstacle_id = rasterize_obstacle_id(sdf_shape, resolution_m, origin, kinds, polygons)
    s_k = assign_sigma(obstacle_id, MANUAL_SIGMA, RANDOM_SIGMA_RANGE, seed)
    rasters = {}
    for idx, sigma in enumerate(s_k, start=1):
        mask = obstacle_id == idx
        if not mask.any():
            continue
        dist_m = distance_transform_edt(~mask) * resolution_m
        rasters[idx] = np.where(mask, 1.0, np.exp(-0.5 * (dist_m / sigma) ** 2))
    return rasters, origin, resolution_m


def risk_to_rgba(risk, max_alpha=MAX_ALPHA):
    """risk in [0,1] -> (H,W,4) uint8, YlOrRd colour (matching plot_result's
    heatmap), alpha scaled by risk value and capped at max_alpha for a light
    wash rather than an opaque overlay."""
    rgba = mpl.colormaps["YlOrRd"](risk)          # (H,W,4) float in [0,1]
    rgba = (rgba * 255).astype(np.uint8)
    rgba[..., 3] = (risk * max_alpha).astype(np.uint8)
    return rgba


def load_pose_traces(path):
    """plot_data.csv is a PlotJuggler-style XY-plot export: each row's 'x value'
    is that pose's position.x at that instant, paired for an XY plot; 'value'
    is its position.y (the row's 'topic' -- e.g. '.../current_pose.pose.
    position.y' -- only names which series' Y-axis it is). Grouping by topic
    and pairing (x value, value) recovers each pose's own (x, y) polyline in
    world metres, sorted by timestamp. Returns {topic: (N,2) array}."""
    df = pd.read_csv(path)
    traces = {}
    for topic in df["topic"].unique():
        sub = df[df["topic"] == topic].sort_values("timestamp")
        traces[topic] = np.column_stack([sub["x value"].to_numpy(), sub["value"].to_numpy()])
    return traces


_DEFAULT_ADJ = dict(dx=0.0, dy=0.0, scale=1.0, angle_deg=0.0)


def _load_adjustments(rock_indices):
    adjustments = {idx: dict(_DEFAULT_ADJ) for idx in rock_indices}
    if os.path.exists(ADJ_PATH):
        saved = np.load(ADJ_PATH)
        for idx in rock_indices:
            key = f"rock_{idx}_dx"
            if key in saved:
                adjustments[idx] = {p: float(saved[f"rock_{idx}_{p}"]) for p in _DEFAULT_ADJ}
        print(f"loaded previous per-rock adjustments -> {ADJ_PATH}")
    return adjustments


def _load_traj_adjustment():
    adj = dict(_DEFAULT_ADJ)
    if os.path.exists(ADJ_PATH):
        saved = np.load(ADJ_PATH)
        if "traj_dx" in saved:
            adj = {p: float(saved[f"traj_{p}"]) for p in _DEFAULT_ADJ}
    return adj


def transform_points(xy, dx, dy, scale, angle_deg, center):
    """Same similarity transform as homography.adjust_overlay, but applied
    directly to (N,2) pixel points instead of warping a raster image -- for
    nudging the plotted trajectory lines as one rigid group."""
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]]) * scale
    return (xy - center) @ R.T + center + np.array([dx, dy])


def main():
    if not os.path.exists(CORR_PATH):
        raise FileNotFoundError(f"{CORR_PATH} does not exist -- run pick_correspondences.py first "
                                 f"to build the world<->pixel point correspondences.")
    
    corr = np.load(CORR_PATH, allow_pickle=True)
    world_pts, pixel_pts = corr["world_pts"], corr["pixel_pts"]
    photo_path = sys.argv[1] if len(sys.argv) > 1 else str(corr["photo_path"])
    if not os.path.exists(photo_path):
        raise FileNotFoundError(f"{photo_path} does not exist.")
    pose_traces = load_pose_traces(PLOT_DATA_PATH) if os.path.exists(PLOT_DATA_PATH) else {}
    if not pose_traces:
        print(f"note: {PLOT_DATA_PATH} not found -- skipping pose-trace overlay")

    rasters, origin, resolution_m = build_per_obstacle_rasters()
    rock_indices = sorted(rasters.keys())

    H_world_to_photo = fit_homography(world_pts, pixel_pts)
    H_raster_to_photo = raster_to_photo_homography(H_world_to_photo, origin, resolution_m)

    photo = Image.open(photo_path).convert("RGBA")
    base_patches = {idx: warp_rgba_raster_to_photo(risk_to_rgba(r), H_raster_to_photo, photo.size)
                     for idx, r in rasters.items()}

    adjustments = _load_adjustments(rock_indices)
    adjusted_patches = {idx: adjust_overlay(base_patches[idx], **adjustments[idx]) for idx in rock_indices}

    state = {"selected": rock_indices[0]}
    background = {"img": None}   # photo + every patch EXCEPT the currently selected rock

    def compute_background(selected):
        bg = photo
        for idx in rock_indices:
            if idx != selected:
                bg = Image.alpha_composite(bg, adjusted_patches[idx])
        return bg

    background["img"] = compute_background(state["selected"])

    def full_composite():
        return Image.alpha_composite(background["img"], adjusted_patches[state["selected"]])

    fig, ax = plt.subplots(figsize=(12, 9))
    plt.subplots_adjust(left=0.30, bottom=0.46)
    im = ax.imshow(full_composite())
    ax.axis("off")
    title = ax.set_title(f"adjusting rock {state['selected']}  (homography fit from "
                          f"{len(world_pts)} point correspondences)")

    # raw (un-nudged) pixel points per pose trace, plus live Line2D/marker handles
    # so the trajectory sliders below can move them without re-plotting from scratch
    traj_raw = {}
    traj_lines = {}
    traj_markers = {}   # "current" topic's start/end marker handles
    traj_adj = dict(_DEFAULT_ADJ)
    traj_center = np.zeros(2)

    if pose_traces:
        # pose traces live in world (x,y) directly -- warp through the SAME base
        # homography as the risk patches (not any per-rock manual nudge, which is
        # a rock-specific correction that doesn't apply to arbitrary path points).
        # A SEPARATE similarity nudge (traj_adj, below) can move the whole
        # trajectory as one rigid group if the homography itself is imperfect.
        traj_center = np.concatenate(
            [warp_points(H_world_to_photo, xy) for xy in pose_traces.values()], axis=0
        ).mean(axis=0)
        traj_adj = _load_traj_adjustment()

        for topic, xy in pose_traces.items():
            key = next((k for k in POSE_TRACE_STYLE if k in topic), None)
            style = POSE_TRACE_STYLE.get(key, dict(color="tab:red", ls="-", label=topic))
            pxy = warp_points(H_world_to_photo, xy)
            traj_raw[topic] = pxy
            line, = ax.plot(pxy[:, 0], pxy[:, 1], style["ls"], color=style["color"], lw=2.0,
                             zorder=5, label=style["label"])
            traj_lines[topic] = line
            if key == "current":
                start_m, = ax.plot(pxy[0, 0], pxy[0, 1], "o", color="lime", ms=10, mec="k",
                                    zorder=6, label="start")
                end_m, = ax.plot(pxy[-1, 0], pxy[-1, 1], "*", color="magenta", ms=16, mec="k",
                                  zorder=6, label="end")
                traj_markers[topic] = (start_m, end_m)
        ax.legend(loc="lower right", fontsize=8)

    ax_dx = fig.add_axes([0.34, 0.17, 0.6, 0.03])
    ax_dy = fig.add_axes([0.34, 0.12, 0.6, 0.03])
    ax_scale = fig.add_axes([0.34, 0.07, 0.6, 0.03])
    ax_angle = fig.add_axes([0.34, 0.02, 0.6, 0.03])
    a0 = adjustments[state["selected"]]
    s_dx = Slider(ax_dx, "rock dx (px)", -400.0, 400.0, valinit=a0["dx"])
    s_dy = Slider(ax_dy, "rock dy (px)", -400.0, 400.0, valinit=a0["dy"])
    s_scale = Slider(ax_scale, "rock scale", 0.5, 2.0, valinit=a0["scale"])
    s_angle = Slider(ax_angle, "rock angle (deg)", -45.0, 45.0, valinit=a0["angle_deg"])

    if pose_traces:
        fig.text(0.34, 0.435, "trajectory adjustment", fontsize=9, fontweight="bold")
        ax_tdx = fig.add_axes([0.34, 0.39, 0.6, 0.03])
        ax_tdy = fig.add_axes([0.34, 0.34, 0.6, 0.03])
        ax_tscale = fig.add_axes([0.34, 0.29, 0.6, 0.03])
        ax_tangle = fig.add_axes([0.34, 0.24, 0.6, 0.03])
        s_tdx = Slider(ax_tdx, "traj dx (px)", -400.0, 400.0, valinit=traj_adj["dx"])
        s_tdy = Slider(ax_tdy, "traj dy (px)", -400.0, 400.0, valinit=traj_adj["dy"])
        s_tscale = Slider(ax_tscale, "traj scale", 0.5, 2.0, valinit=traj_adj["scale"])
        s_tangle = Slider(ax_tangle, "traj angle (deg)", -45.0, 45.0, valinit=traj_adj["angle_deg"])

        def on_traj_slider_change(_val=None):
            traj_adj.update(dx=s_tdx.val, dy=s_tdy.val, scale=s_tscale.val, angle_deg=s_tangle.val)
            for topic, pxy in traj_raw.items():
                new_xy = transform_points(pxy, center=traj_center, **traj_adj)
                traj_lines[topic].set_data(new_xy[:, 0], new_xy[:, 1])
                if topic in traj_markers:
                    start_m, end_m = traj_markers[topic]
                    start_m.set_data([new_xy[0, 0]], [new_xy[0, 1]])
                    end_m.set_data([new_xy[-1, 0]], [new_xy[-1, 1]])
            fig.canvas.draw_idle()

        for s in (s_tdx, s_tdy, s_tscale, s_tangle):
            s.on_changed(on_traj_slider_change)

    fig.text(0.34, 0.195, "rock adjustment", fontsize=9, fontweight="bold")

    ax_radio = fig.add_axes([0.02, 0.46, 0.20, 0.47])
    ax_radio.set_title("select rock", fontsize=9)
    radio = RadioButtons(ax_radio, [str(i) for i in rock_indices])

    guard = {"on": False}   # suppresses slider callbacks while we set_val() them programmatically

    def on_slider_change(_val=None):
        if guard["on"]:
            return
        idx = state["selected"]
        adjustments[idx] = dict(dx=s_dx.val, dy=s_dy.val, scale=s_scale.val, angle_deg=s_angle.val)
        adjusted_patches[idx] = adjust_overlay(base_patches[idx], **adjustments[idx])
        im.set_data(full_composite())
        fig.canvas.draw_idle()

    for s in (s_dx, s_dy, s_scale, s_angle):
        s.on_changed(on_slider_change)

    def on_select(label):
        idx = int(label)
        state["selected"] = idx
        background["img"] = compute_background(idx)
        a = adjustments[idx]
        guard["on"] = True
        s_dx.set_val(a["dx"]); s_dy.set_val(a["dy"])
        s_scale.set_val(a["scale"]); s_angle.set_val(a["angle_deg"])
        guard["on"] = False
        title.set_text(f"adjusting rock {idx}  (homography fit from {len(world_pts)} point correspondences)")
        im.set_data(full_composite())
        fig.canvas.draw_idle()

    radio.on_clicked(on_select)

    ax_save = fig.add_axes([0.87, 0.94, 0.1, 0.05])
    btn_save = Button(ax_save, "Save Adj")

    def save(_event):
        data = {}
        for idx in rock_indices:
            for p, v in adjustments[idx].items():
                data[f"rock_{idx}_{p}"] = v
        for p, v in traj_adj.items():
            data[f"traj_{p}"] = v
        np.savez(ADJ_PATH, **data)
        print(f"saved per-rock adjustments for {len(rock_indices)} rocks, plus trajectory adjustment -> {ADJ_PATH}")

    btn_save.on_clicked(save)

    ax_save_img = fig.add_axes([0.74, 0.94, 0.12, 0.05])
    btn_save_img = Button(ax_save_img, "Save Image")

    def save_image(_event):
        # crop to just the image axes (plot + title + legend) -- excludes the
        # slider/radio/button widgets, which live in their own separate axes
        renderer = fig.canvas.get_renderer()
        bbox = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(IMAGE_OUT_PATH, dpi=200, bbox_inches=bbox)
        print(f"saved image -> {IMAGE_OUT_PATH}")

    btn_save_img.on_clicked(save_image)

    plt.show()


if __name__ == "__main__":
    main()
