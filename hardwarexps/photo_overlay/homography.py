"""
Planar-homography helpers for warping the 2D risk raster (world x,y metres)
onto a phone photo of the physical arena taken from an oblique angle. Pure
numpy + PIL -- no OpenCV dependency (not installed in the motionplanning env).

This is a GROUND-PLANE-ONLY approximation: a homography correctly maps any
point that actually lies on the flat sand surface, but the rocks themselves
have height, so the risk wash will line up on the sand around a rock without
wrapping up over the rock's visible 3D surface. Good enough for a light
overlay; not a full 3D reprojection.
"""
import numpy as np
from PIL import Image


def fit_homography(world_pts, pixel_pts):
    """world_pts, pixel_pts: (N,2) arrays, N>=4, matched row-for-row.
    Returns H (3,3) mapping homogeneous world (x,y,1) -> pixel (u,v,w)
    (pixel = (u/w, v/w)), fit by the normalized Direct Linear Transform
    (DLT): least-squares (SVD) for N>4, exact for N==4."""
    world_pts = np.asarray(world_pts, float)
    pixel_pts = np.asarray(pixel_pts, float)
    n = world_pts.shape[0]
    if n < 4:
        raise ValueError(f"need >=4 point correspondences for a homography fit, got {n}")

    def _normalize(pts):
        # isotropic normalization (Hartley): centroid at origin, mean distance
        # sqrt(2) -- keeps the DLT's SVD numerically well-conditioned.
        centroid = pts.mean(axis=0)
        d = np.sqrt(((pts - centroid) ** 2).sum(axis=1)).mean()
        scale = np.sqrt(2) / d if d > 0 else 1.0
        T = np.array([[scale, 0, -scale * centroid[0]],
                      [0, scale, -scale * centroid[1]],
                      [0, 0, 1]])
        pts_h = np.column_stack([pts, np.ones(len(pts))])
        pts_n = (T @ pts_h.T).T
        return pts_n[:, :2], T

    src_n, T_src = _normalize(world_pts)
    dst_n, T_dst = _normalize(pixel_pts)

    A = np.zeros((2 * n, 9))
    for i in range(n):
        x, y = src_n[i]
        u, v = dst_n[i]
        A[2 * i] = [-x, -y, -1, 0, 0, 0, u * x, u * y, u]
        A[2 * i + 1] = [0, 0, 0, -x, -y, -1, v * x, v * y, v]

    _, _, Vt = np.linalg.svd(A)
    H_n = Vt[-1].reshape(3, 3)
    H = np.linalg.inv(T_dst) @ H_n @ T_src
    return H / H[2, 2]


def warp_points(H_world_to_photo, world_xy):
    """world_xy: (N,2) array in world metres -> (N,2) photo pixel coords,
    through the same fitted homography used for the risk raster. For
    overlaying things that live in world (x,y) directly (e.g. a planned
    trajectory) rather than the raster grid."""
    world_xy = np.asarray(world_xy, float)
    pts_h = np.column_stack([world_xy, np.ones(len(world_xy))])
    proj = (H_world_to_photo @ pts_h.T).T
    return proj[:, :2] / proj[:, 2:3]


def _raster_to_world_affine(origin, res_m):
    """3x3 homogeneous matrix mapping raster (col, row, 1) -> world (x, y, 1),
    matching riskmap1_boundary_risk.rasterize_obstacle_id's own convention
    (x = origin_x + col*res_m, y = origin_y + row*res_m, no +0.5 offset --
    same as src.environment's _bilin_cell)."""
    ox, oy = origin
    return np.array([[res_m, 0, ox],
                      [0, res_m, oy],
                      [0, 0, 1]])


def raster_to_photo_homography(H_world_to_photo, origin, res_m):
    """Compose world->photo with raster-index->world so the result maps
    raster (col, row, 1) directly to photo pixel (u, v, w)."""
    A = _raster_to_world_affine(origin, res_m)
    H = H_world_to_photo @ A
    return H / H[2, 2]


def warp_rgba_raster_to_photo(raster_rgba, H_raster_to_photo, photo_size):
    """raster_rgba: (H,W,4) uint8 array in raster index space.
    H_raster_to_photo: 3x3, raster (col,row,1) -> photo pixel (u,v,w).
    photo_size: (width, height) of the destination canvas.
    Returns a PIL RGBA Image the same size as the photo, with the raster
    perspective-warped into place -- everything outside the raster's own
    footprint comes back fully transparent (PIL's fillcolor), so it composites
    cleanly over the photo with Image.alpha_composite."""
    H_photo_to_raster = np.linalg.inv(H_raster_to_photo)
    M = H_photo_to_raster / H_photo_to_raster[2, 2]
    # PIL's PERSPECTIVE data = (a,b,c,d,e,f,g,h) s.t. for each DESTINATION
    # pixel (x,y): src_x = (a x + b y + c)/(g x + h y + 1), src_y = (d x + e y + f)/(...)
    # -- i.e. exactly M applied to destination homogeneous coords, M[2,2]==1.
    coeffs = (M[0, 0], M[0, 1], M[0, 2],
              M[1, 0], M[1, 1], M[1, 2],
              M[2, 0], M[2, 1])
    src_img = Image.fromarray(raster_rgba, mode="RGBA")
    return src_img.transform(photo_size, Image.Transform.PERSPECTIVE, coeffs,
                              resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0, 0))


def adjust_overlay(warped_rgba, dx, dy, scale, angle_deg):
    """Manual fine-tune on top of an already homography-warped RGBA overlay:
    a similarity transform (uniform scale + rotation about the image centre,
    then translate by (dx, dy) pixels) -- for nudging out residual
    misalignment from imperfect click correspondences, without re-fitting the
    homography. Returns a new PIL RGBA image, same size as the input."""
    w, h = warped_rgba.size
    cx, cy = w / 2.0, h / 2.0
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[scale * c, -scale * s],
                  [scale * s, scale * c]])
    # forward: dest = R @ (src - center) + center + (dx, dy)
    M = np.eye(3)
    M[:2, :2] = R
    M[:2, 2] = np.array([cx, cy]) - R @ np.array([cx, cy]) + np.array([dx, dy])

    M_inv = np.linalg.inv(M)   # PIL's AFFINE data is dest -> src
    coeffs = (M_inv[0, 0], M_inv[0, 1], M_inv[0, 2],
              M_inv[1, 0], M_inv[1, 1], M_inv[1, 2])
    return warped_rgba.transform((w, h), Image.Transform.AFFINE, coeffs,
                                  resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0, 0))
