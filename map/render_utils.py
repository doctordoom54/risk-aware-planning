import numpy as np
import os


def save_maps(surface, features, sdf, out_dir):
    """Save heightmap, featuremap, and SDF as PNG images."""
    os.makedirs(out_dir, exist_ok=True)
    try:
        from PIL import Image
    except ImportError:
        print("[render_utils] PIL not available, skipping image saves.")
        return

    heightmap = ((surface - surface.min()) /
                 (surface.max() - surface.min() + 1e-8) * 255).astype(np.uint8)
    Image.fromarray(heightmap).save(f'{out_dir}/heightmap.png')

    feature_img = np.zeros_like(features, dtype=np.uint8)
    feature_img[features == 1] = 127
    feature_img[features == 2] = 255
    feature_img[features == 3] = 64
    Image.fromarray(feature_img).save(f'{out_dir}/featuremap.png')

    sdf_normalized = ((sdf - sdf.min()) /
                      (sdf.max() - sdf.min() + 1e-8) * 255).astype(np.uint8)
    Image.fromarray(sdf_normalized).save(f'{out_dir}/sdf.png')


def render_moon_like_visual(surface, light_dir=(1, 1, 2),
                            out_file='moon_visual.png'):
    """Render a shaded lunar surface image."""
    try:
        from PIL import Image
    except ImportError:
        print("[render_utils] PIL not available, skipping moon visual.")
        return

    Z = surface
    dx, dy = np.gradient(Z)
    norm = np.sqrt(dx**2 + dy**2 + 1)
    nx = -dx / norm
    ny = -dy / norm
    nz = 1 / norm

    L = np.array(light_dir, dtype=float)
    L = L / np.linalg.norm(L)

    shade = nx * L[0] + ny * L[1] + nz * L[2]
    shade = np.clip(shade, 0, 1)

    img = (shade * 255).astype(np.uint8)
    os.makedirs(os.path.dirname(out_file) or '.', exist_ok=True)
    Image.fromarray(img).save(out_file)
