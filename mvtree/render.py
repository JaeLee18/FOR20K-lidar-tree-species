"""Orthographic multi-view rendering of a single-tree point cloud.

Each tree is rendered to seven views (four side views at 0/90/180/270 degrees
azimuth, one top, one bottom, and one trunk-detail crop). The three image
channels are geometric maps, not colors:

    R = depth along the viewing axis of the front-most point (min-max per view)
    G = log(1 + point count) per pixel                         (min-max per view)
    B = absolute height of the front-most point                (min-max per view)

This module is a vectorized re-implementation of the per-point loop used to
render the training data. ``tests/test_render.py`` checks that both produce
byte-identical images, so renders made here match what the released weights
were trained on.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

VIEW_NAMES = ["side_000", "side_090", "side_180", "side_270", "top", "bottom", "detail"]
SIDE_VIEWS = VIEW_NAMES[:4]
TOPBOT_VIEWS = VIEW_NAMES[4:6]
DETAIL_VIEW = VIEW_NAMES[6]


def get_view_configs() -> List[Dict]:
    """The seven canonical views (azimuth/elevation in degrees)."""
    return [
        {"name": "side_000", "azimuth_deg": 0, "elevation_deg": 0, "is_detail": False, "crop_z_range": None},
        {"name": "side_090", "azimuth_deg": 90, "elevation_deg": 0, "is_detail": False, "crop_z_range": None},
        {"name": "side_180", "azimuth_deg": 180, "elevation_deg": 0, "is_detail": False, "crop_z_range": None},
        {"name": "side_270", "azimuth_deg": 270, "elevation_deg": 0, "is_detail": False, "crop_z_range": None},
        {"name": "top", "azimuth_deg": 0, "elevation_deg": 90, "is_detail": False, "crop_z_range": None},
        {"name": "bottom", "azimuth_deg": 0, "elevation_deg": -90, "is_detail": False, "crop_z_range": None},
        {"name": "detail", "azimuth_deg": 0, "elevation_deg": 0, "is_detail": True, "crop_z_range": (1.0, 1.5)},
    ]


def _rotation_matrix(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Rotation whose rows are (image right, image up, viewing direction)."""
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    cos_az, sin_az = np.cos(az), np.sin(az)
    cos_el, sin_el = np.cos(el), np.sin(el)

    forward = np.array([cos_el * cos_az, cos_el * sin_az, sin_el])
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(sin_el) > 0.999:  # looking straight up or down
        world_up = np.array([0.0, 1.0, 0.0])

    right = np.cross(forward, world_up)
    right_norm = np.linalg.norm(right)
    right = np.array([1.0, 0.0, 0.0]) if right_norm < 1e-9 else right / right_norm
    up = np.cross(right, forward)
    up = up / (np.linalg.norm(up) + 1e-12)
    return np.stack([right, up, forward], axis=0)


def _minmax_to_uint8(values: np.ndarray) -> np.ndarray:
    lo, hi = np.nanmin(values), np.nanmax(values)
    rng = hi - lo
    if rng < 1e-9:
        return np.full(values.shape, 128, dtype=np.uint8)
    return np.clip((values - lo) / rng * 255.0, 0, 255).astype(np.uint8)


def ortho_project(points: np.ndarray, azimuth_deg: float, elevation_deg: float,
                  H: int = 256, W: int = 256) -> np.ndarray:
    """Render one orthographic view. Returns an (H, W, 3) uint8 image."""
    if points.shape[0] == 0:
        return np.zeros((H, W, 3), dtype=np.uint8)

    xyz = np.asarray(points[:, :3], dtype=np.float64)
    original_z = xyz[:, 2].copy()

    R = _rotation_matrix(azimuth_deg, elevation_deg)
    rotated = (R @ xyz.T).T
    u, v, depth = rotated[:, 0], rotated[:, 1], rotated[:, 2]

    margin = 0.02
    u_min, u_max = u.min(), u.max()
    v_min, v_max = v.min(), v.max()
    span = max(u_max - u_min, v_max - v_min)  # isotropic scale, no distortion
    if span < 1e-6:
        return np.zeros((H, W, 3), dtype=np.uint8)

    u_center = 0.5 * (u_min + u_max)
    v_center = 0.5 * (v_min + v_max)
    scale = (1.0 - 2.0 * margin) / span
    px = ((u - u_center) * scale + 0.5) * W
    py = ((v_center - v) * scale + 0.5) * H
    col = np.clip(px.astype(np.int32), 0, W - 1)
    row = np.clip(py.astype(np.int32), 0, H - 1)

    # Per pixel: point count, plus depth and height of the front-most point.
    # Sorting by (pixel, depth, index) picks the smallest depth and, among
    # ties, the earliest point, exactly as a sequential strict-< scan would.
    flat = row.astype(np.int64) * W + col
    count = np.bincount(flat, minlength=H * W).reshape(H, W).astype(np.float64)
    order = np.lexsort((np.arange(flat.size), depth, flat))
    flat_sorted = flat[order]
    first = order[np.r_[True, flat_sorted[1:] != flat_sorted[:-1]]]

    depth_buf = np.full(H * W, np.inf)
    height_buf = np.full(H * W, np.nan)
    depth_buf[flat[first]] = depth[first]
    height_buf[flat[first]] = original_z[first]
    depth_buf = depth_buf.reshape(H, W)
    height_buf = height_buf.reshape(H, W)

    occupied = count > 0
    image = np.zeros((H, W, 3), dtype=np.uint8)
    if occupied.any():
        image[..., 0][occupied] = _minmax_to_uint8(depth_buf[occupied])
        image[..., 1][occupied] = _minmax_to_uint8(np.log1p(count[occupied]))
        image[..., 2][occupied] = _minmax_to_uint8(height_buf[occupied])
    return image


def normalise_cloud(points: np.ndarray) -> np.ndarray:
    """Center XY on the median and shift Z so the lowest point is at 0."""
    pts = points.copy()
    pts[:, 0] -= np.median(pts[:, 0])
    pts[:, 1] -= np.median(pts[:, 1])
    pts[:, 2] -= pts[:, 2].min()
    return pts


def detail_crop(pts: np.ndarray, crop_z_range: Tuple[float, float],
                min_points: int = 20) -> np.ndarray:
    """Trunk slice for the detail view, with fallbacks for sparse trunks."""
    z = pts[:, 2]
    z_lo, z_hi = crop_z_range
    subset = pts[(z >= z_lo) & (z <= z_hi)]
    if subset.shape[0] < min_points:
        subset = pts[(z >= 0.5) & (z <= 2.0)]
    if subset.shape[0] < min_points:
        return pts
    return subset


def render_views(points_xyz: np.ndarray, img_size: int = 256,
                 tree_height: Optional[float] = None) -> Dict[str, np.ndarray]:
    """Render the seven views of one tree.

    Args:
        points_xyz: (N, 3) XYZ in metres, Z up. Pass float32-rounded coordinates
            cast to float64 to match the training renders (see ``mvtree.io``).
        img_size: Output resolution; 256 and 512 match the released models.
        tree_height: Optional height in metres; defaults to the Z extent.

    Returns:
        Mapping from view name to an (img_size, img_size, 3) uint8 image.
    """
    if points_xyz.shape[0] < 3:
        raise ValueError(f"need at least 3 points, got {points_xyz.shape[0]}")

    pts = normalise_cloud(np.asarray(points_xyz[:, :3], dtype=np.float64))
    if tree_height is None:
        tree_height = pts[:, 2].max() - pts[:, 2].min()

    views = {}
    for view in get_view_configs():
        if view["is_detail"]:
            # Fixed 1.0-1.5 m trunk slice for trees >= 5 m; shorter trees use
            # 20-40 % of their height so the slice still hits the stem.
            crop = view["crop_z_range"] if tree_height >= 5.0 else (0.20 * tree_height, 0.40 * tree_height)
            render_pts = detail_crop(pts, crop)
        else:
            render_pts = pts
        if render_pts.shape[0] == 0:
            views[view["name"]] = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        else:
            views[view["name"]] = ortho_project(
                render_pts, view["azimuth_deg"], view["elevation_deg"], H=img_size, W=img_size,
            )
    return views


def save_views(views: Dict[str, np.ndarray], output_dir: str) -> None:
    """Write the seven views as ``<output_dir>/<view>.png``."""
    os.makedirs(output_dir, exist_ok=True)
    for name, img in views.items():
        Image.fromarray(img, mode="RGB").save(os.path.join(output_dir, f"{name}.png"))


def render_tree(points_xyz: np.ndarray, output_dir: str, img_size: int = 256) -> None:
    """Render and save all seven views of one tree."""
    save_views(render_views(points_xyz, img_size=img_size), output_dir)
