"""Handcrafted geometric features + LightGBM (the optional geometric branch).

The official leaderboard files blended these LightGBM probabilities with the
ConvNeXtV2 probabilities (weight 0.35 for the 256 px entry, 0.15 for the
multi-resolution entry). On the public labeled set the blend changes OA by
less than 0.001, so the paper analyses the image model alone; this module is
here so the submitted files can be reproduced exactly.

The feature code is the original implementation, unchanged except that it
takes arrays instead of a laspy object. Requires the ``geometry`` extra
(``pip install -e .[geometry]``).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.spatial import ConvexHull
from scipy.stats import kurtosis, skew

# ---- part segmentation (originally utils/point_cloud_utils.py) ----

def detect_crown_base_height(points_z, n_bins=50):
    """
    Detect crown base height using vertical point density inflection method.

    Method: Create vertical density histogram, smooth it, find the height where
    the density first increases significantly (inflection point) above the trunk region.

    Args:
        points_z: 1D array of Z coordinates (ground-relative, starting at 0)
        n_bins: number of vertical bins

    Returns:
        cbh: estimated crown base height (float)
    """
    tree_h = points_z.max()
    if tree_h <= 0:
        return 0.0

    bin_edges = np.linspace(0, tree_h, n_bins + 1)
    counts, _ = np.histogram(points_z, bins=bin_edges)
    counts = counts.astype(float)

    # Smooth with moving average (window=5)
    kernel = np.ones(5) / 5
    smoothed = np.convolve(counts, kernel, mode='same')

    # Find inflection: where gradient changes from decreasing to increasing
    # in the lower half of the tree
    gradient = np.gradient(smoothed)
    half_idx = n_bins // 2

    # Look for first significant increase after initial decrease
    for i in range(2, half_idx):
        if gradient[i] > 0 and smoothed[i] > smoothed.mean() * 0.3:
            cbh = bin_edges[i]
            return float(cbh)

    # Fallback: 30% of tree height
    return float(tree_h * 0.3)


def segment_tree_parts(points, cbh=None):
    """
    Segment tree point cloud into parts based on height.

    Returns dict with:
        'trunk': points below CBH
        'crown': points above CBH
        'quintiles': list of 5 point arrays (0-20%, 20-40%, ..., 80-100% height)
    """
    z = points[:, 2] - points[:, 2].min()
    tree_h = z.max()

    if cbh is None:
        cbh = detect_crown_base_height(z)

    trunk_mask = z < cbh
    crown_mask = z >= cbh

    quintiles = []
    for q in range(5):
        low = tree_h * q / 5
        high = tree_h * (q + 1) / 5
        mask = (z >= low) & (z < high)
        quintiles.append(points[mask])

    return {
        'trunk': points[trunk_mask],
        'crown': points[crown_mask],
        'quintiles': quintiles,
        'cbh': cbh,
    }


def estimate_dbh_ransac(trunk_points, target_height=1.3, band_width=0.2):
    """
    Estimate Diameter at Breast Height via circle fitting on horizontal slice.

    Args:
        trunk_points: (N, 3) trunk point cloud
        target_height: height for DBH measurement (default 1.3m)
        band_width: thickness of horizontal band to use

    Returns:
        dbh: estimated diameter (float), or -1 if insufficient points
    """
    z = trunk_points[:, 2] - trunk_points[:, 2].min()
    mask = (z >= target_height - band_width/2) & (z <= target_height + band_width/2)

    if mask.sum() < 10:
        # Try wider band
        mask = (z >= target_height - band_width) & (z <= target_height + band_width)

    if mask.sum() < 5:
        return -1.0

    xy = trunk_points[mask, :2]
    center = xy.mean(axis=0)
    radii = np.linalg.norm(xy - center, axis=1)
    dbh = 2.0 * np.median(radii)  # diameter from median radius
    return float(dbh)


def compute_bark_roughness(trunk_points, target_height=1.3, band_width=0.5):
    """
    Compute bark roughness as std of radial distance from fitted cylinder.

    Returns roughness value (float), or -1 if insufficient points.
    """
    z = trunk_points[:, 2] - trunk_points[:, 2].min()
    mask = (z >= target_height - band_width/2) & (z <= target_height + band_width/2)

    if mask.sum() < 20:
        return -1.0

    xy = trunk_points[mask, :2]
    center = xy.mean(axis=0)
    radii = np.linalg.norm(xy - center, axis=1)
    roughness = float(radii.std())
    return roughness


# ---- feature extraction (originally scripts/03_extract_geometric_features.py) ----

def _compute_crown_shape_conical(crown_points):
    """
    Compute conical shape score for crown.

    For each height band in crown, compute XY width. A perfect cone has width
    linearly increasing with depth from top.
    Score = correlation of width vs depth (high = conical).

    Returns:
        conical_residual: float, lower means more conical. -1 if insufficient data.
    """
    if len(crown_points) < 20:
        return -1.0

    z = crown_points[:, 2]
    z_min, z_max = z.min(), z.max()
    crown_h = z_max - z_min
    if crown_h <= 0:
        return -1.0

    n_bands = 10
    band_edges = np.linspace(z_min, z_max, n_bands + 1)
    widths = []
    depths = []

    for i in range(n_bands):
        mask = (z >= band_edges[i]) & (z < band_edges[i + 1])
        if mask.sum() < 3:
            continue
        xy = crown_points[mask, :2]
        width = np.ptp(xy, axis=0).mean()
        depth = z_max - (band_edges[i] + band_edges[i + 1]) / 2  # depth from top
        widths.append(width)
        depths.append(depth)

    if len(widths) < 3:
        return -1.0

    widths = np.array(widths)
    depths = np.array(depths)

    # Correlation of width vs depth: high correlation = conical
    corr = np.corrcoef(depths, widths)[0, 1]
    if np.isnan(corr):
        return -1.0

    # Residual: lower = more conical. Convert correlation to residual.
    residual = float(1.0 - corr)
    return residual


def _compute_crown_shape_spherical(crown_points):
    """
    Compute spherical shape score for crown.

    Score = 1 - CV(width across middle 3 quintiles of crown).
    Low residual = more spherical (constant width).

    Returns:
        spherical_residual: float, lower means more spherical. -1 if insufficient data.
    """
    if len(crown_points) < 20:
        return -1.0

    z = crown_points[:, 2]
    z_min, z_max = z.min(), z.max()
    crown_h = z_max - z_min
    if crown_h <= 0:
        return -1.0

    # Split crown into 5 quintiles, take middle 3
    quintile_edges = np.linspace(z_min, z_max, 6)
    middle_widths = []

    for i in range(1, 4):  # quintiles 1, 2, 3 (0-indexed) = middle 3
        mask = (z >= quintile_edges[i]) & (z < quintile_edges[i + 1])
        if mask.sum() < 3:
            continue
        xy = crown_points[mask, :2]
        width = np.ptp(xy, axis=0).mean()
        middle_widths.append(width)

    if len(middle_widths) < 2:
        return -1.0

    middle_widths = np.array(middle_widths)
    mean_w = middle_widths.mean()
    if mean_w <= 0:
        return -1.0

    cv = middle_widths.std() / mean_w
    residual = float(cv)  # lower = more spherical
    return residual


def _compute_crown_max_width_height(crown_points):
    """
    Compute relative height of maximum crown width.

    Returns:
        relative height (0-1 within crown) where maximum width occurs.
        -1 if insufficient data.
    """
    if len(crown_points) < 20:
        return -1.0

    z = crown_points[:, 2]
    z_min, z_max = z.min(), z.max()
    crown_h = z_max - z_min
    if crown_h <= 0:
        return -1.0

    n_bands = 10
    band_edges = np.linspace(z_min, z_max, n_bands + 1)
    max_width = 0.0
    max_width_z = z_min

    for i in range(n_bands):
        mask = (z >= band_edges[i]) & (z < band_edges[i + 1])
        if mask.sum() < 3:
            continue
        xy = crown_points[mask, :2]
        width = np.ptp(xy, axis=0).mean()
        if width > max_width:
            max_width = width
            max_width_z = (band_edges[i] + band_edges[i + 1]) / 2

    rel_height = float((max_width_z - z_min) / crown_h)
    return rel_height


def _compute_trunk_straightness(trunk_points, n_slices=10):
    """
    Compute trunk straightness as std of XY centroid positions along Z slices.

    Lower value = straighter trunk.
    Returns -1 if insufficient data.
    """
    if len(trunk_points) < 10:
        return -1.0

    z = trunk_points[:, 2]
    z_min, z_max = z.min(), z.max()
    trunk_h = z_max - z_min
    if trunk_h <= 0:
        return -1.0

    slice_edges = np.linspace(z_min, z_max, n_slices + 1)
    centroids = []

    for i in range(n_slices):
        mask = (z >= slice_edges[i]) & (z < slice_edges[i + 1])
        if mask.sum() < 3:
            continue
        xy = trunk_points[mask, :2]
        centroids.append(xy.mean(axis=0))

    if len(centroids) < 3:
        return -1.0

    centroids = np.array(centroids)
    straightness = float(centroids.std())
    return straightness


def extract_features(points: np.ndarray, intensity: Optional[np.ndarray] = None,
                     return_number: Optional[np.ndarray] = None) -> Dict[str, float]:
    """All 105 geometric and part-based features of one tree.

    Args:
        points: (N, 3) XYZ in metres.
        intensity, return_number: per-point LAS attributes; when missing the
            corresponding features take the same fallback values as in training.
    """
    points = np.asarray(points[:, :3], dtype=np.float32).copy()

    # Normalize z to start at 0
    points[:, 2] -= points[:, 2].min()
    tree_h = points[:, 2].max()
    N = len(points)

    feats = {"tree_H": tree_h, "n_points": N}

    # =====================================================================
    # EXISTING FEATURES
    # =====================================================================

    # --- Crown geometry ---
    try:
        hull = ConvexHull(points)
        feats["crown_volume"] = hull.volume
        feats["crown_surface_area"] = hull.area
        hull_2d = ConvexHull(points[:, :2])
        feats["crown_area"] = hull_2d.volume
        feats["crown_perimeter"] = hull_2d.area  # "area" of 2D hull = perimeter
    except Exception:
        feats["crown_volume"] = 0.0
        feats["crown_surface_area"] = 0.0
        feats["crown_area"] = 0.0
        feats["crown_perimeter"] = 0.0

    # Aspect ratios
    xy_range = points[:, :2].max(axis=0) - points[:, :2].min(axis=0)
    feats["crown_aspect_ratio"] = float(xy_range[0] / (xy_range[1] + 1e-6))
    feats["crown_diameter_x"] = float(xy_range[0])
    feats["crown_diameter_y"] = float(xy_range[1])
    feats["crown_diameter_mean"] = float(xy_range.mean())
    feats["height_diameter_ratio"] = float(tree_h / (xy_range.mean() + 1e-6))

    # Point density
    feats["point_density"] = N / (feats["crown_volume"] + 1e-6)
    feats["point_density_2d"] = N / (feats["crown_area"] + 1e-6)

    # --- Vertical profile (20 bins) ---
    if tree_h > 0:
        n_bins = 20
        bin_edges = np.linspace(0, tree_h, n_bins + 1)
        for b in range(n_bins):
            mask = (points[:, 2] >= bin_edges[b]) & (points[:, 2] < bin_edges[b + 1])
            feats[f"vdensity_bin_{b}"] = float(mask.sum() / N)
    else:
        for b in range(20):
            feats[f"vdensity_bin_{b}"] = 0.0

    # --- Height percentiles ---
    z = points[:, 2]
    for p in [5, 10, 25, 50, 75, 90, 95]:
        feats[f"z_p{p}"] = float(np.percentile(z, p))
    feats["z_mean"] = float(z.mean())
    feats["z_std"] = float(z.std())
    feats["z_skew"] = float(skew(z))
    feats["z_kurtosis"] = float(kurtosis(z))
    feats["z_cv"] = float(z.std() / (z.mean() + 1e-6))
    feats["z_iqr"] = float(np.percentile(z, 75) - np.percentile(z, 25))

    # --- Crown vs trunk analysis ---
    trunk_mask = z < min(1.5, tree_h * 0.2)
    crown_mask = z >= tree_h * 0.4
    feats["trunk_crown_ratio"] = float(trunk_mask.sum() / N)
    feats["crown_point_ratio"] = float(crown_mask.sum() / N)

    # Crown width at different heights
    for frac, name in [(0.25, "low"), (0.5, "mid"), (0.75, "high")]:
        h_level = tree_h * frac
        band = (z >= h_level - tree_h * 0.05) & (z <= h_level + tree_h * 0.05)
        if band.sum() > 2:
            xy_band = points[band, :2]
            feats[f"crown_width_{name}"] = float(np.ptp(xy_band, axis=0).mean())
        else:
            feats[f"crown_width_{name}"] = 0.0

    # Crown shape ratio (width at mid / width at base)
    feats["crown_taper"] = float(
        feats["crown_width_high"] / (feats["crown_width_low"] + 1e-6))

    # --- Distance from centroid ---
    centroid = points.mean(axis=0)
    dists = np.linalg.norm(points - centroid, axis=1)
    feats["fractal_proxy"] = float(dists.std())
    feats["dist_mean"] = float(dists.mean())
    feats["dist_max"] = float(dists.max())
    feats["dist_skew"] = float(skew(dists))

    # Horizontal distance from trunk axis
    xy_centroid = points[:, :2].mean(axis=0)
    horiz_dists = np.linalg.norm(points[:, :2] - xy_centroid, axis=1)
    feats["horiz_dist_mean"] = float(horiz_dists.mean())
    feats["horiz_dist_std"] = float(horiz_dists.std())
    feats["horiz_dist_max"] = float(horiz_dists.max())

    # --- Horizontal cross-section analysis ---
    # Crown circularity (at 75% height)
    h75 = tree_h * 0.75
    band75 = (z >= h75 - tree_h * 0.05) & (z <= h75 + tree_h * 0.05)
    if band75.sum() > 10:
        xy75 = points[band75, :2]
        try:
            hull75 = ConvexHull(xy75)
            area75 = hull75.volume
            perim75 = hull75.area
            feats["crown_circularity"] = float(4 * np.pi * area75 / (perim75 ** 2 + 1e-6))
        except Exception:
            feats["crown_circularity"] = 0.0
    else:
        feats["crown_circularity"] = 0.0

    # --- Eigenvalue features (PCA of point cloud) ---
    try:
        cov = np.cov(points.T)
        eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
        eigvals = np.maximum(eigvals, 1e-10)
        feats["eigen_linearity"] = float((eigvals[0] - eigvals[1]) / (eigvals[0] + 1e-6))
        feats["eigen_planarity"] = float((eigvals[1] - eigvals[2]) / (eigvals[0] + 1e-6))
        feats["eigen_sphericity"] = float(eigvals[2] / (eigvals[0] + 1e-6))
        feats["eigen_anisotropy"] = float((eigvals[0] - eigvals[2]) / (eigvals[0] + 1e-6))
        feats["eigen_sum"] = float(eigvals.sum())
        feats["eigen_ratio_12"] = float(eigvals[0] / (eigvals[1] + 1e-6))
        feats["eigen_ratio_13"] = float(eigvals[0] / (eigvals[2] + 1e-6))
    except Exception:
        for k in ["eigen_linearity", "eigen_planarity", "eigen_sphericity",
                   "eigen_anisotropy", "eigen_sum", "eigen_ratio_12", "eigen_ratio_13"]:
            feats[k] = 0.0

    # --- Intensity features ---
    try:
        if intensity is None:
            raise ValueError("no intensity")
        intensity = np.array(intensity, dtype=np.float32)
        feats["intensity_mean"] = float(intensity.mean())
        feats["intensity_std"] = float(intensity.std())
        feats["intensity_skew"] = float(skew(intensity)) if intensity.std() > 0 else 0.0
        feats["intensity_p25"] = float(np.percentile(intensity, 25))
        feats["intensity_p75"] = float(np.percentile(intensity, 75))
    except Exception:
        feats["intensity_mean"] = -1.0
        feats["intensity_std"] = -1.0
        feats["intensity_skew"] = 0.0
        feats["intensity_p25"] = -1.0
        feats["intensity_p75"] = -1.0

    # --- Return number features ---
    try:
        if return_number is None:
            raise ValueError("no return numbers")
        returns = np.array(return_number, dtype=np.float32)
        feats["return_mean"] = float(returns.mean())
        feats["return_max"] = float(returns.max())
        feats["multi_return_ratio"] = float((returns > 1).sum() / N)
    except Exception:
        feats["return_mean"] = 0.0
        feats["return_max"] = 0.0
        feats["multi_return_ratio"] = 0.0

    # =====================================================================
    # NEW PART-BASED FEATURES
    # =====================================================================

    # --- Segment tree into parts ---
    parts = segment_tree_parts(points)
    trunk_pts = parts['trunk']
    crown_pts = parts['crown']
    quintiles = parts['quintiles']
    cbh = parts['cbh']

    # --- Crown base height ---
    feats["cbh"] = float(cbh)
    feats["cbh_ratio"] = float(cbh / (tree_h + 1e-6))

    # --- DBH from trunk ---
    if len(trunk_pts) > 5:
        feats["dbh"] = estimate_dbh_ransac(trunk_pts)
    else:
        feats["dbh"] = -1.0
    feats["dbh_height_ratio"] = float(feats["dbh"] / (tree_h + 1e-6)) if feats["dbh"] > 0 else -1.0

    # --- Bark roughness ---
    if len(trunk_pts) > 20:
        feats["bark_roughness"] = compute_bark_roughness(trunk_pts)
    else:
        feats["bark_roughness"] = -1.0

    # --- Trunk straightness ---
    feats["trunk_straightness"] = _compute_trunk_straightness(trunk_pts)

    # --- Crown ratio (crown height / tree height) ---
    if len(crown_pts) > 0:
        crown_h = crown_pts[:, 2].max() - crown_pts[:, 2].min()
        feats["crown_ratio"] = float(crown_h / (tree_h + 1e-6))
    else:
        feats["crown_ratio"] = 0.0

    # --- Per-quintile features ---
    for qi in range(5):
        q_pts = quintiles[qi]
        n_q = len(q_pts)
        prefix = f"q{qi}"

        # Point ratio: fraction of total points in this quintile
        feats[f"{prefix}_point_ratio"] = float(n_q / N)

        if n_q < 3:
            feats[f"{prefix}_spread"] = 0.0
            feats[f"{prefix}_density"] = 0.0
            feats[f"{prefix}_z_std"] = 0.0
            continue

        # Spread: mean XY spread (mean of XY ranges)
        xy_spread = np.ptp(q_pts[:, :2], axis=0).mean()
        feats[f"{prefix}_spread"] = float(xy_spread)

        # Density: points per volume in quintile
        q_z_range = q_pts[:, 2].max() - q_pts[:, 2].min()
        q_xy_range = np.ptp(q_pts[:, :2], axis=0)
        q_vol = (q_xy_range[0] + 1e-6) * (q_xy_range[1] + 1e-6) * (q_z_range + 1e-6)
        feats[f"{prefix}_density"] = float(n_q / q_vol)

        # Z variation within quintile
        feats[f"{prefix}_z_std"] = float(q_pts[:, 2].std())

    # --- Crown shape features ---
    feats["crown_shape_conical"] = _compute_crown_shape_conical(crown_pts)
    feats["crown_shape_spherical"] = _compute_crown_shape_spherical(crown_pts)
    feats["crown_max_width_height"] = _compute_crown_max_width_height(crown_pts)

    return feats



def read_las_with_attributes(path_or_bytes):
    """Read XYZ, intensity and return number from a .las/.laz path or bytes."""
    import io

    import laspy

    las = laspy.read(io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes) else path_or_bytes)
    xyz = np.stack([las.x, las.y, las.z], axis=-1)
    return xyz, np.asarray(las.intensity), np.asarray(las.return_number)


def load_boosters(model_dir: str):
    """LightGBM fold models and the feature order from ``weights/geometry_lgbm``."""
    import glob
    import json
    import os

    import lightgbm as lgb

    with open(os.path.join(model_dir, "config.json")) as fh:
        cfg = json.load(fh)
    files = sorted(glob.glob(os.path.join(model_dir, "fold*.txt")))
    return [lgb.Booster(model_file=f) for f in files], cfg["feature_cols"]


def predict_proba(features: Dict[str, float], boosters, feature_cols) -> np.ndarray:
    """Fold-averaged LightGBM class probabilities (missing features -> 0)."""
    x = np.array([[features.get(c, 0.0) for c in feature_cols]], dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0)
    return np.mean([b.predict(x)[0] for b in boosters], axis=0)
