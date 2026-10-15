import numpy as np
import pytest

from mvtree.render import VIEW_NAMES, _rotation_matrix, get_view_configs, ortho_project, render_views


def reference_ortho_project(points, azimuth_deg, elevation_deg, H=256, W=256):
    """The original per-point loop the training renders were made with."""
    if points.shape[0] == 0:
        return np.zeros((H, W, 3), dtype=np.uint8)
    xyz = np.asarray(points[:, :3], dtype=np.float64)
    original_z = xyz[:, 2].copy()
    rotated = (_rotation_matrix(azimuth_deg, elevation_deg) @ xyz.T).T
    u, v, depth = rotated[:, 0], rotated[:, 1], rotated[:, 2]
    span = max(u.max() - u.min(), v.max() - v.min())
    if span < 1e-6:
        return np.zeros((H, W, 3), dtype=np.uint8)
    uc, vc = 0.5 * (u.min() + u.max()), 0.5 * (v.min() + v.max())
    scale = (1.0 - 2.0 * 0.02) / span
    col = np.clip((((u - uc) * scale + 0.5) * W).astype(np.int32), 0, W - 1)
    row = np.clip((((vc - v) * scale + 0.5) * H).astype(np.int32), 0, H - 1)
    depth_buf = np.full((H, W), np.inf)
    height_buf = np.full((H, W), np.nan)
    count_buf = np.zeros((H, W))
    for i in range(len(row)):
        r, c = row[i], col[i]
        count_buf[r, c] += 1.0
        if depth[i] < depth_buf[r, c]:
            depth_buf[r, c] = depth[i]
            height_buf[r, c] = original_z[i]
    occ = count_buf > 0
    out = np.zeros((H, W, 3), dtype=np.uint8)
    for ch, vals in enumerate([depth_buf[occ], np.log1p(count_buf[occ]), height_buf[occ]]):
        lo, hi = np.nanmin(vals), np.nanmax(vals)
        out[..., ch][occ] = 128 if hi - lo < 1e-9 else np.clip((vals - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    return out


@pytest.mark.parametrize("view", get_view_configs(), ids=lambda v: v["name"])
@pytest.mark.parametrize("size", [64, 256])
def test_vectorized_matches_reference(cloud, view, size):
    a = ortho_project(cloud, view["azimuth_deg"], view["elevation_deg"], size, size)
    b = reference_ortho_project(cloud, view["azimuth_deg"], view["elevation_deg"], size, size)
    assert np.array_equal(a, b)


def test_render_views_shapes(cloud):
    views = render_views(cloud, img_size=128)
    assert list(views) == VIEW_NAMES
    for img in views.values():
        assert img.shape == (128, 128, 3) and img.dtype == np.uint8
        assert img.any(), "every view of a non-degenerate tree has occupied pixels"


def test_short_tree_detail_crop(cloud):
    short = cloud.copy()
    short[:, 2] *= 0.3  # 2.7 m tree -> height-adaptive trunk slice
    assert render_views(short, img_size=64)["detail"].any()


def test_degenerate_inputs():
    assert not ortho_project(np.zeros((0, 3)), 0, 0, 32, 32).any()
    assert not ortho_project(np.ones((10, 3)), 0, 0, 32, 32).any()
    with pytest.raises(ValueError):
        render_views(np.zeros((2, 3)))
