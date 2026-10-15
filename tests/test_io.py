import numpy as np
import pytest

from mvtree.io import find_point_clouds, read_point_cloud


@pytest.fixture
def xyz():
    rng = np.random.default_rng(1)
    return (rng.normal(size=(500, 3)) * [2, 2, 5] + [600000.0, 5000000.0, 300.0]).astype(np.float32)


def check(path, xyz):
    got = read_point_cloud(str(path))
    assert got.dtype == np.float32 and got.shape == xyz.shape
    np.testing.assert_allclose(got, xyz, rtol=0, atol=0.02)


def test_text_formats(tmp_path, xyz):
    np.savetxt(tmp_path / "a.xyz", xyz, fmt="%.4f")
    np.savetxt(tmp_path / "b.csv", xyz, fmt="%.4f", delimiter=",", header="x,y,z", comments="")
    np.savetxt(tmp_path / "c.txt", np.c_[xyz, np.ones(len(xyz))], fmt="%.4f")
    for name in ("a.xyz", "b.csv", "c.txt"):
        check(tmp_path / name, xyz)


def test_numpy_formats(tmp_path, xyz):
    np.save(tmp_path / "a.npy", xyz)
    np.savez(tmp_path / "b.npz", points=xyz)
    check(tmp_path / "a.npy", xyz)
    check(tmp_path / "b.npz", xyz)


def test_las_laz(tmp_path, xyz):
    laspy = pytest.importorskip("laspy")
    for ext in ("las", "laz"):
        las = laspy.create(point_format=3, file_version="1.2")
        las.header.scales = [0.001] * 3
        las.header.offsets = xyz.min(0).astype(np.float64)
        las.x, las.y, las.z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        las.write(tmp_path / f"a.{ext}")
        check(tmp_path / f"a.{ext}", xyz)


def test_ply(tmp_path, xyz):
    plyfile = pytest.importorskip("plyfile")
    v = np.array([tuple(p) for p in xyz], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    plyfile.PlyData([plyfile.PlyElement.describe(v, "vertex")]).write(str(tmp_path / "a.ply"))
    check(tmp_path / "a.ply", xyz)


def test_pcd_ascii_and_binary(tmp_path, xyz):
    head = ("VERSION .7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
            f"WIDTH {len(xyz)}\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(xyz)}\n")
    (tmp_path / "a.pcd").write_text(head + "DATA ascii\n" + "\n".join(" ".join(f"{c:.4f}" for c in p) for p in xyz))
    (tmp_path / "b.pcd").write_bytes((head + "DATA binary\n").encode() + xyz.astype("<f4").tobytes())
    check(tmp_path / "a.pcd", xyz)
    check(tmp_path / "b.pcd", xyz)


def test_find_and_errors(tmp_path, xyz):
    np.save(tmp_path / "a.npy", xyz)
    (tmp_path / "notes.md").write_text("x")
    assert find_point_clouds([str(tmp_path)]) == [str(tmp_path / "a.npy")]
    with pytest.raises(ValueError):
        read_point_cloud(str(tmp_path / "notes.md"))
