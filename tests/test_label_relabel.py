"""Tests for pygeovision.labeling.relabel.assign_classes (naming SAM instances / DINOv2 clusters)."""
import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio", reason="rasterio not installed")

from pygeovision.labeling.relabel import assign_classes  # noqa: E402


def _w(path, arr, dtype="uint8", nodata=None, bounds=(0, 0, 1, 1)):
    from rasterio.transform import from_bounds
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1, dtype=dtype,
                       crs="EPSG:4326", transform=from_bounds(*bounds, arr.shape[1], arr.shape[0]), nodata=nodata) as d:
        d.write(arr.astype(dtype)[np.newaxis])
    return str(path)


def test_majority_class_under_each_segment(tmp_path):
    seg = np.zeros((10, 10), "uint32")
    seg[:5, :5] = 1       # mostly class 1
    seg[:5, 5:] = 2       # mostly class 2
    seg[5:, :5] = 3       # half/half -> unlabelled at 0.6
    ref = np.zeros((10, 10), "uint8")
    ref[:5, :5] = 1; ref[0, 0] = 2          # 24 of 25 are class 1
    ref[:5, 5:] = 2
    ref[5:, :5] = 1; ref[7:, :5] = 2        # 2 rows class 1, 3 rows class 2 -> 60% class 2
    out = tmp_path / "o.tif"
    r = assign_classes(_w(tmp_path / "s.tif", seg, "uint32"), _w(tmp_path / "r.tif", ref), out, min_share=0.7)
    assert r["mapping"] == {1: 1, 2: 2, 3: 255}
    assert r["n_segments"] == 3 and r["n_unlabelled"] == 1
    with rasterio.open(out) as o:
        a = o.read(1)
        assert o.nodata == 255
    assert (a[:5, :5] == 1).all() and (a[:5, 5:] == 2).all() and (a[5:, :5] == 255).all() and (a[5:, 5:] == 0).all()
    r = assign_classes(_w(tmp_path / "s.tif", seg, "uint32"), _w(tmp_path / "r.tif", ref), out, min_share=0.5)
    assert r["mapping"][3] == 2


def test_reference_nodata_does_not_vote(tmp_path):
    seg = np.full((6, 6), 4, "uint8")
    ref = np.full((6, 6), 255, "uint8"); ref[:2, :2] = 3        # only 4 voting pixels, all class 3
    r = assign_classes(_w(tmp_path / "s.tif", seg), _w(tmp_path / "r.tif", ref, nodata=255), tmp_path / "o.tif")
    assert r["mapping"] == {4: 3}


def test_reference_on_another_grid_is_resampled(tmp_path):
    seg = np.zeros((20, 20), "uint8"); seg[:, :10] = 1; seg[:, 10:] = 2
    ref = np.zeros((5, 5), "uint8"); ref[:, :2] = 7; ref[:, 3:] = 9          # coarser reference
    r = assign_classes(_w(tmp_path / "s.tif", seg), _w(tmp_path / "r.tif", ref), tmp_path / "o.tif")
    assert r["mapping"] == {1: 7, 2: 9}


def test_cluster_maps_keep_zero_as_a_real_id_and_255_as_nodata(tmp_path):
    seg = np.full((8, 8), 255, "uint8"); seg[:, :4] = 0; seg[:, 4:7] = 1
    ref = np.zeros((8, 8), "uint8"); ref[:, :4] = 5; ref[:, 4:] = 6
    r = assign_classes(_w(tmp_path / "s.tif", seg), _w(tmp_path / "r.tif", ref), tmp_path / "o.tif",
                       ignore_segment_ids=(255,))
    assert r["mapping"] == {0: 5, 1: 6}
    with rasterio.open(tmp_path / "o.tif") as o:
        a = o.read(1)
    assert (a[:, :4] == 5).all() and (a[:, 4:7] == 6).all() and (a[:, 7] == 255).all()


def test_no_segments_and_bad_arguments(tmp_path):
    seg = np.zeros((4, 4), "uint8")
    r = assign_classes(_w(tmp_path / "s.tif", seg), _w(tmp_path / "r.tif", seg), tmp_path / "o.tif")
    assert r["n_segments"] == 0 and r["mapping"] == {}
    with pytest.raises(ValueError):
        assign_classes(_w(tmp_path / "s.tif", seg), _w(tmp_path / "r.tif", seg), tmp_path / "o.tif", min_share=0)
