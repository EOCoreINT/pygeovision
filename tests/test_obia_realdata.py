"""OBIA behaviour on real-data conditions: cloud holes (NaN), nodata, CRS mix-ups.

Each test states the answer it expects by construction (a hole of known shape,
a polygon of known extent), not a number copied from a run.
"""
import time

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("skimage")
pytest.importorskip("sklearn")
pytest.importorskip("rasterio")
gpd = pytest.importorskip("geopandas")
from affine import Affine  # noqa: E402
from shapely.geometry import box  # noqa: E402

from pygeovision import obia  # noqa: E402
from tests.obia_testdata import block_scene  # noqa: E402


def _scene_with_holes(size=96):
    sc = block_scene(size)
    img = sc.raster.image.copy()
    hole = np.zeros(img.shape[1:], bool)
    hole[10:30, 20:50] = True            # a cloud block
    hole[70:75, 5:90] = True             # a thin strip (scan-line style)
    img[:, hole] = np.nan
    rd = obia.RasterData(img, sc.raster.band_names, sc.raster.transform, sc.raster.crs)
    return sc, rd, hole


# ------------------------------------------------------------------ valid_mask
def test_valid_mask_nan_and_nodata():
    a = np.ones((2, 3, 3), np.float32)
    a[:, 0, 0] = np.nan                  # all bands NaN
    a[1, 1, 1] = np.inf                  # one band infinite -> pixel invalid
    a[:, 2, 2] = -9999                   # nodata in all bands
    a[0, 2, 0] = -9999                   # nodata in ONE band only -> still valid
    rd = obia.RasterData(a, ["a", "b"], nodata=-9999)
    ok = obia.valid_mask(rd)
    assert (~ok).sum() == 3
    assert not ok[0, 0] and not ok[1, 1] and not ok[2, 2] and ok[2, 0]
    # without a declared nodata the -9999 pixel counts as data
    assert obia.valid_mask(obia.RasterData(a, ["a", "b"])).sum() == 7


def test_rescale_ignores_nan_and_keeps_it():
    from pygeovision.obia.io import rescale_bands
    x = np.arange(100, dtype=float).reshape(1, 10, 10)
    x[0, 0, :5] = np.nan
    y = rescale_bands(x)
    assert np.isnan(y[0, 0, :5]).all() and np.isfinite(y[0, 1:]).all()
    assert np.nanmin(y) == pytest.approx(0) and np.nanmax(y) == pytest.approx(255)
    assert np.isnan(rescale_bands(np.full((1, 4, 4), np.nan))).all()


# ------------------------------------------------------------- segmentation
@pytest.mark.parametrize("method,kw", [("slic", {"n_segments": 120}),
                                       ("felzenszwalb", {"scale": 60}),
                                       ("multires", {"scale": 25})])
def test_segmentation_marks_holes_as_zero(method, kw):
    _, rd, hole = _scene_with_holes()
    seg = obia.segment(rd, method, **kw)
    lab = seg.labels
    assert ((lab == 0) == hole).all(), "label 0 must be exactly the invalid pixels"
    ids = np.unique(lab[lab > 0])
    assert ids.min() == 1 and ids.max() == len(ids), "ids must run 1..n with no gaps"
    assert seg.valid is not None and (seg.valid == ~hole).all()
    # every id is one connected piece
    from scipy import ndimage as ndi
    for i in ids[:40]:
        assert ndi.label(lab == i)[1] == 1
    assert not seg.boundaries()[hole].any()


def test_segmentation_without_holes_has_no_valid_mask():
    sc = block_scene(96)
    seg = obia.segment(sc.raster, "slic", n_segments=60)
    assert seg.valid is None and (seg.labels > 0).all()


def test_all_nan_image_is_refused():
    rd = obia.RasterData(np.full((3, 20, 20), np.nan, np.float32), ["red", "green", "blue"])
    with pytest.raises(ValueError, match="no valid pixels"):
        obia.segment(rd, "slic", n_segments=4)


def test_segment_levels_and_scale_tools_survive_holes():
    _, rd, hole = _scene_with_holes()
    levels = obia.segment_levels(rd, scales=[15, 40])
    for lv in levels.values():
        assert ((lv.labels == 0) == hole).all()
    assert levels[40.0].labels.max() <= levels[15.0].labels.max()
    diag = obia.scale_diagnostics(rd, scales=[10, 20, 40])
    assert list(diag.index) == [10.0, 20.0, 40.0]
    assert (diag["n_objects"].diff().dropna() <= 0).all(), "coarser scale -> no more objects"
    assert np.isfinite(diag["lv"]).all()


def test_relabel_is_fast_with_many_objects():
    from pygeovision.obia.segmentation import _relabel
    rng = np.random.default_rng(0)
    lab = rng.integers(0, 4000, size=(400, 400))          # ~4000 ids, heavily fragmented
    t0 = time.time()
    out = _relabel(lab)
    assert time.time() - t0 < 20
    # fragmenting must keep every output id connected and ids contiguous
    assert out.min() == 1 and out.max() == len(np.unique(out))


def test_from_labels_still_treats_zero_as_an_id():
    lab = np.zeros((4, 4), int); lab[:, 2:] = 1
    rd = obia.RasterData(np.ones((1, 4, 4), np.float32), ["x"])
    seg = obia.from_labels(lab, rd)
    assert seg.labels.min() >= 1 and seg.labels.max() == 2


# ----------------------------------------------------------------- features
def test_features_with_holes_exclude_label_zero():
    _, rd, hole = _scene_with_holes()
    seg = obia.segment(rd, "slic", n_segments=100)
    f = obia.extract_features(seg.labels, rd, texture_layers=["nir"], transform=rd.transform)
    assert (f.index > 0).all() and len(f) == seg.labels.max()
    assert f["mean_nir"].notna().all(), "means are taken over valid pixels only"
    # area_m2 is the pixel count * 4 m2 (2 m pixels); the holes are in no object
    assert f["area_m2"].sum() == pytest.approx((~hole).sum() * 4.0)
    assert f["n_neighbors"].min() >= 0


def test_context_does_not_count_nodata_as_a_neighbour():
    lab = np.array([[1, 1, 0, 2, 2],
                    [1, 1, 0, 2, 2]])
    cls = pd.Series({1: "a", 2: "b"})
    ctx = obia.class_context(lab, cls, ["a", "b"])
    # 1 and 2 never touch (a column of nodata separates them)
    assert ctx.loc[1, "n_nb_b"] == 0 and ctx.loc[2, "n_nb_a"] == 0
    assert (ctx.loc[:, [c for c in ctx if c.startswith("n_nb")]].sum().sum()) == 0
    assert set(ctx.index) == {1, 2}


def test_glcm_rejects_layer_without_finite_pixels():
    lab = np.ones((6, 6), int)
    with pytest.raises(ValueError):
        obia.glcm_texture(lab, np.full((6, 6), np.nan))


# --------------------------------------------------------------- samples / CRS
def _grid():
    tr = Affine(10.0, 0, 500000.0, 0, -10.0, 700000.0)
    lab = np.zeros((20, 20), int)
    lab[:, :10] = 1; lab[:, 10:] = 2
    lab[0:4, :] = 0                                         # no-data band
    return tr, lab


def test_samples_reproject_to_image_crs():
    tr, lab = _grid()
    poly = gpd.GeoDataFrame({"cls": ["x"]}, geometry=[box(500020, 699900, 500080, 699950)], crs="EPSG:32630")
    ll = poly.to_crs(4326)
    s = obia.samples_from_geometries(lab, ll, "cls", tr, min_coverage=0.1, crs="EPSG:32630")  # polygon covers 30 of 160 px
    assert s.to_dict() == {1: "x"}


def test_samples_disjoint_extent_is_a_clear_error():
    tr, lab = _grid()
    poly = gpd.GeoDataFrame({"cls": ["x"]}, geometry=[box(-2, 6, -1.9, 6.1)], crs="EPSG:4326")
    with pytest.raises(ValueError, match="coordinate-system mismatch"):
        obia.samples_from_geometries(lab, poly, "cls", tr)          # degrees against metres
    with pytest.raises(ValueError, match="no coordinate system"):
        obia.samples_from_geometries(lab, poly.set_crs(None, allow_override=True), "cls", tr, crs="EPSG:32630")


def test_object_zero_is_never_a_sample():
    tr, lab = _grid()
    poly = gpd.GeoDataFrame({"cls": ["x"]}, geometry=[box(500000, 699960, 500200, 700000)], crs="EPSG:32630")
    # the polygon lies wholly on the no-data band (rows 0-3 -> y 699960..700000)
    s = obia.samples_from_geometries(lab, poly, "cls", tr)
    assert len(s) == 0 and 0 not in s.index


# ---------------------------------------------------------------- accuracy
def test_segmentation_quality_ignores_label_zero():
    seg = np.zeros((6, 6), int); seg[:, 3:] = 1               # left half is no-data, right half segment 1
    ref = np.zeros((6, 6), int); ref[:, 3:] = 1; ref[0, 0] = 2  # ref 2 sits only on no-data
    q, summ = obia.segmentation_quality(seg, ref)
    assert list(q.index) == [1]                              # ref 2 has no segment to match
    assert q.loc[1, "QR"] == pytest.approx(1.0)
    assert summ["n_segments_total"] == 1
    with pytest.raises(ValueError):
        obia.segmentation_quality(np.zeros((3, 3), int), np.ones((3, 3), int))


# ------------------------------------------------------- weak labels from a map
def test_samples_from_raster_purity_mapping_and_nodata():
    lab = np.zeros((4, 8), int)
    lab[:, 0:2] = 1                      # all code 10                 -> forest, purity 1
    lab[:, 2:4] = 2                      # 6 px code 40 + 2 px code 10  -> farmland, purity 0.75
    lab[:, 4:6] = 3                      # 4 px 10 + 4 px 40            -> tie below 0.6 -> dropped
    lab[:, 6:8] = 4                      # 5 px code 90 (unmapped) + 3 px 10 -> nothing wins
    ras = np.zeros((4, 8), int)
    ras[:, 0:2] = 10
    ras[:, 2:4] = 40; ras[0, 2:4] = 10
    ras[:, 4] = 10; ras[:, 5] = 40
    ras[:, 6:8] = 90; ras[0, 6:8] = 10; ras[1, 6] = 10
    out = obia.samples_from_raster(lab, ras, {10: "forest", 40: "farmland"}, min_purity=0.6)
    assert out["class"].to_dict() == {1: "forest", 2: "farmland"}
    assert out.loc[2, "purity"] == pytest.approx(0.75) and out.loc[1, "n_px"] == 8
    # nodata pixels leave the denominator: object 5 is half nodata but pure in what remains
    lab2 = np.ones((2, 4), int); ras2 = np.array([[10, 10, 0, 0], [10, 10, 0, 0]])
    o2 = obia.samples_from_raster(lab2, ras2, {10: "forest"})
    assert o2.loc[1, "purity"] == 1.0 and o2.loc[1, "n_px"] == 4
    # label 0 is never an object
    lab3 = np.zeros((2, 2), int)
    assert obia.samples_from_raster(lab3, np.full((2, 2), 10)).empty
    with pytest.raises(ValueError, match="same grid"):
        obia.samples_from_raster(np.ones((2, 2), int), np.ones((3, 3), int))
    with pytest.raises(ValueError, match="min_purity"):
        obia.samples_from_raster(lab, ras, min_purity=0)


# ------------------------------------------------------------ OSM polygons
_OVERPASS = {"elements": [
    {"type": "way", "id": 1, "tags": {"landuse": "quarry", "name": "Pit A", "resource": "gold"},
     "geometry": [{"lat": 6.20, "lon": -1.70}, {"lat": 6.20, "lon": -1.69}, {"lat": 6.21, "lon": -1.69},
                  {"lat": 6.21, "lon": -1.70}, {"lat": 6.20, "lon": -1.70}]},
    {"type": "way", "id": 2, "tags": {"landuse": "quarry"},                       # open line: not an area
     "geometry": [{"lat": 6.2, "lon": -1.6}, {"lat": 6.21, "lon": -1.6}, {"lat": 6.21, "lon": -1.59}]},
    {"type": "relation", "id": 3, "tags": {"industrial": "mine", "type": "multipolygon"}, "members": [
        {"type": "way", "role": "outer", "geometry": [{"lat": 6.0, "lon": -1.0}, {"lat": 6.0, "lon": -0.9},
                                                       {"lat": 6.1, "lon": -0.9}, {"lat": 6.1, "lon": -1.0},
                                                       {"lat": 6.0, "lon": -1.0}]},
        {"type": "way", "role": "inner", "geometry": [{"lat": 6.04, "lon": -0.98}, {"lat": 6.04, "lon": -0.94},
                                                       {"lat": 6.06, "lon": -0.94}, {"lat": 6.06, "lon": -0.98},
                                                       {"lat": 6.04, "lon": -0.98}]}]},
    {"type": "node", "id": 4, "tags": {"landuse": "quarry"}},
]}


def test_fetch_osm_polygons_builds_polygons_holes_and_skips_lines(monkeypatch):
    from pygeovision.labeling.osm import OSMLabeler
    seen = {}

    def fake(self, query):
        seen["query"] = query
        return _OVERPASS
    monkeypatch.setattr(OSMLabeler, "_fetch_overpass", fake)
    gdf = obia.fetch_osm_polygons((-1.8, 5.9, -0.8, 6.3))
    assert gdf.crs == "EPSG:4326" and list(gdf["osm_id"]) == [1, 3]           # the open way and the node are skipped
    assert gdf.loc[0, "name"] == "Pit A" and gdf.loc[0, "resource"] == "gold" and gdf.loc[1, "industrial"] == "mine"
    assert gdf.geometry.iloc[0].area == pytest.approx(0.01 * 0.01)             # 0.01 deg x 0.01 deg
    assert gdf.geometry.iloc[1].area == pytest.approx(0.1 * 0.1 - 0.04 * 0.02)  # outer minus the hole
    q = seen["query"]
    assert '(5.9,-1.8,6.3,-0.8)' in q and 'way["landuse"="quarry"]' in q and 'relation["industrial"="mine"]' in q
    assert q.rstrip().endswith("out body geom;")
    # custom filters replace the defaults
    obia.fetch_osm_polygons((-1.8, 5.9, -0.8, 6.3), filters=('["natural"="water"]',))
    assert 'way["natural"="water"]' in seen["query"] and "quarry" not in seen["query"]


def test_fetch_osm_polygons_empty_failure_and_bad_bbox(monkeypatch):
    from pygeovision.labeling.osm import OSMLabeler
    monkeypatch.setattr(OSMLabeler, "_fetch_overpass", lambda self, q: {"elements": []})
    gdf = obia.fetch_osm_polygons((0, 0, 1, 1))
    assert len(gdf) == 0 and gdf.crs == "EPSG:4326" and "osm_id" in gdf.columns      # "nothing mapped" is an answer

    def failing(self, q):
        self._last_error = "HTTP 504 from overpass"
        return None
    monkeypatch.setattr(OSMLabeler, "_fetch_overpass", failing)
    with pytest.raises(RuntimeError, match="HTTP 504"):
        obia.fetch_osm_polygons((0, 0, 1, 1))
    monkeypatch.setattr(OSMLabeler, "_fetch_overpass", lambda self, q: {"elements": [], "remark": "runtime error: timeout"})
    with pytest.raises(RuntimeError, match="timeout"):
        obia.fetch_osm_polygons((0, 0, 1, 1))
    with pytest.raises(ValueError, match="min_lon"):
        obia.fetch_osm_polygons((1, 0, 0, 1))


# --------------------------------------------------------------- client.obia
def test_client_obia_proxy_passes_everything_through(monkeypatch):
    from pygeovision import _ObiaClientProxy
    seen = {}
    monkeypatch.setattr(obia, "fetch_scene", lambda client, bbox, date, out_dir, cloud_cover_max=30.0, **kw:
                        seen.update(scene=(client, bbox, date, str(out_dir), cloud_cover_max, kw)) or "RD")
    monkeypatch.setattr(obia, "fetch_worldcover", lambda client, bbox, out, like=None: seen.update(wc=(bbox, like)) or ("A", out))
    monkeypatch.setattr(obia, "fetch_osm_polygons", lambda bbox, filters, **kw: seen.update(osm=(bbox, filters, kw)) or "G")
    me = object()
    px = _ObiaClientProxy(me)
    assert px.scene((1, 2, 3, 4), "2024-01", "d", bands=["red", "nir"], cloud_cover_max=10, providers=["x"]) == "RD"
    assert seen["scene"] == (me, (1, 2, 3, 4), "2024-01", "d", 10, {"bands": ("red", "nir"), "providers": ["x"]})
    assert px.worldcover((1, 2, 3, 4), like="L")[0] == "A" and seen["wc"] == ((1, 2, 3, 4), "L")
    assert px.osm_polygons((1, 2, 3, 4), timeout=5) == "G"
    assert seen["osm"] == ((1, 2, 3, 4), obia.scene.MINE_OSM_FILTERS, {"timeout": 5})
    assert "scene|worldcover|osm_polygons" in repr(px)


def test_the_package_has_no_synthetic_scene_generators():
    for name in ("make_demo_scene", "make_mine_scene", "DemoScene", "MineScene", "CLASS_NAMES"):
        assert not hasattr(obia, name), name
    import importlib
    for mod in ("demo", "mine_demo"):
        with pytest.raises(ImportError):
            importlib.import_module(f"pygeovision.obia.{mod}")


# -------------------------------------------------------------- live loaders
class _FakeLabeling:
    def __init__(self, tmp, ok=True):
        self.tmp, self.ok, self.kw = tmp, ok, None

    def esa_worldcover(self, bbox, output_path, **kw):
        import rasterio
        self.kw = kw
        if not self.ok:
            return {"success": False, "error": "STAC unreachable"}
        with rasterio.open(kw["reference_raster"]) as src:
            prof = src.profile
        with rasterio.open(output_path, "w", **prof) as dst:
            dst.write(np.full((prof["height"], prof["width"]), 40, np.uint8), 1)
        return {"success": True, "output_path": output_path}


def test_fetch_scene_returns_nan_masked_raster_or_a_clear_error(tmp_path, monkeypatch):
    from pygeovision.ai.pipelines import BasePipeline
    from pygeovision.obia import save_raster
    img = np.random.default_rng(0).uniform(0.02, 0.3, (6, 12, 12)).astype(np.float32)
    img[:, :3, :3] = np.nan
    tif = tmp_path / "s2.tif"
    save_raster(str(tif), img, Affine(10, 0, 640000, 0, -10, 690000), "EPSG:32630")
    seen = {}

    def fake(self, bbox, date, out, **kw):
        seen.update(bbox=bbox, date=date, **kw)
        return tif
    monkeypatch.setattr(BasePipeline, "_search_and_download", fake)
    rd = obia.fetch_scene(object(), (-1.7, 6.18, -1.65, 6.22), "2024-01", tmp_path / "dl", cloud_cover_max=15)
    assert rd.band_names == list(obia.scene.DEFAULT_BANDS) and rd.image.shape == (6, 12, 12)
    assert (~obia.valid_mask(rd)).sum() == 9, "cloud holes stay NaN"
    assert seen["bands"] == tuple(obia.scene.DEFAULT_BANDS) and seen["cloud_cover_max"] == 15
    assert seen["bbox"] == (-1.7, 6.18, -1.65, 6.22)

    monkeypatch.setattr(BasePipeline, "_search_and_download", lambda self, *a, **k: None)
    with pytest.raises(RuntimeError, match="no usable Sentinel-2 scene"):
        obia.fetch_scene(object(), (-1.7, 6.18, -1.65, 6.22), "2024-01", tmp_path / "dl")


def test_fetch_worldcover_aligns_to_the_image_grid(tmp_path):
    rd = obia.RasterData(np.zeros((1, 7, 9), np.float32), ["x"], Affine(10, 0, 640000, 0, -10, 690000), "EPSG:32630")

    class Client:
        labeling = _FakeLabeling(tmp_path)
    arr, path = obia.fetch_worldcover(Client(), (-1.7, 6.18, -1.65, 6.22), tmp_path / "wc" / "wc.tif", like=rd)
    assert arr.shape == (7, 9) and (arr == 40).all() and path.exists()
    assert "reference_raster" in Client.labeling.kw

    class Bad:
        labeling = _FakeLabeling(tmp_path, ok=False)
    with pytest.raises(RuntimeError, match="STAC unreachable"):
        obia.fetch_worldcover(Bad(), (-1.7, 6.18, -1.65, 6.22), tmp_path / "wc.tif")


# --------------------------------------------- progress, heartbeat and failure reasons
@pytest.fixture(autouse=True)
def _default_logging(monkeypatch):
    """Other tests (and the client) configure the root logger; these tests need the library's default state."""
    import logging
    root = logging.getLogger()
    monkeypatch.setattr(root, "level", logging.WARNING)
    monkeypatch.setattr(root, "handlers", [h for h in root.handlers if "pytest" in type(h).__module__ or "LogCapture" in type(h).__name__])
    for n in ("pygeovision", "pygeofetch"):
        monkeypatch.setattr(logging.getLogger(n), "level", logging.NOTSET)


def _tif(tmp_path):
    from pygeovision.obia import save_raster
    img = np.random.default_rng(0).uniform(0.02, 0.3, (6, 12, 12)).astype(np.float32)
    tif = tmp_path / "s2.tif"
    save_raster(str(tif), img, Affine(10, 0, 640000, 0, -10, 690000), "EPSG:32630")
    return tif


def test_fetch_scene_reports_progress_and_the_library_messages(tmp_path, monkeypatch, capsys):
    """A download used to be silent: nothing in the output said what was happening, so it looked frozen."""
    import logging

    from pygeovision.ai.pipelines import BasePipeline
    tif = _tif(tmp_path)

    def slow(self, bbox, date, out, **kw):
        logging.getLogger("pygeovision.data.fetch").info("Searching 1 provider(s) [planetary_computer]")
        (out / "imagery").mkdir(exist_ok=True)
        (out / "imagery" / "B04.tif").write_bytes(b"x" * 2_000_000)     # bytes arriving
        time.sleep(0.35)
        logging.getLogger("pygeovision.ai.pipelines").info("downloading scene S2_X")
        return tif
    monkeypatch.setattr(BasePipeline, "_search_and_download", slow)
    obia.fetch_scene(object(), (-1.7, 6.18, -1.65, 6.22), "2024-01", tmp_path / "dl", heartbeat=0.1)
    out = capsys.readouterr().out
    assert "Sentinel-2 L2A, bbox=" in out and "search and download ..." in out          # what is about to happen
    assert "Searching 1 provider(s)" in out and "downloading scene S2_X" in out          # the library's own INFO lines
    assert "still working" in out and "MB received so far" in out                        # heartbeat with the bytes
    assert "search and download: done after" in out and "scene ready: 12 x 12 px" in out


def test_a_stalled_download_says_so_in_the_heartbeat(tmp_path, monkeypatch, capsys):
    from pygeovision.ai.pipelines import BasePipeline
    tif = _tif(tmp_path)

    def stalled(self, bbox, date, out, **kw):
        time.sleep(0.5)                                  # no bytes ever arrive
        return tif
    monkeypatch.setattr(BasePipeline, "_search_and_download", stalled)
    obia.fetch_scene(object(), (-1.7, 6.18, -1.65, 6.22), "2024-01", tmp_path / "dl", heartbeat=0.05)
    assert "nothing new for" in capsys.readouterr().out


def test_a_failed_fetch_names_the_reason_the_download_code_logged(tmp_path, monkeypatch, capsys):
    import logging

    from pygeovision.ai.pipelines import BasePipeline

    def failing(self, bbox, date, out, **kw):
        logging.getLogger("pygeovision.ai.pipelines").warning("the search found no scene for bbox=X")
        logging.getLogger("pygeovision.ai.pipelines").error("_search_and_download failed: provider timed out")
        return None
    monkeypatch.setattr(BasePipeline, "_search_and_download", failing)
    for verbose in (True, False):
        with pytest.raises(RuntimeError) as ei:
            obia.fetch_scene(object(), (-1.7, 6.18, -1.65, 6.22), "2024-01", tmp_path / "dl", verbose=verbose)
        assert "no scene for bbox=X" in str(ei.value) and "provider timed out" in str(ei.value)
    out = capsys.readouterr().out
    assert out.count("FAILED after") == 1, "only the verbose call prints"
    assert "ERROR: _search_and_download failed: provider timed out" in out


def test_verbose_false_is_silent_and_logging_is_restored(tmp_path, monkeypatch, capsys):
    import logging

    from pygeovision.ai.pipelines import BasePipeline
    tif = _tif(tmp_path)
    monkeypatch.setattr(BasePipeline, "_search_and_download",
                        lambda self, *a, **k: logging.getLogger("pygeovision.x").info("chatter") or tif)
    before = {n: (list(logging.getLogger(n).handlers), logging.getLogger(n).level) for n in ("pygeovision", "pygeofetch")}
    obia.fetch_scene(object(), (-1.7, 6.18, -1.65, 6.22), "2024-01", tmp_path / "dl", verbose=False)
    assert capsys.readouterr().out == ""
    obia.fetch_scene(object(), (-1.7, 6.18, -1.65, 6.22), "2024-01", tmp_path / "dl")
    monkeypatch.setattr(BasePipeline, "_search_and_download", lambda self, *a, **k: None)
    with pytest.raises(RuntimeError):
        obia.fetch_scene(object(), (-1.7, 6.18, -1.65, 6.22), "2024-01", tmp_path / "dl")
    after = {n: (list(logging.getLogger(n).handlers), logging.getLogger(n).level) for n in ("pygeovision", "pygeofetch")}
    assert after == before, "handlers and levels are put back, also after a failure"


def test_user_logging_is_not_echoed_twice(tmp_path, monkeypatch, capsys):
    import logging

    from pygeovision.ai.pipelines import BasePipeline
    tif = _tif(tmp_path)
    monkeypatch.setattr(BasePipeline, "_search_and_download",
                        lambda self, *a, **k: logging.getLogger("pygeovision.y").info("only-once") or tif)
    root, seen = logging.getLogger(), []

    class Keep(logging.Handler):
        def emit(self, record):
            seen.append(record.getMessage())
    h, old = Keep(), root.level
    root.addHandler(h)
    root.setLevel(logging.INFO)
    logging.getLogger("pygeovision").setLevel(logging.INFO)
    try:
        obia.fetch_scene(object(), (-1.7, 6.18, -1.65, 6.22), "2024-01", tmp_path / "dl")
    finally:
        root.removeHandler(h)
        root.setLevel(old)
        logging.getLogger("pygeovision").setLevel(logging.NOTSET)
    assert seen.count("only-once") == 1 and "only-once" not in capsys.readouterr().out


def test_worldcover_and_osm_report_progress_and_bound_their_waits(tmp_path, monkeypatch, capsys):
    rd = obia.RasterData(np.zeros((1, 7, 9), np.float32), ["x"], Affine(10, 0, 640000, 0, -10, 690000), "EPSG:32630")

    class Client:
        labeling = _FakeLabeling(tmp_path)
    obia.fetch_worldcover(Client(), (-1.7, 6.18, -1.65, 6.22), tmp_path / "wc.tif", like=rd)
    out = capsys.readouterr().out
    assert "ESA WorldCover download ..." in out and "WorldCover ready: 7 x 9 px, class codes {40: 63}" in out

    from pygeovision.labeling.osm import OSMLabeler
    got = {}

    def fake(self, query):
        got.update(timeout=self.timeout, retries=self.retry_attempts)
        self._last_error = "HTTP 504 from overpass"
        return None
    monkeypatch.setattr(OSMLabeler, "_fetch_overpass", fake)
    with pytest.raises(RuntimeError, match="HTTP 504"):
        obia.fetch_osm_polygons((0, 0, 1, 1))
    assert got == {"timeout": 60, "retries": 2}, "worst case is about timeout * 3 mirrors * retries = 6 minutes"
    assert "OpenStreetMap (Overpass) request ..." in capsys.readouterr().out


def test_a_search_that_finds_nothing_says_so(tmp_path, caplog):
    """`_search_and_download` returned None without a word when the search came back empty."""
    import logging
    class Client:
        def search(self, **kw):
            return []
    with caplog.at_level(logging.WARNING, logger="pygeovision"):
        with pytest.raises(RuntimeError) as ei:
            obia.fetch_scene(Client(), (-1.7, 6.18, -1.65, 6.22), "2024-01", tmp_path / "dl", verbose=False)
    assert "the search found no scene" in str(ei.value) and "cloud cover <= 30" in str(ei.value)


def test_an_empty_search_blames_the_cloud_limit_when_that_is_the_reason(tmp_path):
    """Rainy season: 54 scenes found by the provider, all cloudier than the limit, 0 results. The error said only
    'no scene'; it now says which cloud cover to raise the limit above."""
    from types import SimpleNamespace
    calls = []

    class Client:
        def search(self, **kw):
            calls.append(kw["cloud_cover_max"])
            if kw["cloud_cover_max"] >= 100:
                return [SimpleNamespace(cloud_cover=c) for c in (63.2, 41.7, 88.0)]
            return []
    with pytest.raises(RuntimeError) as ei:
        obia.fetch_scene(Client(), (-2.5, 6.8, -2.0, 7.2), "2025-06", tmp_path / "dl", cloud_cover_max=10, verbose=False)
    msg = str(ei.value)
    assert calls == [10, 100]
    assert "the cloud limit is the reason" in msg and "3 scene(s) exist" in msg and "clearest has 42%" in msg
    assert "at least 42" in msg


def test_the_cli_search_call_has_a_timeout():
    """`_run_cli` had no timeout: an unreachable provider hung the caller for ever."""
    import subprocess

    import pygeovision.data.fetch as fetch
    f = object.__new__(fetch.SatelliteFetcher)
    old = fetch._PYGEOFETCH_CLI_EXE
    fetch._PYGEOFETCH_CLI_EXE = "sleep"
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            f._run_cli(["5"], timeout=0.2)
    finally:
        fetch._PYGEOFETCH_CLI_EXE = old


# ------------------------------------------------------------ CLI entry points
def test_python_dash_m_runs_every_command_group():
    """`python -m pygeovision.cli.main` used to stop before the later command groups (obia ...) were registered."""
    import subprocess
    import sys
    r = subprocess.run([sys.executable, "-m", "pygeovision.cli.main", "obia", "--help"], capture_output=True, text=True,
                       env={**__import__("os").environ, "PYTHONPATH": "."})
    assert r.returncode == 0, r.stderr[-400:]
    assert "segment" in r.stdout and "assess-classification" in r.stdout
