# ruff: noqa: E702, N802, C408, E731
"""docs/_examples/foundation/embedding_workflow.py and the TESSERA wrapper, on stand-in data (no network)."""
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("embedding_workflow", ROOT / "docs/_examples/foundation/embedding_workflow.py")
wf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wf)


def _field(h=20, w=30, c=128, seed=0):
    """Left half one cluster, right half another, plus noise."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(2, c))
    emb = np.empty((c, h, w), dtype=np.float32)
    truth = np.zeros((h, w), dtype=np.int32)
    truth[:, w // 2:] = 1
    for k in (0, 1):
        emb[:, truth == k] = (centres[k][:, None] + 0.1 * rng.normal(size=(c, int((truth == k).sum())))).astype(np.float32)
    return emb, truth


def test_classify_from_a_few_points_recovers_the_two_regions():
    emb, truth = _field()
    points = [(2, 3), (10, 5), (17, 8), (3, 22), (12, 25), (18, 28)]
    labels = [int(truth[r, c]) for r, c in points]
    pred = wf.classify_pixels(emb, points, labels)
    assert pred.shape == truth.shape and (pred == truth).mean() > 0.99


def test_nan_pixels_get_minus_one_and_nan_training_points_are_refused():
    emb, truth = _field()
    emb[:, 0, 0] = np.nan
    pts = [(2, 3), (10, 5), (3, 22), (12, 25)]
    pred = wf.classify_pixels(emb, pts, [int(truth[r, c]) for r, c in pts])
    assert pred[0, 0] == -1
    with pytest.raises(ValueError, match="without an embedding"):
        wf.classify_pixels(emb, [(0, 0), (3, 22)], [0, 1])


def test_similarity_map_is_one_at_the_reference_and_high_in_its_region():
    emb, truth = _field()
    sim = wf.similarity_map(emb, 5, 5)
    assert sim.shape == truth.shape and abs(sim[5, 5] - 1.0) < 1e-5
    assert sim[truth == 0].mean() > sim[truth == 1].mean() + 0.3


@pytest.fixture
def fake_geotessera(monkeypatch):
    emb, _ = _field(h=12, w=14)
    emb_hwc = np.transpose(emb, (1, 2, 0))
    calls = {}

    class GeoTessera:
        def __init__(self, dataset_version="v1", cache_dir=None):
            calls["init"] = (dataset_version, cache_dir)

        def fetch_mosaic_for_region(self, bbox, year, target_crs="EPSG:4326"):
            from rasterio.transform import from_bounds
            calls["fetch"] = (bbox, year, target_crs)
            return emb_hwc, from_bounds(*bbox, 14, 12), "EPSG:4326"

        def sample_embeddings_at_points(self, points, year):
            return np.zeros((len(points), 128), dtype=np.float32)

    mod = types.ModuleType("geotessera")
    mod.GeoTessera = GeoTessera
    monkeypatch.setitem(sys.modules, "geotessera", mod)
    return calls


def test_tessera_wrapper_returns_channel_first_array_and_a_georeferenced_tif(tmp_path, fake_geotessera):
    import rasterio

    from pygeovision.models.foundation.tessera import TesseraGeo

    out = TesseraGeo().embeddings_for_bbox((0.20, 6.70, 0.45, 6.95), year=2023, output_path=str(tmp_path / "t.tif"))
    assert out["success"] and out["shape"] == (128, 12, 14) and out["n_channels"] == 128 and out["embedding"].shape == (128, 12, 14)
    assert fake_geotessera["fetch"] == ((0.20, 6.70, 0.45, 6.95), 2023, "EPSG:4326")
    with rasterio.open(tmp_path / "t.tif") as src:
        assert src.count == 128 and (src.height, src.width) == (12, 14)
        assert src.tags()["source"] == "TESSERA" and src.tags()["year"] == "2023"
        assert abs(src.bounds.left - 0.20) < 1e-6 and abs(src.bounds.top - 6.95) < 1e-6


def test_tessera_pca_picture_is_three_uint8_bands_with_variance_ratios(tmp_path, fake_geotessera):
    import rasterio

    from pygeovision.models.foundation.tessera import TesseraGeo

    res = TesseraGeo().pca_visualisation((0.20, 6.70, 0.45, 6.95), str(tmp_path / "pca.tif"))
    assert res["success"] and len(res["explained_variance_ratio"]) == 3 and res["explained_variance_ratio"][0] > 0.5
    with rasterio.open(tmp_path / "pca.tif") as src:
        assert src.count == 3 and src.dtypes[0] == "uint8"


def test_tessera_without_the_package_says_what_to_install(monkeypatch):
    from pygeovision.models.foundation.tessera import TesseraGeo
    monkeypatch.setitem(sys.modules, "geotessera", None)
    with pytest.raises(ImportError, match="pip install geotessera"):
        TesseraGeo().embeddings_at_points([(0, 0)])


def test_cosine_similarity_and_nearest_refuse_to_mix_dimensions_and_rank_best_first():
    from pygeovision.models.embeddings import cosine_similarity, nearest

    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0) and cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([0, 0], [1, 0]) == 0.0
    with pytest.raises(ValueError, match="different real dimensions"):
        cosine_similarity(np.ones(128), np.ones(64))
    ranked = nearest(np.array([1.0, 0.0]), {"a": np.array([0.0, 1.0]), "b": np.array([1.0, 0.1]), "c": np.array([-1.0, 0.0])}, top_k=2)
    assert [n for n, _ in ranked] == ["b", "a"]


def _fake_ee(monkeypatch, features=None, fail_export=False):
    ee = types.ModuleType("ee")
    log = {}

    class _Img:
        def filterDate(self, *a): log["dates"] = a; return self
        def filterBounds(self, *a): return self
        def mosaic(self): return self
        def clip(self, *a): return self
        def sampleRegions(self, collection, scale, geometries): log["scale"] = scale; return self
        def getInfo(self): return {"features": features or []}

    ee.ImageCollection = lambda name: (log.__setitem__("collection", name), _Img())[1]
    ee.Feature = lambda geom, props: props
    ee.FeatureCollection = lambda feats: feats
    ee.Geometry = types.SimpleNamespace(Point=lambda c: c, Rectangle=lambda c: c)

    class _Task:
        id = "TASK123"
        def start(self): log["started"] = True

    ee.batch = types.SimpleNamespace(Export=types.SimpleNamespace(image=types.SimpleNamespace(toDrive=lambda **kw: (log.__setitem__("export", kw), _Task())[1])))
    monkeypatch.setitem(sys.modules, "ee", ee)
    return log


def test_alphaearth_points_come_back_in_order_with_nan_rows_for_missing_points(monkeypatch):
    from pygeovision.models.foundation.alphaearth_geo import BAND_NAMES, AlphaEarthGeo

    feats = [{"idx": 1, **{b: float(i) for i, b in enumerate(BAND_NAMES)}}]
    feats = [{"properties": f} for f in feats]
    log = _fake_ee(monkeypatch, features=feats)
    out = AlphaEarthGeo().embeddings_at_points([(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)], year=2022)
    assert out.shape == (3, 64) and np.isnan(out[0]).all() and np.isnan(out[2]).all()
    assert out[1, 5] == 5.0 and log["collection"] == "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL" and log["dates"] == ("2022-01-01", "2022-12-31")


def test_alphaearth_region_export_returns_a_task_not_a_raster(monkeypatch):
    from pygeovision.models.foundation.alphaearth_geo import AlphaEarthGeo

    log = _fake_ee(monkeypatch)
    res = AlphaEarthGeo().embeddings_for_bbox((0, 0, 1, 1), year=2024, drive_folder="pgv")
    assert res["success"] and res["ee_task_id"] == "TASK123" and res["status"] == "started" and res["n_bands"] == 64
    assert log["started"] and log["export"]["scale"] == 10 and log["export"]["folder"] == "pgv"


def test_alphaearth_without_the_package_says_what_to_install(monkeypatch):
    from pygeovision.models.foundation.alphaearth_geo import AlphaEarthGeo

    monkeypatch.setitem(sys.modules, "ee", None)
    assert AlphaEarthGeo().embeddings_for_bbox((0, 0, 1, 1)) == {"success": False, "error": "pip install earthengine-api"}
    with pytest.raises(ImportError, match="earthengine-api"):
        AlphaEarthGeo().embeddings_at_points([(0, 0)])


def test_the_documented_end_to_end_recipe_runs_on_the_geotiff_the_wrapper_writes(tmp_path, fake_geotessera):
    """Mirrors the worked example on docs/foundation-models/embeddings.md."""
    import rasterio
    from sklearn.metrics import accuracy_score

    from pygeovision.models.foundation.tessera import TesseraGeo

    bbox = (0.20, 6.70, 0.45, 6.95)
    TesseraGeo().embeddings_for_bbox(bbox, year=2024, output_path=str(tmp_path / "tessera.tif"))
    with rasterio.open(tmp_path / "tessera.tif") as src:
        emb = src.read()                                              # (128, H, W)
        lon = np.linspace(0.21, 0.44, 12); lat = np.linspace(6.71, 6.94, 12)
        rows_cols = [src.index(x, y) for x, y in zip(lon, lat)]
        truth_by_col = [0 if c < 7 else 1 for _, c in rows_cols]
    train = list(range(0, 12, 2)); held = list(range(1, 12, 2))
    pred = wf.classify_pixels(emb, [rows_cols[i] for i in train], [truth_by_col[i] for i in train])
    acc = accuracy_score([truth_by_col[i] for i in held], [int(pred[rows_cols[i]]) for i in held])
    assert acc > 0.9
