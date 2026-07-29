"""
tests/test_viz.py
==================
Tests for pygeovision.viz — Map, RasterViewer, VectorViewer, ChangeViewer, TimeSeriesViewer.

All tests run on synthetic rasters / GeoJSON; no real satellite files required.
"""
from __future__ import annotations

import json
import pathlib
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))


# ── synthetic raster helpers ────────────────────────────────────────────────

def make_raster(path: str, width: int = 64, height: int = 64,
                bands: int = 6, crs: str = "EPSG:4326") -> str:
    try:
        import rasterio
        from rasterio.transform import from_bounds
    except ImportError:
        pytest.skip("rasterio not installed")

    data = np.random.uniform(0.01, 0.9, (bands, height, width)).astype("float32")
    transform = from_bounds(-0.3, 5.5, -0.05, 5.7, width, height)
    profile = {
        "driver": "GTiff", "dtype": "float32", "width": width, "height": height,
        "count": bands, "crs": rasterio.crs.CRS.from_epsg(4326),
        "transform": transform,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
    return path


def make_geojson(path: str, n: int = 5) -> str:
    features = []
    for i in range(n):
        lon, lat = -0.2 + i * 0.02, 5.6 + i * 0.01
        features.append({
            "type": "Feature",
            "properties": {"id": i, "area": (i + 1) * 100.0, "name": f"Feature {i}"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[lon, lat], [lon+0.01, lat], [lon+0.01, lat+0.01],
                                  [lon, lat+0.01], [lon, lat]]]
            }
        })
    gj = {"type": "FeatureCollection", "features": features}
    with open(path, "w") as f:
        json.dump(gj, f)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# 1. Map
# ══════════════════════════════════════════════════════════════════════════════

class TestMap:

    def test_map_constructs(self):
        from pygeovision.viz import Map
        m = Map(center=(5.6, -0.2), zoom=12, basemap="streets")
        assert m._center == (5.6, -0.2)
        assert m._zoom == 12
        assert m._basemap == "streets"

    def test_map_default_values(self):
        from pygeovision.viz import Map
        m = Map()
        assert m._zoom == 12
        assert len(m._layers) == 0

    def test_add_raster_returns_self(self):
        from pygeovision.viz import Map
        m = Map()
        with tempfile.TemporaryDirectory() as tmp:
            path = make_raster(f"{tmp}/scene.tif")
            result = m.add_raster(path)
        assert result is m

    def test_add_raster_creates_layer(self):
        from pygeovision.viz import Map, RasterLayer
        m = Map()
        with tempfile.TemporaryDirectory() as tmp:
            path = make_raster(f"{tmp}/scene.tif")
            m.add_raster(path, colormap="viridis", band=0, opacity=0.7)
        assert len(m.layers) == 1
        assert isinstance(m.layers[0], RasterLayer)
        assert m.layers[0].colormap == "viridis"
        assert m.layers[0].opacity == 0.7

    def test_add_vector_creates_layer(self):
        from pygeovision.viz import Map, VectorLayer
        with tempfile.TemporaryDirectory() as tmp:
            gj_path = make_geojson(f"{tmp}/buildings.geojson")
            m = Map()
            m.add_vector(gj_path, color="red", fill_opacity=0.3)
        assert len(m.layers) == 1
        assert isinstance(m.layers[0], VectorLayer)

    def test_add_vector_dict_input(self):
        from pygeovision.viz import Map
        gj_dict = {"type": "FeatureCollection", "features": []}
        m = Map()
        m.add_vector(gj_dict)
        assert len(m.layers) == 1

    def test_add_basemap_changes_basemap(self):
        from pygeovision.viz import Map
        m = Map(basemap="streets")
        m.add_basemap("satellite")
        assert m._basemap == "satellite"

    def test_remove_layer(self):
        from pygeovision.viz import Map
        with tempfile.TemporaryDirectory() as tmp:
            path = make_raster(f"{tmp}/s.tif")
            m = Map()
            m.add_raster(path, name="flood")
            assert len(m.layers) == 1
            m.remove_layer("flood")
            assert len(m.layers) == 0

    def test_export_html(self):
        from pygeovision.viz import Map
        with tempfile.TemporaryDirectory() as tmp:
            m = Map(center=(5.6, -0.2), zoom=11)
            html_path = f"{tmp}/map.html"
            result = m.export(html_path)
            assert pathlib.Path(html_path).exists()
            assert "leaflet" in pathlib.Path(html_path).read_text().lower()
            assert result == html_path

    def test_export_html_with_vector(self):
        from pygeovision.viz import Map
        with tempfile.TemporaryDirectory() as tmp:
            gj = make_geojson(f"{tmp}/data.geojson")
            m = Map(center=(5.6, -0.2))
            m.add_vector(gj, color="#ff0000")
            m.export(f"{tmp}/map.html")
            html = pathlib.Path(f"{tmp}/map.html").read_text()
            assert "ff0000" in html or "Feature" in html

    def test_map_repr(self):
        from pygeovision.viz import Map
        m = Map(center=(5.6, -0.2), zoom=12)
        r = repr(m)
        assert "Map" in r
        assert "12" in r

    def test_split_view_creates_split_map(self):
        from pygeovision.viz import Map
        from pygeovision.viz.map import SplitMap
        with tempfile.TemporaryDirectory() as tmp:
            pre  = make_raster(f"{tmp}/pre.tif")
            post = make_raster(f"{tmp}/post.tif")
            split = Map.split_view(pre, post, center=(5.6, -0.2))
            assert isinstance(split, SplitMap)

    def test_split_map_export_html(self):
        from pygeovision.viz import Map
        with tempfile.TemporaryDirectory() as tmp:
            pre  = make_raster(f"{tmp}/pre.tif")
            post = make_raster(f"{tmp}/post.tif")
            split = Map.split_view(pre, post)
            out = f"{tmp}/split.html"
            split.export(out)
            assert pathlib.Path(out).exists()


# ══════════════════════════════════════════════════════════════════════════════
# 2. RasterViewer
# ══════════════════════════════════════════════════════════════════════════════

class TestRasterViewer:

    def test_constructs(self):
        from pygeovision.viz import RasterViewer
        rv = RasterViewer("dummy.tif", figsize=(8, 6))
        assert rv.path == "dummy.tif"
        assert rv.figsize == (8, 6)

    def test_normalize_percentile_stretch(self):
        from pygeovision.viz.raster import RasterViewer
        rv = RasterViewer("dummy.tif")
        arr = np.array([0.0, 0.1, 0.5, 0.9, 1.0], dtype="float32")
        norm = rv._normalize(arr)
        assert float(norm.min()) >= 0.0 - 1e-4
        assert float(norm.max()) <= 1.0 + 1e-4

    def test_rgb_sets_figure(self):
        from pygeovision.viz import RasterViewer
        with tempfile.TemporaryDirectory() as tmp:
            path = make_raster(f"{tmp}/s2.tif", bands=6)
            rv = RasterViewer(path)
            rv.rgb(red=2, green=1, blue=0)
            assert rv._fig is not None

    def test_rgb_returns_self(self):
        from pygeovision.viz import RasterViewer
        with tempfile.TemporaryDirectory() as tmp:
            path = make_raster(f"{tmp}/s2.tif", bands=6)
            rv = RasterViewer(path)
            result = rv.rgb(red=2, green=1, blue=0)
            assert result is rv

    def test_ndvi_sets_figure(self):
        from pygeovision.viz import RasterViewer
        with tempfile.TemporaryDirectory() as tmp:
            path = make_raster(f"{tmp}/s2.tif", bands=6)
            rv = RasterViewer(path)
            rv.ndvi(nir_band=3, red_band=2)
            assert rv._fig is not None

    def test_ndwi_sets_figure(self):
        from pygeovision.viz import RasterViewer
        with tempfile.TemporaryDirectory() as tmp:
            path = make_raster(f"{tmp}/s2.tif", bands=6)
            rv = RasterViewer(path)
            rv.ndwi(green_band=1, nir_band=3)
            assert rv._fig is not None

    def test_ndbi_sets_figure(self):
        from pygeovision.viz import RasterViewer
        with tempfile.TemporaryDirectory() as tmp:
            path = make_raster(f"{tmp}/s2.tif", bands=6)
            rv = RasterViewer(path)
            rv.ndbi(swir_band=4, nir_band=3)
            assert rv._fig is not None

    def test_single_band_sets_figure(self):
        from pygeovision.viz import RasterViewer
        with tempfile.TemporaryDirectory() as tmp:
            path = make_raster(f"{tmp}/s2.tif", bands=6)
            rv = RasterViewer(path)
            rv.single_band(band=0, colormap="gray")
            assert rv._fig is not None

    def test_histogram_all_bands(self):
        from pygeovision.viz import RasterViewer
        with tempfile.TemporaryDirectory() as tmp:
            path = make_raster(f"{tmp}/s2.tif", bands=4)
            rv = RasterViewer(path)
            rv.histogram()
            assert rv._fig is not None

    def test_histogram_single_band(self):
        from pygeovision.viz import RasterViewer
        with tempfile.TemporaryDirectory() as tmp:
            path = make_raster(f"{tmp}/s2.tif", bands=4)
            rv = RasterViewer(path)
            rv.histogram(band=1)
            assert rv._fig is not None

    def test_profile_sets_figure(self):
        from pygeovision.viz import RasterViewer
        with tempfile.TemporaryDirectory() as tmp:
            path = make_raster(f"{tmp}/s2.tif", bands=4)
            rv = RasterViewer(path)
            rv.profile(start_px=(5, 5), end_px=(50, 50))
            assert rv._fig is not None

    def test_export_png(self):
        from pygeovision.viz import RasterViewer
        with tempfile.TemporaryDirectory() as tmp:
            path = make_raster(f"{tmp}/s2.tif", bands=6)
            out  = f"{tmp}/output.png"
            rv   = RasterViewer(path)
            rv.rgb(red=2, green=1, blue=0)
            result = rv.export(out, dpi=72)
            assert pathlib.Path(out).exists()
            assert result == out

    def test_export_without_figure_raises(self):
        from pygeovision.viz import RasterViewer
        rv = RasterViewer("dummy.tif")
        with pytest.raises(RuntimeError, match="Nothing to export"):
            rv.export("/tmp/out.png")

    def test_data_cache_used(self):
        from pygeovision.viz import RasterViewer
        with tempfile.TemporaryDirectory() as tmp:
            path = make_raster(f"{tmp}/s2.tif", bands=3)
            rv   = RasterViewer(path)
            arr1 = rv._read(0)
            arr2 = rv._read(0)
            assert arr1 is arr2  # same object from cache

    def test_repr(self):
        from pygeovision.viz import RasterViewer
        rv = RasterViewer("/tmp/scene.tif")
        assert "RasterViewer" in repr(rv)
        assert "scene.tif" in repr(rv)


# ══════════════════════════════════════════════════════════════════════════════
# 3. ChangeViewer
# ══════════════════════════════════════════════════════════════════════════════

class TestChangeViewer:

    def test_constructs(self):
        from pygeovision.viz import ChangeViewer
        cv = ChangeViewer("before.tif", "after.tif", band=0)
        assert cv._before == "before.tif"
        assert cv._after  == "after.tif"
        assert cv._band   == 0

    def test_statistics_uses_pixel_area_m2_fallback_without_mask(self, tmp_path):
        """Regression test for a real bug: pixel_area_m2 was accepted but
        never referenced anywhere in the function body — statistics()
        called without a mask_path never produced any area-based stats
        at all, regardless of what pixel_area_m2 was set to."""
        rasterio = pytest.importorskip("rasterio", reason="rasterio not installed")
        pytest.importorskip("matplotlib", reason="matplotlib not installed")
        import numpy as np
        from rasterio.transform import from_bounds
        from pygeovision.viz import ChangeViewer

        before_p = tmp_path / "before.tif"
        after_p = tmp_path / "after.tif"
        transform = from_bounds(0, 0, 500, 500, 50, 50)
        rng = np.random.default_rng(0)
        before_data = rng.random((1, 50, 50)).astype("float32")
        after_data = before_data.copy()
        after_data[0, :10, :10] += 0.5

        for p, d in [(before_p, before_data), (after_p, after_data)]:
            with rasterio.open(p, "w", driver="GTiff", height=50, width=50, count=1,
                                dtype="float32", crs="EPSG:32630", transform=transform) as dst:
                dst.write(d)

        cv = ChangeViewer(str(before_p), str(after_p))
        stats = cv.statistics(mask_path=None, pixel_area_m2=100.0)

        assert "changed_area_m2" in stats, "pixel_area_m2 fallback not wired in"
        assert stats["changed_area_m2"] > 0
        assert stats["changed_area_m2"] == stats["changed_pixels"] * 100.0

    def test_split_sets_figure(self):
        from pygeovision.viz import ChangeViewer
        with tempfile.TemporaryDirectory() as tmp:
            pre  = make_raster(f"{tmp}/pre.tif", bands=1)
            post = make_raster(f"{tmp}/post.tif", bands=1)
            cv = ChangeViewer(pre, post)
            cv.split()
            assert cv._fig is not None

    def test_split_returns_self(self):
        from pygeovision.viz import ChangeViewer
        with tempfile.TemporaryDirectory() as tmp:
            pre  = make_raster(f"{tmp}/pre.tif", bands=1)
            post = make_raster(f"{tmp}/post.tif", bands=1)
            cv = ChangeViewer(pre, post)
            assert cv.split() is cv

    def test_difference_sets_figure(self):
        from pygeovision.viz import ChangeViewer
        with tempfile.TemporaryDirectory() as tmp:
            pre  = make_raster(f"{tmp}/pre.tif", bands=1)
            post = make_raster(f"{tmp}/post.tif", bands=1)
            cv = ChangeViewer(pre, post)
            cv.difference()
            assert cv._fig is not None

    def test_statistics_returns_dict(self):
        from pygeovision.viz import ChangeViewer
        with tempfile.TemporaryDirectory() as tmp:
            pre  = make_raster(f"{tmp}/pre.tif", bands=1)
            post = make_raster(f"{tmp}/post.tif", bands=1)
            cv = ChangeViewer(pre, post)
            stats = cv.statistics()
            assert isinstance(stats, dict)
            assert "mean_change" in stats
            assert "pct_increased" in stats
            assert "pct_stable" in stats
            assert "pct_decreased" in stats

    def test_statistics_percentages_sum_to_100(self):
        from pygeovision.viz import ChangeViewer
        with tempfile.TemporaryDirectory() as tmp:
            pre  = make_raster(f"{tmp}/pre.tif", bands=1)
            post = make_raster(f"{tmp}/post.tif", bands=1)
            cv = ChangeViewer(pre, post)
            stats = cv.statistics()
            total = stats["pct_increased"] + stats["pct_stable"] + stats["pct_decreased"]
            assert abs(total - 100.0) < 0.5

    def test_export_png(self):
        from pygeovision.viz import ChangeViewer
        with tempfile.TemporaryDirectory() as tmp:
            pre  = make_raster(f"{tmp}/pre.tif", bands=1)
            post = make_raster(f"{tmp}/post.tif", bands=1)
            out  = f"{tmp}/change.png"
            cv   = ChangeViewer(pre, post)
            cv.split()
            cv.export(out)
            assert pathlib.Path(out).exists()

    def test_repr(self):
        from pygeovision.viz import ChangeViewer
        cv = ChangeViewer("before.tif", "after.tif")
        assert "ChangeViewer" in repr(cv)
        assert "before.tif" in repr(cv)


# ══════════════════════════════════════════════════════════════════════════════
# 4. TimeSeriesViewer
# ══════════════════════════════════════════════════════════════════════════════

class TestTimeSeriesViewer:

    def _make_stack(self, tmp, n=4, bands=1):
        paths = []
        for i in range(n):
            p = make_raster(f"{tmp}/raster_{i}.tif", bands=bands)
            paths.append(p)
        return paths

    def test_constructs(self):
        from pygeovision.viz import TimeSeriesViewer
        tsv = TimeSeriesViewer(["a.tif", "b.tif"], dates=["2023-01", "2024-01"])
        assert len(tsv._paths) == 2
        assert tsv._dates == ["2023-01", "2024-01"]

    def test_default_dates_generated(self):
        from pygeovision.viz import TimeSeriesViewer
        tsv = TimeSeriesViewer(["a.tif", "b.tif", "c.tif"])
        assert tsv._dates == ["T0", "T1", "T2"]

    def test_load_stack_shape(self):
        from pygeovision.viz import TimeSeriesViewer
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._make_stack(tmp, n=3)
            tsv   = TimeSeriesViewer(paths, dates=["2021","2022","2023"])
            stack = tsv._load_stack()
            assert stack.shape[0] == 3
            assert stack.shape[1] == 64
            assert stack.shape[2] == 64

    def test_stack_cached(self):
        from pygeovision.viz import TimeSeriesViewer
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._make_stack(tmp, n=2)
            tsv   = TimeSeriesViewer(paths)
            s1 = tsv._load_stack()
            s2 = tsv._load_stack()
            assert s1 is s2

    def test_trend_sets_figure(self):
        from pygeovision.viz import TimeSeriesViewer
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._make_stack(tmp, n=4)
            dates = ["2021-01","2022-01","2023-01","2024-01"]
            tsv   = TimeSeriesViewer(paths, dates=dates)
            tsv.trend()
            assert tsv._fig is not None

    def test_trend_returns_self(self):
        from pygeovision.viz import TimeSeriesViewer
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._make_stack(tmp, n=3)
            tsv   = TimeSeriesViewer(paths)
            assert tsv.trend() is tsv

    def test_anomaly_sets_figure(self):
        from pygeovision.viz import TimeSeriesViewer
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._make_stack(tmp, n=5)
            tsv   = TimeSeriesViewer(paths)
            tsv.anomaly(n_sigma=1.5)
            assert tsv._fig is not None

    def test_mosaic_sets_figure(self):
        from pygeovision.viz import TimeSeriesViewer
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._make_stack(tmp, n=4)
            tsv   = TimeSeriesViewer(paths)
            tsv.mosaic(max_cols=2)
            assert tsv._fig is not None

    def test_export_png(self):
        from pygeovision.viz import TimeSeriesViewer
        with tempfile.TemporaryDirectory() as tmp:
            paths = self._make_stack(tmp, n=3)
            tsv   = TimeSeriesViewer(paths)
            tsv.trend()
            out   = f"{tmp}/trend.png"
            tsv.export(out)
            assert pathlib.Path(out).exists()

    def test_repr(self):
        from pygeovision.viz import TimeSeriesViewer
        tsv = TimeSeriesViewer(["a.tif","b.tif","c.tif"])
        assert "TimeSeriesViewer" in repr(tsv)
        assert "n=3" in repr(tsv)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
