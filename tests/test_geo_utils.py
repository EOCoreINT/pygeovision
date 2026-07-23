"""Tests for pygeovision.data.boundary, Preprocessor.mosaic, and
PostProcessor.clip_vector_to_polygon — capabilities added so notebooks can
delegate mosaicking, AOI-boundary-fetching, and vector clipping to the
library instead of writing raw rasterio/shapely/geopandas code.
"""
import json
import pathlib

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds


# ── utm_epsg_for — pure function, no network ────────────────────────────────

class TestUTMEpsgDetection:
    def test_accra_ghana_is_zone_30n(self):
        from pygeovision.data.boundary import utm_epsg_for
        # Accra: ~0.19°W, 5.58°N -> zone 30N
        assert utm_epsg_for(-0.19, 5.58) == 32630

    def test_southern_hemisphere_uses_700_series(self):
        from pygeovision.data.boundary import utm_epsg_for
        # Sao Paulo: ~46.6°W, 23.5°S -> zone 23S
        assert utm_epsg_for(-46.6, -23.5) == 32723

    def test_zone_boundaries_clamped(self):
        from pygeovision.data.boundary import utm_epsg_for
        assert 32601 <= utm_epsg_for(-180.0, 10.0) <= 32660
        assert 32601 <= utm_epsg_for(179.9, 10.0) <= 32660


# ── fetch_admin_boundary — mocked Nominatim response ────────────────────────

class TestFetchAdminBoundary:
    @staticmethod
    def _fake_response(monkeypatch, area_deg=0.035, lon=-0.19, lat=5.58):
        import shapely.geometry as sgeom
        poly = sgeom.box(lon - area_deg, lat - area_deg, lon + area_deg, lat + area_deg)
        fc = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": sgeom.mapping(poly),
                "properties": {"category": "boundary", "osm_type": "relation",
                               "display_name": "Fake Accra Metropolitan District, Ghana"},
            }],
        }

        class _FakeResp:
            status_code = 200
            def raise_for_status(self): pass
            def json(self): return fc

        import requests as requests_mod
        monkeypatch.setattr(requests_mod, "get", lambda *a, **kw: _FakeResp())

    def test_returns_bbox_and_area(self, monkeypatch):
        from pygeovision.data.boundary import fetch_admin_boundary
        self._fake_response(monkeypatch)
        result = fetch_admin_boundary("Accra Metropolitan District, Ghana")
        assert len(result.bbox) == 4
        assert result.area_km2 > 0
        assert result.utm_epsg == 32630
        assert result.validated is True

    def test_reference_area_check_passes_for_matching_area(self, monkeypatch):
        from pygeovision.data.boundary import fetch_admin_boundary
        self._fake_response(monkeypatch)  # ~60 km2 by construction
        result = fetch_admin_boundary(
            "Accra Metropolitan District, Ghana", reference_area_km2=60.0,
        )
        assert result.validated is True

    def test_reference_area_mismatch_raises_by_default(self, monkeypatch):
        from pygeovision.data.boundary import fetch_admin_boundary
        self._fake_response(monkeypatch)  # ~60 km2
        with pytest.raises(ValueError):
            fetch_admin_boundary(
                "Accra Metropolitan District, Ghana", reference_area_km2=5000.0,
            )

    def test_reference_area_mismatch_no_raise_when_disabled(self, monkeypatch):
        from pygeovision.data.boundary import fetch_admin_boundary
        self._fake_response(monkeypatch)
        result = fetch_admin_boundary(
            "Accra Metropolitan District, Ghana",
            reference_area_km2=5000.0, raise_on_mismatch=False,
        )
        assert result.validated is False
        assert len(result.warnings) == 1

    def test_saves_geojson_when_output_path_given(self, monkeypatch, tmp_path):
        from pygeovision.data.boundary import fetch_admin_boundary
        self._fake_response(monkeypatch)
        out = tmp_path / "aoi.geojson"
        result = fetch_admin_boundary("Accra Metropolitan District, Ghana", output_path=str(out))
        assert out.exists()
        assert result.geojson_path == str(out)
        saved = json.loads(out.read_text())
        assert saved["type"] == "Feature"

    def test_client_boundary_method(self, monkeypatch):
        from pygeovision import PyGeoVision
        self._fake_response(monkeypatch)
        client = PyGeoVision()
        result = client.boundary("Accra Metropolitan District, Ghana", reference_area_km2=60.0)
        assert result.validated is True
        assert result.utm_epsg == 32630

    def test_prefers_boundary_category_over_other_polygon_matches(self, monkeypatch):
        """Regression test for a real bug: Nominatim's format=geojson uses the
        property key `category` (not `class`, which is only used in
        format=json/jsonv2). A prior implementation checked `class`, which
        doesn't exist in geojson responses, so the 'prefer administrative
        boundary' filter silently never matched anything and always fell
        back to whichever polygon Nominatim ranked first — regardless of
        whether it was actually the right entity."""
        import shapely.geometry as sgeom
        from pygeovision.data.boundary import fetch_admin_boundary

        # Two polygon candidates: a non-boundary match ranked first (e.g. a
        # park or other polygon-tagged feature) and the real admin boundary
        # ranked second — exactly the shape of a real Nominatim response
        # where the filter needs to actually do something.
        wrong_poly  = sgeom.box(-0.30, 5.50, -0.29, 5.51)   # tiny, wrong entity
        correct_poly = sgeom.box(-0.225, 5.545, -0.155, 5.615)  # ~60 km2, correct

        fc = {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "geometry": sgeom.mapping(wrong_poly),
                    "properties": {"category": "leisure", "osm_type": "way",
                                   "display_name": "Some Park, Accra, Ghana"},
                },
                {
                    "type": "Feature",
                    "geometry": sgeom.mapping(correct_poly),
                    "properties": {"category": "boundary", "osm_type": "relation",
                                   "display_name": "Accra Metropolitan District, Ghana"},
                },
            ],
        }

        class _FakeResp:
            status_code = 200
            def raise_for_status(self): pass
            def json(self): return fc

        import requests as requests_mod
        monkeypatch.setattr(requests_mod, "get", lambda *a, **kw: _FakeResp())

        result = fetch_admin_boundary(
            "Accra Metropolitan District, Ghana", reference_area_km2=60.0,
        )
        # Must have picked the boundary-category feature (~60 km2), not the
        # first-ranked non-boundary one (~1.2 km2) — this only passes if the
        # category filter actually works.
        assert result.osm_class == "boundary"
        assert result.validated is True
        assert 40 <= result.area_km2 <= 80


# ── select_covering_scenes ───────────────────────────────────────────────────

class TestSelectCoveringScenes:
    def test_picks_scenes_until_bbox_covered(self):
        from pygeovision import PyGeoVision
        client = PyGeoVision()

        class FakeResult:
            def __init__(self, geom_bounds):
                import shapely.geometry as sgeom
                self.geometry = sgeom.mapping(sgeom.box(*geom_bounds))

        bbox = (0.0, 0.0, 2.0, 1.0)
        # Two scenes, each covering half the bbox
        left  = FakeResult((0.0, 0.0, 1.0, 1.0))
        right = FakeResult((1.0, 0.0, 2.0, 1.0))
        selected = client.select_covering_scenes([left, right], bbox)
        assert len(selected) == 2

    def test_stops_early_once_covered(self):
        from pygeovision import PyGeoVision
        client = PyGeoVision()

        class FakeResult:
            def __init__(self, geom_bounds):
                import shapely.geometry as sgeom
                self.geometry = sgeom.mapping(sgeom.box(*geom_bounds))

        bbox = (0.0, 0.0, 1.0, 1.0)
        full = FakeResult((-0.1, -0.1, 1.1, 1.1))  # covers the whole bbox alone
        extra = FakeResult((5.0, 5.0, 6.0, 6.0))    # irrelevant, shouldn't be needed
        selected = client.select_covering_scenes([full, extra], bbox)
        assert len(selected) == 1

    def test_respects_max_scenes_cap(self):
        from pygeovision import PyGeoVision
        client = PyGeoVision()

        class FakeResult:
            def __init__(self, geom_bounds):
                import shapely.geometry as sgeom
                self.geometry = sgeom.mapping(sgeom.box(*geom_bounds))

        bbox = (0.0, 0.0, 100.0, 100.0)  # never fully covered by the tiny scenes below
        tiny_scenes = [FakeResult((i, i, i + 0.1, i + 0.1)) for i in range(10)]
        selected = client.select_covering_scenes(tiny_scenes, bbox, max_scenes=3)
        assert len(selected) == 3


# ── Preprocessor.mosaic ──────────────────────────────────────────────────────

def _make_raster(path, bounds, w=16, h=16, value=1):
    transform = from_bounds(*bounds, w, h)
    data = np.full((1, h, w), value, dtype="uint16")
    with rasterio.open(path, "w", driver="GTiff", height=h, width=w, count=1,
                        dtype="uint16", crs="EPSG:4326", transform=transform) as dst:
        dst.write(data)


class TestPreprocessorMosaic:
    def test_single_input_is_noop(self, tmp_path):
        from pygeovision.preprocess.core import Preprocessor
        p = tmp_path / "a.tif"
        _make_raster(p, (0, 0, 1, 1))
        pre = Preprocessor()
        result = pre.mosaic([str(p)])
        assert result == str(p)

    def test_two_adjacent_rasters_merge(self, tmp_path):
        from pygeovision.preprocess.core import Preprocessor
        a = tmp_path / "a.tif"
        b = tmp_path / "b.tif"
        _make_raster(a, (0, 0, 1, 1), value=1)
        _make_raster(b, (1, 0, 2, 1), value=2)
        pre = Preprocessor()
        out = tmp_path / "mosaic.tif"
        result = pre.mosaic([str(a), str(b)], output_path=str(out))
        assert pathlib.Path(result).exists()
        with rasterio.open(result) as src:
            assert src.width >= 16  # wider than either single input alone
            data = src.read(1)
            assert set(np.unique(data)) <= {0, 1, 2}

    def test_empty_list_raises(self):
        from pygeovision.preprocess.core import Preprocessor
        pre = Preprocessor()
        with pytest.raises(ValueError):
            pre.mosaic([])

    def test_multiband_inputs_preserve_band_count(self, tmp_path):
        """A real-world regression case: mosaicking 4-band scenes must
        produce a 4-band output, not silently collapse to 1 band."""
        from pygeovision.preprocess.core import Preprocessor

        def make_multiband(path, bounds, bands=4, dtype="uint16"):
            transform = from_bounds(*bounds, 16, 16)
            with rasterio.open(path, "w", driver="GTiff", height=16, width=16,
                                count=bands, dtype=dtype, crs="EPSG:4326",
                                transform=transform) as dst:
                dst.write(np.full((bands, 16, 16), 1000, dtype=dtype))

        a = tmp_path / "a.tif"
        b = tmp_path / "b.tif"
        make_multiband(a, (0, 0, 1, 1))
        make_multiband(b, (1, 0, 2, 1))

        pre = Preprocessor()
        out = tmp_path / "mosaic.tif"
        result = pre.mosaic([str(a), str(b)], output_path=str(out))

        with rasterio.open(result) as src:
            assert src.count == 4
            assert src.dtypes[0] == "uint16"

    def test_output_metadata_matches_actual_merged_array_not_stale_profile(self, tmp_path, monkeypatch):
        """Regression test: the output GeoTIFF's count/dtype must come from
        what rasterio.merge actually produced, not be blindly copied from
        srcs[0].profile — those can disagree, and trusting the stale profile
        either raises a shape-mismatch error or silently writes a
        wrong-shaped file."""
        from pygeovision.preprocess import core as core_mod
        from pygeovision.preprocess.core import Preprocessor

        def make_multiband(path, bounds, bands=4):
            transform = from_bounds(*bounds, 16, 16)
            with rasterio.open(path, "w", driver="GTiff", height=16, width=16,
                                count=bands, dtype="uint16", crs="EPSG:4326",
                                transform=transform) as dst:
                dst.write(np.full((bands, 16, 16), 1000, dtype="uint16"))

        a = tmp_path / "a.tif"
        b = tmp_path / "b.tif"
        make_multiband(a, (0, 0, 1, 1))
        make_multiband(b, (1, 0, 2, 1))

        # Simulate rio_merge returning something that DISAGREES with
        # srcs[0].profile (1 band, float32) — exactly the shape of the real
        # bug report, regardless of why rio_merge might do this in practice.
        real_merge = core_mod._require_rasterio  # keep for other calls
        import rasterio.merge as merge_mod
        real_rio_merge = merge_mod.merge

        def fake_merge(srcs, method="first"):
            arr, transform = real_rio_merge(srcs, method=method)
            # Collapse to 1 band, cast to float32 — simulating the reported bug
            return arr[:1].astype("float32"), transform

        monkeypatch.setattr(merge_mod, "merge", fake_merge)

        pre = Preprocessor()
        out = tmp_path / "mosaic.tif"
        result = pre.mosaic([str(a), str(b)], output_path=str(out))

        # The written file must honestly reflect what merge returned (1 band,
        # float32) rather than crashing or silently keeping srcs[0]'s stale
        # 4-band/uint16 profile — this makes the corruption visible/debuggable
        # instead of hidden behind mismatched metadata.
        with rasterio.open(result) as src:
            assert src.count == 1
            assert src.dtypes[0] == "float32"

    def test_raises_clearly_on_inconsistent_input_band_counts(self, tmp_path):
        """If inputs disagree on band count, mosaic() must fail with a clear,
        actionable error — rasterio.merge cannot reconcile mismatched band
        counts and would otherwise raise a cryptic low-level
        DatasetIOShapeError instead."""
        from pygeovision.preprocess.core import Preprocessor

        def make_raster(path, bounds, bands):
            transform = from_bounds(*bounds, 16, 16)
            with rasterio.open(path, "w", driver="GTiff", height=16, width=16,
                                count=bands, dtype="uint16", crs="EPSG:4326",
                                transform=transform) as dst:
                dst.write(np.full((bands, 16, 16), 1000, dtype="uint16"))

        a = tmp_path / "a.tif"
        b = tmp_path / "b.tif"
        make_raster(a, (0, 0, 1, 1), bands=4)
        make_raster(b, (1, 0, 2, 1), bands=1)  # inconsistent!

        pre = Preprocessor()
        with pytest.raises(ValueError, match="inconsistent band counts"):
            pre.mosaic([str(a), str(b)], output_path=str(tmp_path / "mosaic.tif"))


# ── PostProcessor.clip_vector_to_polygon ─────────────────────────────────────

class TestClipVectorToPolygon:
    def test_drops_features_outside_aoi(self, tmp_path):
        from pygeovision.data.postprocess import PostProcessor
        import shapely.geometry as sgeom

        # Two building polygons: one inside the AOI, one outside
        inside  = sgeom.box(0.1, 0.1, 0.2, 0.2)
        outside = sgeom.box(5.0, 5.0, 5.1, 5.1)
        buildings_gj = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "geometry": sgeom.mapping(inside), "properties": {"id": 1}},
                {"type": "Feature", "geometry": sgeom.mapping(outside), "properties": {"id": 2}},
            ],
        }
        buildings_path = tmp_path / "buildings.geojson"
        buildings_path.write_text(json.dumps(buildings_gj))

        aoi = sgeom.box(0.0, 0.0, 1.0, 1.0)
        aoi_gj = {"type": "Feature", "geometry": sgeom.mapping(aoi), "properties": {}}

        post = PostProcessor()
        out = tmp_path / "clipped.geojson"
        result = post.clip_vector_to_polygon(str(buildings_path), aoi_gj, str(out))

        assert result["n_before"] == 2
        assert result["n_after"] == 1
        assert out.exists()

    def test_accepts_aoi_as_file_path(self, tmp_path):
        from pygeovision.data.postprocess import PostProcessor
        import shapely.geometry as sgeom

        inside = sgeom.box(0.1, 0.1, 0.2, 0.2)
        buildings_gj = {
            "type": "FeatureCollection",
            "features": [{"type": "Feature", "geometry": sgeom.mapping(inside), "properties": {}}],
        }
        buildings_path = tmp_path / "buildings.geojson"
        buildings_path.write_text(json.dumps(buildings_gj))

        aoi_path = tmp_path / "aoi.geojson"
        aoi = sgeom.box(0.0, 0.0, 1.0, 1.0)
        aoi_path.write_text(json.dumps({"type": "Feature", "geometry": sgeom.mapping(aoi), "properties": {}}))

        post = PostProcessor()
        out = tmp_path / "clipped.geojson"
        result = post.clip_vector_to_polygon(str(buildings_path), str(aoi_path), str(out))
        assert result["n_after"] == 1



        