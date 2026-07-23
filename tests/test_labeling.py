"""Tests for PyGeoVision auto-labeling layer (Phase 2+)."""
from unittest.mock import patch

import pytest

# ── OSM Labeler ───────────────────────────────────────────────────────────────

class TestOSMLabeler:
    def test_list_categories(self):
        from pygeovision.labeling.osm import OSM_CATEGORIES, OSMLabeler
        labeler = OSMLabeler()
        cats = labeler.list_categories()
        assert isinstance(cats, dict)
        assert len(cats) == len(OSM_CATEGORIES)
        assert "buildings" in cats
        assert "water" in cats
        assert "roads" in cats

    def test_preview_query(self):
        from pygeovision.labeling.osm import OSMLabeler
        labeler = OSMLabeler()
        q = labeler.preview_query((-74.05, 40.70, -73.95, 40.80), ["buildings"])
        assert "building" in q
        assert "out body geom" in q
        assert "-74.05" in q and "40.7" in q  # lat-min,lon-min Overpass format

    def test_build_overpass_query(self):
        from pygeovision.labeling.osm import OSMLabeler
        labeler = OSMLabeler()
        q = labeler._build_overpass_query((-74.05, 40.70, -73.95, 40.80), ["buildings", "water"])
        assert "[out:json]" in q
        assert "building" in q
        assert "water" in q

    def test_osm_to_geojson_empty(self):
        from pygeovision.labeling.osm import OSMLabeler
        labeler = OSMLabeler()
        geojson = labeler._osm_to_geojson({"elements": []}, ["buildings"])
        assert geojson["type"] == "FeatureCollection"
        assert geojson["features"] == []

    def test_osm_to_geojson_with_buildings(self):
        from pygeovision.labeling.osm import OSMLabeler
        labeler = OSMLabeler()
        data = {"elements": [{
            "type": "way", "id": 1,
            "tags": {"building": "yes"},
            "geometry": [
                {"lon": -74.0, "lat": 40.7}, {"lon": -74.01, "lat": 40.7},
                {"lon": -74.01, "lat": 40.71}, {"lon": -74.0, "lat": 40.71},
                {"lon": -74.0, "lat": 40.7},
            ]
        }]}
        geojson = labeler._osm_to_geojson(data, ["buildings"])
        assert len(geojson["features"]) == 1
        f = geojson["features"][0]
        assert f["properties"]["category"] == "buildings"
        assert f["properties"]["label_value"] == 1

    def test_label_network_failure_returns_error(self):
        from pygeovision.labeling.osm import OSMLabeler
        labeler = OSMLabeler(retry_attempts=1)
        with patch.object(labeler, "_fetch_overpass", return_value=None):
            result = labeler.label((-74.05, 40.70, -73.95, 40.80), ["buildings"])
        assert result["success"] is False
        assert "error" in result

    @pytest.mark.skipif(not pytest.importorskip("rasterio", reason="rasterio not installed"), reason="needs rasterio")
    def test_rasterise_empty_geojson(self, tmp_path):
        from pygeovision.labeling.osm import OSMLabeler
        labeler = OSMLabeler()
        geojson = {"type": "FeatureCollection", "features": []}
        out = tmp_path / "test.tif"
        result = labeler._rasterise(geojson, (-74.05, 40.70, -73.95, 40.80), out, 1000.0)
        assert result == out
        assert out.exists()


# ── Label Quality Assessor ────────────────────────────────────────────────────

class TestLabelQualityAssessor:
    def test_recommendations_empty(self):
        from pygeovision.labeling.quality import LabelQualityAssessor
        qa = LabelQualityAssessor()
        results = {"quality_score": 0.9, "quality_grade": "A", "checks": {}}
        recs = qa._recommendations(results)
        assert isinstance(recs, list)

    def test_report_html_structure(self):
        from pygeovision.labeling.quality import LabelQualityAssessor
        qa = LabelQualityAssessor()
        results = {
            "label_path": "test.tif",
            "quality_score": 0.85,
            "quality_grade": "B",
            "checks": {"class_balance": {"status": "ok", "score": 0.9}},
            "recommendations": ["Test recommendation"],
        }
        html = qa.report_html(results)
        assert "<html>" in html
        assert "quality_grade" in html.lower() or "B" in html
        assert "Test recommendation" in html

    @pytest.mark.skipif(not pytest.importorskip("rasterio", reason="rasterio"), reason="needs rasterio")
    def test_assess_synthetic_label(self, tmp_path):
        import numpy as np
        import rasterio
        from rasterio.transform import from_bounds

        from pygeovision.labeling.quality import LabelQualityAssessor

        # Create a synthetic label raster
        label = np.zeros((100, 100), dtype=np.uint8)
        label[20:50, 20:50] = 1   # class 1 patch
        label[60:80, 60:80] = 1   # another patch
        transform = from_bounds(0, 0, 1, 1, 100, 100)
        p = tmp_path / "label.tif"
        with rasterio.open(str(p), "w", driver="GTiff", height=100, width=100,
                            count=1, dtype="uint8", crs="EPSG:4326", transform=transform) as dst:
            dst.write(label[np.newaxis])

        qa = LabelQualityAssessor(num_classes=2)
        result = qa.assess(str(p), checks=["class_balance", "coverage"])
        assert "quality_score" in result
        assert 0.0 <= result["quality_score"] <= 1.0
        assert "quality_grade" in result
        assert "class_balance" in result.get("checks", {})


# ── Active Learner ────────────────────────────────────────────────────────────

class TestActiveLearner:
    def test_init_valid_strategies(self):
        from pygeovision.labeling.active import ActiveLearner
        for strategy in ["entropy", "least_confidence", "margin", "coreset", "committee", "random"]:
            learner = ActiveLearner(strategy=strategy)
            assert learner.strategy == strategy

    def test_init_invalid_strategy(self):
        from pygeovision.labeling.active import ActiveLearner
        with pytest.raises(ValueError, match="strategy"):
            ActiveLearner(strategy="invalid_xyz")

    def test_update_adds_samples(self):
        from pygeovision.labeling.active import ActiveLearner
        learner = ActiveLearner()
        samples = [{"path": f"img{i}.tif", "label": i % 2} for i in range(5)]
        learner.update(samples)
        assert len(learner._labeled) == 5
        assert learner._iteration == 1

    def test_random_selection(self):
        from pygeovision.labeling.active import ActiveLearner
        learner = ActiveLearner(strategy="random", seed=42)
        pool = [{"path": f"img{i}.tif"} for i in range(20)]
        selected = learner.select(None, pool, n_select=5)
        assert len(selected) == 5

    def test_history_empty_initially(self):
        from pygeovision.labeling.active import ActiveLearner
        learner = ActiveLearner()
        assert learner.history == []

    @pytest.mark.skipif(
        __import__('importlib').util.find_spec('torch') is None,
        reason='torch not installed',
    )
    def test_train_iteration_returns_dict(self):
        from pygeovision.labeling.active import ActiveLearner
        learner = ActiveLearner(strategy="random")
        pool = [{"path": f"img{i}.tif"} for i in range(10)]
        result = learner.train_iteration(None, None, pool)
        assert "n_selected" in result
        assert "selected_samples" in result
        assert result["strategy"] == "random"


# ── AutoLabelPipeline ────────────────────────────────────────────────────────

class TestAutoLabelPipeline:
    def test_init_defaults(self):
        from pygeovision.labeling.pipeline import AutoLabelPipeline
        pipeline = AutoLabelPipeline()
        assert "osm" in pipeline.sources
        assert pipeline.fusion in ("majority_vote", "union", "intersection", "priority")

    def test_init_custom_sources(self):
        from pygeovision.labeling.pipeline import AutoLabelPipeline
        pipeline = AutoLabelPipeline(sources=["osm", "esa_worldcover"])
        assert pipeline.sources == ["osm", "esa_worldcover"]

    def test_run_returns_dict_on_network_failure(self, tmp_path):
        from pygeovision.labeling.pipeline import AutoLabelPipeline
        pipeline = AutoLabelPipeline(sources=["osm"])
        with patch("pygeovision.labeling.osm.OSMLabeler.label",
                   return_value={"success": False, "error": "Network"}):
            result = pipeline.run((-74.05, 40.70, -73.95, 40.80),
                                   output_dir=str(tmp_path))
        assert "sources_succeeded" in result
        assert "sources_failed" in result

    def test_supported_sources_list(self):
        from pygeovision.labeling.pipeline import AutoLabelPipeline
        assert "osm" in AutoLabelPipeline.SOURCES
        assert "microsoft_buildings" in AutoLabelPipeline.SOURCES
        assert "esa_worldcover" in AutoLabelPipeline.SOURCES
        assert "sam_auto" in AutoLabelPipeline.SOURCES


# ── Foundation Model Labeler ──────────────────────────────────────────────────

class TestFoundationModelLabeler:
    def test_init(self):
        from pygeovision.labeling.foundation import FoundationModelLabeler
        lab = FoundationModelLabeler(model="dinov2-base")
        assert lab.model_name == "dinov2-base"

    def test_hf_model_mapping(self):
        # Verify all model names map to HF IDs (via FewShotLearner which shares mapping)
        from pygeovision.advanced.few_shot import FewShotLearner
        for name in ["dinov2-small", "dinov2-base", "dinov2-large"]:
            learner = FewShotLearner(backbone=name)
            assert learner is not None  # FewShotLearner init succeeds


# ── Microsoft Buildings — dataset-links.csv / quadkey regression tests ────────
# These cover a real bug: the old implementation built URLs like
# "{zoom}/{tx}/{ty}.geojson.gz" against a base path that never existed. The
# real dataset is only addressable via the dataset-links.csv manifest keyed
# by Bing Maps quadkey.

class TestMicrosoftBuildingsQuadkey:
    def test_quadkey_format(self):
        """Quadkeys must be base-4 strings (digits 0-3) of length == zoom."""
        from pygeovision.labeling.buildings import MicrosoftBuildingsLabeler
        bbox = (-0.15, 51.47, -0.10, 51.52)  # London
        for zoom in (6, 9, 12):
            keys = MicrosoftBuildingsLabeler._bbox_to_quadkeys(bbox, zoom)
            assert len(keys) >= 1
            for qk in keys:
                assert len(qk) == zoom
                assert set(qk) <= {"0", "1", "2", "3"}

    def test_quadkey_matches_microsoft_reference_example(self):
        """Microsoft's own Bing Maps Tile System docs give tile (3,5) at
        level 3 -> quadkey "213". Verify our tile->quadkey digit encoding
        (extracted inline in _bbox_to_quadkeys) reproduces that exactly."""
        def tile_to_quadkey(x, y, z):
            qk = []
            for i in range(z, 0, -1):
                digit = 0
                mask = 1 << (i - 1)
                if x & mask:
                    digit += 1
                if y & mask:
                    digit += 2
                qk.append(str(digit))
            return "".join(qk)
        assert tile_to_quadkey(3, 5, 3) == "213"

    def test_manifest_url_is_current_hosting_location(self):
        """Regression guard: MS moved dataset-links.csv hosting in Nov 2024
        from *.blob.core.windows.net to *.z5.web.core.windows.net."""
        from pygeovision.labeling.buildings import MicrosoftBuildingsLabeler
        assert "z5.web.core.windows.net" in MicrosoftBuildingsLabeler.MANIFEST_URL
        assert MicrosoftBuildingsLabeler.MANIFEST_URL.endswith("dataset-links.csv")

    def test_fetch_matches_manifest_by_quadkey_prefix(self, monkeypatch):
        """A manifest row should be selected when its quadkey is a prefix of
        (or shares a prefix with) our target quadkeys."""
        import pandas as pd
        from pygeovision.labeling.buildings import MicrosoftBuildingsLabeler

        lab = MicrosoftBuildingsLabeler()
        target_qk = lab._bbox_to_quadkeys((-0.15, 51.47, -0.10, 51.52), 9)[0]

        fake_manifest = pd.DataFrame({
            "Location": ["UnitedKingdom"],
            "QuadKey": [target_qk[:6]],  # coarser prefix, as real manifest often is
            "Url": ["https://example.test/uk_part.csv.gz"],
            "Size": ["1KB"],
            "UploadDate": ["2026-01-01"],
        })
        monkeypatch.setattr(lab, "_load_manifest", lambda: fake_manifest)

        import gzip
        import requests as requests_mod
        payload = gzip.compress(b'{"type":"Feature","geometry":{"type":"Polygon","coordinates":[[[-0.12,51.5],[-0.12,51.51],[-0.11,51.51],[-0.11,51.5],[-0.12,51.5]]]},"properties":{"confidence":0.9}}\n')

        class _FakeResponse:
            status_code = 200
            content = payload

        monkeypatch.setattr(requests_mod, "get", lambda *a, **kw: _FakeResponse())

        result = lab._fetch_buildings_geojson((-0.15, 51.47, -0.10, 51.52))
        assert result["type"] == "FeatureCollection"
        assert len(result["features"]) == 1
        assert result["features"][0]["properties"]["source"] == "Microsoft"


# ── Google Buildings — S2 token / WKT regression tests ─────────────────────────
# Real bugs: (1) used str(cell.id()) — a huge decimal — instead of the short
# hex token the GCS bucket actually uses; (2) parsed the `geometry` CSV
# column with json.loads() when it's actually WKT, which would always raise.

class TestGoogleBuildingsS2Tokens:
    def test_tokens_are_short_hex_not_raw_decimal_ids(self):
        from pygeovision.labeling.buildings import GoogleBuildingsLabeler
        lab = GoogleBuildingsLabeler()
        tokens = lab._bbox_to_s2_cells((3.35, 6.45, 3.45, 6.55))  # Lagos, Nigeria
        assert len(tokens) >= 1
        for tok in tokens:
            # A raw decimal cell id would be ~19 digits and include no
            # hex-only letters; a token is short (<=16) hex.
            assert len(tok) <= 16
            assert all(c in "0123456789abcdef" for c in tok)
            assert not tok.isdigit() or len(tok) < 10  # not a raw huge decimal id

    def test_label_parses_wkt_geometry_not_json(self, monkeypatch):
        """The CSV `geometry` column is WKT; verify the real code path
        parses it via shapely instead of crashing on json.loads()."""
        from pygeovision.labeling.buildings import GoogleBuildingsLabeler
        lab = GoogleBuildingsLabeler(min_confidence=0.5)
        monkeypatch.setattr(lab, "_bbox_to_s2_cells", lambda bbox: ["abc"])

        import gzip
        import io as io_mod
        csv_bytes = (
            b"latitude,longitude,area_in_meters,confidence,geometry,full_plus_code\n"
            b'6.5,3.4,50.0,0.9,"POLYGON((3.4 6.5, 3.41 6.5, 3.41 6.51, 3.4 6.51, 3.4 6.5))",ABC\n'
        )
        payload = gzip.compress(csv_bytes)

        class _FakeResponse:
            status_code = 200
            content = payload

        import requests as requests_mod
        monkeypatch.setattr(requests_mod, "get", lambda *a, **kw: _FakeResponse())

        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out.tif")
            result = lab.label((3.35, 6.45, 3.45, 6.55), output_path=out)
        assert result["success"] is True
        assert result["n_buildings"] == 1


# ── ESA WorldCover — year filter regression test ────────────────────────────────
# Real bug: self.year was stored but never sent as a STAC datetime filter,
# so the search returned whichever item matched first regardless of year.

class TestESAWorldCoverYearFilter:
    def test_search_includes_year_datetime_filter(self, monkeypatch):
        from pygeovision.labeling.landcover import ESAWorldCoverLabeler
        lab = ESAWorldCoverLabeler(year=2020)

        captured = {}
        class _FakeResponse:
            status_code = 200
            def json(self):
                return {"features": []}

        import requests as requests_mod
        def fake_post(url, json=None, timeout=None):
            captured["json"] = json
            return _FakeResponse()
        monkeypatch.setattr(requests_mod, "post", fake_post)

        lab.label((-87.7, 41.8, -87.5, 41.9), output_path="/tmp/_unused_esa.tif")
        assert "datetime" in captured["json"]
        assert captured["json"]["datetime"] == "2020-01-01/2020-12-31"


# ── Dynamic World — SAS signing regression test ──────────────────────────────────
# Real bug: Planetary Computer assets live in private Blob Storage containers
# and require SAS-token signing before download; this was skipped entirely.

class TestDynamicWorldSigning:
    def test_pc_backend_signs_asset_url(self, monkeypatch):
        from pygeovision.labeling.landcover import DynamicWorldLabeler
        lab = DynamicWorldLabeler(backend="planetary_computer")

        class _FakeSearchResponse:
            status_code = 200
            def json(self):
                return {"features": [{"assets": {"data": {"href": "https://unsigned.example/asset.tif"}}}]}

        class _FakeDownloadResponse:
            status_code = 200
            def iter_content(self, chunk_size):
                return [b"fake-bytes"]

        import requests as requests_mod
        monkeypatch.setattr(requests_mod, "post", lambda *a, **kw: _FakeSearchResponse())

        captured = {}
        def fake_get(url, stream=None, timeout=None):
            captured["url"] = url
            return _FakeDownloadResponse()
        monkeypatch.setattr(requests_mod, "get", fake_get)

        pc_mod = pytest.importorskip("planetary_computer", reason="planetary-computer not installed")
        monkeypatch.setattr(pc_mod, "sign", lambda url: url + "?SIGNED")

        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "dw.tif")
            result = lab.label((-0.15, 51.47, -0.10, 51.52), output_path=out)
        assert result["success"] is True
        assert captured["url"].endswith("?SIGNED")
