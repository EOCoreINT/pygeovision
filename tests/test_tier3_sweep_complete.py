"""Regression tests for the full Tier 3 sweep: all 16 originally-
unverified named pipeline classes in domains.py.

Category A (9 classes): confirmed silent-fallback bugs where a bare
`except Exception:` caught the real processing call's failure and
silently substituted a wrong result (raw imagery, or the output
directory itself) while still returning success=True. Fixed by
removing the silent inner catch -- real failures now propagate to the
outer handler, which already correctly returns success=False.

Category B (4 classes): the same broken post_process=["ndvi"/"ndwi",...]
pattern found and fixed in CarbonEstimationPipeline earlier this
session -- these pipelines claimed specific indices were computed via
stats like {"computed": ["NDVI", "NDWI"]} while the mechanism that was
supposed to compute them never actually ran. Fixed with real, direct
index computation.

Category C (4 classes): success=True despite an honest-sounding note
admitting no real processing happened (misleading for anyone checking
result.success programmatically). Fixed either by real implementation
(reusing already-verified real computations via inheritance:
LandSurfaceTemperaturePipeline, ForestFirePipeline, VolcanoMonitoringPipeline
all reuse UrbanHeatIslandPipeline's/WildfireSeverityPipeline's real,
verified logic) or by honestly returning success=False
(TreeSpeciesPipeline).
"""
import pytest

rasterio = pytest.importorskip("rasterio", reason="rasterio not installed")

import numpy as np
from rasterio.transform import from_bounds
from unittest.mock import MagicMock, patch


# ── Category A: silent-fallback bugs, now fail loudly ──────────────────

class TestCategoryASilentFallbacksFixed:
    """For each, the real sub-call is mocked to raise -- confirms the
    pipeline now correctly returns success=False instead of silently
    substituting a wrong result."""

    def _make_pipeline(self, cls_name, tmp_path):
        import pygeovision.ai.pipelines.domains as domains
        cls = getattr(domains, cls_name)
        pipeline = cls.__new__(cls)
        mock_pgv = MagicMock()
        pipeline.pgv = mock_pgv
        pipeline._pgv = mock_pgv
        return pipeline, mock_pgv

    def test_crop_type_mapping_fails_loudly_not_silently(self, tmp_path):
        pipeline, mock_pgv = self._make_pipeline("CropTypeMappingPipeline", tmp_path)
        pipeline._search = MagicMock(return_value=[MagicMock()])
        fake_download = MagicMock(success=True, path=str(tmp_path / "scene.tif"))
        (tmp_path / "scene.tif").write_bytes(b"fake")
        mock_pgv.download.return_value = [fake_download]
        mock_pgv.segmentation.custom.side_effect = RuntimeError("model not found")

        result = pipeline.run(bbox=(0, 0, 1, 1), output_dir=str(tmp_path / "out"), date="2024-06")
        assert not result.success
        assert "model not found" in result.error

    def test_irrigation_detection_fails_loudly_not_silently(self, tmp_path):
        pipeline, mock_pgv = self._make_pipeline("IrrigationDetectionPipeline", tmp_path)
        pipeline._search = MagicMock(return_value=[MagicMock()])
        fake_download = MagicMock(success=True, path=str(tmp_path / "scene.tif"))
        mock_pgv.download.return_value = [fake_download]
        mock_pgv.segmentation.water.side_effect = RuntimeError("water model failed")

        result = pipeline.run(bbox=(0, 0, 1, 1), output_dir=str(tmp_path / "out"), date="2024-06")
        assert not result.success
        assert "water model failed" in result.error

    def test_road_extraction_fails_loudly_not_silently(self, tmp_path):
        pipeline, mock_pgv = self._make_pipeline("RoadExtractionPipeline", tmp_path)
        pipeline._search = MagicMock(return_value=[MagicMock()])
        fake_path = tmp_path / "scene.tif"; fake_path.write_bytes(b"fake")
        fake_download = MagicMock(success=True, path=str(fake_path))
        mock_pgv.download.return_value = [fake_download]
        mock_pgv.detection.generic.side_effect = RuntimeError("detector unavailable")

        result = pipeline.run(bbox=(0, 0, 1, 1), output_dir=str(tmp_path / "out"), date="2024-06")
        assert not result.success
        assert "detector unavailable" in result.error

    def test_ocean_ship_detection_fails_loudly_not_silently(self, tmp_path):
        pipeline, mock_pgv = self._make_pipeline("OceanShipDetectionPipeline", tmp_path)
        pipeline._search = MagicMock(return_value=[MagicMock()])
        fake_path = tmp_path / "scene.tif"; fake_path.write_bytes(b"fake")
        fake_download = MagicMock(success=True, path=str(fake_path))
        mock_pgv.download.return_value = [fake_download]
        mock_pgv.detection.ships.side_effect = RuntimeError("ships detector failed")

        result = pipeline.run(bbox=(0, 0, 1, 1), output_dir=str(tmp_path / "out"), date="2024-06")
        assert not result.success
        assert "ships detector failed" in result.error

    def test_flood_mapping_fails_loudly_not_silently(self, tmp_path):
        pipeline, mock_pgv = self._make_pipeline("FloodMappingPipeline", tmp_path)
        pipeline._search = MagicMock(return_value=[MagicMock()])
        fake_path = tmp_path / "scene.tif"; fake_path.write_bytes(b"fake")
        fake_download = MagicMock(success=True, path=str(fake_path))
        mock_pgv.download.return_value = [fake_download]
        mock_pgv.segmentation.water.side_effect = RuntimeError("flood model failed")

        result = pipeline.run(bbox=(0, 0, 1, 1), output_dir=str(tmp_path / "out"), date="2024-06")
        assert not result.success

    def test_landslide_detection_fails_loudly_not_silently(self, tmp_path):
        pipeline, mock_pgv = self._make_pipeline("LandslideDetectionPipeline", tmp_path)
        pipeline._search = MagicMock(return_value=[MagicMock()])
        fake_path = tmp_path / "scene.tif"; fake_path.write_bytes(b"fake")
        fake_download = MagicMock(success=True, path=str(fake_path))
        mock_pgv.download.return_value = [fake_download]
        mock_pgv.segmentation.custom.side_effect = RuntimeError("landslide model failed")

        result = pipeline.run(bbox=(0, 0, 1, 1), output_dir=str(tmp_path / "out"), date="2024-06")
        assert not result.success

    def test_zero_bare_except_exception_remain(self):
        """Direct source check -- confirms all 9 silent-catch sites
        were genuinely removed, not just individually tested."""
        import inspect
        import pygeovision.ai.pipelines.domains as domains
        source = inspect.getsource(domains)
        assert source.count("except Exception:") == 0


# ── Category B: broken post_process claims, now real computation ───────

class TestCategoryBRealComputation:
    def test_crop_health_computes_real_ndvi_anomaly(self, tmp_path):
        from pygeovision.ai.pipelines.domains import CropHealthPipeline
        size = 10
        transform = from_bounds(-95.0, 41.0, -94.9, 41.1, size, size)

        def make_scene(path, nir, red):
            with rasterio.open(path, "w", driver="GTiff", height=size, width=size, count=2,
                                dtype="float32", crs="EPSG:4326", transform=transform) as dst:
                dst.write(np.full((size, size), nir, dtype="float32"), 1)
                dst.write(np.full((size, size), red, dtype="float32"), 2)

        baseline = tmp_path / "baseline.tif"; make_scene(baseline, 0.5, 0.088)
        current = tmp_path / "current.tif"; make_scene(current, 0.3, 0.166)

        pipeline = CropHealthPipeline.__new__(CropHealthPipeline)
        pipeline.pgv = MagicMock(); pipeline._pgv = pipeline.pgv
        pipeline._search_and_download_pair = MagicMock(return_value=(baseline, current))

        result = pipeline.run(bbox=(-95.0, 41.0, -94.9, 41.1), output_dir=str(tmp_path / "out"),
                               date_before="2023-06", date_after="2024-06")
        assert result.success
        expected = ((0.5 - 0.088) / (0.5 + 0.088)) - ((0.3 - 0.166) / (0.3 + 0.166))
        computed = result.stats["mean_ndvi_baseline"] - result.stats["mean_ndvi_current"]
        assert abs(computed - expected) < 0.01

    def test_water_quality_computes_real_ndci(self, tmp_path):
        from pygeovision.ai.pipelines.domains import WaterQualityPipeline
        size = 10
        transform = from_bounds(-81.5, 26.5, -81.4, 26.6, size, size)
        img_path = tmp_path / "scene.tif"
        # green=0.1, red=0.05, rededge=0.15, nir=0.03 -> water (NDWI=(0.1-0.03)/(0.1+0.03)=0.538>0)
        with rasterio.open(img_path, "w", driver="GTiff", height=size, width=size, count=4,
                            dtype="float32", crs="EPSG:4326", transform=transform) as dst:
            dst.write(np.full((size, size), 0.1, dtype="float32"), 1)
            dst.write(np.full((size, size), 0.05, dtype="float32"), 2)
            dst.write(np.full((size, size), 0.15, dtype="float32"), 3)
            dst.write(np.full((size, size), 0.03, dtype="float32"), 4)

        pipeline = WaterQualityPipeline.__new__(WaterQualityPipeline)
        pipeline.pgv = MagicMock(); pipeline._pgv = pipeline.pgv
        pipeline._search_and_download = MagicMock(return_value=img_path)

        result = pipeline.run(bbox=(-81.5, 26.5, -81.4, 26.6), output_dir=str(tmp_path / "out"), date="2024-06")
        assert result.success
        expected_ndci = (0.15 - 0.05) / (0.15 + 0.05)
        assert abs(result.stats["mean_ndci_in_water"] - expected_ndci) < 0.01
        assert "UNCALIBRATED" in result.stats["note"]

    def test_vegetation_indices_computes_real_multi_date_series(self, tmp_path):
        from pygeovision.ai.pipelines.domains import VegetationIndicesPipeline
        size = 10
        transform = from_bounds(-95.0, 41.0, -94.9, 41.1, size, size)

        def make_scene(path, blue, red, nir, green):
            with rasterio.open(path, "w", driver="GTiff", height=size, width=size, count=4,
                                dtype="float32", crs="EPSG:4326", transform=transform) as dst:
                dst.write(np.full((size, size), blue, dtype="float32"), 1)
                dst.write(np.full((size, size), red, dtype="float32"), 2)
                dst.write(np.full((size, size), nir, dtype="float32"), 3)
                dst.write(np.full((size, size), green, dtype="float32"), 4)

        d1 = tmp_path / "d1.tif"; make_scene(d1, 0.05, 0.08, 0.4, 0.1)
        d2 = tmp_path / "d2.tif"; make_scene(d2, 0.05, 0.06, 0.55, 0.1)

        pipeline = VegetationIndicesPipeline.__new__(VegetationIndicesPipeline)
        pipeline.pgv = MagicMock(); pipeline._pgv = pipeline.pgv
        pipeline._search_and_download = MagicMock(side_effect=[d1, d2])

        result = pipeline.run(bbox=(-95.0, 41.0, -94.9, 41.1), output_dir=str(tmp_path / "out"),
                               dates=["2024-05", "2024-07"])
        assert result.success
        assert result.stats["n_dates_used"] == 2
        expected_ndvi_d1 = (0.4 - 0.08) / (0.4 + 0.08)
        assert abs(result.stats["per_date"]["2024-05"]["mean_ndvi"] - expected_ndvi_d1) < 0.01

    def test_vegetation_indices_requires_real_dates(self, tmp_path):
        from pygeovision.ai.pipelines.domains import VegetationIndicesPipeline
        pipeline = VegetationIndicesPipeline.__new__(VegetationIndicesPipeline)
        pipeline.pgv = MagicMock(); pipeline._pgv = pipeline.pgv
        result = pipeline.run(bbox=(0, 0, 1, 1), output_dir=str(tmp_path / "out"))
        assert not result.success


# ── Category C: honesty and real-reuse fixes ─────────────────────────────

class TestCategoryCFixes:
    def test_tree_species_now_honestly_fails(self, tmp_path):
        from pygeovision.ai.pipelines.domains import TreeSpeciesPipeline
        pipeline = TreeSpeciesPipeline.__new__(TreeSpeciesPipeline)
        mock_pgv = MagicMock()
        pipeline.pgv = mock_pgv; pipeline._pgv = mock_pgv
        pipeline._search = MagicMock(return_value=[MagicMock()])
        fake_download = MagicMock(success=True, path=str(tmp_path / "scene.tif"))
        mock_pgv.download.return_value = [fake_download]

        result = pipeline.run(bbox=(0, 0, 1, 1), output_dir=str(tmp_path / "out"), date="2024-06")
        assert not result.success  # previously True, misleadingly

    def test_land_surface_temperature_reuses_real_verified_computation(self, tmp_path):
        from pygeovision.ai.pipelines.domains import LandSurfaceTemperaturePipeline
        size = 10
        transform = from_bounds(-74.05, 40.65, -73.95, 40.75, size, size)
        data = np.full((size, size), 46565, dtype="uint16")
        thermal_path = tmp_path / "thermal.tif"
        with rasterio.open(thermal_path, "w", driver="GTiff", height=size, width=size, count=1,
                            dtype="uint16", crs="EPSG:4326", transform=transform) as dst:
            dst.write(data, 1)

        pipeline = LandSurfaceTemperaturePipeline.__new__(LandSurfaceTemperaturePipeline)
        pipeline.pgv = MagicMock(); pipeline._pgv = pipeline.pgv
        pipeline._search_and_download = MagicMock(return_value=thermal_path)
        result = pipeline.run(bbox=(-74.05, 40.65, -73.95, 40.75), output_dir=str(tmp_path / "out"), date="2024-07")

        expected_c = 46565 * 0.00341802 + 149.0 - 273.15
        assert result.success
        assert abs(result.stats["mean_temp_celsius"] - expected_c) < 0.01

    def test_forest_fire_reuses_real_verified_dnbr(self, tmp_path):
        from pygeovision.ai.pipelines.domains import ForestFirePipeline
        size = 10
        transform = from_bounds(-119.5, 37.5, -119.4, 37.6, size, size)

        def make_scene(path, nir, swir2):
            with rasterio.open(path, "w", driver="GTiff", height=size, width=size, count=2,
                                dtype="float32", crs="EPSG:4326", transform=transform) as dst:
                dst.write(np.full((size, size), nir, dtype="float32"), 1)
                dst.write(np.full((size, size), swir2, dtype="float32"), 2)

        pre = tmp_path / "pre.tif"; make_scene(pre, 0.45, 0.15)
        post = tmp_path / "post.tif"; make_scene(post, 0.15, 0.35)

        pipeline = ForestFirePipeline.__new__(ForestFirePipeline)
        pipeline.pgv = MagicMock(); pipeline._pgv = pipeline.pgv
        pipeline._search_and_download_pair = MagicMock(return_value=(pre, post))
        result = pipeline.run(bbox=(-119.5, 37.5, -119.4, 37.6), output_dir=str(tmp_path / "out"),
                               date_before="2024-05", date_after="2024-08")

        assert result.success
        assert abs(result.stats["mean_dnbr"] - 0.9) < 0.01

    def test_volcano_monitoring_uses_distinct_higher_threshold(self):
        from pygeovision.ai.pipelines.domains import VolcanoMonitoringPipeline, UrbanHeatIslandPipeline
        assert VolcanoMonitoringPipeline._HOTSPOT_THRESHOLD_C > UrbanHeatIslandPipeline._HOTSPOT_THRESHOLD_C
        assert VolcanoMonitoringPipeline._HOTSPOT_STAT_KEY == "pct_thermal_anomaly"

    def test_volcano_monitoring_detects_known_hotspot(self, tmp_path):
        from pygeovision.ai.pipelines.domains import VolcanoMonitoringPipeline
        size = 20
        transform = from_bounds(-155.3, 19.4, -155.2, 19.5, size, size)
        data = np.full((size, size), 44000, dtype="uint16")
        data[:5, :5] = 60000
        thermal_path = tmp_path / "thermal.tif"
        with rasterio.open(thermal_path, "w", driver="GTiff", height=size, width=size, count=1,
                            dtype="uint16", crs="EPSG:4326", transform=transform) as dst:
            dst.write(data, 1)

        pipeline = VolcanoMonitoringPipeline.__new__(VolcanoMonitoringPipeline)
        pipeline.pgv = MagicMock(); pipeline._pgv = pipeline.pgv
        pipeline._search_and_download = MagicMock(return_value=thermal_path)
        result = pipeline.run(bbox=(-155.3, 19.4, -155.2, 19.5), output_dir=str(tmp_path / "out"), date="2024-07")

        assert result.success
        assert abs(result.stats["pct_thermal_anomaly"] - (25/400)) < 0.01
        assert "pct_heat_island" not in result.stats


class TestClassOrderingCorrect:
    """The real, structural fix needed for the inheritance-based Category
    C fixes -- Python requires base classes defined before subclasses."""

    def test_urban_heat_island_defined_before_its_subclasses(self):
        import inspect
        import pygeovision.ai.pipelines.domains as domains
        source_lines = inspect.getsource(domains).split("\n")
        def line_of(pattern):
            return next(i for i, l in enumerate(source_lines) if l.startswith(pattern))

        base_line = line_of("class UrbanHeatIslandPipeline")
        assert line_of("class LandSurfaceTemperaturePipeline") > base_line
        assert line_of("class VolcanoMonitoringPipeline") > base_line

    def test_wildfire_severity_defined_before_forest_fire(self):
        import inspect
        import pygeovision.ai.pipelines.domains as domains
        source_lines = inspect.getsource(domains).split("\n")
        def line_of(pattern):
            return next(i for i, l in enumerate(source_lines) if l.startswith(pattern))
        assert line_of("class ForestFirePipeline") > line_of("class WildfireSeverityPipeline")
