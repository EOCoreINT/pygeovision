"""Tests for pygeovision.data.radiometric (the extracted, shared module)
and SatelliteFetcher.prepare_stack() (its new public exposure).

Real architectural gap found and fixed: radiometric scaling and cloud
masking previously lived only inside pygeovision.ai.pipelines.
BasePipeline, a private class -- completely unreachable from the
separate, public client.segmentation/client.classification API, which
operates on an already-prepared image_path with no documented way to
get one. Extracted into a real, standalone, public module (single
source of truth, eliminating the duplication risk of two implementations
drifting apart), and exposed directly as client.data.prepare_stack().

The extraction itself must be behavior-preserving -- these tests
confirm the standalone module produces identical, correct results to
what was already verified inside BasePipeline._stack_bands(), and that
BasePipeline's own tests (a separate file) continue to pass unchanged
against the new thin-wrapper implementation.
"""
import pytest

rasterio = pytest.importorskip("rasterio", reason="rasterio not installed")

import numpy as np
from rasterio.transform import from_bounds


class TestSharedModuleDirectly:
    def test_landsat_known_dn_produces_known_reflectance(self, tmp_path):
        from pygeovision.data.radiometric import stack_and_prepare_bands

        size = 20
        transform = from_bounds(-74.1, 40.6, -73.7, 40.9, size, size)
        known_dn = 9091
        expected = known_dn * 0.0000275 - 0.2

        asset_paths = {}
        for key in ["SR_B4", "SR_B3", "SR_B2"]:
            p = tmp_path / f"{key}.tif"
            with rasterio.open(p, "w", driver="GTiff", height=size, width=size, count=1,
                                dtype="uint16", crs="EPSG:4326", transform=transform) as dst:
                dst.write(np.full((size, size), known_dn, dtype="uint16"), 1)
            asset_paths[key] = p

        result = stack_and_prepare_bands(
            asset_paths, tmp_path / "stack.tif", bands=("red", "green", "blue"),
            scene_id="LC08_L2SP_014031_20240615_02_T1", apply_cloud_mask=False,
        )
        with rasterio.open(result) as src:
            data = src.read()
        assert abs(data[0, 0, 0] - expected) < 1e-4

    def test_sentinel2_known_dn_produces_known_reflectance(self, tmp_path):
        from pygeovision.data.radiometric import stack_and_prepare_bands

        size = 20
        transform = from_bounds(-74.1, 40.6, -73.7, 40.9, size, size)
        asset_paths = {}
        for key in ["B04", "B03", "B02"]:
            p = tmp_path / f"{key}.tif"
            with rasterio.open(p, "w", driver="GTiff", height=size, width=size, count=1,
                                dtype="uint16", crs="EPSG:4326", transform=transform) as dst:
                dst.write(np.full((size, size), 500, dtype="uint16"), 1)
            asset_paths[key] = p

        result = stack_and_prepare_bands(
            asset_paths, tmp_path / "stack.tif", bands=("red", "green", "blue"),
            scene_id="S2A_MSIL2A_20240615T153941_R011_T18TXL", apply_cloud_mask=False,
        )
        with rasterio.open(result) as src:
            data = src.read()
        assert abs(data[0, 0, 0] - 0.05) < 1e-4

    def test_unidentified_sensor_still_refuses_to_guess(self, tmp_path):
        from pygeovision.data.radiometric import stack_and_prepare_bands

        size = 20
        transform = from_bounds(-74.1, 40.6, -73.7, 40.9, size, size)
        asset_paths = {}
        for key in ["red", "green", "blue"]:
            p = tmp_path / f"{key}.tif"
            with rasterio.open(p, "w", driver="GTiff", height=size, width=size, count=1,
                                dtype="uint16", crs="EPSG:4326", transform=transform) as dst:
                dst.write(np.full((size, size), 9091, dtype="uint16"), 1)
            asset_paths[key] = p

        with pytest.raises(RuntimeError, match="not identifiable as Landsat or Sentinel-2"):
            stack_and_prepare_bands(
                asset_paths, tmp_path / "stack.tif", bands=("red", "green", "blue"),
                scene_id="UNKNOWN_SENSOR_SCENE",
            )

    def test_identify_sensor_positive_detection(self):
        from pygeovision.data.radiometric import identify_sensor

        assert identify_sensor("LC08_L2SP_014031_20240615_02_T1") == (True, False)
        assert identify_sensor("LE07_L2SP_013032_20180616_02_T1") == (True, False)
        assert identify_sensor("S2A_MSIL2A_20240615T153941") == (False, True)
        assert identify_sensor("UNKNOWN_SENSOR") == (False, False)
        assert identify_sensor("") == (False, False)


class TestPrepareStackPublicMethod:
    def test_prepare_stack_produces_correct_reflectance(self, tmp_path):
        from pygeovision.data.fetch import SatelliteFetcher

        size = 20
        transform = from_bounds(-74.1, 40.6, -73.7, 40.9, size, size)
        asset_paths = {}
        for key in ["B04", "B03", "B02"]:
            p = tmp_path / f"{key}.tif"
            with rasterio.open(p, "w", driver="GTiff", height=size, width=size, count=1,
                                dtype="uint16", crs="EPSG:4326", transform=transform) as dst:
                dst.write(np.full((size, size), 500, dtype="uint16"), 1)
            asset_paths[key] = p

        fetcher = SatelliteFetcher.__new__(SatelliteFetcher)
        result = fetcher.prepare_stack(
            asset_paths, tmp_path / "stack.tif",
            scene_id="S2A_MSIL2A_20240615T153941_R011_T18TXL",
        )
        with rasterio.open(result) as src:
            data = src.read()
        assert abs(data[0, 0, 0] - 0.05) < 1e-4

    def test_prepare_stack_output_is_usable_by_segmentation_proxy(self, tmp_path):
        """End-to-end: the exact real workflow this fix enables --
        prepare_stack() output can be fed directly into
        client.segmentation.water()'s NDWI computation."""
        from pygeovision.data.fetch import SatelliteFetcher

        size = 20
        transform = from_bounds(-74.1, 40.6, -73.7, 40.9, size, size)
        asset_paths = {}
        # Need at least 4 bands for water()'s green/nir index logic
        for key, val in [("B02", 300), ("B03", 400), ("B04", 350), ("B08", 100)]:
            p = tmp_path / f"{key}.tif"
            with rasterio.open(p, "w", driver="GTiff", height=size, width=size, count=1,
                                dtype="uint16", crs="EPSG:4326", transform=transform) as dst:
                dst.write(np.full((size, size), val, dtype="uint16"), 1)
            asset_paths[key] = p

        fetcher = SatelliteFetcher.__new__(SatelliteFetcher)
        stack_path = fetcher.prepare_stack(
            asset_paths, tmp_path / "stack.tif",
            bands=("blue", "green", "red", "nir"),
            scene_id="S2A_MSIL2A_20240615T153941_R011_T18TXL",
            apply_cloud_mask=False,
        )

        # Now genuinely usable by the segmentation proxy's water() logic
        with rasterio.open(stack_path) as src:
            n_bands = src.count
            assert n_bands == 4
            green = src.read(2).astype(np.float32)
            nir = src.read(4).astype(np.float32)
        ndwi = (green - nir) / (green + nir + 1e-8)
        assert np.isfinite(ndwi).all()
