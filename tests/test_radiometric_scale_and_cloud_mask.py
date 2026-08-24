"""Tests for real radiometric scaling and cloud masking in
BasePipeline._stack_bands().

Real gap found and fixed, from a production report that final output
looked "completely dark" with "a lot of clouds" and no visible
correction: _stack_bands() previously wrote raw sensor DN values
straight through with zero radiometric correction and zero cloud
handling. Landsat Collection 2 SR bands are scaled 16-bit integers, not
directly-usable reflectance -- using them as-is (as raw values in the
thousands) is not remotely close to real 0-1 reflectance, and nothing
was ever removing cloud-contaminated pixels.

Verified against the real, documented conversion formulas:
  - Landsat Collection 2 SR (USGS Level-2 Science Product Guide):
    reflectance = DN * 0.0000275 - 0.2
  - Sentinel-2 L2A (ESA product spec): reflectance = DN / 10000
  - Landsat QA_PIXEL cloud bits (USGS spec): bits 1/2/3/4
    (dilated cloud / cirrus / cloud / cloud shadow)
  - Sentinel-2 SCL cloud classes (ESA spec): 3/8/9/10
    (cloud shadow / cloud medium / cloud high / thin cirrus)
"""
import pytest

rasterio = pytest.importorskip("rasterio", reason="rasterio not installed")

import numpy as np
from rasterio.transform import from_bounds


def _make_pipeline():
    from pygeovision.ai.pipelines import BasePipeline

    class FakePipeline(BasePipeline):
        def run(self, bbox, output_dir, **kwargs):
            pass

    return FakePipeline.__new__(FakePipeline)


class TestLandsatRadiometricScaling:
    def test_known_dn_produces_exact_known_reflectance(self, tmp_path):
        pipeline = _make_pipeline()
        size = 20
        transform = from_bounds(-74.1, 40.6, -73.7, 40.9, size, size)
        # reflectance = DN * 0.0000275 - 0.2; for target 0.05 -> DN=9090.9 -> 9091
        known_dn = 9091
        expected = known_dn * 0.0000275 - 0.2

        asset_paths = {}
        for key in ["SR_B4", "SR_B3", "SR_B2"]:
            p = tmp_path / f"{key}.tif"
            with rasterio.open(p, "w", driver="GTiff", height=size, width=size, count=1,
                                dtype="uint16", crs="EPSG:4326", transform=transform) as dst:
                dst.write(np.full((size, size), known_dn, dtype="uint16"), 1)
            asset_paths[key] = p

        stack_path = pipeline._stack_bands(
            asset_paths, tmp_path / "stack.tif", bands=("red", "green", "blue"),
            scene_id="LC08_L2SP_014031_20240615_02_T1", apply_cloud_mask=False,
        )
        with rasterio.open(stack_path) as src:
            assert src.dtypes[0] == "float32"
            data = src.read()
        assert abs(data[0, 0, 0] - expected) < 1e-4

    def test_output_dtype_is_float32_not_raw_uint16(self, tmp_path):
        pipeline = _make_pipeline()
        size = 20
        transform = from_bounds(-74.1, 40.6, -73.7, 40.9, size, size)
        asset_paths = {}
        for key in ["SR_B4", "SR_B3", "SR_B2"]:
            p = tmp_path / f"{key}.tif"
            with rasterio.open(p, "w", driver="GTiff", height=size, width=size, count=1,
                                dtype="uint16", crs="EPSG:4326", transform=transform) as dst:
                dst.write(np.full((size, size), 9091, dtype="uint16"), 1)
            asset_paths[key] = p

        stack_path = pipeline._stack_bands(
            asset_paths, tmp_path / "stack.tif", bands=("red", "green", "blue"),
            scene_id="LC08_L2SP_014031_20240615_02_T1", apply_cloud_mask=False,
        )
        with rasterio.open(stack_path) as src:
            assert src.dtypes[0] == "float32"

    def test_apply_scale_false_preserves_raw_dn(self, tmp_path):
        """The opt-out must genuinely skip scaling, for callers who want
        raw DN for their own reasons."""
        pipeline = _make_pipeline()
        size = 20
        transform = from_bounds(-74.1, 40.6, -73.7, 40.9, size, size)
        known_dn = 9091
        asset_paths = {}
        for key in ["SR_B4", "SR_B3", "SR_B2"]:
            p = tmp_path / f"{key}.tif"
            with rasterio.open(p, "w", driver="GTiff", height=size, width=size, count=1,
                                dtype="uint16", crs="EPSG:4326", transform=transform) as dst:
                dst.write(np.full((size, size), known_dn, dtype="uint16"), 1)
            asset_paths[key] = p

        stack_path = pipeline._stack_bands(
            asset_paths, tmp_path / "stack.tif", bands=("red", "green", "blue"),
            scene_id="LC08_L2SP_014031_20240615_02_T1",
            apply_scale=False, apply_cloud_mask=False,
        )
        with rasterio.open(stack_path) as src:
            data = src.read()
        assert data[0, 0, 0] == known_dn


class TestSentinel2RadiometricScaling:
    def test_known_dn_produces_exact_known_reflectance(self, tmp_path):
        pipeline = _make_pipeline()
        size = 20
        transform = from_bounds(-74.1, 40.6, -73.7, 40.9, size, size)
        known_dn = 500  # reflectance = DN/10000 = 0.05
        asset_paths = {}
        for key in ["B04", "B03", "B02"]:
            p = tmp_path / f"{key}.tif"
            with rasterio.open(p, "w", driver="GTiff", height=size, width=size, count=1,
                                dtype="uint16", crs="EPSG:4326", transform=transform) as dst:
                dst.write(np.full((size, size), known_dn, dtype="uint16"), 1)
            asset_paths[key] = p

        stack_path = pipeline._stack_bands(
            asset_paths, tmp_path / "s2_stack.tif", bands=("red", "green", "blue"),
            scene_id="S2A_MSIL2A_20240615T153941_R011_T18TXL", apply_cloud_mask=False,
        )
        with rasterio.open(stack_path) as src:
            data = src.read()
        assert abs(data[0, 0, 0] - 0.05) < 1e-4


class TestLandsatCloudMasking:
    def test_cloud_flagged_pixels_are_masked_clear_pixels_are_not(self, tmp_path):
        pipeline = _make_pipeline()
        size = 40
        transform = from_bounds(-74.1, 40.6, -73.7, 40.9, size, size)
        known_dn = 9091
        asset_paths = {}
        for key in ["SR_B4", "SR_B3", "SR_B2"]:
            p = tmp_path / f"{key}.tif"
            with rasterio.open(p, "w", driver="GTiff", height=size, width=size, count=1,
                                dtype="uint16", crs="EPSG:4326", transform=transform) as dst:
                dst.write(np.full((size, size), known_dn, dtype="uint16"), 1)
            asset_paths[key] = p

        qa_data = np.zeros((size, size), dtype="uint16")
        qa_data[:20, :] = 8    # bit 3 (Cloud) set
        qa_data[20:, :] = 64   # bit 6 (Clear) set, no cloud bits
        qa_path = tmp_path / "QA_PIXEL.tif"
        with rasterio.open(qa_path, "w", driver="GTiff", height=size, width=size, count=1,
                            dtype="uint16", crs="EPSG:4326", transform=transform) as dst:
            dst.write(qa_data, 1)
        asset_paths["QA_PIXEL"] = qa_path

        stack_path = pipeline._stack_bands(
            asset_paths, tmp_path / "stack.tif", bands=("red", "green", "blue"),
            scene_id="LC08_L2SP_014031_20240615_02_T1",
        )
        with rasterio.open(stack_path) as src:
            data = src.read()
        assert np.isnan(data[0, 5, 0])    # cloud region
        assert not np.isnan(data[0, 25, 0])  # clear region

    def test_dilated_cloud_and_cirrus_bits_are_also_masked(self, tmp_path):
        """Not just the literal 'Cloud' bit -- dilated cloud (bit 1) and
        cirrus (bit 2) are real, separate flags that should also be masked."""
        pipeline = _make_pipeline()
        size = 20
        transform = from_bounds(-74.1, 40.6, -73.7, 40.9, size, size)
        asset_paths = {}
        for key in ["SR_B4", "SR_B3", "SR_B2"]:
            p = tmp_path / f"{key}.tif"
            with rasterio.open(p, "w", driver="GTiff", height=size, width=size, count=1,
                                dtype="uint16", crs="EPSG:4326", transform=transform) as dst:
                dst.write(np.full((size, size), 9091, dtype="uint16"), 1)
            asset_paths[key] = p

        qa_data = np.zeros((size, size), dtype="uint16")
        qa_data[:10, :] = 2   # bit 1 (Dilated Cloud)
        qa_data[10:, :] = 4   # bit 2 (Cirrus)
        qa_path = tmp_path / "QA_PIXEL.tif"
        with rasterio.open(qa_path, "w", driver="GTiff", height=size, width=size, count=1,
                            dtype="uint16", crs="EPSG:4326", transform=transform) as dst:
            dst.write(qa_data, 1)
        asset_paths["QA_PIXEL"] = qa_path

        stack_path = pipeline._stack_bands(
            asset_paths, tmp_path / "stack.tif", bands=("red", "green", "blue"),
            scene_id="LC08_L2SP_014031_20240615_02_T1",
        )
        with rasterio.open(stack_path) as src:
            data = src.read()
        assert np.isnan(data[0, 5, 0])    # dilated cloud region
        assert np.isnan(data[0, 15, 0])   # cirrus region

    def test_missing_qa_band_warns_but_does_not_crash(self, tmp_path):
        pipeline = _make_pipeline()
        size = 20
        transform = from_bounds(-74.1, 40.6, -73.7, 40.9, size, size)
        asset_paths = {}
        for key in ["SR_B4", "SR_B3", "SR_B2"]:
            p = tmp_path / f"{key}.tif"
            with rasterio.open(p, "w", driver="GTiff", height=size, width=size, count=1,
                                dtype="uint16", crs="EPSG:4326", transform=transform) as dst:
                dst.write(np.full((size, size), 9091, dtype="uint16"), 1)
            asset_paths[key] = p
        # No QA_PIXEL asset present at all
        result = pipeline._stack_bands(
            asset_paths, tmp_path / "stack.tif", bands=("red", "green", "blue"),
            scene_id="LC08_L2SP_014031_20240615_02_T1",
        )
        assert result is not None


class TestSentinel2CloudMasking:
    def test_scl_cloud_classes_are_masked(self, tmp_path):
        pipeline = _make_pipeline()
        size = 40
        transform = from_bounds(-74.1, 40.6, -73.7, 40.9, size, size)
        asset_paths = {}
        for key in ["B04", "B03", "B02"]:
            p = tmp_path / f"{key}.tif"
            with rasterio.open(p, "w", driver="GTiff", height=size, width=size, count=1,
                                dtype="uint16", crs="EPSG:4326", transform=transform) as dst:
                dst.write(np.full((size, size), 500, dtype="uint16"), 1)
            asset_paths[key] = p

        scl_data = np.zeros((size, size), dtype="uint8")
        scl_data[:20, :] = 9   # cloud high probability
        scl_data[20:, :] = 4   # vegetation (clear)
        scl_path = tmp_path / "SCL.tif"
        with rasterio.open(scl_path, "w", driver="GTiff", height=size, width=size, count=1,
                            dtype="uint8", crs="EPSG:4326", transform=transform) as dst:
            dst.write(scl_data, 1)
        asset_paths["SCL"] = scl_path

        stack_path = pipeline._stack_bands(
            asset_paths, tmp_path / "s2_stack.tif", bands=("red", "green", "blue"),
            scene_id="S2A_MSIL2A_20240615T153941_R011_T18TXL",
        )
        with rasterio.open(stack_path) as src:
            data = src.read()
        assert np.isnan(data[0, 5, 0])
        assert not np.isnan(data[0, 25, 0])
