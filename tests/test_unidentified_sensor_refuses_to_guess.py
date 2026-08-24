"""Tests for real sensor identification in _stack_bands().

Real, active correctness bug found while directly auditing whether
radiometric correction is genuinely applied for "whether Sentinel or
Landsat data": sensor detection was binary (is_landsat / else assume
Sentinel-2), so any THIRD sensor (NAIP, PlanetScope, MODIS, commercial
VHR, etc.) would silently get the Sentinel-2 DN/10000 formula applied --
not a crash, just quietly wrong reflectance values with no error at all.
Fixed to positively identify both Landsat and Sentinel-2, and refuse to
guess for anything else.
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


def _make_generic_assets(tmp_path, size=20):
    transform = from_bounds(-74.1, 40.6, -73.7, 40.9, size, size)
    asset_paths = {}
    for key in ["red", "green", "blue"]:
        p = tmp_path / f"{key}.tif"
        with rasterio.open(p, "w", driver="GTiff", height=size, width=size, count=1,
                            dtype="uint16", crs="EPSG:4326", transform=transform) as dst:
            dst.write(np.full((size, size), 9091, dtype="uint16"), 1)
        asset_paths[key] = p
    return asset_paths


class TestUnidentifiedSensorRefusesToGuess:
    def test_unrecognised_scene_id_raises_instead_of_silently_scaling(self, tmp_path):
        pipeline = _make_pipeline()
        asset_paths = _make_generic_assets(tmp_path)
        with pytest.raises(RuntimeError, match="not identifiable as Landsat or Sentinel-2"):
            pipeline._stack_bands(
                asset_paths, tmp_path / "stack.tif", bands=("red", "green", "blue"),
                scene_id="NAIP_2024_SOMETHING_UNRECOGNIZED",
            )

    def test_empty_scene_id_raises_instead_of_silently_scaling(self, tmp_path):
        pipeline = _make_pipeline()
        asset_paths = _make_generic_assets(tmp_path)
        with pytest.raises(RuntimeError, match="not identifiable as Landsat or Sentinel-2"):
            pipeline._stack_bands(
                asset_paths, tmp_path / "stack.tif", bands=("red", "green", "blue"),
            )

    def test_apply_scale_false_bypasses_the_requirement(self, tmp_path):
        """An unidentified sensor with scaling explicitly disabled
        should succeed -- honestly unscaled, not silently guessed."""
        pipeline = _make_pipeline()
        asset_paths = _make_generic_assets(tmp_path)
        result = pipeline._stack_bands(
            asset_paths, tmp_path / "stack.tif", bands=("red", "green", "blue"),
            scene_id="NAIP_2024_SOMETHING_UNRECOGNIZED", apply_scale=False,
        )
        with rasterio.open(result) as src:
            assert src.dtypes[0] == "uint16"  # raw DN, genuinely unscaled
            data = src.read()
        assert data[0, 0, 0] == 9091

    def test_sentinel2_scene_id_is_positively_recognised(self, tmp_path):
        """Confirms the fix didn't just add a blanket rejection -- real
        Sentinel-2 scene IDs must still work correctly."""
        pipeline = _make_pipeline()
        size = 20
        transform = from_bounds(-74.1, 40.6, -73.7, 40.9, size, size)
        asset_paths = {}
        for key in ["B04", "B03", "B02"]:
            p = tmp_path / f"{key}.tif"
            with rasterio.open(p, "w", driver="GTiff", height=size, width=size, count=1,
                                dtype="uint16", crs="EPSG:4326", transform=transform) as dst:
                dst.write(np.full((size, size), 500, dtype="uint16"), 1)
            asset_paths[key] = p

        result = pipeline._stack_bands(
            asset_paths, tmp_path / "stack.tif", bands=("red", "green", "blue"),
            scene_id="S2A_MSIL2A_20240615T153941_R011_T18TXL", apply_cloud_mask=False,
        )
        with rasterio.open(result) as src:
            data = src.read()
        assert abs(data[0, 0, 0] - 0.05) < 1e-4

    def test_landsat_scene_id_is_still_positively_recognised(self, tmp_path):
        """Confirms Landsat detection is unaffected by this fix."""
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

        result = pipeline._stack_bands(
            asset_paths, tmp_path / "stack.tif", bands=("red", "green", "blue"),
            scene_id="LC08_L2SP_014031_20240615_02_T1", apply_cloud_mask=False,
        )
        with rasterio.open(result) as src:
            data = src.read()
        assert abs(data[0, 0, 0] - 0.05) < 1e-4
