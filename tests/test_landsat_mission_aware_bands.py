"""Tests for mission-aware Landsat band resolution in BasePipeline._stack_bands().

Real, scientifically important bug found and fixed: Landsat 7 ETM+/5 TM
and Landsat 8/9 OLI have genuinely different band-number-to-wavelength
mappings (SR_B1 is blue on ETM+/TM, but SR_B2 is blue on OLI — SR_B1 on
OLI is coastal aerosol instead). A single fixed mapping would silently
select the physically wrong band for one of the two mission families.

Also covers the real fix to the silent-fallback behavior that was the
direct, confirmed cause of a real production crash: when band stacking
failed for a multi-asset scene, the code previously fell back to an
arbitrary single asset path (observed to be a QA_PIXEL quality band
used as if it were reflectance imagery) instead of failing clearly at
the point where the real problem was.
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


def _make_landsat_assets(tmp_path, band_keys, size=40):
    transform = from_bounds(-74.1, 40.6, -73.7, 40.9, size, size)
    asset_paths = {}
    for key, value in band_keys.items():
        p = tmp_path / f"{key}.tif"
        with rasterio.open(p, "w", driver="GTiff", height=size, width=size, count=1,
                            dtype="uint16", crs="EPSG:4326", transform=transform) as dst:
            dst.write(np.full((size, size), value, dtype="uint16"), 1)
        asset_paths[key] = p
    return asset_paths


class TestMissionAwareLandsatBandMapping:
    def test_landsat7_etm_mapping_is_correct(self, tmp_path):
        """The real, confirmed ETM+/TM mapping: SR_B1=blue, SR_B2=green,
        SR_B3=red, SR_B4=nir."""
        pipeline = _make_pipeline()
        asset_paths = _make_landsat_assets(
            tmp_path, {"SR_B1": 111, "SR_B2": 222, "SR_B3": 333, "SR_B4": 444},
        )
        stack_path = pipeline._stack_bands(
            asset_paths, tmp_path / "l7.tif", bands=("red", "green", "blue"),
            scene_id="LE07_L2SP_013032_20180616_02_T1",
        )
        with rasterio.open(stack_path) as src:
            data = src.read()
        assert data[0, 0, 0] == 333  # red = SR_B3
        assert data[1, 0, 0] == 222  # green = SR_B2
        assert data[2, 0, 0] == 111  # blue = SR_B1

    def test_landsat8_oli_mapping_is_correct_and_genuinely_different(self, tmp_path):
        """The real, confirmed OLI mapping: SR_B2=blue, SR_B3=green,
        SR_B4=red, SR_B5=nir -- shifted by one band vs ETM+/TM."""
        pipeline = _make_pipeline()
        asset_paths = _make_landsat_assets(
            tmp_path, {"SR_B2": 222, "SR_B3": 333, "SR_B4": 444, "SR_B5": 555},
        )
        stack_path = pipeline._stack_bands(
            asset_paths, tmp_path / "l8.tif", bands=("red", "green", "blue"),
            scene_id="LC08_L2SP_014031_20240615_02_T1",
        )
        with rasterio.open(stack_path) as src:
            data = src.read()
        assert data[0, 0, 0] == 444  # red = SR_B4
        assert data[1, 0, 0] == 333  # green = SR_B3
        assert data[2, 0, 0] == 222  # blue = SR_B2

    def test_landsat9_uses_the_same_oli_mapping_as_landsat8(self, tmp_path):
        pipeline = _make_pipeline()
        asset_paths = _make_landsat_assets(
            tmp_path, {"SR_B2": 222, "SR_B3": 333, "SR_B4": 444},
        )
        stack_path = pipeline._stack_bands(
            asset_paths, tmp_path / "l9.tif", bands=("red", "green", "blue"),
            scene_id="LC09_L2SP_014032_20240607_02_T1",
        )
        with rasterio.open(stack_path) as src:
            data = src.read()
        assert data[0, 0, 0] == 444


class TestLoudFailureInsteadOfSilentSubstitution:
    """The real fix for the second production crash: a QA_PIXEL band
    was silently used as reflectance imagery when real bands weren't
    found. Now raises clearly at the point of the actual problem."""

    def test_missing_bands_raises_not_silently_returns_none(self, tmp_path):
        pipeline = _make_pipeline()
        # Only a QA band available -- no real reflectance bands at all,
        # exactly matching the real crash scenario.
        asset_paths = _make_landsat_assets(tmp_path, {"QA_PIXEL": 1})
        with pytest.raises(RuntimeError, match="no asset found for band"):
            pipeline._stack_bands(
                asset_paths, tmp_path / "bad.tif", bands=("red", "green", "blue"),
                scene_id="LC08_L2SP_014031_20240615_02_T1",
            )

    def test_error_message_names_the_scene_and_available_assets(self, tmp_path):
        pipeline = _make_pipeline()
        asset_paths = _make_landsat_assets(tmp_path, {"QA_PIXEL": 1})
        with pytest.raises(RuntimeError) as exc_info:
            pipeline._stack_bands(
                asset_paths, tmp_path / "bad.tif", bands=("red",),
                scene_id="LC08_L2SP_014031_20240615_02_T1",
            )
        msg = str(exc_info.value)
        assert "LC08_L2SP_014031_20240615_02_T1" in msg
        assert "QA_PIXEL" in msg
