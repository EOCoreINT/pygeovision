"""Tests for the classification (LandCoverPipeline) fixes.

Real, production-confirmed bugs found and fixed, traced from an actual
CLI failure log:

  1. _search_and_download() returned downloads[0].path -- for a
     multi-asset Sentinel-2 scene (15 separate band files), this was
     whichever file happened to be downloaded first (confirmed from the
     real log: AOT, a 1-band aerosol auxiliary layer), not a usable
     multi-band composite. DownloadResult now preserves ALL real asset
     paths (asset_paths), and _stack_bands() builds a genuine,
     correctly-ordered, correctly-resampled multi-band GeoTIFF from them.

  2. LandCoverPipeline's TileMetadata construction used the wrong
     field name ('path' instead of 'source_file') and omitted five
     required fields (tile_id, transform, row_off, col_off, bands)
     entirely -- would never have worked regardless of the imagery fix.

  3. ESAWorldCoverLabeler and DynamicWorldLabeler's LabelingResult
     construction used field names that don't exist on the real
     dataclass at all (tile_path, success, labeler, metadata) --
     'success' is actually a computed property, not a settable field.
     This made both labelers crash unconditionally on EVERY call,
     success or failure alike.

  4. Both labelers called _write_label_geotiff() with arguments that
     don't match its real (TileMetadata, output_directory) signature at
     all -- they were calling it as if it took (output_path, profile_dict).
     Fixed by adding a correctly-scoped _write_mask_geotiff() helper
     matching what they actually need.
"""
import pytest

rasterio = pytest.importorskip("rasterio", reason="rasterio not installed")

import numpy as np
from pathlib import Path
from rasterio.transform import from_bounds


def _make_asset_paths(tmp_path, size=60):
    """Reproduces the exact real Sentinel-2 filenames from a real
    production failure log."""
    transform = from_bounds(-74.1, 40.6, -73.7, 40.9, size, size)
    real_filenames = {
        "AOT": "S2A_MSIL2A_20240615T153941_R011_T18TXL_AOT_10m.tif",
        "B02": "S2A_MSIL2A_20240615T153941_R011_T18TXL_B02_10m.tif",
        "B03": "S2A_MSIL2A_20240615T153941_R011_T18TXL_B03_10m.tif",
        "B04": "S2A_MSIL2A_20240615T153941_R011_T18TXL_B04_10m.tif",
        "B08": "S2A_MSIL2A_20240615T153941_R011_T18TXL_B08_10m.tif",
    }
    rng = np.random.default_rng(0)
    for key, fname in real_filenames.items():
        val = rng.integers(500, 3000, (size, size)).astype("uint16")
        p = tmp_path / fname
        with rasterio.open(p, "w", driver="GTiff", height=size, width=size, count=1,
                            dtype="uint16", crs="EPSG:4326", transform=transform) as dst:
            dst.write(val, 1)

    def derive_key(p):
        p = Path(p)
        stem_parts = p.stem.split("_")
        if len(stem_parts) >= 2 and stem_parts[-1].rstrip("m").isdigit():
            return stem_parts[-2]
        return stem_parts[-1]

    return {derive_key(tmp_path / f): tmp_path / f for f in real_filenames.values()}


class TestDownloadResultAssetPaths:
    def test_asset_paths_field_exists_and_defaults_empty(self):
        from pygeovision.data.fetch import DownloadResult
        r = DownloadResult(scene_id="x")
        assert r.asset_paths == {}

    def test_asset_paths_can_hold_multiple_real_bands(self, tmp_path):
        from pygeovision.data.fetch import DownloadResult
        asset_paths = _make_asset_paths(tmp_path)
        r = DownloadResult(scene_id="x", asset_paths=asset_paths)
        assert set(r.asset_paths.keys()) == {"AOT", "B02", "B03", "B04", "B08"}


class TestStackBandsBuildsRealComposite:
    def test_real_filenames_parse_to_correct_asset_keys(self, tmp_path):
        asset_paths = _make_asset_paths(tmp_path)
        assert set(asset_paths.keys()) == {"AOT", "B02", "B03", "B04", "B08"}

    def test_stack_selects_rgb_not_the_auxiliary_aot_band(self, tmp_path):
        """The core regression test: the fix must build a real RGB
        composite from B04/B03/B02, not return the AOT auxiliary file
        that caused the real 'expected 3 channels, got 1' crash."""
        from pygeovision.ai.pipelines import BasePipeline

        class FakePipeline(BasePipeline):
            def run(self, bbox, output_dir, **kwargs):
                pass

        asset_paths = _make_asset_paths(tmp_path)
        pipeline = FakePipeline.__new__(FakePipeline)
        stack_path = pipeline._stack_bands(
            asset_paths, tmp_path / "stack.tif", bands=("red", "green", "blue"),
        )
        assert stack_path is not None
        with rasterio.open(stack_path) as src:
            assert src.count == 3
            assert src.descriptions == ("red", "green", "blue")

    def test_missing_band_raises_clearly_not_a_silent_none(self, tmp_path):
        """Updated for the loud-failure fix: silently returning None (and
        letting the caller fall back to an arbitrary single asset) was
        the direct, confirmed cause of a real production crash -- a
        QA_PIXEL quality band got used as reflectance imagery. Now
        raises clearly at the point of the actual problem instead."""
        from pygeovision.ai.pipelines import BasePipeline

        class FakePipeline(BasePipeline):
            def run(self, bbox, output_dir, **kwargs):
                pass

        asset_paths = _make_asset_paths(tmp_path)
        del asset_paths["B04"]  # remove red -- an incomplete scene
        pipeline = FakePipeline.__new__(FakePipeline)
        with pytest.raises(RuntimeError, match="no asset found for band"):
            pipeline._stack_bands(
                asset_paths, tmp_path / "stack.tif", bands=("red", "green", "blue"),
            )

    def test_mismatched_resolution_bands_are_resampled_to_common_grid(self, tmp_path):
        """A real Sentinel-2 characteristic: 10m and 20m bands genuinely
        ship at different native resolutions in the same product."""
        from pygeovision.ai.pipelines import BasePipeline

        class FakePipeline(BasePipeline):
            def run(self, bbox, output_dir, **kwargs):
                pass

        size_10m, size_20m = 60, 30
        t10 = from_bounds(-74.1, 40.6, -73.7, 40.9, size_10m, size_10m)
        t20 = from_bounds(-74.1, 40.6, -73.7, 40.9, size_20m, size_20m)
        rng = np.random.default_rng(1)
        asset_paths = {}
        for key, (size, transform) in {
            "B02": (size_10m, t10), "B03": (size_10m, t10),
            "B04": (size_10m, t10), "B11": (size_20m, t20),
        }.items():
            val = rng.integers(500, 3000, (size, size)).astype("uint16")
            p = tmp_path / f"{key}.tif"
            with rasterio.open(p, "w", driver="GTiff", height=size, width=size, count=1,
                                dtype="uint16", crs="EPSG:4326", transform=transform) as dst:
                dst.write(val, 1)
            asset_paths[key] = p

        pipeline = FakePipeline.__new__(FakePipeline)
        pipeline._S2_BAND_ALIASES["swir1"] = ("B11",)
        stack_path = pipeline._stack_bands(asset_paths, tmp_path / "mixed.tif", bands=("red", "swir1"))

        with rasterio.open(stack_path) as src:
            assert src.height == size_10m and src.width == size_10m
            data = src.read()
            assert len(np.unique(data[1])) > 5, "resampled band looks degenerate"


class TestLabelingResultRealFieldNames:
    """Regression tests for the real dataclass-mismatch bug: 'success'
    is a computed property (not settable), and 'tile_path'/'labeler'/
    'metadata' don't exist as fields at all -- both labelers crashed
    unconditionally on every single call before this fix."""

    def test_labeling_result_success_is_computed_not_settable(self):
        from pygeovision.ai.labeling.base_labeler import LabelingResult
        r = LabelingResult(tile_id="t1", label_path=Path("/tmp/x.tif"))
        assert r.success is True  # no error, has a label_path -> success

        r2 = LabelingResult(tile_id="t1", error="something failed")
        assert r2.success is False

    def test_labeling_result_rejects_nonexistent_fields(self):
        from pygeovision.ai.labeling.base_labeler import LabelingResult
        with pytest.raises(TypeError):
            LabelingResult(tile_path="x", success=True, labeler="y", metadata={})


class TestWriteMaskGeotiffHelper:
    def test_writes_to_the_exact_given_path_with_the_given_profile(self, tmp_path):
        from pygeovision.ai.labeling.base_labeler import BaseLabeler

        class FakeLabeler(BaseLabeler):
            name = "fake"
            supported_tasks = ["land_cover"]
            def label_tile(self, *a, **kw): pass

        labeler = FakeLabeler.__new__(FakeLabeler)
        mask = np.ones((20, 20), dtype=np.uint8) * 3
        profile = {
            "driver": "GTiff", "dtype": "uint8", "width": 20, "height": 20, "count": 1,
            "crs": "EPSG:4326", "transform": from_bounds(-74.1, 40.6, -73.7, 40.9, 20, 20),
            "compress": "lzw",
        }
        out_path = tmp_path / "exact_output.tif"
        result_path = labeler._write_mask_geotiff(mask, out_path, profile)

        assert result_path == out_path
        assert out_path.exists()
        with rasterio.open(out_path) as src:
            assert src.count == 1
            np.testing.assert_array_equal(src.read(1), mask)
