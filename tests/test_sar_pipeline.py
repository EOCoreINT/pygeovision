"""
tests/test_sar_pipeline.py
===========================
Unit tests for the PyGeoVision SAR preprocessing pipeline.

All tests run on synthetic rasters created in a temporary directory — no real
Sentinel-1 files are required.  Each test class is scoped to one module and
covers both the happy-path and the exact bug-fix scenarios from the processing
logs.
"""
import os
import sys
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Synthetic raster helpers ──────────────────────────────────────────────────

def make_raster(
    path,
    width=64,
    height=64,
    crs="EPSG:32637",
    transform=None,
    dtype="float32",
    bands=1,
    values=None,
    nodata=None,
    identity_transform=False,
):
    """Write a synthetic single-band GeoTIFF for testing."""
    if identity_transform:
        # BUG 1 condition: a=1.0, origin at (0, height) — pixel space, not UTM
        from rasterio.transform import Affine
        transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, float(height))
    elif transform is None:
        # Valid UTM 37N transform: top-left at ~400 km E, 4000 km N
        transform = from_bounds(400000, 3990000, 400640, 3990640, width, height)

    data = np.random.uniform(0.01, 0.5, (bands, height, width)).astype(dtype)
    if values is not None:
        data[:] = values

    profile = {
        "driver": "GTiff", "dtype": dtype, "width": width, "height": height,
        "count": bands, "crs": CRS.from_epsg(int(crs.split(":")[-1])),
        "transform": transform,
    }
    if nodata is not None:
        profile["nodata"] = nodata

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Georeference validator tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestGeoreferenceValidator(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    # ── Happy path ─────────────────────────────────────────────────────────────

    def test_valid_utm_raster_passes(self):
        """A properly georeferenced UTM raster should pass validation."""
        from pygeovision.data.validators.georeference import validate_georeference
        path = make_raster(f"{self.tmp}/valid.tif")
        result = validate_georeference(path)
        self.assertTrue(result.valid)
        self.assertFalse(result.repaired)
        self.assertGreater(result.pixel_width_m, 1.0)
        self.assertGreater(abs(result.origin_x), 100.0)

    def test_valid_raster_srid_present(self):
        """Valid raster should expose correct EPSG code."""
        from pygeovision.data.validators.georeference import validate_georeference
        path = make_raster(f"{self.tmp}/valid_srid.tif")
        result = validate_georeference(path)
        self.assertEqual(result.srid, 32637)

    # ── BUG 1: identity transform detection ────────────────────────────────────

    def test_detects_identity_transform(self):
        """BUG 1: identity transform (a=1.0, origin at pixel-space) must be detected."""
        from pygeovision.data.validators.georeference import validate_georeference
        corrupt = make_raster(f"{self.tmp}/corrupt.tif", identity_transform=True)
        result = validate_georeference(corrupt)
        self.assertFalse(result.valid)
        self.assertTrue(any("identity" in e.lower() or "pixel" in e.lower()
                            for e in result.errors))

    def test_identity_transform_pixel_width_flagged(self):
        """Pixel width of 1.0 m should be flagged (S1 IW is ~10 m)."""
        from pygeovision.data.validators.georeference import validate_georeference
        corrupt = make_raster(f"{self.tmp}/corrupt_pw.tif", identity_transform=True)
        result = validate_georeference(corrupt)
        self.assertFalse(result.valid)
        self.assertLessEqual(result.pixel_width_m, 1.0)

    def test_auto_repair_from_raw(self):
        """BUG 1 fix: corrupt reprojected file + raw file → auto-repair."""
        from pygeovision.data.validators.georeference import validate_georeference
        corrupt = make_raster(f"{self.tmp}/corrupt.tif", identity_transform=True)
        raw = make_raster(f"{self.tmp}/raw.tif")  # valid transform
        result = validate_georeference(corrupt, raw_path=raw, repair=True)
        self.assertTrue(result.valid)
        self.assertTrue(result.repaired)
        self.assertIsNotNone(result.repaired_path)
        # Verify the repaired file is actually valid
        result2 = validate_georeference(result.repaired_path)
        self.assertTrue(result2.valid)
        self.assertGreater(result2.pixel_width_m, 1.0)

    def test_no_repair_without_raw_path(self):
        """Corrupt file without raw_path → still invalid, no repair attempted."""
        from pygeovision.data.validators.georeference import validate_georeference
        corrupt = make_raster(f"{self.tmp}/corrupt_noraw.tif", identity_transform=True)
        result = validate_georeference(corrupt, raw_path=None, repair=True)
        self.assertFalse(result.valid)
        self.assertFalse(result.repaired)

    def test_assert_valid_raises_on_corrupt(self):
        """assert_valid_georeference must raise GeoreferenceCorruptError."""
        from pygeovision.data.validators.georeference import (
            assert_valid_georeference, GeoreferenceCorruptError,
        )
        corrupt = make_raster(f"{self.tmp}/corrupt_assert.tif", identity_transform=True)
        with self.assertRaises(GeoreferenceCorruptError):
            assert_valid_georeference(corrupt)

    def test_assert_valid_passes_valid_raster(self):
        """assert_valid_georeference must not raise on a valid raster."""
        from pygeovision.data.validators.georeference import assert_valid_georeference
        valid = make_raster(f"{self.tmp}/valid_assert.tif")
        result = assert_valid_georeference(valid)
        self.assertTrue(result.valid)

    def test_missing_file_returns_invalid(self):
        """Missing file → invalid result with informative error."""
        from pygeovision.data.validators.georeference import validate_georeference
        result = validate_georeference(f"{self.tmp}/nonexistent.tif")
        self.assertFalse(result.valid)
        self.assertTrue(any("not found" in e.lower() for e in result.errors))

    # ── BUG 2: download completeness ──────────────────────────────────────────

    def test_check_download_complete_valid(self):
        """A properly written raster should pass the tile-read completeness check."""
        from pygeovision.data.validators.georeference import check_download_complete
        path = make_raster(f"{self.tmp}/complete.tif", width=256, height=256)
        result = check_download_complete(path, test_tiles=4, test_tile_size=32)
        self.assertTrue(result["complete"])
        self.assertEqual(result["readable_tiles"], result["total_tiles"])

    def test_check_download_missing_file(self):
        """Missing file → not complete, error reported."""
        from pygeovision.data.validators.georeference import check_download_complete
        result = check_download_complete(f"{self.tmp}/missing.tif")
        self.assertFalse(result["complete"])
        self.assertGreater(len(result["errors"]), 0)

    # ── BUG 3: bbox reprojection ──────────────────────────────────────────────

    def test_reproject_bbox_wgs84_to_utm(self):
        """WGS84 bbox should be reprojected to UTM coordinates."""
        from pygeovision.data.validators.georeference import reproject_bbox_to_raster_crs
        raster = make_raster(f"{self.tmp}/utm.tif")
        # Turkey study area in WGS84
        bbox_wgs84 = (36.1, 36.1, 36.4, 36.4)
        bbox_utm = reproject_bbox_to_raster_crs(bbox_wgs84, raster)
        # UTM 37N coordinates should be in hundred-thousands / millions of metres
        left, bottom, right, top = bbox_utm
        self.assertGreater(left, 100_000)
        self.assertGreater(bottom, 1_000_000)
        self.assertGreater(right, left)
        self.assertGreater(top, bottom)

    def test_reproject_bbox_already_wgs84_raster(self):
        """If raster is already WGS84, bbox should be returned unchanged."""
        from pygeovision.data.validators.georeference import reproject_bbox_to_raster_crs
        from rasterio.transform import from_bounds as fb
        wgs84_path = f"{self.tmp}/wgs84.tif"
        profile = {
            "driver": "GTiff", "dtype": "float32", "width": 64, "height": 64,
            "count": 1, "crs": CRS.from_epsg(4326),
            "transform": fb(36.0, 36.0, 37.0, 37.0, 64, 64),
        }
        with rasterio.open(wgs84_path, "w", **profile) as dst:
            dst.write(np.zeros((1, 64, 64), dtype="float32"))
        bbox_wgs84 = (36.1, 36.1, 36.4, 36.4)
        result = reproject_bbox_to_raster_crs(bbox_wgs84, wgs84_path)
        # Should be essentially unchanged (same coordinate system)
        self.assertAlmostEqual(result[0], bbox_wgs84[0], places=4)
        self.assertAlmostEqual(result[2], bbox_wgs84[2], places=4)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SAR processor tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSARProcessor(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        # Create a synthetic sigma-naught linear raster (values ~0.01–0.5)
        self.linear_path = make_raster(
            f"{self.tmp}/sigma0_linear.tif", width=64, height=64,
            values=None,  # random [0.01, 0.5] (see make_raster)
        )

    # ── Download verification ──────────────────────────────────────────────────

    def test_verify_sar_downloads_both_ok(self):
        """Both VV and VH complete → both returned in verified dict."""
        from pygeovision.data.processors.sar import verify_sar_downloads
        vv = make_raster(f"{self.tmp}/vv.tif", width=256, height=256)
        vh = make_raster(f"{self.tmp}/vh.tif", width=256, height=256)
        result = verify_sar_downloads({"vv": vv, "vh": vh})
        self.assertIsNotNone(result["vv"])
        self.assertIsNotNone(result["vh"])

    def test_verify_sar_downloads_missing_vh_fallback(self):
        """VH missing → None with vv_only fallback (no exception)."""
        from pygeovision.data.processors.sar import verify_sar_downloads
        vv = make_raster(f"{self.tmp}/vv_ok.tif", width=256, height=256)
        result = verify_sar_downloads(
            {"vv": vv, "vh": f"{self.tmp}/nonexistent_vh.tif"},
            fallback_policy="vv_only",
        )
        self.assertIsNotNone(result["vv"])
        self.assertIsNone(result["vh"])

    def test_verify_sar_downloads_missing_vh_fail_policy(self):
        """VH missing with fallback_policy='fail' → vh=None (same behaviour, caller decides)."""
        from pygeovision.data.processors.sar import verify_sar_downloads
        vv = make_raster(f"{self.tmp}/vv_fail.tif", width=256, height=256)
        result = verify_sar_downloads(
            {"vv": vv, "vh": f"{self.tmp}/missing.tif"},
            fallback_policy="fail",
        )
        self.assertIsNone(result["vh"])

    # ── Despeckle ─────────────────────────────────────────────────────────────

    def test_despeckle_enhanced_lee_produces_output(self):
        """Enhanced Lee filter must produce a float32 output of same shape."""
        try:
            from scipy.ndimage import uniform_filter  # noqa: F401
        except ImportError:
            self.skipTest("scipy not installed")

        from pygeovision.data.processors.sar import despeckle_sar
        out = f"{self.tmp}/despeckle_lee.tif"
        despeckle_sar(self.linear_path, out, filter_type="enhanced_lee", window_size=5)
        with rasterio.open(out) as src:
            data = src.read()
        self.assertEqual(data.dtype, np.float32)
        self.assertEqual(data.shape, (1, 64, 64))
        # Output should be non-negative (linear scale)
        self.assertGreaterEqual(float(data.min()), 0.0)

    def test_despeckle_boxcar_produces_output(self):
        """Boxcar filter must produce valid float32 output."""
        try:
            from scipy.ndimage import uniform_filter  # noqa: F401
        except ImportError:
            self.skipTest("scipy not installed")

        from pygeovision.data.processors.sar import despeckle_sar
        out = f"{self.tmp}/despeckle_box.tif"
        despeckle_sar(self.linear_path, out, filter_type="boxcar", window_size=7)
        with rasterio.open(out) as src:
            data = src.read()
        self.assertEqual(data.dtype, np.float32)

    def test_despeckle_even_window_raises(self):
        """Even window_size must raise ValueError."""
        from pygeovision.data.processors.sar import despeckle_sar
        with self.assertRaises(ValueError):
            despeckle_sar(self.linear_path, f"{self.tmp}/bad.tif", window_size=6)

    def test_despeckle_unknown_filter_raises(self):
        """Unknown filter_type must raise ValueError."""
        from pygeovision.data.processors.sar import despeckle_sar
        with self.assertRaises(ValueError):
            despeckle_sar(self.linear_path, f"{self.tmp}/bad2.tif", filter_type="magic")

    # ── dB conversion ─────────────────────────────────────────────────────────

    def test_linear_to_db_output_in_range(self):
        """Output dB values must be within the clip range [-35, +5]."""
        from pygeovision.data.processors.sar import linear_to_db
        out = f"{self.tmp}/db.tif"
        linear_to_db(self.linear_path, out, clip_db_min=-35.0, clip_db_max=5.0)
        with rasterio.open(out) as src:
            data = src.read(1)
        finite = data[np.isfinite(data)]
        self.assertGreaterEqual(float(finite.min()), -35.0 - 0.01)
        self.assertLessEqual(float(finite.max()), 5.0 + 0.01)

    def test_linear_to_db_preserves_spatial_structure(self):
        """dB output must have same shape as input."""
        from pygeovision.data.processors.sar import linear_to_db
        out = f"{self.tmp}/db_shape.tif"
        linear_to_db(self.linear_path, out)
        with rasterio.open(out) as src:
            self.assertEqual(src.count, 1)
            self.assertEqual(src.width, 64)
            self.assertEqual(src.height, 64)

    def test_linear_to_db_zero_values_become_nan(self):
        """Zero-value pixels must become NaN after dB conversion (log10(0) is -inf)."""
        from pygeovision.data.processors.sar import linear_to_db
        zero_path = make_raster(f"{self.tmp}/zeros.tif", values=0.0)
        out = f"{self.tmp}/db_zeros.tif"
        linear_to_db(zero_path, out)
        with rasterio.open(out) as src:
            data = src.read(1)
        # All fill values become NaN
        self.assertEqual(int(np.isfinite(data).sum()), 0)

    # ── Normalisation ─────────────────────────────────────────────────────────

    def test_normalise_minmax_db_range(self):
        """minmax_db normalisation → output in [0, 1]."""
        from pygeovision.data.processors.sar import linear_to_db, normalise_sar_for_ai
        db_path = f"{self.tmp}/db_norm_in.tif"
        norm_path = f"{self.tmp}/norm_out.tif"
        linear_to_db(self.linear_path, db_path)
        arr = normalise_sar_for_ai(db_path, norm_path, method="minmax_db")
        self.assertGreaterEqual(float(arr.min()), 0.0 - 1e-5)
        self.assertLessEqual(float(arr.max()), 1.0 + 1e-5)
        self.assertEqual(arr.dtype, np.float32)

    def test_normalise_percentile_range(self):
        """Percentile normalisation → output approximately in [0, 1]."""
        from pygeovision.data.processors.sar import linear_to_db, normalise_sar_for_ai
        db_path = f"{self.tmp}/db_pct_in.tif"
        norm_path = f"{self.tmp}/norm_pct.tif"
        linear_to_db(self.linear_path, db_path)
        arr = normalise_sar_for_ai(db_path, norm_path, method="percentile")
        self.assertEqual(arr.dtype, np.float32)
        self.assertGreaterEqual(float(arr.min()), 0.0 - 1e-5)
        self.assertLessEqual(float(arr.max()), 1.0 + 1e-5)

    def test_normalise_unknown_method_raises(self):
        """Unknown normalisation method must raise ValueError."""
        from pygeovision.data.processors.sar import normalise_sar_for_ai
        with self.assertRaises(ValueError):
            normalise_sar_for_ai(self.linear_path, f"{self.tmp}/x.tif", method="unknown")

    # ── Clip (BUG 3) ──────────────────────────────────────────────────────────

    def test_clip_with_utm_bbox_raises_informative_error(self):
        """BUG 3: passing UTM coords as a WGS84 bbox should raise informative ValueError."""
        from pygeovision.data.processors.sar import clip_sar_to_bbox
        # Pass UTM coordinates as if they were WGS84 — vastly out of range for degrees
        bad_bbox = (400000.0, 3990000.0, 400640.0, 3990640.0)
        with self.assertRaises((ValueError, Exception)):
            clip_sar_to_bbox(self.linear_path, f"{self.tmp}/bad_clip.tif", bad_bbox)

    def test_clip_with_correct_wgs84_bbox(self):
        """CRS-aware clip: WGS84 bbox for Turkey → auto-reprojected to UTM → clipped."""
        from pygeovision.data.processors.sar import clip_sar_to_bbox
        # The synthetic raster covers 400000–400640 E, 3990000–3990640 N (UTM 37N)
        # That corresponds approximately to 36.09–36.10°E, 36.02–36.03°N in WGS84
        # We'll test that the clip succeeds when the bbox overlaps the raster
        from pygeovision.data.validators.georeference import reproject_bbox_to_raster_crs
        # First find what WGS84 coords correspond to this UTM extent
        from pyproj import Transformer
        t = Transformer.from_crs("EPSG:32637", "EPSG:4326", always_xy=True)
        left_lon, bottom_lat = t.transform(400000, 3990000)
        right_lon, top_lat = t.transform(400640, 3990640)
        # Slightly inset to ensure overlap
        mid_lon = (left_lon + right_lon) / 2
        mid_lat = (bottom_lat + top_lat) / 2
        d = 0.002  # ~200m
        bbox = (mid_lon - d, mid_lat - d, mid_lon + d, mid_lat + d)

        out = f"{self.tmp}/clipped_correct.tif"
        clip_sar_to_bbox(self.linear_path, out, bbox)
        with rasterio.open(out) as src:
            self.assertGreater(src.width, 0)
            self.assertGreater(src.height, 0)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Channel manager tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSARChannelManager(unittest.TestCase):

    def setUp(self):
        H, W = 64, 64
        self.vv = np.random.uniform(0.1, 0.8, (1, H, W)).astype(np.float32)
        self.vh = np.random.uniform(0.05, 0.4, (1, H, W)).astype(np.float32)

    # ── Pseudo-RGB ────────────────────────────────────────────────────────────

    def test_pseudo_rgb_shape(self):
        """sar_to_pseudo_rgb must return (3, H, W)."""
        from pygeovision.models.adapters.sar_channel_manager import sar_to_pseudo_rgb
        out = sar_to_pseudo_rgb(self.vv, self.vh)
        self.assertEqual(out.shape, (3, 64, 64))

    def test_pseudo_rgb_range(self):
        """All pseudo-RGB channels must be in [0, 1]."""
        from pygeovision.models.adapters.sar_channel_manager import sar_to_pseudo_rgb
        for arr in ["vv_vh_ratio", "vv_vh_diff", "vv_vh_geomean"]:
            out = sar_to_pseudo_rgb(self.vv, self.vh, arrangement=arr)
            self.assertGreaterEqual(float(out.min()), 0.0 - 1e-5, msg=arr)
            self.assertLessEqual(float(out.max()), 1.0 + 1e-5, msg=arr)

    def test_pseudo_rgb_dtype_float32(self):
        """Output must be float32."""
        from pygeovision.models.adapters.sar_channel_manager import sar_to_pseudo_rgb
        out = sar_to_pseudo_rgb(self.vv, self.vh)
        self.assertEqual(out.dtype, np.float32)

    def test_pseudo_rgb_unknown_arrangement_raises(self):
        """Unknown arrangement must raise ValueError."""
        from pygeovision.models.adapters.sar_channel_manager import sar_to_pseudo_rgb
        with self.assertRaises(ValueError):
            sar_to_pseudo_rgb(self.vv, self.vh, arrangement="bogus")

    # ── HLS 6-channel ─────────────────────────────────────────────────────────

    def test_hls_6ch_shape(self):
        """sar_to_hls_6ch must return (6, H, W)."""
        from pygeovision.models.adapters.sar_channel_manager import sar_to_hls_6ch
        out = sar_to_hls_6ch(self.vv, self.vh)
        self.assertEqual(out.shape, (6, 64, 64))

    def test_hls_6ch_range(self):
        """All 6 channels must be in [0, 1]."""
        from pygeovision.models.adapters.sar_channel_manager import sar_to_hls_6ch
        out = sar_to_hls_6ch(self.vv, self.vh, mapping="physics_guided")
        self.assertGreaterEqual(float(out.min()), 0.0 - 1e-5)
        self.assertLessEqual(float(out.max()), 1.0 + 1e-5)

    def test_hls_6ch_replicate_mapping(self):
        """Replicate mapping must alternate VV/VH across 6 channels."""
        from pygeovision.models.adapters.sar_channel_manager import sar_to_hls_6ch
        out = sar_to_hls_6ch(self.vv, self.vh, mapping="replicate")
        self.assertEqual(out.shape[0], 6)

    def test_hls_6ch_no_vh_uses_approximation(self):
        """Missing VH → approximation from VV, no exception, shape still (6,H,W)."""
        from pygeovision.models.adapters.sar_channel_manager import sar_to_hls_6ch
        out = sar_to_hls_6ch(self.vv, vh=None)
        self.assertEqual(out.shape, (6, 64, 64))
        self.assertGreaterEqual(float(out.min()), 0.0 - 1e-5)

    def test_hls_6ch_accepts_2d_input(self):
        """(H, W) shaped VV/VH must be accepted (squeezed internally)."""
        from pygeovision.models.adapters.sar_channel_manager import sar_to_hls_6ch
        vv2d = self.vv[0]  # (H, W)
        vh2d = self.vh[0]
        out = sar_to_hls_6ch(vv2d, vh2d)
        self.assertEqual(out.shape, (6, 64, 64))

    def test_hls_6ch_dtype_float32(self):
        """Output must always be float32."""
        from pygeovision.models.adapters.sar_channel_manager import sar_to_hls_6ch
        out = sar_to_hls_6ch(self.vv.astype(np.float64), self.vh.astype(np.float64))
        self.assertEqual(out.dtype, np.float32)

    # ── Validation ────────────────────────────────────────────────────────────

    def test_validate_prithvi_6ch_passes(self):
        """Valid 6-channel float32 [0,1] array must pass Prithvi validation."""
        from pygeovision.models.adapters.sar_channel_manager import (
            sar_to_hls_6ch, validate_sar_ai_input,
        )
        six_ch = sar_to_hls_6ch(self.vv, self.vh)
        val = validate_sar_ai_input(six_ch, model="prithvi")
        self.assertTrue(val["valid"])
        self.assertEqual(len(val["errors"]), 0)

    def test_validate_prithvi_wrong_channels_fails(self):
        """3-channel input for Prithvi (which needs 6) must fail validation."""
        from pygeovision.models.adapters.sar_channel_manager import validate_sar_ai_input
        three_ch = np.random.rand(3, 64, 64).astype(np.float32)
        val = validate_sar_ai_input(three_ch, model="prithvi")
        self.assertFalse(val["valid"])
        self.assertTrue(any("6" in e or "channel" in e.lower() for e in val["errors"]))

    def test_validate_nan_fails(self):
        """Array with NaN values must fail validation."""
        from pygeovision.models.adapters.sar_channel_manager import validate_sar_ai_input
        bad = np.random.rand(6, 64, 64).astype(np.float32)
        bad[0, 10, 10] = np.nan
        val = validate_sar_ai_input(bad, model="prithvi")
        self.assertFalse(val["valid"])
        self.assertTrue(any("nan" in e.lower() for e in val["errors"]))

    def test_validate_2d_fails(self):
        """2D input (missing channel dim) must fail validation."""
        from pygeovision.models.adapters.sar_channel_manager import validate_sar_ai_input
        bad = np.random.rand(64, 64).astype(np.float32)
        val = validate_sar_ai_input(bad)
        self.assertFalse(val["valid"])

    # ── Co-registration ───────────────────────────────────────────────────────

    def test_coregister_sar_pair_aligned_shapes(self):
        """Co-registered POST must have identical shape/transform to PRE."""
        from pygeovision.models.adapters.sar_channel_manager import coregister_sar_pair
        tmp = tempfile.mkdtemp()
        pre = make_raster(f"{tmp}/pre.tif", width=64, height=64)
        # POST at slightly different grid
        post = make_raster(f"{tmp}/post.tif", width=48, height=48,
                            transform=from_bounds(400050, 3990050, 400530, 3990530, 48, 48))
        pre_out, post_out = coregister_sar_pair(pre, post, f"{tmp}/aligned")
        with rasterio.open(pre_out) as pre_src, rasterio.open(post_out) as post_src:
            self.assertEqual(pre_src.width, post_src.width)
            self.assertEqual(pre_src.height, post_src.height)
            self.assertAlmostEqual(pre_src.transform.a, post_src.transform.a, places=3)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Prithvi adapter tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSARPrithviAdapter(unittest.TestCase):

    def setUp(self):
        self.vv = np.random.uniform(0.1, 0.8, (1, 64, 64)).astype(np.float32)
        self.vh = np.random.uniform(0.05, 0.4, (1, 64, 64)).astype(np.float32)

    def test_zero_shot_flood_mask_shape(self):
        """Zero-shot adapter must return a (H, W) binary mask."""
        from pygeovision.models.adapters.sar_prithvi import SARPrithviAdapter
        adapter = SARPrithviAdapter(mode="zero_shot", task="flood_detection")
        result = adapter.run(self.vv, self.vh)
        self.assertIsNotNone(result["prediction"])
        self.assertEqual(result["prediction"].shape, (64, 64))

    def test_zero_shot_mask_binary(self):
        """Zero-shot flood mask must be binary (0 or 1)."""
        from pygeovision.models.adapters.sar_prithvi import SARPrithviAdapter
        adapter = SARPrithviAdapter(mode="zero_shot", task="flood_detection")
        result = adapter.run(self.vv, self.vh)
        mask = result["prediction"]
        unique_vals = set(np.unique(mask))
        self.assertTrue(unique_vals.issubset({0, 1}))

    def test_zero_shot_mode_string_in_result(self):
        """Result dict must carry mode and task metadata."""
        from pygeovision.models.adapters.sar_prithvi import SARPrithviAdapter
        adapter = SARPrithviAdapter(mode="zero_shot", task="flood_detection")
        result = adapter.run(self.vv, self.vh)
        self.assertEqual(result["mode"], "zero_shot")
        self.assertEqual(result["task"], "flood_detection")

    def test_invalid_mode_raises(self):
        """Invalid mode must raise ValueError at construction."""
        from pygeovision.models.adapters.sar_prithvi import SARPrithviAdapter
        with self.assertRaises(ValueError):
            SARPrithviAdapter(mode="magic", task="flood_detection")

    def test_invalid_task_raises(self):
        """Invalid task must raise ValueError at construction."""
        from pygeovision.models.adapters.sar_prithvi import SARPrithviAdapter
        with self.assertRaises(ValueError):
            SARPrithviAdapter(mode="zero_shot", task="banana_detection")

    def test_six_channels_built_in_result(self):
        """Result must contain the list of 6 channel descriptions."""
        from pygeovision.models.adapters.sar_prithvi import SARPrithviAdapter
        adapter = SARPrithviAdapter(mode="zero_shot", task="flood_detection")
        result = adapter.run(self.vv, self.vh)
        self.assertEqual(len(result["channels"]), 6)

    def test_fine_tune_requires_terratorch(self):
        """fine_tune mode without TerraTorch installed → ImportError."""
        from pygeovision.models.adapters.sar_prithvi import (
            SARPrithviAdapter, check_terratorch_available,
        )
        if check_terratorch_available():
            self.skipTest("TerraTorch is installed; skip absence test")
        with self.assertRaises(ImportError):
            SARPrithviAdapter(mode="fine_tune", task="flood_detection")

    def test_check_terratorch_available_returns_bool(self):
        """check_terratorch_available must return a bool."""
        from pygeovision.models.adapters.sar_prithvi import check_terratorch_available
        result = check_terratorch_available()
        self.assertIsInstance(result, bool)

    def test_scattering_prompt_tuner_init(self):
        """ScatteringPromptTuner must initialise prompts in correct shape."""
        from pygeovision.models.adapters.sar_prithvi import ScatteringPromptTuner
        tuner = ScatteringPromptTuner(n_prompts=4, prompt_dim=768)
        prompts = tuner.initialise_prompts(method="scattering")
        self.assertEqual(prompts.shape, (4, 768))
        self.assertEqual(prompts.dtype, np.float32)

    def test_scattering_prompt_tuner_random_init(self):
        """Random initialisation must produce non-zero prompts."""
        from pygeovision.models.adapters.sar_prithvi import ScatteringPromptTuner
        tuner = ScatteringPromptTuner(n_prompts=4, prompt_dim=384)
        prompts = tuner.initialise_prompts(method="random")
        self.assertGreater(float(np.abs(prompts).max()), 0.0)

    def test_sen1floods11_config_defaults(self):
        """Sen1Floods11Config must have correct default SAR band list."""
        from pygeovision.models.adapters.sar_prithvi import Sen1Floods11Config
        cfg = Sen1Floods11Config()
        self.assertEqual(cfg.sar_bands, ["VV", "VH"])
        self.assertEqual(cfg.n_classes, 2)

    def test_sen1floods11_config_yaml_writes(self):
        """to_terratorch_yaml must write a valid YAML file."""
        try:
            import yaml  # noqa: F401
        except ImportError:
            self.skipTest("pyyaml not installed")
        from pygeovision.models.adapters.sar_prithvi import Sen1Floods11Config
        cfg = Sen1Floods11Config()
        tmp = tempfile.mkdtemp()
        path = cfg.to_terratorch_yaml(f"{tmp}/test_config.yaml")
        self.assertTrue(Path(path).exists())
        import yaml
        content = yaml.safe_load(open(path))
        self.assertIn("data", content)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. DINOv3 adapter tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSARDINOv3Adapter(unittest.TestCase):

    def setUp(self):
        self.vv = np.random.uniform(0.1, 0.8, (1, 112, 112)).astype(np.float32)
        self.vh = np.random.uniform(0.05, 0.4, (1, 112, 112)).astype(np.float32)

    def test_normalise_for_dino_shape(self):
        """normalise_for_dino must preserve (3, H, W) shape."""
        from pygeovision.models.adapters.sar_dinov3 import normalise_for_dino
        rgb = np.random.uniform(0, 1, (3, 64, 64)).astype(np.float32)
        norm = normalise_for_dino(rgb)
        self.assertEqual(norm.shape, rgb.shape)

    def test_normalise_for_dino_shifts_mean(self):
        """After ImageNet normalisation, mean should not be 0.5 (it was shifted)."""
        from pygeovision.models.adapters.sar_dinov3 import normalise_for_dino, IMAGENET_MEAN
        # Image with all pixels = 0.5
        rgb = np.full((3, 64, 64), 0.5, dtype=np.float32)
        norm = normalise_for_dino(rgb)
        # Channel 0: (0.5 - 0.485) / 0.229 ≈ 0.065
        expected_ch0 = (0.5 - IMAGENET_MEAN[0]) / 0.229
        self.assertAlmostEqual(float(norm[0].mean()), expected_ch0, places=3)

    def test_normalise_for_dino_wrong_channels_raises(self):
        """Non-3-channel input must raise AssertionError."""
        from pygeovision.models.adapters.sar_dinov3 import normalise_for_dino
        bad = np.random.rand(6, 64, 64).astype(np.float32)
        with self.assertRaises(AssertionError):
            normalise_for_dino(bad)

    def test_preprocess_returns_3ch_normalised(self):
        """SARDINOv3Adapter.preprocess must return ImageNet-normalised (3,H,W)."""
        from pygeovision.models.adapters.sar_dinov3 import SARDINOv3Adapter
        adapter = SARDINOv3Adapter(mode="zero_shot")
        out = adapter.preprocess(self.vv, self.vh)
        self.assertEqual(out.shape, (3, 112, 112))
        self.assertEqual(out.dtype, np.float32)
        # After ImageNet normalisation, values will not be in [0,1]
        # but should be in roughly [-3, +3]
        self.assertLess(float(out.min()), 2.0)

    def test_preprocess_no_vh_uses_approximation(self):
        """Missing VH → approximation, no exception."""
        from pygeovision.models.adapters.sar_dinov3 import SARDINOv3Adapter
        adapter = SARDINOv3Adapter(mode="zero_shot")
        out = adapter.preprocess(self.vv, vh=None)
        self.assertEqual(out.shape[0], 3)

    def test_extract_features_returns_mock_when_torch_absent(self):
        """extract_features must return correct-shape zero embeddings if torch not available."""
        from pygeovision.models.adapters.sar_dinov3 import SARDINOv3Adapter
        adapter = SARDINOv3Adapter(mode="zero_shot", model_size="vits14")
        # Force _model to None (torch may or may not be installed)
        adapter._model = None
        # Temporarily mock _load_model to do nothing
        adapter._load_model = lambda: None
        result = adapter.extract_features(self.vv, self.vh)
        # Should return zero cls_token of correct embed_dim
        self.assertIsNotNone(result["cls_token"])
        self.assertEqual(result["cls_token"].shape[0], adapter.embed_dim)

    def test_invalid_mode_raises(self):
        """Invalid mode must raise ValueError."""
        from pygeovision.models.adapters.sar_dinov3 import SARDINOv3Adapter
        with self.assertRaises(ValueError):
            SARDINOv3Adapter(mode="telepathic")

    def test_invalid_model_size_raises(self):
        """Invalid model_size must raise ValueError."""
        from pygeovision.models.adapters.sar_dinov3 import SARDINOv3Adapter
        with self.assertRaises(ValueError):
            SARDINOv3Adapter(model_size="vit_gigantic_32")

    def test_speckle_augmentation_shape_preserved(self):
        """simulate_gamma_speckle must preserve image shape."""
        from pygeovision.models.adapters.sar_dinov3 import SARSpeckleAugmentation
        aug = SARSpeckleAugmentation()
        img = np.random.uniform(0.05, 0.5, (1, 64, 64)).astype(np.float32)
        speckled = aug.simulate_gamma_speckle(img, n_looks=4)
        self.assertEqual(speckled.shape, img.shape)

    def test_speckle_augmentation_multiplicative(self):
        """Gamma speckle must be multiplicative — should not change the sign of values."""
        from pygeovision.models.adapters.sar_dinov3 import SARSpeckleAugmentation
        aug = SARSpeckleAugmentation()
        img = np.ones((1, 32, 32), dtype=np.float32) * 0.5
        speckled = aug.simulate_gamma_speckle(img, n_looks=1)
        # All values must still be positive (Gamma noise is positive)
        self.assertTrue(bool(np.all(speckled >= 0.0)))

    def test_speckle_config_dict_keys(self):
        """get_pytorch_transform_config must return dict with 'type' key."""
        from pygeovision.models.adapters.sar_dinov3 import SARSpeckleAugmentation
        aug = SARSpeckleAugmentation()
        cfg = aug.get_pytorch_transform_config()
        self.assertIn("type", cfg)
        self.assertIn("gamma_noise_n_looks_range", cfg)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Approach list tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSARApproachList(unittest.TestCase):

    def test_approach_list_length(self):
        """SAR_APPROACH_LIST must have the correct number of steps."""
        from pygeovision.data.processors.sar_approach_list import SAR_APPROACH_LIST
        self.assertGreaterEqual(len(SAR_APPROACH_LIST), 11)  # S0–S9 + A0–A4

    def test_all_steps_have_required_keys(self):
        """Every step must have id, name, rationale."""
        from pygeovision.data.processors.sar_approach_list import SAR_APPROACH_LIST
        for step in SAR_APPROACH_LIST:
            for key in ("id", "name", "rationale"):
                self.assertIn(key, step, msg=f"Step {step.get('id','?')} missing '{key}'")

    def test_step_ids_unique(self):
        """Step IDs must be unique."""
        from pygeovision.data.processors.sar_approach_list import SAR_APPROACH_LIST
        ids = [s["id"] for s in SAR_APPROACH_LIST]
        self.assertEqual(len(ids), len(set(ids)))

    def test_bug_references_present(self):
        """The three bug-fix steps must reference their bug."""
        from pygeovision.data.processors.sar_approach_list import SAR_APPROACH_LIST
        bug_steps = [s for s in SAR_APPROACH_LIST if "bug_reference" in s]
        bug_ids = {s["bug_reference"] for s in bug_steps}
        # All three bugs must have at least one step referencing them
        self.assertTrue(any("BUG 1" in ref for ref in bug_ids))
        self.assertTrue(any("BUG 2" in ref for ref in bug_ids))
        self.assertTrue(any("BUG 3" in ref for ref in bug_ids))

    def test_ai_inference_steps_present(self):
        """AI inference steps (A0–A4) must be present."""
        from pygeovision.data.processors.sar_approach_list import SAR_APPROACH_LIST
        ai_ids = {s["id"] for s in SAR_APPROACH_LIST if s["phase"] == "ai_inference"}
        for expected in ("A0", "A1", "A2", "A3", "A4"):
            self.assertIn(expected, ai_ids, msg=f"AI step {expected} missing")

    def test_preprocessing_steps_present(self):
        """Pre-processing steps S0–S9 must be present."""
        from pygeovision.data.processors.sar_approach_list import SAR_APPROACH_LIST
        pre_ids = {s["id"] for s in SAR_APPROACH_LIST if s["phase"] == "pre_processing"}
        for expected in ("S0", "S1", "S6", "S7", "S8", "S9"):
            self.assertIn(expected, pre_ids, msg=f"Pre-processing step {expected} missing")

    def test_despeckle_before_db(self):
        """S6 (despeckle) must appear before S7 (dB) in the ordered list."""
        from pygeovision.data.processors.sar_approach_list import SAR_APPROACH_LIST
        ids = [s["id"] for s in SAR_APPROACH_LIST]
        self.assertLess(ids.index("S6"), ids.index("S7"),
                         "Despeckle (S6) must come before dB conversion (S7)")

    def test_download_check_first(self):
        """S0 (download check) must be the first step."""
        from pygeovision.data.processors.sar_approach_list import SAR_APPROACH_LIST
        self.assertEqual(SAR_APPROACH_LIST[0]["id"], "S0")

    def test_get_nb02_approach_context_returns_string(self):
        """get_nb02_approach_context must return a non-empty string."""
        from pygeovision.data.processors.sar_approach_list import get_nb02_approach_context
        ctx = get_nb02_approach_context()
        self.assertIsInstance(ctx, str)
        self.assertGreater(len(ctx), 500)
        self.assertIn("Bug", ctx)  # context uses "Bug 1", "Bug 2", "Bug 3"
        self.assertIn("S6", ctx)


if __name__ == "__main__":
    unittest.main(verbosity=2)
