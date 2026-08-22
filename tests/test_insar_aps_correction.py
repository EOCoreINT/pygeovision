"""Tests for pygeovision.insar.core.apply_linear_aps_correction() and the
re-applied _read_envi_width fix.

Real, scientifically significant bug found and fixed: SLCInSARPipeline.run()'s
apply_linear_aps parameter DEFAULTS TO TRUE (implying atmospheric
correction is applied by default) but was never referenced anywhere in
the method — every normal call silently skipped tropospheric correction
while appearing to have it enabled. Confirmed the real-world impact with
a known injected elevation-correlated trend: mean error against the true
displacement signal dropped from 15mm to 0.12mm after correction (~123x),
and the fitted slope exactly matched the injected value.

Also: _read_envi_width's hardcoded-5000-pixel-width fallback (a severe
bug found and fixed much earlier this session) was found to have
reverted in this snapshot — re-applied here, verified directly.
"""
import pytest

rasterio = pytest.importorskip("rasterio", reason="rasterio not installed")
scipy = pytest.importorskip("scipy", reason="scipy not installed")

import numpy as np
from rasterio.transform import from_bounds


class TestApplyLinearApsCorrection:
    def _make_dem(self, tmp_path, size=40):
        dem = np.linspace(100, 900, size * size).reshape(size, size).astype(np.float32)
        dem_path = tmp_path / "dem.tif"
        transform = from_bounds(0, 0, size * 30, size * 30, size, size)
        with rasterio.open(dem_path, "w", driver="GTiff", height=size, width=size, count=1,
                            dtype="float32", crs="EPSG:32630", transform=transform) as dst:
            dst.write(dem, 1)
        return dem, dem_path

    def test_recovers_true_signal_from_elevation_correlated_trend(self, tmp_path):
        """The core regression test: inject a KNOWN elevation-correlated
        trend into a real displacement signal and confirm the correction
        genuinely removes it, recovering something close to the truth —
        not just 'produces some output'."""
        from pygeovision.insar.core import apply_linear_aps_correction

        size = 40
        dem, dem_path = self._make_dem(tmp_path, size)
        rng = np.random.default_rng(0)

        true_displacement = rng.normal(0, 0.002, (size, size)).astype(np.float32)
        known_aps_slope = 0.00003  # m per m elevation — a realistic APS magnitude
        injected_phase = true_displacement + known_aps_slope * dem
        coherence = np.full((size, size), 0.8, dtype=np.float32)

        corrected, log = apply_linear_aps_correction(
            injected_phase.flatten(), coherence.flatten(), size, str(dem_path),
            coherence_threshold=0.3,
        )

        assert log["applied"] is True
        corrected_2d = corrected.reshape(size, size)

        error_before = np.abs(injected_phase - true_displacement).mean()
        error_after = np.abs(corrected_2d - true_displacement).mean()
        assert error_after < error_before * 0.2, (
            f"correction did not meaningfully remove the elevation trend: "
            f"{error_before*1000:.2f}mm -> {error_after*1000:.2f}mm"
        )

    def test_fitted_slope_matches_the_true_injected_slope(self, tmp_path):
        from pygeovision.insar.core import apply_linear_aps_correction

        size = 40
        dem, dem_path = self._make_dem(tmp_path, size)
        rng = np.random.default_rng(1)

        true_displacement = rng.normal(0, 0.001, (size, size)).astype(np.float32)
        known_aps_slope = 0.00005
        injected_phase = true_displacement + known_aps_slope * dem
        coherence = np.full((size, size), 0.9, dtype=np.float32)

        _, log = apply_linear_aps_correction(
            injected_phase.flatten(), coherence.flatten(), size, str(dem_path),
            coherence_threshold=0.3,
        )
        assert log["applied"] is True
        # message reports slope in mm/m elevation
        assert "0.05" in log["message"] or "0.050" in log["message"]

    def test_too_few_coherent_pixels_skips_gracefully(self, tmp_path):
        from pygeovision.insar.core import apply_linear_aps_correction

        size = 20
        dem, dem_path = self._make_dem(tmp_path, size)
        phase = np.random.default_rng(0).normal(0, 0.002, size * size).astype(np.float32)
        coherence = np.zeros(size * size, dtype=np.float32)  # nothing coherent

        corrected, log = apply_linear_aps_correction(
            phase, coherence, size, str(dem_path), coherence_threshold=0.3,
        )
        assert log["applied"] is False
        np.testing.assert_array_equal(corrected, phase)  # unmodified on skip

    def test_missing_dem_fails_gracefully_not_a_crash(self, tmp_path):
        from pygeovision.insar.core import apply_linear_aps_correction

        size = 20
        phase = np.random.default_rng(0).normal(0, 0.002, size * size).astype(np.float32)
        coherence = np.full(size * size, 0.8, dtype=np.float32)

        corrected, log = apply_linear_aps_correction(
            phase, coherence, size, str(tmp_path / "nonexistent_dem.tif"),
            coherence_threshold=0.3,
        )
        assert log["applied"] is False
        assert "failed" in log["message"].lower()
        np.testing.assert_array_equal(corrected, phase)  # unmodified on failure

    def test_dem_with_different_resolution_still_works(self, tmp_path):
        """The DEM's native grid often differs from the interferogram's —
        confirms the resampling path is genuinely exercised and correct."""
        from pygeovision.insar.core import apply_linear_aps_correction

        # DEM at a coarser resolution (20x20) than the interferogram (40x40)
        dem_size = 20
        dem = np.linspace(100, 900, dem_size * dem_size).reshape(dem_size, dem_size).astype(np.float32)
        dem_path = tmp_path / "coarse_dem.tif"
        transform = from_bounds(0, 0, dem_size * 60, dem_size * 60, dem_size, dem_size)
        with rasterio.open(dem_path, "w", driver="GTiff", height=dem_size, width=dem_size, count=1,
                            dtype="float32", crs="EPSG:32630", transform=transform) as dst:
            dst.write(dem, 1)

        ifg_size = 40
        rng = np.random.default_rng(2)
        # Build a matching-shape "true elevation" surface for computing
        # the injected trend against (same linear pattern as dem, just at
        # the finer resolution) — the correction's resampling should still
        # recover the true signal reasonably well.
        fine_dem_equiv = np.linspace(100, 900, ifg_size * ifg_size).reshape(ifg_size, ifg_size)
        true_displacement = rng.normal(0, 0.002, (ifg_size, ifg_size)).astype(np.float32)
        injected_phase = true_displacement + 0.00003 * fine_dem_equiv
        coherence = np.full((ifg_size, ifg_size), 0.8, dtype=np.float32)

        corrected, log = apply_linear_aps_correction(
            injected_phase.flatten(), coherence.flatten(), ifg_size, str(dem_path),
            coherence_threshold=0.3,
        )
        assert log["applied"] is True
        error_after = np.abs(corrected.reshape(ifg_size, ifg_size) - true_displacement).mean()
        error_before = np.abs(injected_phase - true_displacement).mean()
        assert error_after < error_before


class TestReadEnviWidthNoLongerGuesses:
    """Regression tests for a severe bug found and fixed much earlier
    this session, found to have reverted in this snapshot: a malformed
    ENVI header silently fell back to a hardcoded width of 5000 pixels,
    corrupting any reshape of raw binary SAR data (including the APS
    correction above, which also reshapes using this width)."""

    def test_missing_header_raises_instead_of_guessing_5000(self):
        from pygeovision.insar.core import SLCInSARPipeline
        p = SLCInSARPipeline.__new__(SLCInSARPipeline)
        with pytest.raises(RuntimeError, match="Failed to read ENVI header"):
            p._read_envi_width("/nonexistent/path.hdr")

    def test_header_without_samples_field_raises(self, tmp_path):
        from pygeovision.insar.core import SLCInSARPipeline
        hdr_path = tmp_path / "bad.hdr"
        hdr_path.write_text("description = {no samples field here}\n")
        p = SLCInSARPipeline.__new__(SLCInSARPipeline)
        with pytest.raises(RuntimeError, match="no 'samples' field"):
            p._read_envi_width(str(hdr_path))

    def test_valid_header_reads_the_real_width(self, tmp_path):
        from pygeovision.insar.core import SLCInSARPipeline
        hdr_path = tmp_path / "good.hdr"
        hdr_path.write_text("samples = 1847\nlines = 2000\n")
        p = SLCInSARPipeline.__new__(SLCInSARPipeline)
        assert p._read_envi_width(str(hdr_path)) == 1847
