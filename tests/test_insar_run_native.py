"""Tests for SLCInSARPipeline.run_native() — the new, additive SNAP-free
InSAR path built on pygeofetch's real InSAR suite.

Honest scope: these tests verify the CONTROL FLOW and DATA HANDLING using
realistic mocked pygeofetch return objects (matching the real, verified
dataclass shapes of InterferogramResult/UnwrapResult) — not a full
end-to-end run against real Sentinel-1 SLC data, since no real SLC test
fixture or SNAPHU binary is available in this environment. Real bugs
found and fixed while building this, all covered below:

  1. InSARProcessingResult.interferogram/.unwrapped_phase are typed
     `str | None` (file paths) and used that way elsewhere in the
     codebase — an early draft assigned raw InterferogramResult/array
     objects to these fields instead, a real type-contract violation.
  2. Interferograms are complex-valued; a plain "write the array to a
     GeoTIFF" helper would either crash or silently lose the imaginary
     component. The real fix writes real+imaginary as two bands.
  3. The atmospherically-corrected phase was computed but never actually
     used for the final output — an unconditional assignment further
     down silently discarded the correction that had just been computed.
  4. InSARProcessingResult had no declared `metadata` field at all,
     despite an early draft assigning to `result.metadata` (which would
     have worked as an undeclared dynamic attribute on a plain
     @dataclass, but is a real design smell) — fixed by adding a proper
     declared field.
"""
import pytest

pygeofetch = pytest.importorskip("pygeofetch", reason="pygeofetch not installed")
rasterio = pytest.importorskip("rasterio", reason="rasterio not installed")

import numpy as np
from rasterio.transform import from_bounds
from unittest.mock import patch


def _make_mocks(size=32):
    from pygeofetch.insar.interferogram import InterferogramResult
    from pygeofetch.insar.unwrap import UnwrapResult

    profile = {
        "driver": "GTiff", "height": size, "width": size, "crs": "EPSG:32630",
        "transform": from_bounds(0, 0, size * 10, size * 10, size, size), "nodata": None,
    }
    rng = np.random.default_rng(0)
    complex_ifg = (rng.standard_normal((size, size)) + 1j * rng.standard_normal((size, size))).astype(np.complex64)
    coherence_arr = rng.random((size, size)).astype(np.float32)

    ifg_result = InterferogramResult(
        interferogram=complex_ifg, coherence=coherence_arr, amplitude=np.abs(complex_ifg),
        profile=profile, reference_date="2024-01-15", secondary_date="2024-01-27",
        perpendicular_baseline_m=45.2, temporal_baseline_days=12,
    )

    unwrapped_arr = (rng.standard_normal((size, size)) * 10).astype(np.float32)
    unwrap_result = UnwrapResult(
        unwrapped_phase=unwrapped_arr, conncomp=np.ones((size, size), dtype=np.int32),
        coherence=coherence_arr, profile=profile,
        reference_date="2024-01-15", secondary_date="2024-01-27",
        nlooks=1.0, cost_mode="defo", init_method="mcf", reliable_fraction=0.85,
    )
    return ifg_result, unwrap_result, complex_ifg, unwrapped_arr


class TestRunNativeControlFlow:
    def test_success_with_full_mocked_pipeline(self, tmp_path):
        from pygeovision.insar.core import SLCInSARPipeline
        ifg_result, unwrap_result, complex_ifg, unwrapped_arr = _make_mocks()
        corrected_arr = unwrapped_arr - 0.5

        pipeline = SLCInSARPipeline(output_dir=str(tmp_path))
        with patch("pygeofetch.insar.extraction.SLCExtractor.extract_pair",
                   return_value=(tmp_path / "ref.tif", tmp_path / "sec.tif")), \
             patch("pygeofetch.insar.interferogram.InterferogramGenerator.process_pair",
                   return_value=ifg_result), \
             patch("pygeofetch.insar.unwrap.PhaseUnwrapper.unwrap_pair",
                   return_value=unwrap_result), \
             patch("pygeofetch.insar.atmosphere.AtmosphericCorrector.correct",
                   return_value=corrected_arr):
            result = pipeline.run_native(
                primary_zip="primary.zip", secondary_zip="secondary.zip",
                bbox=(-0.25, 5.52, -0.20, 5.60),
            )

        assert result.success is True
        assert result.errors == []

    def test_result_fields_are_file_paths_not_raw_objects(self, tmp_path):
        """Regression test for the real type-contract bug: interferogram/
        unwrapped_phase must be strings (file paths), matching how
        run()'s (the SNAP-based path's) results are used elsewhere,
        not raw InterferogramResult/array objects."""
        from pygeovision.insar.core import SLCInSARPipeline
        ifg_result, unwrap_result, complex_ifg, unwrapped_arr = _make_mocks()

        pipeline = SLCInSARPipeline(output_dir=str(tmp_path))
        with patch("pygeofetch.insar.extraction.SLCExtractor.extract_pair",
                   return_value=(tmp_path / "ref.tif", tmp_path / "sec.tif")), \
             patch("pygeofetch.insar.interferogram.InterferogramGenerator.process_pair",
                   return_value=ifg_result), \
             patch("pygeofetch.insar.unwrap.PhaseUnwrapper.unwrap_pair",
                   return_value=unwrap_result), \
             patch("pygeofetch.insar.atmosphere.AtmosphericCorrector.correct",
                   side_effect=RuntimeError("no correction for this test")):
            result = pipeline.run_native(
                primary_zip="primary.zip", secondary_zip="secondary.zip",
                bbox=(-0.25, 5.52, -0.20, 5.60),
            )

        assert isinstance(result.interferogram, str)
        assert isinstance(result.coherence, str)
        assert isinstance(result.unwrapped_phase, str)


class TestComplexInterferogramGeoTIFFHandling:
    """Regression tests for correct complex-valued array handling —
    interferograms are complex (amplitude * e^(i*phase)), and a naive
    'write the array' helper would crash or silently drop the imaginary
    component entirely."""

    def test_interferogram_written_as_two_bands_real_and_imaginary(self, tmp_path):
        from pygeovision.insar.core import SLCInSARPipeline
        ifg_result, unwrap_result, complex_ifg, unwrapped_arr = _make_mocks()

        pipeline = SLCInSARPipeline(output_dir=str(tmp_path))
        with patch("pygeofetch.insar.extraction.SLCExtractor.extract_pair",
                   return_value=(tmp_path / "ref.tif", tmp_path / "sec.tif")), \
             patch("pygeofetch.insar.interferogram.InterferogramGenerator.process_pair",
                   return_value=ifg_result), \
             patch("pygeofetch.insar.unwrap.PhaseUnwrapper.unwrap_pair",
                   return_value=unwrap_result), \
             patch("pygeofetch.insar.atmosphere.AtmosphericCorrector.correct",
                   side_effect=RuntimeError("skip correction")):
            result = pipeline.run_native(
                primary_zip="primary.zip", secondary_zip="secondary.zip",
                bbox=(-0.25, 5.52, -0.20, 5.60), apply_atmospheric_correction=False,
            )

        with rasterio.open(result.interferogram) as src:
            assert src.count == 2
            real_band = src.read(1)
            imag_band = src.read(2)
        np.testing.assert_allclose(real_band, np.real(complex_ifg), atol=1e-3)
        np.testing.assert_allclose(imag_band, np.imag(complex_ifg), atol=1e-3)

    def test_real_valued_coherence_written_as_single_band(self, tmp_path):
        from pygeovision.insar.core import SLCInSARPipeline
        ifg_result, unwrap_result, complex_ifg, unwrapped_arr = _make_mocks()

        pipeline = SLCInSARPipeline(output_dir=str(tmp_path))
        with patch("pygeofetch.insar.extraction.SLCExtractor.extract_pair",
                   return_value=(tmp_path / "ref.tif", tmp_path / "sec.tif")), \
             patch("pygeofetch.insar.interferogram.InterferogramGenerator.process_pair",
                   return_value=ifg_result), \
             patch("pygeofetch.insar.unwrap.PhaseUnwrapper.unwrap_pair",
                   return_value=unwrap_result), \
             patch("pygeofetch.insar.atmosphere.AtmosphericCorrector.correct",
                   side_effect=RuntimeError("skip correction")):
            result = pipeline.run_native(
                primary_zip="primary.zip", secondary_zip="secondary.zip",
                bbox=(-0.25, 5.52, -0.20, 5.60), apply_atmospheric_correction=False,
            )

        with rasterio.open(result.coherence) as src:
            assert src.count == 1


class TestAtmosphericCorrectionNotDiscarded:
    """Regression test for the real bug: the corrected phase was computed
    but an unconditional assignment further down silently used the raw,
    uncorrected phase instead — the correction step ran and logged
    success but its output was never actually used."""

    def test_corrected_phase_is_what_gets_written(self, tmp_path):
        from pygeovision.insar.core import SLCInSARPipeline
        ifg_result, unwrap_result, complex_ifg, unwrapped_arr = _make_mocks()
        corrected_arr = unwrapped_arr - 0.5  # deliberately different from the raw phase

        pipeline = SLCInSARPipeline(output_dir=str(tmp_path))
        with patch("pygeofetch.insar.extraction.SLCExtractor.extract_pair",
                   return_value=(tmp_path / "ref.tif", tmp_path / "sec.tif")), \
             patch("pygeofetch.insar.interferogram.InterferogramGenerator.process_pair",
                   return_value=ifg_result), \
             patch("pygeofetch.insar.unwrap.PhaseUnwrapper.unwrap_pair",
                   return_value=unwrap_result), \
             patch("pygeofetch.insar.atmosphere.AtmosphericCorrector.correct",
                   return_value=corrected_arr):
            result = pipeline.run_native(
                primary_zip="primary.zip", secondary_zip="secondary.zip",
                bbox=(-0.25, 5.52, -0.20, 5.60),
            )

        with rasterio.open(result.unwrapped_phase) as src:
            written = src.read(1)
        # Must match the CORRECTED array, not the raw uncorrected one
        np.testing.assert_allclose(written, corrected_arr, atol=1e-3)
        assert not np.allclose(written, unwrapped_arr, atol=1e-3)
        assert result.metadata.get("atmospheric_correction") == "elevation"

    def test_correction_failure_falls_back_to_uncorrected_phase(self, tmp_path):
        from pygeovision.insar.core import SLCInSARPipeline
        ifg_result, unwrap_result, complex_ifg, unwrapped_arr = _make_mocks()

        pipeline = SLCInSARPipeline(output_dir=str(tmp_path))
        with patch("pygeofetch.insar.extraction.SLCExtractor.extract_pair",
                   return_value=(tmp_path / "ref.tif", tmp_path / "sec.tif")), \
             patch("pygeofetch.insar.interferogram.InterferogramGenerator.process_pair",
                   return_value=ifg_result), \
             patch("pygeofetch.insar.unwrap.PhaseUnwrapper.unwrap_pair",
                   return_value=unwrap_result), \
             patch("pygeofetch.insar.atmosphere.AtmosphericCorrector.correct",
                   side_effect=RuntimeError("simulated correction failure")):
            result = pipeline.run_native(
                primary_zip="primary.zip", secondary_zip="secondary.zip",
                bbox=(-0.25, 5.52, -0.20, 5.60),
            )

        assert result.success is True  # correction failure shouldn't fail the whole pipeline
        assert any("Atmospheric correction failed" in w for w in result.warnings)
        with rasterio.open(result.unwrapped_phase) as src:
            written = src.read(1)
        np.testing.assert_allclose(written, unwrapped_arr, atol=1e-3)


class TestExtractionFailureHandling:
    def test_no_overlap_gives_clear_error_not_crash(self, tmp_path):
        from pygeovision.insar.core import SLCInSARPipeline

        pipeline = SLCInSARPipeline(output_dir=str(tmp_path))
        with patch("pygeofetch.insar.extraction.SLCExtractor.extract_pair",
                   return_value=(None, None)):
            result = pipeline.run_native(
                primary_zip="primary.zip", secondary_zip="secondary.zip",
                bbox=(-0.25, 5.52, -0.20, 5.60),
            )
        assert result.success is False
        assert any("overlapping burst" in e for e in result.errors)


class TestMetadataFieldIsRealAndDeclared:
    def test_metadata_field_exists_on_a_fresh_instance(self):
        from pygeovision.insar.core import InSARProcessingResult
        r = InSARProcessingResult(success=True)
        assert r.metadata == {}
        r.metadata["k"] = "v"
        assert r.metadata == {"k": "v"}
