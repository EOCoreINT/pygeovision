"""Tests for perpendicular baseline handling in pygeovision.insar.core.

Real, scientifically meaningful gap found and fixed: select_insar_pair()'s
docstring claimed perpendicular baseline as the third of three selection
criteria ("avoids both geometric decorrelation and poor topographic
sensitivity"), but the actual pair-scoring loop never computed or checked
it at all — `score = tb` (temporal baseline only). The "best" pair could
have had an excessive or too-short perpendicular baseline and nothing
would have flagged it.

Computing a real perpendicular baseline genuinely requires orbit state
vector data this function's lightweight scene dicts don't carry — so
rather than fake a check, the fix (1) makes the docstring honest about
this limitation and warns clearly when a non-default value is requested
but can't be honoured, and (2) adds a REAL check where genuine baseline
data actually exists: SLCInSARPipeline.run_native(), after
InterferogramGenerator computes it for real from actual orbit data.
"""
import pytest

pygeofetch = pytest.importorskip("pygeofetch", reason="pygeofetch not installed")
rasterio = pytest.importorskip("rasterio", reason="rasterio not installed")

import numpy as np
from rasterio.transform import from_bounds
from unittest.mock import patch


class TestSelectInsarPairHonestWarning:
    def test_default_value_produces_no_warning(self, caplog):
        import logging
        from pygeovision.insar.core import select_insar_pair
        scenes = [
            {"path": "a.zip", "date": "2024-01-01", "relative_orbit": 143, "pass_direction": "DESCENDING"},
            {"path": "b.zip", "date": "2024-01-13", "relative_orbit": 143, "pass_direction": "DESCENDING"},
        ]
        with caplog.at_level(logging.WARNING):
            select_insar_pair(scenes)
        assert "max_perpendicular_baseline_m" not in caplog.text

    def test_non_default_value_warns_it_is_not_applied(self, caplog):
        import logging
        from pygeovision.insar.core import select_insar_pair
        scenes = [
            {"path": "a.zip", "date": "2024-01-01", "relative_orbit": 143, "pass_direction": "DESCENDING"},
            {"path": "b.zip", "date": "2024-01-13", "relative_orbit": 143, "pass_direction": "DESCENDING"},
        ]
        with caplog.at_level(logging.WARNING):
            select_insar_pair(scenes, max_perpendicular_baseline_m=50.0)
        assert "not applied as a filter" in caplog.text.lower()
        assert "50.0" in caplog.text

    def test_still_selects_a_valid_pair_regardless(self):
        """The honesty fix must not break the function's real, working
        selection logic (orbit/pass-direction/temporal-baseline)."""
        from pygeovision.insar.core import select_insar_pair
        scenes = [
            {"path": "a.zip", "date": "2024-01-01", "relative_orbit": 143, "pass_direction": "DESCENDING"},
            {"path": "b.zip", "date": "2024-01-13", "relative_orbit": 143, "pass_direction": "DESCENDING"},
        ]
        pair = select_insar_pair(scenes)
        assert pair is not None
        assert pair.temporal_baseline_days == 12


def _make_mocks(perp_baseline_m, size=16):
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
        perpendicular_baseline_m=perp_baseline_m, temporal_baseline_days=12,
    )
    unwrapped_arr = (rng.standard_normal((size, size))).astype(np.float32)
    unwrap_result = UnwrapResult(
        unwrapped_phase=unwrapped_arr, conncomp=np.ones((size, size), dtype=np.int32),
        coherence=coherence_arr, profile=profile,
        reference_date="2024-01-15", secondary_date="2024-01-27",
        nlooks=1.0, cost_mode="defo", init_method="mcf", reliable_fraction=0.85,
    )
    return ifg_result, unwrap_result


class TestRunNativeRealBaselineCheck:
    """Verifies the REAL check — using the genuinely computed baseline
    from InterferogramGenerator, not a fabricated one."""

    def test_excessive_baseline_produces_a_warning(self, tmp_path):
        from pygeovision.insar.core import SLCInSARPipeline
        ifg_result, unwrap_result = _make_mocks(perp_baseline_m=300.0)

        pipeline = SLCInSARPipeline(output_dir=str(tmp_path))
        with patch("pygeofetch.insar.extraction.SLCExtractor.extract_pair",
                   return_value=(tmp_path / "ref.tif", tmp_path / "sec.tif")), \
             patch("pygeofetch.insar.interferogram.InterferogramGenerator.process_pair",
                   return_value=ifg_result), \
             patch("pygeofetch.insar.unwrap.PhaseUnwrapper.unwrap_pair",
                   return_value=unwrap_result), \
             patch("pygeofetch.insar.atmosphere.AtmosphericCorrector.correct",
                   side_effect=RuntimeError("skip")):
            result = pipeline.run_native(
                primary_zip="p.zip", secondary_zip="s.zip",
                bbox=(-0.25, 5.52, -0.20, 5.60), max_perpendicular_baseline_m=200.0,
            )

        assert any("exceeds max_perpendicular_baseline_m" in w for w in result.warnings)
        assert result.success is True  # a warning, not a hard failure

    def test_acceptable_baseline_produces_no_warning(self, tmp_path):
        from pygeovision.insar.core import SLCInSARPipeline
        ifg_result, unwrap_result = _make_mocks(perp_baseline_m=45.0)

        pipeline = SLCInSARPipeline(output_dir=str(tmp_path))
        with patch("pygeofetch.insar.extraction.SLCExtractor.extract_pair",
                   return_value=(tmp_path / "ref.tif", tmp_path / "sec.tif")), \
             patch("pygeofetch.insar.interferogram.InterferogramGenerator.process_pair",
                   return_value=ifg_result), \
             patch("pygeofetch.insar.unwrap.PhaseUnwrapper.unwrap_pair",
                   return_value=unwrap_result), \
             patch("pygeofetch.insar.atmosphere.AtmosphericCorrector.correct",
                   side_effect=RuntimeError("skip")):
            result = pipeline.run_native(
                primary_zip="p.zip", secondary_zip="s.zip",
                bbox=(-0.25, 5.52, -0.20, 5.60), max_perpendicular_baseline_m=200.0,
            )

        assert not any("exceeds max_perpendicular_baseline_m" in w for w in result.warnings)

    def test_negative_baseline_is_handled_via_absolute_value(self, tmp_path):
        """Perpendicular baseline can legitimately be negative (sign
        indicates geometry direction) — the check must compare magnitude."""
        from pygeovision.insar.core import SLCInSARPipeline
        ifg_result, unwrap_result = _make_mocks(perp_baseline_m=-300.0)

        pipeline = SLCInSARPipeline(output_dir=str(tmp_path))
        with patch("pygeofetch.insar.extraction.SLCExtractor.extract_pair",
                   return_value=(tmp_path / "ref.tif", tmp_path / "sec.tif")), \
             patch("pygeofetch.insar.interferogram.InterferogramGenerator.process_pair",
                   return_value=ifg_result), \
             patch("pygeofetch.insar.unwrap.PhaseUnwrapper.unwrap_pair",
                   return_value=unwrap_result), \
             patch("pygeofetch.insar.atmosphere.AtmosphericCorrector.correct",
                   side_effect=RuntimeError("skip")):
            result = pipeline.run_native(
                primary_zip="p.zip", secondary_zip="s.zip",
                bbox=(-0.25, 5.52, -0.20, 5.60), max_perpendicular_baseline_m=200.0,
            )

        assert any("exceeds max_perpendicular_baseline_m" in w for w in result.warnings)
