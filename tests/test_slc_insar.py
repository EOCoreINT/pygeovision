"""
tests/test_slc_insar.py
========================
Tests for the SLC InSAR pipeline (pygeovision.insar.slc).

All tests run without SNAP, snapista, or snaphu installed.
External tool availability is checked via the check_* functions and
tests that require them are skipped gracefully.
"""
from __future__ import annotations

import json
import re
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

# ── Module imports ────────────────────────────────────────────────────────────

from pygeovision.insar.slc.sentinel1 import (
    S1SLCProduct,
    parse_s1_slc_filename,
    validate_slc_pair,
    SENTINEL1_WAVELENGTH_M,
    SUBSWATH_INCIDENCE_DEG,
)
from pygeovision.insar.slc.snap_graph import (
    SNAPGraph,
    check_snap,
    check_snapista,
    check_environment,
)
from pygeovision.insar.slc.snaphu import (
    SnaphuConfig,
    SnaphuUnwrapper,
    unwrap_phase,
)
from pygeovision.insar.slc.displacement import (
    slc_phase_to_displacement,
    los_to_vertical,
    dual_pass_decomposition,
    compute_deformation_rate,
)
from pygeovision.insar.slc.pipeline import SLCInSARPipeline, SLCInSARResult


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def s1c_master_product():
    """Minimal S1C product descriptor for pairing tests."""
    return S1SLCProduct(
        path            = "S1C_IW_SLC__1SDV_20260601T053000_20260601T053027_067421_0A1B2C_1234.zip",
        satellite       = "S1C",
        start_time      = datetime(2026, 6, 1, 5, 30, 0),
        stop_time       = datetime(2026, 6, 1, 5, 30, 27),
        absolute_orbit  = 67421,
        polarisations   = ["VV", "VH"],
        subswaths       = ["IW1", "IW2", "IW3"],
        pass_direction  = "ASCENDING",
    )


@pytest.fixture
def s1c_slave_product():
    """Matching S1C slave (12-day repeat)."""
    return S1SLCProduct(
        path            = "S1C_IW_SLC__1SDV_20260613T053000_20260613T053027_067596_0A1B2D_5678.zip",
        satellite       = "S1C",
        start_time      = datetime(2026, 6, 13, 5, 30, 0),
        stop_time       = datetime(2026, 6, 13, 5, 30, 27),
        absolute_orbit  = 67596,
        polarisations   = ["VV", "VH"],
        subswaths       = ["IW1", "IW2", "IW3"],
        pass_direction  = "ASCENDING",
    )


@pytest.fixture
def synthetic_phase():
    """Small synthetic wrapped phase (64×64 pixels, radians)."""
    rng  = np.random.default_rng(42)
    h, w = 64, 64
    # Simulate a deformation bowl: smooth quadratic signal + noise
    x = np.linspace(-np.pi, np.pi, w)
    y = np.linspace(-np.pi, np.pi, h)
    X, Y = np.meshgrid(x, y)
    true_phase = 3.0 * np.exp(-(X**2 + Y**2) / 2)          # Gaussian bowl
    noise      = rng.normal(0, 0.1, (h, w))
    wrapped    = np.angle(np.exp(1j * (true_phase + noise)))  # wrap to [-π, π]
    return wrapped.astype(np.float32)


@pytest.fixture
def synthetic_coherence():
    """Corresponding coherence (high centre, lower edges)."""
    h, w = 64, 64
    x = np.linspace(-3, 3, w)
    y = np.linspace(-3, 3, h)
    X, Y = np.meshgrid(x, y)
    coh = 0.9 * np.exp(-(X**2 + Y**2) / 8) + 0.1
    return coh.astype(np.float32)


# ── sentinel1.py tests ────────────────────────────────────────────────────────

class TestS1SLCProduct:

    def test_parse_s1c_filename(self):
        name = "S1C_IW_SLC__1SDV_20260601T053000_20260601T053027_067421_0A1B2C_1234.zip"
        prod = parse_s1_slc_filename(name)
        assert prod is not None
        assert prod.satellite == "S1C"
        assert prod.start_time == datetime(2026, 6, 1, 5, 30, 0)
        assert "VV" in prod.polarisations
        assert "VH" in prod.polarisations
        assert prod.absolute_orbit == 67421
        assert prod.subswaths == ["IW1", "IW2", "IW3"]

    def test_parse_s1d_filename(self):
        """Sentinel-1D naming is identical — only satellite letter differs."""
        name = "S1D_IW_SLC__1SDV_20260613T053000_20260613T053027_067596_0A1B2D_5678.zip"
        prod = parse_s1_slc_filename(name)
        assert prod is not None
        assert prod.satellite == "S1D"

    def test_parse_s1a_filename_legacy(self):
        """Old Sentinel-1A products still parse correctly."""
        name = "S1A_IW_SLC__1SDV_20230206T153200_20230206T153227_047042_05A777_ABCD.zip"
        prod = parse_s1_slc_filename(name)
        assert prod is not None
        assert prod.satellite == "S1A"

    def test_parse_invalid_filename_returns_none(self):
        prod = parse_s1_slc_filename("not_a_sentinel_product.tif")
        assert prod is None

    def test_relative_orbit(self, s1c_master_product):
        orbit = s1c_master_product.relative_orbit
        assert 1 <= orbit <= 175

    def test_incidence_angle(self, s1c_master_product):
        assert abs(s1c_master_product.incidence_angle("IW2") - 38.3) < 0.1
        assert abs(s1c_master_product.incidence_angle("IW1") - 32.9) < 0.1
        assert abs(s1c_master_product.incidence_angle("IW3") - 43.1) < 0.1

    def test_wavelength_constant(self, s1c_master_product):
        assert abs(s1c_master_product.wavelength_m - SENTINEL1_WAVELENGTH_M) < 1e-8

    def test_repr(self, s1c_master_product):
        r = repr(s1c_master_product)
        assert "S1C" in r
        assert "20260601" in r

    def test_date_str(self, s1c_master_product):
        assert s1c_master_product.date_str == "20260601"

    def test_name_from_path(self, s1c_master_product):
        assert "S1C" in s1c_master_product.name


class TestValidateSLCPair:

    def test_valid_pair(self, s1c_master_product, s1c_slave_product):
        v = validate_slc_pair(s1c_master_product, s1c_slave_product)
        assert v["valid"] is True
        assert len(v["errors"]) == 0
        assert v["temporal_baseline_days"] == pytest.approx(12.0, abs=0.1)

    def test_pass_direction_mismatch(self, s1c_master_product, s1c_slave_product):
        s1c_slave_product.pass_direction = "DESCENDING"
        v = validate_slc_pair(s1c_master_product, s1c_slave_product)
        assert v["valid"] is False
        assert any("pass direction" in e.lower() for e in v["errors"])

    def test_large_temporal_baseline_warns(self, s1c_master_product, s1c_slave_product):
        # Move slave far in time
        from datetime import timedelta
        s1c_slave_product.start_time = s1c_master_product.start_time + timedelta(days=400)
        v = validate_slc_pair(s1c_master_product, s1c_slave_product)
        assert any("Temporal" in w for w in v["warnings"])

    def test_cross_satellite_pair_warns(self, s1c_master_product, s1c_slave_product):
        s1c_slave_product.satellite = "S1D"
        v = validate_slc_pair(s1c_master_product, s1c_slave_product)
        # Cross-satellite is a warning, not an error
        assert v["valid"] is True
        assert any("Cross-satellite" in w for w in v["warnings"])

    def test_common_polarisations(self, s1c_master_product, s1c_slave_product):
        v = validate_slc_pair(s1c_master_product, s1c_slave_product)
        assert "VV" in v["common_polarisations"]
        assert "VH" in v["common_polarisations"]

    def test_same_date_is_error(self, s1c_master_product):
        clone = S1SLCProduct(
            path           = s1c_master_product.path,
            satellite      = s1c_master_product.satellite,
            start_time     = s1c_master_product.start_time,
            stop_time      = s1c_master_product.stop_time,
            absolute_orbit = s1c_master_product.absolute_orbit,
            polarisations  = s1c_master_product.polarisations,
            pass_direction = s1c_master_product.pass_direction,
        )
        v = validate_slc_pair(s1c_master_product, clone)
        assert v["valid"] is False


# ── snap_graph.py tests ───────────────────────────────────────────────────────

class TestCheckEnvironment:

    def test_check_snap_returns_dict(self):
        result = check_snap()
        assert "available" in result
        assert "message" in result
        assert isinstance(result["available"], bool)

    def test_check_snapista_returns_dict(self):
        result = check_snapista()
        assert "available" in result
        assert "message" in result

    def test_check_environment_returns_dict(self):
        env = check_environment()
        assert "ready" in env
        assert "snap" in env
        assert "snapista" in env
        assert "snaphu" in env
        assert "summary" in env

    def test_check_snap_message_is_helpful_when_absent(self):
        result = check_snap()
        if not result["available"]:
            assert "step.esa.int" in result["message"] or "SNAP" in result["message"]


class TestSNAPGraphValidation:

    def test_invalid_subswath_raises(self):
        g = SNAPGraph()
        with pytest.raises(ValueError, match="subswath"):
            g._validate_inputs("IW9", (1, 3), "VV")

    def test_invalid_polarisation_raises(self):
        g = SNAPGraph()
        with pytest.raises(ValueError, match="polarisation"):
            g._validate_inputs("IW2", (1, 3), "XX")

    def test_invalid_burst_range_raises(self):
        g = SNAPGraph()
        with pytest.raises(ValueError, match="burst"):
            g._validate_inputs("IW2", (5, 3), "VV")   # first > last

    def test_invalid_burst_range_out_of_bounds(self):
        g = SNAPGraph()
        with pytest.raises(ValueError, match="burst"):
            g._validate_inputs("IW2", (0, 3), "VV")   # first < 1

    def test_valid_inputs_pass(self):
        g = SNAPGraph()
        g._validate_inputs("IW2", (2, 4), "VV")  # should not raise

    def test_xml_graph_structure(self):
        g = SNAPGraph()
        xml = g._build_insar_xml(
            master     = "master.zip",
            slave      = "slave.zip",
            snaphu_dir = "/tmp/snaphu",
            ifg_dim    = "/tmp/ifg",
            subswath   = "IW2",
            first_burst = 2,
            last_burst  = 4,
            pol         = "VV",
        )
        # Must be parseable XML
        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml)
        assert root.tag == "graph"

        # Must contain all required operators
        operators = {n.find("operator").text for n in root.findall("node")}
        required = {
            "Read", "TOPSAR-Split", "Apply-Orbit-File",
            "Back-Geocoding", "Enhanced-Spectral-Diversity",
            "Interferogram", "TOPSAR-Deburst", "TopoPhaseRemoval",
            "GoldsteinPhaseFiltering", "SnaphuExport", "Write",
        }
        assert required.issubset(operators), f"Missing: {required - operators}"

    def test_tc_xml_structure(self):
        g = SNAPGraph()
        xml = g._build_tc_xml("input.dim", "output.tif", 20.0)
        import xml.etree.ElementTree as ET
        root = ET.fromstring(xml)
        operators = {n.find("operator").text for n in root.findall("node")}
        assert "RangeDoppler-Terrain-Correction" in operators
        assert "Write" in operators

    def test_require_snap_raises_when_absent(self):
        g = SNAPGraph(gpt_path="/nonexistent/gpt")
        g._snap_ok = False
        with pytest.raises(RuntimeError, match="SNAP"):
            g._require_snap()


# ── snaphu.py tests ───────────────────────────────────────────────────────────

class TestSnaphuConfig:

    def test_conf_string_contains_required_keys(self):
        cfg = SnaphuConfig()
        conf = cfg.to_conf_string(
            wrapped_file   = "/tmp/phase.img",
            unwrapped_file = "/tmp/unwrap.img",
            coherence_file = "/tmp/coh.img",
            n_lines        = 1000,
            line_length    = 2000,
        )
        assert "STATCOSTMODE" in conf
        assert "DEFO"         in conf
        assert "LINELENGTH"   in conf
        assert "2000"         in conf
        assert "NLINES"       in conf
        assert "1000"         in conf

    def test_default_config_values(self):
        cfg = SnaphuConfig()
        assert cfg.stat_cost_mode == "DEFO"
        assert cfg.init_method    == "MCF"
        assert 0 < cfg.coh_threshold < 1


class TestSnaphuUnwrapper:

    def test_available_property_false_when_absent(self):
        u = SnaphuUnwrapper(snaphu_exe="/nonexistent/snaphu")
        assert u.available is False

    def test_run_raises_when_absent(self, tmp_path):
        u = SnaphuUnwrapper(snaphu_exe="/nonexistent/snaphu")
        with pytest.raises(RuntimeError, match="snaphu"):
            u.run(str(tmp_path))

    def test_parse_dimensions_from_conf(self, tmp_path):
        conf = tmp_path / "snaphu.conf"
        conf.write_text("NLINES     512\nLINELENGTH  1024\nSTATCOSTMODE DEFO\n")
        n, l = SnaphuUnwrapper._parse_dimensions(str(conf))
        assert n == 512
        assert l == 1024

    def test_parse_dimensions_missing_raises(self, tmp_path):
        conf = tmp_path / "snaphu.conf"
        conf.write_text("STATCOSTMODE DEFO\n")   # no dimensions
        with pytest.raises(ValueError, match="NLINES"):
            SnaphuUnwrapper._parse_dimensions(str(conf))

    def test_find_file_by_glob(self, tmp_path):
        (tmp_path / "phase.snaphu.img").touch()
        result = SnaphuUnwrapper._find_file(tmp_path, "*.snaphu.img")
        assert result is not None
        assert "snaphu.img" in result

    def test_run_from_rasters_raises_without_snaphu(self, synthetic_phase, synthetic_coherence):
        u = SnaphuUnwrapper(snaphu_exe="/nonexistent/snaphu")
        with pytest.raises(RuntimeError, match="snaphu"):
            u.run_from_rasters(synthetic_phase, synthetic_coherence, "/tmp/test_snaphu")


# ── displacement.py tests ─────────────────────────────────────────────────────

class TestSLCPhaseToDisplacement:

    def test_zero_phase_gives_zero_displacement(self):
        phase = np.zeros((32, 32), dtype=np.float32)
        result = slc_phase_to_displacement(phase)
        assert np.allclose(result["los_m"], 0, atol=1e-6)

    def test_scale_factor_is_wavelength_over_4pi(self):
        phase = np.ones((4, 4), dtype=np.float32) * np.pi
        result = slc_phase_to_displacement(phase, incidence_angle_deg=38.3)
        expected = SENTINEL1_WAVELENGTH_M / (4 * np.pi) * np.pi
        assert abs(float(np.nanmean(result["los_m"])) - expected) < 1e-5

    def test_coherence_mask_applied(self, synthetic_phase, synthetic_coherence):
        # Force low coherence everywhere → all masked
        low_coh = np.zeros_like(synthetic_coherence)
        result  = slc_phase_to_displacement(
            synthetic_phase, coherence=low_coh, coherence_threshold=0.3
        )
        assert np.all(np.isnan(result["los_m"]))

    def test_high_coherence_not_masked(self, synthetic_phase):
        high_coh = np.ones_like(synthetic_phase)
        result   = slc_phase_to_displacement(
            synthetic_phase, coherence=high_coh, coherence_threshold=0.3
        )
        assert not np.all(np.isnan(result["los_m"]))

    def test_result_has_required_keys(self, synthetic_phase):
        result = slc_phase_to_displacement(synthetic_phase)
        for key in ["los_m", "vertical_m", "masked", "phase_rad",
                    "incidence_deg", "scale_factor"]:
            assert key in result, f"Missing key: {key}"

    def test_los_shape_matches_input(self, synthetic_phase):
        result = slc_phase_to_displacement(synthetic_phase)
        assert result["los_m"].shape == synthetic_phase.shape

    def test_vertical_is_larger_than_los(self, synthetic_phase):
        # d_V = d_LOS / cos(θ) → |d_V| > |d_LOS| for θ > 0
        result   = slc_phase_to_displacement(synthetic_phase, incidence_angle_deg=38.3)
        los_rms  = np.sqrt(np.nanmean(result["los_m"] ** 2))
        vert_rms = np.sqrt(np.nanmean(result["vertical_m"] ** 2))
        assert vert_rms > los_rms

    def test_negative_phase_gives_negative_displacement(self):
        phase  = np.full((8, 8), -2 * np.pi, dtype=np.float32)
        result = slc_phase_to_displacement(phase)
        assert np.nanmean(result["los_m"]) < 0

    def test_output_dtype_is_float32(self, synthetic_phase):
        result = slc_phase_to_displacement(synthetic_phase)
        assert result["los_m"].dtype == np.float32


class TestLosToVertical:

    def test_nadir_incidence_gives_equal_result(self):
        los = np.array([0.1, -0.2, 0.3])
        vert = los_to_vertical(los, incidence_angle_deg=0.0)
        np.testing.assert_allclose(vert, los, atol=1e-6)

    def test_typical_incidence_amplifies(self):
        los  = np.ones((4, 4), dtype=np.float32) * 0.1
        vert = los_to_vertical(los, incidence_angle_deg=38.3)
        # 0.1 / cos(38.3°) ≈ 0.128
        assert abs(float(np.mean(vert)) - 0.1 / np.cos(np.deg2rad(38.3))) < 1e-4


class TestDualPassDecomposition:

    # Ascending: heading ~10°, incidence ~38.3°
    # Descending: heading ~170°, incidence ~38.3°
    # These are geometrically distinct and well-conditioned for decomposition.

    def test_pure_vertical_deformation(self):
        """Equal LOS signal in ascending and descending → dominated by vertical."""
        asc = np.full((8, 8), 0.1, dtype=np.float32)
        dsc = np.full((8, 8), 0.1, dtype=np.float32)
        result = dual_pass_decomposition(
            asc, dsc,
            asc_inc_deg=38.3, dsc_inc_deg=38.3,
            asc_head_deg=10.0, dsc_head_deg=350.0,
        )
        # Vertical should be non-zero; E-W near zero for equal asc/dsc
        v_mean  = float(np.nanmean(result["vertical_m"]))
        ew_mean = float(np.nanmean(result["ew_m"]))
        assert abs(v_mean)  > 0.05      # non-zero vertical signal
        assert abs(ew_mean) < abs(v_mean)   # EW smaller than vertical

    def test_output_shapes_match_input(self):
        h, w = 16, 24
        asc = np.random.rand(h, w).astype(np.float32) * 0.05
        dsc = np.random.rand(h, w).astype(np.float32) * 0.05
        result = dual_pass_decomposition(
            asc, dsc,
            asc_inc_deg=38.3, dsc_inc_deg=38.3,
            asc_head_deg=10.0, dsc_head_deg=350.0,
        )
        assert result["vertical_m"].shape == (h, w)
        assert result["ew_m"].shape        == (h, w)

    def test_nan_propagation(self):
        asc = np.ones((4, 4), dtype=np.float32) * 0.1
        dsc = np.ones((4, 4), dtype=np.float32) * 0.1
        asc[0, 0] = np.nan
        result = dual_pass_decomposition(
            asc, dsc,
            asc_inc_deg=38.3, dsc_inc_deg=38.3,
            asc_head_deg=10.0, dsc_head_deg=350.0,
        )
        assert np.isnan(result["vertical_m"][0, 0])
        assert np.isnan(result["ew_m"][0, 0])


class TestComputeDeformationRate:

    def test_linear_signal_recovers_rate(self):
        h, w = 8, 8
        # Create a scene subsiding at 30 mm/year = 0.03 m/year
        rate_true = 0.03   # m/year
        days      = [0, 12, 24, 36]
        stack     = [np.full((h, w), rate_true * d / 365.25, dtype=np.float32)
                     for d in days]
        result    = compute_deformation_rate(stack, days)
        rate_est  = float(np.nanmean(result["rate_m_per_year"]))
        assert abs(rate_est - rate_true) < 0.002   # within 2 mm/year

    def test_requires_at_least_2_maps(self):
        with pytest.raises(ValueError, match="at least 2"):
            compute_deformation_rate(
                [np.zeros((4, 4))], [0]
            )

    def test_output_shapes(self):
        h, w  = 12, 16
        stack = [np.random.rand(h, w).astype(np.float32) * 0.01 for _ in range(4)]
        days  = [0, 12, 24, 36]
        result = compute_deformation_rate(stack, days)
        assert result["rate_m_per_year"].shape == (h, w)
        assert result["residual_rms_mm"].shape  == (h, w)


# ── SLCInSARPipeline tests ────────────────────────────────────────────────────

class TestSLCInSARPipeline:

    def test_init_defaults(self):
        p = SLCInSARPipeline(
            master_zip = "master.zip",
            slave_zip  = "slave.zip",
        )
        assert p.subswath     == "IW2"
        assert p.polarisation == "VV"
        assert p.coherence_threshold == pytest.approx(0.3)
        assert p.incidence_deg == pytest.approx(SUBSWATH_INCIDENCE_DEG["IW2"])

    def test_iw1_incidence(self):
        p = SLCInSARPipeline("m.zip", "s.zip", subswath="IW1")
        assert p.incidence_deg == pytest.approx(SUBSWATH_INCIDENCE_DEG["IW1"])

    def test_check_environment_returns_dict(self):
        env = SLCInSARPipeline.check_environment()
        assert "ready" in env
        assert "snap" in env

    def test_run_raises_when_snap_absent(self, tmp_path):
        p = SLCInSARPipeline(
            master_zip = "nonexistent_master.zip",
            slave_zip  = "nonexistent_slave.zip",
            output_dir = str(tmp_path),
            gpt_path   = "/nonexistent/gpt",
        )
        with pytest.raises(Exception):
            p.run()

    def test_result_dataclass_defaults(self):
        r = SLCInSARResult()
        assert r.success is False
        assert r.displacement_path is None
        assert r.processing_log == []

    def test_result_summary_no_crash(self):
        r = SLCInSARResult(
            los_min_m       = -0.05,
            los_max_m       =  0.03,
            los_mean_m      = -0.01,
            los_std_m       =  0.02,
            mean_coherence  =  0.72,
            masked_fraction =  0.15,
            temporal_baseline = 12.0,
            elapsed_sec     = 1800.0,
            success         = True,
        )
        summary = r.summary()
        assert "LOS displacement" in summary
        assert "PyGeoVision" in summary


# ── Integration: imports through main insar module ────────────────────────────

class TestInSARModuleExports:

    def test_slc_pipeline_importable_from_insar(self):
        from pygeovision.insar import SLCInSARPipeline
        assert SLCInSARPipeline is not None

    def test_slc_result_importable_from_insar(self):
        from pygeovision.insar import SLCInSARResult
        assert SLCInSARResult is not None

    def test_check_snap_importable_from_insar(self):
        from pygeovision.insar import check_snap
        result = check_snap()
        assert "available" in result

    def test_check_snapista_importable_from_insar(self):
        from pygeovision.insar import check_snapista
        result = check_snapista()
        assert "available" in result

    def test_snaphu_unwrapper_importable_from_insar(self):
        from pygeovision.insar import SnaphuUnwrapper
        assert SnaphuUnwrapper is not None

    def test_slc_displacement_importable_from_insar(self):
        from pygeovision.insar import slc_phase_to_displacement, los_to_vertical
        assert callable(slc_phase_to_displacement)
        assert callable(los_to_vertical)

    def test_s1_product_importable_from_insar(self):
        from pygeovision.insar import S1SLCProduct, parse_s1_slc_manifest
        assert S1SLCProduct is not None
        assert callable(parse_s1_slc_manifest)

    def test_all_original_exports_still_present(self):
        from pygeovision.insar import (
    
            generate_interferogram, amplitude_coherence,
            estimate_coherence, coherence_mask,
            phase_to_displacement, displacement_rate,
            InSARInterpreter, DeformationReport,
            InSARViz,
        )
        # All original exports intact
        
        assert InSARViz is not None
