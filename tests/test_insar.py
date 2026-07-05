"""
tests/test_insar.py
====================
Tests for pygeovision.insar — interferogram, coherence, displacement,
interpretation, visualization, and the InSARProcessor façade.

All tests run on synthetic rasters; no real SAR data required.
"""
from __future__ import annotations

import json
import sys
import tempfile
import pathlib

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))


def make_sar_raster(path: str, width: int = 64, height: int = 64,
                    values: float = None) -> str:
    """Create a synthetic single-band SAR raster (normalised float32 [0,1])."""
    try:
        import rasterio
        from rasterio.transform import from_bounds
    except ImportError:
        pytest.skip("rasterio not installed")

    data = (np.random.uniform(0.05, 0.8, (1, height, width))
            if values is None
            else np.full((1, height, width), values)).astype("float32")

    transform = from_bounds(-0.3, 5.5, -0.05, 5.7, width, height)
    profile = {
        "driver": "GTiff", "dtype": "float32", "width": width,
        "height": height, "count": 1,
        "crs": rasterio.crs.CRS.from_epsg(4326),
        "transform": transform,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data[0], 1)   # data is (1,H,W); write band 1 with 2D array
    return path


def make_disp_raster(path: str, width: int = 32, height: int = 32,
                     vmin: float = -0.10, vmax: float = 0.05) -> str:
    """Create a synthetic displacement raster (metres, float32)."""
    try:
        import rasterio
        from rasterio.transform import from_bounds
    except ImportError:
        pytest.skip("rasterio not installed")

    data = np.random.uniform(vmin, vmax, (1, height, width)).astype("float32")
    transform = from_bounds(-0.3, 5.5, -0.05, 5.7, width, height)
    profile = {
        "driver": "GTiff", "dtype": "float32", "width": width,
        "height": height, "count": 1,
        "crs": rasterio.crs.CRS.from_epsg(4326),
        "transform": transform, "nodata": -9999.0,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# 1. Interferogram generation
# ══════════════════════════════════════════════════════════════════════════════

class TestInterferogram:

    def test_generate_interferogram_produces_two_files(self):
        from pygeovision.insar.interferogram import generate_interferogram
        with tempfile.TemporaryDirectory() as tmp:
            pre  = make_sar_raster(f"{tmp}/pre.tif")
            post = make_sar_raster(f"{tmp}/post.tif")
            result = generate_interferogram(pre, post, output_dir=f"{tmp}/out/",
                                            filter_type="none")
            assert pathlib.Path(result["interferogram_path"]).exists()
            assert pathlib.Path(result["coherence_path"]).exists()

    def test_interferogram_wrapped_phase_range(self):
        from pygeovision.insar.interferogram import generate_interferogram
        import rasterio
        with tempfile.TemporaryDirectory() as tmp:
            pre  = make_sar_raster(f"{tmp}/pre.tif")
            post = make_sar_raster(f"{tmp}/post.tif")
            result = generate_interferogram(pre, post, output_dir=f"{tmp}/out/",
                                            filter_type="none")
            with rasterio.open(result["interferogram_path"]) as src:
                data = src.read(1)
            # Wrapped phase should be in [-π, π]
            assert float(data.min()) >= -np.pi - 0.01
            assert float(data.max()) <= np.pi + 0.01

    def test_coherence_in_0_1_range(self):
        from pygeovision.insar.interferogram import generate_interferogram
        import rasterio
        with tempfile.TemporaryDirectory() as tmp:
            pre  = make_sar_raster(f"{tmp}/pre.tif")
            post = make_sar_raster(f"{tmp}/post.tif")
            result = generate_interferogram(pre, post, output_dir=f"{tmp}/out/",
                                            filter_type="none")
            with rasterio.open(result["coherence_path"]) as src:
                coh = src.read(1)
            assert float(coh.min()) >= 0.0 - 0.01
            assert float(coh.max()) <= 1.0 + 0.01

    def test_result_has_stats_keys(self):
        from pygeovision.insar.interferogram import generate_interferogram
        with tempfile.TemporaryDirectory() as tmp:
            pre  = make_sar_raster(f"{tmp}/pre.tif")
            post = make_sar_raster(f"{tmp}/post.tif")
            result = generate_interferogram(pre, post, output_dir=f"{tmp}/out/",
                                            filter_type="none")
            assert "mean_coherence" in result["stats"]
            assert "changed_pct"    in result["stats"]
            assert "pixel_spacing_m" in result["stats"]

    def test_identical_pre_post_high_coherence(self):
        from pygeovision.insar.interferogram import generate_interferogram
        import rasterio
        with tempfile.TemporaryDirectory() as tmp:
            # Identical arrays → amplitude coherence should be near 1
            pre = make_sar_raster(f"{tmp}/pre.tif", values=0.4)
            make_sar_raster(f"{tmp}/post.tif", values=0.4)
            post = f"{tmp}/post.tif"
            result = generate_interferogram(pre, post, output_dir=f"{tmp}/out/",
                                            filter_type="none")
            assert result["stats"]["mean_coherence"] > 0.7

    def test_amplitude_coherence_shape(self):
        from pygeovision.insar.interferogram import amplitude_coherence
        pre  = np.random.uniform(0.1, 0.8, (64, 64)).astype("float32")
        post = np.random.uniform(0.1, 0.8, (64, 64)).astype("float32")
        coh  = amplitude_coherence(pre, post, window_size=5)
        assert coh.shape == (64, 64)
        assert coh.dtype == np.float32

    def test_amplitude_coherence_range(self):
        from pygeovision.insar.interferogram import amplitude_coherence
        pre  = np.random.uniform(0.1, 0.8, (32, 32)).astype("float32")
        post = pre.copy()  # identical → high coherence
        coh  = amplitude_coherence(pre, post, window_size=5)
        assert float(coh.min()) >= 0.0 - 0.01
        assert float(coh.max()) <= 1.0 + 0.01

    def test_goldstein_filter_preserves_shape(self):
        from pygeovision.insar.interferogram import _goldstein_filter
        phase = np.random.uniform(-np.pi, np.pi, (64, 64)).astype("float32")
        coh   = np.random.uniform(0.3, 0.9, (64, 64)).astype("float32")
        filtered = _goldstein_filter(phase, coh, alpha=0.5, block=16)
        assert filtered.shape == phase.shape


# ══════════════════════════════════════════════════════════════════════════════
# 2. Coherence
# ══════════════════════════════════════════════════════════════════════════════

class TestCoherence:

    def test_estimate_coherence_produces_file(self):
        from pygeovision.insar.coherence import estimate_coherence
        with tempfile.TemporaryDirectory() as tmp:
            pre  = make_sar_raster(f"{tmp}/pre.tif")
            post = make_sar_raster(f"{tmp}/post.tif")
            out  = f"{tmp}/coherence.tif"
            coh  = estimate_coherence(pre, post, output_path=out, window_size=5)
            assert pathlib.Path(out).exists()
            assert coh.dtype == np.float32
            assert float(coh.min()) >= 0.0 - 0.01
            assert float(coh.max()) <= 1.0 + 0.01

    def test_coherence_mask_above_threshold(self):
        from pygeovision.insar.coherence import coherence_mask
        coh  = np.array([[0.2, 0.5, 0.8], [0.3, 0.6, 0.9]], dtype="float32")
        mask = coherence_mask(coh, threshold=0.5)
        assert mask.dtype == np.uint8
        assert mask[0, 0] == 0  # 0.2 < 0.5
        assert mask[0, 2] == 1  # 0.8 >= 0.5
        assert mask[1, 1] == 1  # 0.6 >= 0.5

    def test_coherence_mask_shape(self):
        from pygeovision.insar.coherence import coherence_mask
        coh  = np.random.uniform(0, 1, (32, 32)).astype("float32")
        mask = coherence_mask(coh, threshold=0.4)
        assert mask.shape == coh.shape

    def test_temporal_coherence_shape(self):
        from pygeovision.insar.coherence import temporal_coherence
        stack = np.random.uniform(-np.pi, np.pi, (5, 32, 32)).astype("float32")
        tc    = temporal_coherence(stack, window_size=3)
        assert tc.shape == (32, 32)
        assert tc.dtype == np.float32


# ══════════════════════════════════════════════════════════════════════════════
# 3. Displacement
# ══════════════════════════════════════════════════════════════════════════════

class TestDisplacement:

    def test_slc_phase_conversion(self):
        from pygeovision.insar.displacement import phase_to_displacement
        with tempfile.TemporaryDirectory() as tmp:
            # Synthetic unwrapped phase (radians)
            path = make_sar_raster(f"{tmp}/phase.tif", values=np.pi)
            out  = f"{tmp}/disp.tif"
            disp = phase_to_displacement(path, out, mode="slc_phase",
                                          wavelength=0.05546576)
            # d_LOS = λ/(4π) × φ = 0.05546576/(4π) × π ≈ 0.01386 m
            expected = 0.05546576 / (4 * np.pi) * np.pi
            assert abs(float(np.nanmean(disp)) - expected) < 1e-4

    def test_amplitude_proxy_conversion(self):
        from pygeovision.insar.displacement import phase_to_displacement
        with tempfile.TemporaryDirectory() as tmp:
            path = make_sar_raster(f"{tmp}/change.tif")
            out  = f"{tmp}/disp.tif"
            disp = phase_to_displacement(path, out, mode="amplitude_proxy")
            assert disp.dtype == np.float32
            # Values should be in ±0.5 m range for normalized input
            assert float(np.nanmax(np.abs(disp))) <= 0.5 + 0.01

    def test_output_file_created(self):
        from pygeovision.insar.displacement import phase_to_displacement
        with tempfile.TemporaryDirectory() as tmp:
            path = make_sar_raster(f"{tmp}/phase.tif")
            out  = f"{tmp}/disp.tif"
            phase_to_displacement(path, out)
            assert pathlib.Path(out).exists()

    def test_vertical_projection(self):
        from pygeovision.insar.displacement import phase_to_displacement
        with tempfile.TemporaryDirectory() as tmp:
            # Vertical projected displacement > LOS displacement
            path = make_sar_raster(f"{tmp}/phase.tif", values=0.5)
            out_los  = f"{tmp}/disp_los.tif"
            out_vert = f"{tmp}/disp_vert.tif"
            disp_los  = phase_to_displacement(path, out_los, mode="amplitude_proxy",
                                               vertical_only=False)
            disp_vert = phase_to_displacement(path, out_vert, mode="amplitude_proxy",
                                               vertical_only=True, incidence_angle_deg=39.0)
            # Vertical component > LOS for incidence > 0
            assert float(np.nanmean(np.abs(disp_vert))) > float(np.nanmean(np.abs(disp_los)))

    def test_displacement_rate_linear_trend(self):
        from pygeovision.insar.displacement import displacement_rate
        with tempfile.TemporaryDirectory() as tmp:
            # Create 4 displacement rasters with known linear trend
            paths = []
            for i, val in enumerate([-0.01, -0.02, -0.03, -0.04]):
                p = make_disp_raster(f"{tmp}/disp_{i}.tif", vmin=val, vmax=val)
                paths.append(p)
            dates  = ["2023-01-01","2023-04-01","2023-07-01","2023-10-01"]
            out    = f"{tmp}/rate.tif"
            rate   = displacement_rate(paths, dates, output_path=out)
            assert pathlib.Path(out).exists()
            # Should be negative (subsidence)
            assert float(np.nanmean(rate)) < 0

    def test_displacement_rate_requires_at_least_2(self):
        from pygeovision.insar.displacement import displacement_rate
        with pytest.raises(AssertionError):
            displacement_rate(["a.tif"], ["2023-01-01"])

    def test_displacement_rate_mismatched_lengths_raises(self):
        from pygeovision.insar.displacement import displacement_rate
        with pytest.raises(AssertionError):
            displacement_rate(["a.tif","b.tif"], ["2023-01-01"])


# ══════════════════════════════════════════════════════════════════════════════
# 4. Interpretation
# ══════════════════════════════════════════════════════════════════════════════

class TestInterpretation:

    def test_interpret_returns_report(self):
        from pygeovision.insar.interpretation import InSARInterpreter, DeformationReport
        with tempfile.TemporaryDirectory() as tmp:
            path   = make_disp_raster(f"{tmp}/disp.tif")
            interp = InSARInterpreter()
            report = interp.interpret(path, study_area="Test Area")
            assert isinstance(report, DeformationReport)

    def test_report_has_required_fields(self):
        from pygeovision.insar.interpretation import InSARInterpreter
        with tempfile.TemporaryDirectory() as tmp:
            path   = make_disp_raster(f"{tmp}/disp.tif", vmin=-0.05, vmax=0.02)
            interp = InSARInterpreter()
            report = interp.interpret(path)
            assert hasattr(report, "max_subsidence_m")
            assert hasattr(report, "max_uplift_m")
            assert hasattr(report, "subsidence_extent_km2")
            assert hasattr(report, "zones")
            assert hasattr(report, "flags")
            assert isinstance(report.zones, list)
            assert isinstance(report.flags, list)

    def test_subsidence_detected(self):
        from pygeovision.insar.interpretation import InSARInterpreter
        with tempfile.TemporaryDirectory() as tmp:
            # Large subsidence signal (all pixels below -0.05 threshold)
            path   = make_disp_raster(f"{tmp}/disp.tif", vmin=-0.15, vmax=-0.10)
            # min_zone_km2=0.0 so tiny synthetic rasters are not filtered out
            interp = InSARInterpreter(subsidence_threshold_m=-0.05, min_zone_km2=0.0)
            report = interp.interpret(path)
            # All pixels are between -0.15 and -0.10 (below threshold -0.05)
            assert report.max_subsidence_m < -0.05
            sub_zones = [z for z in report.zones if z.zone_type == "subsidence"]
            assert len(sub_zones) > 0

    def test_critical_flag_on_large_subsidence(self):
        from pygeovision.insar.interpretation import InSARInterpreter
        with tempfile.TemporaryDirectory() as tmp:
            # > 20 cm subsidence should trigger critical flag
            path   = make_disp_raster(f"{tmp}/disp.tif", vmin=-0.30, vmax=-0.25)
            interp = InSARInterpreter(min_zone_km2=0.0)
            report = interp.interpret(path)
            # max_subsidence should be < -0.20 (i.e. > 20 cm)
            assert report.max_subsidence_m < -0.20
            assert any("CRITICAL" in f or "subsidence" in f.lower()
                       or "cm" in f for f in report.flags)

    def test_summary_is_string(self):
        from pygeovision.insar.interpretation import InSARInterpreter
        with tempfile.TemporaryDirectory() as tmp:
            path   = make_disp_raster(f"{tmp}/disp.tif")
            interp = InSARInterpreter()
            report = interp.interpret(path, study_area="Jakarta")
            s = report.summary()
            assert isinstance(s, str)
            assert "Jakarta" in s
            assert "subsidence" in s.lower() or "displacement" in s.lower()

    def test_report_export_json(self):
        from pygeovision.insar.interpretation import InSARInterpreter
        with tempfile.TemporaryDirectory() as tmp:
            path   = make_disp_raster(f"{tmp}/disp.tif")
            interp = InSARInterpreter()
            report = interp.interpret(path)
            out    = f"{tmp}/report.json"
            result = report.export(out, format="json")
            assert pathlib.Path(out).exists()
            data = json.load(open(out))
            assert "study_area" in data
            assert "zones" in data

    def test_report_export_markdown(self):
        from pygeovision.insar.interpretation import InSARInterpreter
        with tempfile.TemporaryDirectory() as tmp:
            path   = make_disp_raster(f"{tmp}/disp.tif")
            interp = InSARInterpreter()
            report = interp.interpret(path, study_area="Test")
            out    = f"{tmp}/report.md"
            report.export(out, format="md")
            assert pathlib.Path(out).exists()
            md = pathlib.Path(out).read_text()
            assert "Test" in md

    def test_report_to_dict_json_serialisable(self):
        from pygeovision.insar.interpretation import InSARInterpreter
        import json as _json
        with tempfile.TemporaryDirectory() as tmp:
            path   = make_disp_raster(f"{tmp}/disp.tif")
            interp = InSARInterpreter()
            report = interp.interpret(path)
            d = report.to_dict()
            _json.dumps(d, default=str)   # must not raise


# ══════════════════════════════════════════════════════════════════════════════
# 5. InSARViz
# ══════════════════════════════════════════════════════════════════════════════

class TestInSARViz:

    def test_constructs(self):
        from pygeovision.insar.visualization import InSARViz
        viz = InSARViz(figsize=(10, 8))
        assert viz._fig is None

    def test_interferogram_sets_figure(self):
        from pygeovision.insar.visualization import InSARViz
        with tempfile.TemporaryDirectory() as tmp:
            path = make_sar_raster(f"{tmp}/ifg.tif")
            viz  = InSARViz()
            viz.interferogram(path)
            assert viz._fig is not None

    def test_coherence_sets_figure(self):
        from pygeovision.insar.visualization import InSARViz
        with tempfile.TemporaryDirectory() as tmp:
            path = make_sar_raster(f"{tmp}/coh.tif")
            viz  = InSARViz()
            viz.coherence(path)
            assert viz._fig is not None

    def test_displacement_sets_figure(self):
        from pygeovision.insar.visualization import InSARViz
        with tempfile.TemporaryDirectory() as tmp:
            path = make_disp_raster(f"{tmp}/disp.tif")
            viz  = InSARViz()
            viz.displacement(path)
            assert viz._fig is not None

    def test_deformation_rate_sets_figure(self):
        from pygeovision.insar.visualization import InSARViz
        with tempfile.TemporaryDirectory() as tmp:
            path = make_disp_raster(f"{tmp}/rate.tif")
            viz  = InSARViz()
            viz.deformation_rate(path)
            assert viz._fig is not None

    def test_export_png(self):
        from pygeovision.insar.visualization import InSARViz
        with tempfile.TemporaryDirectory() as tmp:
            path = make_sar_raster(f"{tmp}/coh.tif")
            out  = f"{tmp}/coh.png"
            viz  = InSARViz()
            viz.coherence(path)
            result = viz.export(out)
            assert pathlib.Path(out).exists()

    def test_export_without_figure_raises(self):
        from pygeovision.insar.visualization import InSARViz
        viz = InSARViz()
        with pytest.raises(RuntimeError, match="Nothing to export"):
            viz.export("/tmp/out.png")

    def test_dashboard_partial_inputs(self):
        from pygeovision.insar.visualization import InSARViz
        with tempfile.TemporaryDirectory() as tmp:
            disp = make_disp_raster(f"{tmp}/disp.tif")
            viz  = InSARViz()
            # Should not crash if some paths are None
            viz.dashboard(displacement_path=disp)
            assert viz._fig is not None

    def test_time_series_spatial_mean(self):
        from pygeovision.insar.visualization import InSARViz
        with tempfile.TemporaryDirectory() as tmp:
            paths = [make_disp_raster(f"{tmp}/d_{i}.tif") for i in range(4)]
            dates = ["2023-01-01","2023-04-01","2023-07-01","2023-10-01"]
            viz   = InSARViz()
            viz.time_series(paths, dates)
            assert viz._fig is not None


# ══════════════════════════════════════════════════════════════════════════════
# 6. InSARProcessor (integration)
# ══════════════════════════════════════════════════════════════════════════════

class TestInSARProcessor:

    def test_constructs(self):
        from pygeovision.insar import InSARProcessor
        proc = InSARProcessor(output_dir="/tmp/pgv_test_insar/")
        assert proc._wl    == 0.05546576
        assert proc._angle == 39.0

    def test_interferogram_produces_result(self):
        from pygeovision.insar import InSARProcessor, InSARResult
        with tempfile.TemporaryDirectory() as tmp:
            pre  = make_sar_raster(f"{tmp}/pre.tif")
            post = make_sar_raster(f"{tmp}/post.tif")
            proc = InSARProcessor(output_dir=f"{tmp}/insar/")
            result = proc.interferogram(pre, post, filter_type="none")
            assert isinstance(result, InSARResult)
            assert result.interferogram_path is not None
            assert result.coherence_path is not None

    def test_displacement_produces_result(self):
        from pygeovision.insar import InSARProcessor, InSARResult
        with tempfile.TemporaryDirectory() as tmp:
            pre  = make_sar_raster(f"{tmp}/pre.tif")
            post = make_sar_raster(f"{tmp}/post.tif")
            proc = InSARProcessor(output_dir=f"{tmp}/insar/")
            ifg  = proc.interferogram(pre, post, filter_type="none")
            disp = proc.displacement(ifg)
            assert isinstance(disp, InSARResult)
            assert disp.displacement_path is not None
            assert pathlib.Path(disp.displacement_path).exists()

    def test_full_pipeline_produces_report(self):
        from pygeovision.insar import InSARProcessor
        with tempfile.TemporaryDirectory() as tmp:
            pre  = make_sar_raster(f"{tmp}/pre.tif")
            post = make_sar_raster(f"{tmp}/post.tif")
            proc = InSARProcessor(output_dir=f"{tmp}/insar/")
            result = proc.full_pipeline(pre, post, study_area="Test Area")
            assert result.report is not None
            assert result.displacement_path is not None
            assert "Test Area" in result.report.study_area

    def test_repr(self):
        from pygeovision.insar import InSARProcessor
        proc = InSARProcessor()
        assert "InSARProcessor" in repr(proc)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
