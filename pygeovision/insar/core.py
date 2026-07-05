"""
pygeovision.insar.core
=======================
High-level InSAR processing façade.

InSARProcessor chains all InSAR modules behind a single object,
mirrors the PyGeoVision client's ergonomic API, and handles the
GRD-amplitude vs SLC-phase mode selection automatically.

Usage::

    from pygeovision.insar import InSARProcessor

    proc = InSARProcessor(output_dir="./insar_results/")

    # Single-pair interferogram from preprocessed SAR rasters
    result = proc.interferogram("sar_pre_ready.tif", "sar_post_ready.tif")
    print(result.interferogram_path)

    # Full pipeline: interferogram → coherence → displacement → interpret
    full  = proc.full_pipeline("sar_pre_ready.tif", "sar_post_ready.tif",
                                study_area="Istanbul, Turkey")
    print(full.report.summary())

    # Time-series deformation rate
    rate  = proc.deformation_rate(
        displacement_paths=["disp_t0.tif", "disp_t1.tif", "disp_t2.tif"],
        dates=["2023-01-01", "2023-07-01", "2024-01-01"],
    )
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("pygeovision.insar.core")


@dataclass
class InSARResult:
    """Results container returned by InSARProcessor methods."""
    interferogram_path: Optional[str] = None
    coherence_path:     Optional[str] = None
    displacement_path:  Optional[str] = None
    rate_path:          Optional[str] = None
    report:             Any           = None   # DeformationReport
    stats:              Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        parts = [f"interferogram={Path(self.interferogram_path).name}" if self.interferogram_path else ""]
        parts += [f"coherence={Path(self.coherence_path).name}" if self.coherence_path else ""]
        parts += [f"displacement={Path(self.displacement_path).name}" if self.displacement_path else ""]
        return f"InSARResult({', '.join(p for p in parts if p)})"


class InSARProcessor:
    """
    High-level InSAR processing pipeline.

    Parameters
    ----------
    output_dir : str
        Default output directory for all products.
    wavelength : float
        SAR wavelength in metres (Sentinel-1 C-band = 0.05546576).
    incidence_angle_deg : float
        Mean incidence angle of the acquisition.
    coherence_threshold : float
        Pixels below this coherence level are masked in displacement maps.
    """

    def __init__(
        self,
        output_dir:           str   = "./insar/",
        wavelength:           float = 0.05546576,
        incidence_angle_deg:  float = 39.0,
        coherence_threshold:  float = 0.4,
    ) -> None:
        self._out   = Path(output_dir)
        self._wl    = wavelength
        self._angle = incidence_angle_deg
        self._coh_t = coherence_threshold
        self._out.mkdir(parents=True, exist_ok=True)

        # Sub-module instances (lazy import)
        self._viz:  Optional[Any] = None
        self._interp: Optional[Any] = None

    @property
    def viz(self):
        if self._viz is None:
            from pygeovision.insar.visualization import InSARViz
            self._viz = InSARViz()
        return self._viz

    @property
    def interpret(self):
        if self._interp is None:
            from pygeovision.insar.interpretation import InSARInterpreter
            self._interp = InSARInterpreter(coherence_threshold=self._coh_t)
        return self._interp

    # ── Single interferogram pair ──────────────────────────────────────────────

    def interferogram(
        self,
        pre_path:   str,
        post_path:  str,
        multilook:  int  = 1,
        filter_type: str = "goldstein",
        name:        str = "pair",
    ) -> InSARResult:
        """Generate interferogram + coherence from a pre/post SAR pair."""
        from pygeovision.insar.interferogram import generate_interferogram

        out_dir = self._out / name
        result  = generate_interferogram(
            pre_path, post_path,
            output_dir=str(out_dir),
            multilook=multilook,
            filter_type=filter_type,
        )
        return InSARResult(
            interferogram_path=result["interferogram_path"],
            coherence_path    =result["coherence_path"],
            stats             =result["stats"],
        )

    # ── Displacement ──────────────────────────────────────────────────────────

    def displacement(
        self,
        interferogram_result: InSARResult,
        mode: str = "amplitude_proxy",
    ) -> InSARResult:
        """Convert interferogram to displacement map."""
        from pygeovision.insar.displacement import phase_to_displacement

        if not interferogram_result.interferogram_path:
            raise ValueError("InSARResult must have interferogram_path")

        out_path = str(self._out / "displacement.tif")
        disp = phase_to_displacement(
            interferogram_result.interferogram_path,
            output_path=out_path,
            mode=mode,
            wavelength=self._wl,
            incidence_angle_deg=self._angle,
        )
        return InSARResult(
            interferogram_path=interferogram_result.interferogram_path,
            coherence_path    =interferogram_result.coherence_path,
            displacement_path =out_path,
            stats={**interferogram_result.stats,
                   "max_displacement_m": round(float(abs(disp).max()), 4)},
        )

    # ── Deformation rate (time-series) ────────────────────────────────────────

    def deformation_rate(
        self,
        displacement_paths: List[str],
        dates:              List[str],
        coherence_path:     Optional[str] = None,
    ) -> InSARResult:
        """Compute pixel-wise deformation rate from a displacement time-series."""
        from pygeovision.insar.displacement import displacement_rate

        out_path = str(self._out / "deformation_rate_mm_yr.tif")
        rate = displacement_rate(
            displacement_paths, dates,
            output_path=out_path,
            mask_path=coherence_path,
        )
        mean_rate = float(rate[~(rate == -9999.0)].mean()) if rate.size > 0 else 0.0
        return InSARResult(rate_path=out_path, stats={"mean_rate_mm_yr": round(mean_rate, 2)})

    # ── Full pipeline ─────────────────────────────────────────────────────────

    def full_pipeline(
        self,
        pre_path:    str,
        post_path:   str,
        study_area:  str = "Study Area",
        multilook:   int = 1,
        filter_type: str = "goldstein",
    ) -> InSARResult:
        """
        Run the complete InSAR chain:
          interferogram → coherence → displacement → interpretation

        Returns an ``InSARResult`` with all product paths and a
        ``DeformationReport`` attached as ``.report``.
        """
        logger.info("Running full InSAR pipeline for %s", study_area)

        # Step 1: Interferogram
        ifg_result = self.interferogram(pre_path, post_path,
                                         multilook=multilook, filter_type=filter_type)
        logger.info("  Interferogram: done  coh=%.3f", ifg_result.stats.get("mean_coherence", 0))

        # Step 2: Displacement
        disp_result = self.displacement(ifg_result)
        logger.info("  Displacement: done  max=%.3f m", disp_result.stats.get("max_displacement_m", 0))

        # Step 3: Interpretation
        report = self.interpret.interpret(
            displacement_path=disp_result.displacement_path,
            coherence_path   =disp_result.coherence_path,
            study_area=study_area,
        )
        logger.info("  Interpretation: %d zones detected, %d flags",
                    len(report.zones), len(report.flags))

        # Export report
        report_path = str(self._out / "insar_report.json")
        report.export(report_path)

        return InSARResult(
            interferogram_path=disp_result.interferogram_path,
            coherence_path    =disp_result.coherence_path,
            displacement_path =disp_result.displacement_path,
            report=report,
            stats={**disp_result.stats, "report_path": report_path},
        )

    def __repr__(self) -> str:
        return (f"InSARProcessor(output_dir={str(self._out)!r}, "
                f"wavelength={self._wl}, angle={self._angle}°)")
