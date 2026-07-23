"""
pygeovision.insar.processor
=============================
High-level ``InSARProcessor`` facade combining interferogram generation,
displacement estimation, interpretation, and visualization into a single
object, as documented in ``pygeovision.insar``'s module docstring::

    from pygeovision.insar import InSARProcessor

    proc = InSARProcessor(output_dir="./insar/")
    ifg  = proc.interferogram("pre_sar.tif", "post_sar.tif")
    disp = proc.displacement(ifg)
    result = proc.full_pipeline("pre_sar.tif", "post_sar.tif", study_area="Jakarta")
    print(result.report.summary())
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pygeovision.insar.displacement import SENTINEL1_WAVELENGTH, phase_to_displacement
from pygeovision.insar.interferogram import generate_interferogram
from pygeovision.insar.interpretation import DeformationReport, InSARInterpreter
from pygeovision.insar.visualization import InSARViz

logger = logging.getLogger("pygeovision.insar.processor")


@dataclass
class InSARResult:
    """Container for the outputs produced at any stage of ``InSARProcessor``."""
    interferogram_path: str | None = None
    coherence_path:     str | None = None
    displacement_path:  str | None = None
    stats:               dict[str, Any] = field(default_factory=dict)
    report:               DeformationReport | None = None


class InSARProcessor:
    """
    End-to-end InSAR processing facade (GRD amplitude-proxy mode by default).

    Parameters
    ----------
    output_dir : str
        Directory for generated interferogram/coherence/displacement rasters.
    wavelength : float
        SAR wavelength in metres. Defaults to Sentinel-1 C-band.
    incidence_angle : float
        Mean incidence angle in degrees, used for LOS→vertical projection.
    """

    def __init__(
        self,
        output_dir: str = "./insar/",
        wavelength: float = SENTINEL1_WAVELENGTH,
        incidence_angle: float = 39.0,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._wl = wavelength
        self._angle = incidence_angle
        self.interpreter = InSARInterpreter()
        self.viz = InSARViz()

    def interferogram(
        self,
        pre_path: str,
        post_path: str,
        multilook: int = 1,
        filter_type: str = "goldstein",
        filter_strength: float = 0.5,
    ) -> InSARResult:
        """Generate an interferogram (amplitude-change proxy) + coherence map."""
        out = generate_interferogram(
            pre_path, post_path,
            output_dir=str(self.output_dir),
            multilook=multilook,
            filter_type=filter_type,
            filter_strength=filter_strength,
        )
        return InSARResult(
            interferogram_path=out["interferogram_path"],
            coherence_path=out["coherence_path"],
            stats=out.get("stats", {}),
        )

    def displacement(
        self,
        ifg: InSARResult,
        mode: str = "amplitude_proxy",
        vertical_only: bool = False,
    ) -> InSARResult:
        """Convert an interferogram result to a ground-displacement raster."""
        if ifg.interferogram_path is None:
            raise ValueError("InSARResult has no interferogram_path — run interferogram() first")

        displacement_path = str(self.output_dir / "displacement.tif")
        phase_to_displacement(
            ifg.interferogram_path,
            output_path=displacement_path,
            mode=mode,
            wavelength=self._wl,
            incidence_angle_deg=self._angle,
            vertical_only=vertical_only,
        )
        return InSARResult(
            interferogram_path=ifg.interferogram_path,
            coherence_path=ifg.coherence_path,
            displacement_path=displacement_path,
            stats=ifg.stats,
        )

    def interpret(
        self,
        disp: InSARResult,
        study_area: str = "Study Area",
    ) -> DeformationReport:
        """Run interpretation on a displacement result, producing a report."""
        if disp.displacement_path is None:
            raise ValueError("InSARResult has no displacement_path — run displacement() first")
        return self.interpreter.interpret(
            displacement_path=disp.displacement_path,
            coherence_path=disp.coherence_path,
            study_area=study_area,
        )

    def full_pipeline(
        self,
        pre_path: str,
        post_path: str,
        study_area: str = "Study Area",
        filter_type: str = "goldstein",
    ) -> InSARResult:
        """Run interferogram → displacement → interpretation in one call."""
        ifg = self.interferogram(pre_path, post_path, filter_type=filter_type)
        disp = self.displacement(ifg)
        report = self.interpret(disp, study_area=study_area)
        disp.report = report
        return disp

    def __repr__(self) -> str:
        return (
            f"InSARProcessor(output_dir='{self.output_dir}', "
            f"wavelength={self._wl}, incidence_angle={self._angle})"
        )
