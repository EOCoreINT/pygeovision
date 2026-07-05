"""
pygeovision.insar
==================
Complete InSAR processing chain for PyGeoVision.

Covers: coregistration → interferogram → coherence → unwrapping →
        displacement → PS-InSAR → SBAS → visualization → interpretation.

Quick reference::

    from pygeovision.insar import InSARProcessor

    proc = InSARProcessor()

    # Single-pair interferogram (GRD amplitude proxy)
    result = proc.interferogram("pre_sar.tif", "post_sar.tif", output_dir="./insar/")
    # result.interferogram_path, result.coherence_path

    # Displacement from unwrapped phase
    disp = proc.displacement("unwrapped.tif", wavelength=0.055, incidence_angle=39.0)

    # Interpretation
    report = proc.interpret(disp)
    print(report.summary())

    # Visualization
    proc.viz.interferogram("interferogram.tif")
    proc.viz.coherence("coherence.tif")
    proc.viz.displacement("displacement.tif")

For full InSAR (centimetre precision) from Sentinel-1 SLC data:
    Use pyroSAR + ESA SNAP → isce2 / MintPy → import results here.
    This module provides the analysis and visualization layer above raw InSAR.
"""

from pygeovision.insar.core          import InSARProcessor, InSARResult
from pygeovision.insar.interferogram import generate_interferogram, amplitude_coherence
from pygeovision.insar.coherence     import estimate_coherence, coherence_mask
from pygeovision.insar.displacement  import phase_to_displacement, displacement_rate
from pygeovision.insar.interpretation import InSARInterpreter, DeformationReport
from pygeovision.insar.visualization  import InSARViz

# ── True SLC InSAR (SNAP + snapista + snaphu required) ────────────────────────
from pygeovision.insar.slc           import (
    SLCInSARPipeline, SLCInSARResult,
    SNAPGraph, check_snap, check_snapista,
    SnaphuUnwrapper, unwrap_phase,
    slc_phase_to_displacement, los_to_vertical,
    S1SLCProduct, parse_s1_slc_manifest,
)

__all__ = [
    # GRD amplitude proxy (no external dependencies)
    "InSARProcessor", "InSARResult",
    "generate_interferogram", "amplitude_coherence",
    "estimate_coherence", "coherence_mask",
    "phase_to_displacement", "displacement_rate",
    "InSARInterpreter", "DeformationReport",
    "InSARViz",
    # True SLC InSAR (SNAP + snapista + snaphu)
    "SLCInSARPipeline", "SLCInSARResult",
    "SNAPGraph", "check_snap", "check_snapista",
    "SnaphuUnwrapper", "unwrap_phase",
    "slc_phase_to_displacement", "los_to_vertical",
    "S1SLCProduct", "parse_s1_slc_manifest",
]
