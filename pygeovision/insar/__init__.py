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

from pygeovision.insar.coherence import coherence_mask, estimate_coherence
from pygeovision.insar.displacement import displacement_rate, phase_to_displacement
from pygeovision.insar.interferogram import amplitude_coherence, generate_interferogram
from pygeovision.insar.interpretation import DeformationReport, InSARInterpreter
from pygeovision.insar.processor import InSARProcessor, InSARResult

# ── True SLC InSAR (SNAP + snapista + snaphu required) ────────────────────────
from pygeovision.insar.slc import (
    S1SLCProduct,
    SLCInSARPipeline,
    SLCInSARResult,
    SNAPGraph,
    SnaphuUnwrapper,
    check_snap,
    check_snapista,
    los_to_vertical,
    parse_s1_slc_manifest,
    slc_phase_to_displacement,
    unwrap_phase,
)
from pygeovision.insar.visualization import InSARViz

__all__ = [
    # GRD amplitude proxy (no external dependencies)

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
