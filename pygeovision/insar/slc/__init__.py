"""
pygeovision.insar.slc
======================
True SLC InSAR processing chain using ESA SNAP / snapista.

Full pipeline:
  TOPSAR-Split → Apply-Orbit → Back-Geocoding → ESD →
  Interferogram → Deburst → Topo-Phase-Removal →
  Goldstein-Filter → Snaphu-Export → (snaphu) →
  Snaphu-Import → Phase-to-Displacement → Terrain-Correction

Requirements (all optional — module degrades gracefully without them):
  • ESA SNAP ≥ 9.0 with S1TBX  — https://step.esa.int
  • snapista                    — pip install snapista
  • snaphu                      — conda install -c conda-forge snaphu
                                  or  sudo apt install snaphu

Quick start::

    from pygeovision.insar.slc import SLCInSARPipeline

    pipeline = SLCInSARPipeline(
        master_zip  = './S1C_IW_SLC__1SDV_20260601.zip',
        slave_zip   = './S1C_IW_SLC__1SDV_20260613.zip',
        output_dir  = './slc_insar/',
        subswath    = 'IW2',
        bursts      = (2, 4),
        polarisation= 'VV',
    )
    result = pipeline.run()
    print(result.displacement_m.min(), result.displacement_m.max())
"""

from pygeovision.insar.slc.pipeline      import SLCInSARPipeline, SLCInSARResult
from pygeovision.insar.slc.snap_graph    import SNAPGraph, check_snap, check_snapista
from pygeovision.insar.slc.snaphu        import SnaphuUnwrapper, unwrap_phase
from pygeovision.insar.slc.displacement  import slc_phase_to_displacement, los_to_vertical
from pygeovision.insar.slc.sentinel1     import S1SLCProduct, parse_s1_slc_manifest

__all__ = [
    "SLCInSARPipeline", "SLCInSARResult",
    "SNAPGraph", "check_snap", "check_snapista",
    "SnaphuUnwrapper", "unwrap_phase",
    "slc_phase_to_displacement", "los_to_vertical",
    "S1SLCProduct", "parse_s1_slc_manifest",
]
