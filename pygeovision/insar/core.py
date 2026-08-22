"""
pygeovision.insar.core
~~~~~~~~~~~~~~~~~~~~~~~
State-of-the-art Sentinel-1 TOPSAR InSAR processing pipeline.

Scientific foundation
---------------------
This implementation follows:

1. ESA Sentinel-1 Toolbox (SNAP) official processing chain
   - S-1 TOPS Coregistration with Enhanced Spectral Diversity (ESD)
   - Goldstein adaptive phase filter (alpha=0.5, 32x32 window)
   - SNAPHU statistical-cost network-flow phase unwrapping in DEFO mode
   - Copernicus DEM GLO-30 for topographic phase removal
   Reference: ESA TM-19 "InSAR Principles: Guidelines for SAR
   Interferometry Processing and Interpretation"
   https://earth.esa.int/documents/10174/1616536/InSAR_Principles.pdf

2. NASA Earthdata InSAR data recipes (2025)
   https://www.earthdata.nasa.gov/learn/data-recipes/create-interferogram-using-esas-sentinel-1-toolbox
   https://www.earthdata.nasa.gov/learn/data-recipes/phase-unwrap-interferogram

3. ESD coregistration requirements for TOPS mode
   - Required accuracy: 1/1000 pixel in azimuth (Scheiber & Moreira 2000)
   - Burst-overlap ESD for azimuth misregistration correction
   - Geometric coregistration as initial step to resolve ESD phase ambiguity
   Reference: Sentinel-1 TOPS coregistration (MDPI Remote Sensing, 2018)
   https://doi.org/10.3390/rs10091405

4. SNAPHU phase unwrapping (Chen & Zebker 2000)
   - DEFO mode for surface displacement (not TOPO)
   - Coherence-weighted cost functions
   - Coherence mask >= 0.3 (standard practice, multiple studies 2022-2024)
   Reference: https://step.esa.int/main/snap-supported-plugins/snaphu/
   SNAPHU man page: https://manpages.ubuntu.com/manpages/bionic/man1/snaphu.1.html

5. Phase-to-displacement conversion
   - d_LOS = -(lambda/4*pi) * phi_unwrapped
   - Sentinel-1 C-band: lambda = 0.05546576 m (5.405 GHz)
   - Vertical decomposition: d_vert = d_LOS / cos(theta_incidence)
   - Full 3D decomposition requires ascending + descending geometry
   Reference: Brouwer & Hanssen (2024) Journal of Geodesy 98(12)
   https://arxiv.org/pdf/2412.10282

6. Atmospheric correction
   - ERA5 reanalysis (ECMWF) for tropospheric delay correction
   - GACOS (Generic Atmospheric Correction Online Service) — highest quality
   - Linear elevation-phase correction as fallback
   Reference: Multiple studies 2023-2025 comparing GACOS, ERA5, WRF
   https://www.mdpi.com/2072-4292/15/4/990

7. DEM
   - Copernicus DEM GLO-30 (30 m) is the current operational standard
   - Replaces SRTM which has known voids and is no longer recommended
   Reference: https://dataspace.copernicus.eu/explore-data/data-collections/
   copernicus-contributing-missions/collections-description/COP-DEM

Key physical constants
----------------------
- C-band frequency: 5.405 GHz
- C-band wavelength: lambda = c/f = 2.998e8 / 5.405e9 = 0.05546576 m
- Half-wavelength: 2.773 cm = displacement per full 2pi phase cycle
  (factor of 2 because two-way travel path)
- Sentinel-1 IW incidence angle range: 29.1° (near range) to 46.0° (far range)
  - IW1: 29.1° – 36.9°
  - IW2: 34.9° – 41.7°
  - IW3: 38.5° – 46.0°
- 12-day repeat cycle (single satellite); 6-day with A+B constellation
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# ── Physical constants ────────────────────────────────────────────────────────
SENTINEL1_WAVELENGTH_M = 0.05546576    # C-band lambda (m), 5.405 GHz
SENTINEL1_FREQUENCY_HZ = 5.405e9      # C-band centre frequency
SPEED_OF_LIGHT         = 2.99792458e8  # m/s

# Sentinel-1 IW subswath incidence angle ranges (degrees)
# Source: Sentinel-1 Product Specification, ESA-EOPG-CSCOP-TN-0002
IW_INCIDENCE_ANGLES = {
    "IW1": (29.1, 36.9),
    "IW2": (34.9, 41.7),
    "IW3": (38.5, 46.0),
}

# Coherence threshold for phase unwrapping mask (Chen & Zebker 2000,
# confirmed by multiple studies 2022-2024)
COHERENCE_THRESHOLD_UNWRAP = 0.30

# Goldstein filter parameters (ESA recommended defaults)
GOLDSTEIN_ALPHA  = 0.5   # filter strength (0=no filter, 1=max)
GOLDSTEIN_WINDOW = 32    # window size in pixels
GOLDSTEIN_OVERLAP = 0.5  # fractional overlap between windows

# Minimum perpendicular baseline for valid interferogram (m)
MIN_PERPENDICULAR_BASELINE_M = 5.0
# Maximum perpendicular baseline before critical decorrelation (m)
# C-band Sentinel-1: B_crit ≈ 4800 m (rarely exceeded in practice: ~150 m typical)
MAX_PERPENDICULAR_BASELINE_M = 4800.0

# Temporal baseline thresholds (days)
# Beyond 72 days, vegetation decorrelation is severe in tropical regions
MAX_TEMPORAL_BASELINE_TROPICAL_DAYS = 72
# Urban areas maintain coherence longer
MAX_TEMPORAL_BASELINE_URBAN_DAYS    = 180


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class InSARPairInfo:
    """
    Metadata for a Sentinel-1 SLC pair selected for interferometric processing.

    The perpendicular baseline B_perp determines topographic sensitivity:
        delta_phi_topo = (4*pi/lambda) * (B_perp/R*sin(theta)) * delta_h
    For C-band with B_perp=100m, R=850km, theta=38°, delta_h=10m:
        delta_phi_topo ≈ 0.04 rad (negligible — topographic phase removal critical)
    """
    primary_path:   str
    secondary_path: str
    primary_date:   str   # YYYYMMDD
    secondary_date: str   # YYYYMMDD
    temporal_baseline_days: int
    perpendicular_baseline_m: float | None = None
    subswath:       str = "IW2"
    relative_orbit: int | None = None
    pass_direction: str = "DESCENDING"
    warnings:       list[str] = field(default_factory=list)

    @property
    def is_suitable(self) -> bool:
        """Return True if this pair is likely to produce useful coherence."""
        if self.perpendicular_baseline_m is not None:
            if self.perpendicular_baseline_m < MIN_PERPENDICULAR_BASELINE_M:
                return False
            if self.perpendicular_baseline_m > MAX_PERPENDICULAR_BASELINE_M:
                return False
        return True

    def coherence_expectation(self, land_cover: str = "mixed") -> str:
        """Qualitative coherence expectation based on temporal baseline and land cover."""
        tb = self.temporal_baseline_days
        if land_cover in ("vegetation", "tropical_forest"):
            if tb <= 12:   return "good (0.4-0.7)"
            if tb <= 24:   return "moderate (0.2-0.5)"
            return "poor (<0.3) — consider shorter baseline"
        elif land_cover in ("urban", "built"):
            if tb <= 36:   return "good (0.6-0.9)"
            if tb <= 72:   return "moderate (0.4-0.7)"
            return "moderate (0.3-0.6)"
        else:  # mixed
            if tb <= 12:   return "moderate-good"
            if tb <= 24:   return "moderate"
            return "poor-moderate"


@dataclass
class InSARProcessingResult:
    """Result of the full InSAR processing chain."""
    success:             bool
    # Intermediate products
    coregistered_stack:  str | None = None
    interferogram:       str | None = None
    coherence:           str | None = None
    filtered_phase:      str | None = None
    unwrapped_phase:     str | None = None
    # Final displacement products
    los_displacement_m:  str | None = None   # line-of-sight displacement (m)
    vertical_disp_m:     str | None = None   # vertical component (m), approx
    geocoded_product:    str | None = None   # geocoded to WGS84
    # Quality metrics
    mean_coherence:      float | None = None
    valid_pixel_fraction: float | None = None
    unwrapping_success:  bool = False
    # Processing metadata
    errors:              list[str] = field(default_factory=list)
    warnings:            list[str] = field(default_factory=list)
    processing_log:      list[str] = field(default_factory=list)
    metadata:            dict = field(default_factory=dict)


# ── Step 1: Pair selection ────────────────────────────────────────────────────

def select_insar_pair(
    scenes: list[dict],
    max_temporal_baseline_days: int = 24,
    max_perpendicular_baseline_m: float = 200.0,
    preferred_subswath: str = "IW2",
    land_cover: str = "mixed",
) -> InSARPairInfo | None:
    """
    Select the optimal primary/secondary SLC pair from a list of scenes.

    Selection criteria actually applied here, in order:
    1. Same relative orbit and pass direction (mandatory — different orbits
       have completely different viewing geometry and cannot form interferograms)
    2. Temporal baseline <= max_temporal_baseline_days (shorter = more coherence)

    HONEST LIMITATION — perpendicular baseline is NOT checked here, despite
    `max_perpendicular_baseline_m` being accepted: computing a real
    perpendicular baseline requires orbit state vectors (satellite
    position/velocity at acquisition time), which this function's
    lightweight scene dicts (path/date/relative_orbit/pass_direction) do
    not carry — that data only becomes available after downloading and
    coregistering the actual SLC products. `max_perpendicular_baseline_m`
    is accepted here for forward compatibility with callers who already
    have it precomputed, but is NOT applied as a filter in this function.
    The REAL, computed perpendicular baseline is checked post-hoc in
    `SLCInSARPipeline.run_native()` (via pygeofetch's
    InterferogramGenerator, which computes it for real from actual orbit
    data after coregistration) — see its `max_perpendicular_baseline_m`
    warning there for the genuine check.

    For tropical West Africa (Accra):
    - Vegetation decorrelates rapidly — prefer 6 or 12 day baseline
    - Urban areas (Accra CBD, Tema) maintain coherence to 72 days
    - Rainy season dramatically reduces coherence even at 6 days

    Args:
        scenes:                      List of scene dicts with 'path', 'date',
                                     'relative_orbit', 'pass_direction' keys.
        max_temporal_baseline_days:  Maximum temporal separation.
        max_perpendicular_baseline_m: NOT currently applied here — see the
                                     honest limitation note above.
        preferred_subswath:          'IW1', 'IW2', or 'IW3'.
        land_cover:                  'urban', 'vegetation', 'tropical_forest', 'mixed'.

    Returns:
        InSARPairInfo or None if no suitable pair found.
    """
    from datetime import datetime

    if len(scenes) < 2:
        logger.warning("select_insar_pair: need at least 2 scenes")
        return None

    if max_perpendicular_baseline_m != 200.0:
        logger.warning(
            "select_insar_pair: max_perpendicular_baseline_m=%.1f was "
            "requested but is NOT applied as a filter here (this function's "
            "scene dicts don't carry orbit state vector data) — the real, "
            "computed perpendicular baseline is only available and checked "
            "post-hoc in SLCInSARPipeline.run_native().",
            max_perpendicular_baseline_m,
        )

    # Sort by date
    def parse_date(s):
        d = s.get("date", s.get("datetime", ""))
        for fmt in ("%Y%m%d", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
            try:
                return datetime.strptime(str(d)[:10].replace("-", ""), "%Y%m%d")
            except ValueError:
                continue
        return datetime.min

    scenes_sorted = sorted(scenes, key=parse_date)

    best_pair = None
    best_score = float("inf")

    for i in range(len(scenes_sorted)):
        for j in range(i + 1, len(scenes_sorted)):
            s1, s2 = scenes_sorted[i], scenes_sorted[j]

            # Must share same orbit and pass direction
            o1 = s1.get("relative_orbit") or s1.get("extra", {}).get("relative_orbit")
            o2 = s2.get("relative_orbit") or s2.get("extra", {}).get("relative_orbit")
            p1 = s1.get("pass_direction") or s1.get("extra", {}).get("pass_direction", "")
            p2 = s2.get("pass_direction") or s2.get("extra", {}).get("pass_direction", "")

            if o1 and o2 and o1 != o2:
                continue
            if p1 and p2 and p1.upper() != p2.upper():
                continue

            # Temporal baseline
            d1 = parse_date(s1)
            d2 = parse_date(s2)
            tb = abs((d2 - d1).days)
            if tb > max_temporal_baseline_days:
                continue

            # Score: shorter baseline is better
            score = tb
            if score < best_score:
                best_score = score
                pair = InSARPairInfo(
                    primary_path   = s1.get("path", s1.get("download_path", "")),
                    secondary_path = s2.get("path", s2.get("download_path", "")),
                    primary_date   = d1.strftime("%Y%m%d"),
                    secondary_date = d2.strftime("%Y%m%d"),
                    temporal_baseline_days = tb,
                    subswath       = preferred_subswath,
                    relative_orbit = o1,
                    pass_direction = (p1 or "DESCENDING").upper(),
                )
                if tb > 36 and land_cover in ("vegetation", "tropical_forest"):
                    pair.warnings.append(
                        f"Temporal baseline {tb}d may cause decorrelation in tropical vegetation"
                    )
                best_pair = pair

    if best_pair is None:
        logger.warning(
            f"No pair found within {max_temporal_baseline_days}d baseline. "
            f"Try increasing max_temporal_baseline_days."
        )

    return best_pair


# ── Step 2: SNAP processing chain ────────────────────────────────────────────

def generate_snap_graph(
    primary_zip:    str,
    secondary_zip:  str,
    output_dir:     str,
    subswath:       str = "IW2",
    bursts:         list[int] | None = None,
    dem_path:       str | None = None,
    polarisation:   str = "VV",
    goldstein_alpha: float = GOLDSTEIN_ALPHA,
) -> str:
    """
    Generate a SNAP GPT XML graph for the full Sentinel-1 TOPSAR InSAR chain.

    Processing steps implemented (following ESA TM-19 and NASA Earthdata recipe):
    1.  TOPSAR-Split     — extract target subswath and burst range
    2.  Apply-Orbit-File — update precise (restituted) orbit state vectors
    3.  Back-Geocoding   — coregister secondary to primary using CopDEM
    4.  ESD              — Enhanced Spectral Diversity azimuth correction
                           (required for TOPS: achieves 1/1000 pixel accuracy)
    5.  Interferogram    — cross-multiply: primary x conj(secondary)
                           outputs: complex ifg, coherence, intensity
    6.  TOPSAR-Deburst   — stitch burst boundaries into continuous image
    7.  TopoPhaseRemoval — subtract flat-earth + topographic phase using DEM
    8.  GoldsteinPhaseFilter — adaptive filter to reduce phase noise
    9.  Write            — save filtered interferogram + coherence

    WHY ESD IS MANDATORY FOR TOPSAR:
    The TOPS azimuth beam sweep introduces a Doppler centroid that varies
    within each burst. A 1/100 pixel azimuth misregistration causes a
    phase error of ~0.3 rad at burst edges. ESD achieves <1/1000 pixel
    accuracy using the burst-overlap double-difference interferogram.
    Reference: Sentinel-1 TOPS coregistration study, MDPI 2018.

    Args:
        primary_zip:      Path to primary SLC .zip file.
        secondary_zip:    Path to secondary SLC .zip file.
        output_dir:       Directory for output products.
        subswath:         'IW1', 'IW2', or 'IW3'. IW2 covers Accra.
        bursts:           List of burst indices (1-indexed). None = all bursts.
        dem_path:         Path to DEM GeoTIFF. None = auto-download CopDEM GLO-30.
        polarisation:     'VV' (recommended for deformation) or 'VH'.
        goldstein_alpha:  Goldstein filter strength [0, 1]. Default 0.5 (ESA standard).

    Returns:
        Path to generated XML graph file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Burst selection string for SNAP
    if bursts:
        burst_str = ",".join(str(b) for b in bursts)
    else:
        burst_str = "1,2,3,4,5,6,7,8,9"  # all bursts (SNAP clips automatically)

    # DEM configuration
    if dem_path and Path(dem_path).exists():
        dem_config = f"""
            <parameter name="demName">External DEM</parameter>
            <parameter name="externalDEMFile">{dem_path}</parameter>
            <parameter name="externalDEMNoDataValue">-32768.0</parameter>
            <parameter name="externalDEMApplyEGM">true</parameter>"""
    else:
        # Copernicus DEM GLO-30 is the current operational standard
        # Replaces SRTM (legacy, with known voids)
        dem_config = """
            <parameter name="demName">Copernicus 30m Global DEM</parameter>"""

    # Output file paths
    ifg_output   = str(output_dir / "interferogram_goldstein.dim")
    str(output_dir / "coregistered_stack.dim")

    graph_xml = f"""<graph id="S1_TOPSAR_InSAR_Chain">
  <version>1.0</version>

  <!-- ═══════════════════════════════════════════════════════════════════════
       ESA Sentinel-1 TOPSAR InSAR Processing Graph
       Following: ESA TM-19 InSAR Principles + NASA Earthdata recipe (2025)

       KEY PHYSICS:
       - C-band wavelength: {SENTINEL1_WAVELENGTH_M:.8f} m
       - Phase sensitivity: 1 fringe = {SENTINEL1_WAVELENGTH_M/2*100:.2f} cm LOS displacement
       - ESD requirement: <1/1000 pixel azimuth coregistration
       - Coherence threshold for unwrapping: {COHERENCE_THRESHOLD_UNWRAP}

       STEP 1: TOPSAR-Split — extract subswath {subswath}, bursts [{burst_str}]
       STEP 2: Apply-Orbit — precise restituted orbit state vectors
       STEP 3: Back-Geocoding — DEM-assisted coregistration (geometric initial step)
       STEP 4: ESD — Enhanced Spectral Diversity (azimuth fine correction)
       STEP 5: Interferogram — cross-multiply primary x conj(secondary)
       STEP 6: Deburst — merge burst images
       STEP 7: TopoPhaseRemoval — remove flat-earth + topographic phase
       STEP 8: GoldsteinPhaseFilter — alpha={goldstein_alpha}, window=32px
  ═══════════════════════════════════════════════════════════════════════ -->

  <!-- Read primary SLC -->
  <node id="Read_Primary">
    <operator>Read</operator>
    <sources/>
    <parameters>
      <parameter name="file">{primary_zip}</parameter>
    </parameters>
  </node>

  <!-- Read secondary SLC -->
  <node id="Read_Secondary">
    <operator>Read</operator>
    <sources/>
    <parameters>
      <parameter name="file">{secondary_zip}</parameter>
    </parameters>
  </node>

  <!-- STEP 1a: Split primary — extract subswath and burst range -->
  <node id="Split_Primary">
    <operator>TOPSAR-Split</operator>
    <sources>
      <source refid="Read_Primary"/>
    </sources>
    <parameters>
      <parameter name="subswath">{subswath}</parameter>
      <parameter name="selectedPolarisations">{polarisation}</parameter>
      <parameter name="firstBurstIndex">{bursts[0] if bursts else 1}</parameter>
      <parameter name="lastBurstIndex">{bursts[-1] if bursts else 9}</parameter>
    </parameters>
  </node>

  <!-- STEP 1b: Split secondary — MUST match primary geometry exactly -->
  <node id="Split_Secondary">
    <operator>TOPSAR-Split</operator>
    <sources>
      <source refid="Read_Secondary"/>
    </sources>
    <parameters>
      <parameter name="subswath">{subswath}</parameter>
      <parameter name="selectedPolarisations">{polarisation}</parameter>
      <parameter name="firstBurstIndex">{bursts[0] if bursts else 1}</parameter>
      <parameter name="lastBurstIndex">{bursts[-1] if bursts else 9}</parameter>
    </parameters>
  </node>

  <!-- STEP 2a: Apply precise orbit file to primary
       Uses ESA restituted orbit files (accuracy: ~5 cm RMS vs precise orbits)
       POD (Precise Orbit Determination) preferred if available.
       Critical for accurate geometric coregistration. -->
  <node id="Orbit_Primary">
    <operator>Apply-Orbit-File</operator>
    <sources>
      <source refid="Split_Primary"/>
    </sources>
    <parameters>
      <parameter name="orbitType">Sentinel Precise (Auto Download)</parameter>
      <parameter name="polyDegree">3</parameter>
      <parameter name="continueOnFail">true</parameter>
    </parameters>
  </node>

  <!-- STEP 2b: Apply precise orbit file to secondary -->
  <node id="Orbit_Secondary">
    <operator>Apply-Orbit-File</operator>
    <sources>
      <source refid="Split_Secondary"/>
    </sources>
    <parameters>
      <parameter name="orbitType">Sentinel Precise (Auto Download)</parameter>
      <parameter name="polyDegree">3</parameter>
      <parameter name="continueOnFail">true</parameter>
    </parameters>
  </node>

  <!-- STEP 3 + STEP 4: Back-Geocoding + ESD coregistration
       Back-Geocoding: DEM-assisted geometric coregistration (initial step)
       ESD: Enhanced Spectral Diversity for 1/1000 pixel azimuth accuracy
       WHY BOTH: Geometric coregistration provides initial estimate needed
       to resolve 2pi phase ambiguity in ESD cross-interferogram.
       outputDerampDemodPhase=true: required output for subsequent ESD step.
       Reference: Scheiber & Moreira 2000, MDPI RS 2018. -->
  <node id="BackGeocoding_ESD">
    <operator>S1-Back-Geocoding</operator>
    <sources>
      <source refid="Orbit_Primary"/>
      <source refid="Orbit_Secondary"/>
    </sources>
    <parameters>{dem_config}
      <parameter name="demResamplingMethod">BILINEAR_INTERPOLATION</parameter>
      <parameter name="resamplingType">BILINEAR_INTERPOLATION</parameter>
      <parameter name="maskOutAreaWithoutElevation">false</parameter>
      <parameter name="outputRangeAzimuthOffset">false</parameter>
      <parameter name="outputDerampDemodPhase">true</parameter>
      <parameter name="disableRerampingInSelectedBursts">false</parameter>
    </parameters>
  </node>

  <!-- ESD: Fine azimuth coregistration using burst-overlap interferometry
       Corrects the residual azimuth offset remaining after geometric coregistration.
       The TOPS azimuth beam sweep causes phase discontinuities at burst boundaries
       if azimuth coregistration is not accurate to 1/1000 pixel. -->
  <node id="ESD">
    <operator>Enhanced-Spectral-Diversity</operator>
    <sources>
      <source refid="BackGeocoding_ESD"/>
    </sources>
    <parameters>
      <parameter name="fineWinWidthStr">512</parameter>
      <parameter name="fineWinHeightStr">512</parameter>
      <parameter name="fineWinAccAzimuth">16</parameter>
      <parameter name="fineWinAccRange">16</parameter>
      <parameter name="fineWinOversampling">128</parameter>
      <parameter name="xCorrThreshold">0.1</parameter>
      <parameter name="cohThreshold">0.15</parameter>
      <parameter name="numBlocksPerOverlap">10</parameter>
      <parameter name="useSuppliedShift">false</parameter>
      <parameter name="overallAzimuthShift">0.0</parameter>
    </parameters>
  </node>

  <!-- STEP 5: Form complex interferogram
       phi = angle(primary * conj(secondary))
       Outputs: interferogram phase, coherence, intensity (for visual check)
       subtract_flat_earth_phase=true: removes orbital fringe pattern
       cohWin: coherence estimation window (10x10 is SNAP default) -->
  <node id="Interferogram">
    <operator>Interferogram</operator>
    <sources>
      <source refid="ESD"/>
    </sources>
    <parameters>
      <parameter name="subtractFlatEarthPhase">true</parameter>
      <parameter name="srpPolynomialDegree">5</parameter>
      <parameter name="srpNumberPoints">501</parameter>
      <parameter name="orbitDegree">3</parameter>
      <parameter name="includeCoherence">true</parameter>
      <parameter name="cohWinAz">10</parameter>
      <parameter name="cohWinRg">10</parameter>
      <parameter name="squarePixel">true</parameter>
    </parameters>
  </node>

  <!-- STEP 6: Merge burst boundaries into a continuous image -->
  <node id="Deburst">
    <operator>TOPSAR-Deburst</operator>
    <sources>
      <source refid="Interferogram"/>
    </sources>
    <parameters>
      <parameter name="selectedPolarisations">{polarisation}</parameter>
    </parameters>
  </node>

  <!-- STEP 7: Remove topographic phase
       Uses DEM to compute and subtract the phase contribution from terrain.
       Without this step, terrain fringes dominate the interferogram and
       deformation signal is invisible.
       tilingEnabled=true: required for large scenes (Accra IW2 ~250 km swath) -->
  <node id="TopoPhaseRemoval">
    <operator>TopoPhaseRemoval</operator>
    <sources>
      <source refid="Deburst"/>
    </sources>
    <parameters>{dem_config}
      <parameter name="tileExtensionPercent">100</parameter>
      <parameter name="outputTopoBandPhase">false</parameter>
      <parameter name="outputElevationBand">true</parameter>
    </parameters>
  </node>

  <!-- STEP 8: Goldstein adaptive phase filter
       Reduces phase noise while preserving fringe signal.
       Algorithm: FFT-based multiplicative filter in overlapping windows.
       alpha={goldstein_alpha}: ESA recommended (0=no filter, 1=maximum smoothing)
       FFTSize=64: larger FFT = more averaging = smoother but less detailed
       windowSize=3: kernel size for averaging alpha exponent
       Reference: Goldstein & Werner 1998, Radio Science. -->
  <node id="GoldsteinFilter">
    <operator>GoldsteinPhaseFiltering</operator>
    <sources>
      <source refid="TopoPhaseRemoval"/>
    </sources>
    <parameters>
      <parameter name="alpha">{goldstein_alpha}</parameter>
      <parameter name="FFTSizeString">64</parameter>
      <parameter name="windowSizeString">3</parameter>
      <parameter name="useCoherenceMask">true</parameter>
      <parameter name="coherenceThreshold">{COHERENCE_THRESHOLD_UNWRAP}</parameter>
    </parameters>
  </node>

  <!-- Write final interferogram + coherence -->
  <node id="Write">
    <operator>Write</operator>
    <sources>
      <source refid="GoldsteinFilter"/>
    </sources>
    <parameters>
      <parameter name="file">{ifg_output}</parameter>
      <parameter name="formatName">BEAM-DIMAP</parameter>
    </parameters>
  </node>

</graph>"""

    graph_path = str(output_dir / "s1_topsar_insar_graph.xml")
    with open(graph_path, "w", encoding="utf-8") as f:
        f.write(graph_xml)

    logger.info(f"SNAP graph written: {graph_path}")
    return graph_path


def run_snap_graph(
    graph_path: str,
    snap_gpt:   str = "gpt",
    java_heap_gb: int = 16,
    n_threads:  int | None = None,
    timeout_h:  float = 12.0,
) -> bool:
    """
    Execute a SNAP GPT graph file.

    Args:
        graph_path:   Path to XML graph file.
        snap_gpt:     Path to SNAP gpt binary. Defaults to 'gpt' on PATH.
        java_heap_gb: Java heap memory in GB. 16 GB recommended for IW mode.
        n_threads:    CPU threads. None = auto (SNAP uses all available).
        timeout_h:    Maximum runtime in hours.

    Returns:
        True on success, False on failure.
    """
    cmd = [snap_gpt, graph_path, f"-J-Xmx{java_heap_gb}g"]
    if n_threads:
        cmd.extend(["-q", str(n_threads)])

    logger.info(f"Running SNAP graph: {Path(graph_path).name}")
    logger.info(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True, text=True,
            timeout=timeout_h * 3600,
        )
        if result.returncode == 0:
            logger.info("SNAP processing completed successfully")
            return True
        else:
            logger.error(f"SNAP failed (returncode={result.returncode}):\n{result.stderr[:2000]}")
            return False
    except subprocess.TimeoutExpired:
        logger.error(f"SNAP timed out after {timeout_h}h")
        return False
    except FileNotFoundError:
        logger.error(
            f"SNAP GPT binary not found: '{snap_gpt}'. "
            f"Install SNAP from https://step.esa.int/main/download/snap-download/"
        )
        return False


# ── Step 3: SNAPHU phase unwrapping ──────────────────────────────────────────

def prepare_snaphu_config(
    wrapped_phase_path: str,
    coherence_path:     str,
    output_dir:         str,
    width:              int,
    mode:               str = "DEFO",
    coherence_threshold: float = COHERENCE_THRESHOLD_UNWRAP,
    n_processors:       int = 4,
) -> tuple[str, str]:
    """
    Prepare SNAPHU configuration and return (config_path, cmd_string).

    SNAPHU mode selection (critical):
    - DEFO: for surface displacement interferograms (tectonic, subsidence, flood)
             Statistical model assumes deformation signal — correct for most EO applications
    - TOPO: for DEM generation — assumes signal represents topography
    - SMOOTH: generic smooth data — use only if neither DEFO nor TOPO applies

    For Accra flood/subsidence monitoring: always use DEFO.

    Coherence masking:
    Standard practice (confirmed by multiple 2022-2024 studies) is to apply
    a coherence mask >= 0.3 before unwrapping. Pixels below threshold are
    excluded from the network flow optimisation to prevent error propagation
    from incoherent areas (water bodies, vegetation in rainy season).

    Reference:
    - Chen & Zebker 2000 (original SNAPHU paper)
    - SNAPHU man page: https://manpages.ubuntu.com/manpages/bionic/man1/snaphu.1.html
    - ISCE2 documentation (NASA JPL)
    - ESA SNAP-SNAPHU integration tutorial

    Args:
        wrapped_phase_path: Path to wrapped phase GeoTIFF or binary.
        coherence_path:     Path to coherence GeoTIFF or binary (values 0-1).
        output_dir:         Directory for SNAPHU input/output files.
        width:              Number of range pixels (columns) in interferogram.
        mode:               'DEFO' | 'TOPO' | 'SMOOTH'. Use 'DEFO' for displacement.
        coherence_threshold: Mask pixels below this coherence. Default 0.3.
        n_processors:       Number of parallel CPU processors.

    Returns:
        (config_file_path, snaphu_command_string)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    unwrapped_path  = str(output_dir / "unwrapped_phase.snaphu")
    config_path     = str(output_dir / "snaphu.conf")

    config_content = f"""# SNAPHU Configuration
# Generated by pygeovision.insar.core
# Reference: Chen & Zebker 2000, SNAPHU v1.4.2
#
# MODE: {mode}
# DEFO = deformation mode (surface displacement)
# Required: flat-earth AND topographic phase must be removed before unwrapping
# The interferogram passed to SNAPHU should contain ONLY deformation + atmosphere + noise

STATCOSTMODE    {mode}

# Input/output format
INFILEFORMAT    FLOAT_DATA
OUTFILEFORMAT   FLOAT_DATA
CORRFILEFORMAT  FLOAT_DATA

# Image dimensions
LINELENGTH      {width}

# Parallel processing ({n_processors} processors)
NPROC           {n_processors}

# Coherence threshold: mask pixels below {coherence_threshold}
# Standard practice: >= 0.3 (confirmed by multiple studies 2022-2024)
# Lower values allow more pixels but propagate phase errors from incoherent regions
CORR_THRESHOLD  {coherence_threshold}

# Tile parameters (for large scenes)
# NTILEROW and NTILECOL: tile the image for memory efficiency
# Accra IW2 typical size: ~1500 x 20000 pixels
NTILEROW        4
NTILECOL        4
ROWOVRLP        256
COLOVRLP        256

# Connected components
# Outputs a file identifying regions successfully unwrapped
CONNCOMPFILE    {str(output_dir / "connected_components.snaphu")}

# Verbose output for quality assessment
VERBOSE         TRUE

# Physical parameters for DEFO mode (Sentinel-1 C-band)
# DEFOMAX_CYCLE: maximum expected deformation in half-wavelength units (cycles)
# For typical urban subsidence (<5 cm): DEFOMAX_CYCLE = 5
# For seismic events (up to metres): DEFOMAX_CYCLE = 100
DEFOMAX_CYCLE   5.0

# Amplitude scaling (important for correct cost function)
# AMPLITUDE_DATA should be TRUE if passing magnitude alongside phase
AMPLITUDE       FALSE
"""

    with open(config_path, "w") as f:
        f.write(config_content)

    # SNAPHU command
    cmd = (
        f"snaphu -f {config_path} "
        f"-c {coherence_path} "
        f"-o {unwrapped_path} "
        f"{wrapped_phase_path} "
        f"{width}"
    )

    logger.info(f"SNAPHU config written: {config_path}")
    logger.info(f"SNAPHU command: {cmd}")

    return config_path, cmd


def run_snaphu(
    wrapped_phase_path: str,
    coherence_path:     str,
    output_dir:         str,
    width:              int,
    mode:               str = "DEFO",
    coherence_threshold: float = COHERENCE_THRESHOLD_UNWRAP,
    n_processors:       int = 4,
    timeout_h:          float = 12.0,
) -> str | None:
    """
    Run SNAPHU phase unwrapping and return path to unwrapped phase file.

    Returns path to unwrapped phase or None on failure.
    """
    config_path, cmd = prepare_snaphu_config(
        wrapped_phase_path, coherence_path, output_dir,
        width, mode, coherence_threshold, n_processors,
    )
    unwrapped_path = str(Path(output_dir) / "unwrapped_phase.snaphu")

    logger.info(f"Running SNAPHU ({mode} mode, coherence mask >= {coherence_threshold})")
    try:
        result = subprocess.run(
            cmd.split(), capture_output=True, text=True,
            timeout=timeout_h * 3600,
        )
        if result.returncode == 0:
            logger.info("SNAPHU unwrapping completed")
            if Path(unwrapped_path).exists():
                return unwrapped_path
            logger.error("SNAPHU ran but output file not found")
            return None
        else:
            logger.error(f"SNAPHU failed:\n{result.stderr[:1000]}")
            return None
    except subprocess.TimeoutExpired:
        logger.error(f"SNAPHU timed out after {timeout_h}h")
        return None
    except FileNotFoundError:
        logger.error(
            "SNAPHU binary not found. Install via: "
            "apt install snaphu  OR  conda install -c conda-forge snaphu"
        )
        return None


# ── Step 4: Phase to displacement conversion ─────────────────────────────────

def phase_to_los_displacement(
    unwrapped_phase_rad: np.ndarray,
    wavelength_m: float = SENTINEL1_WAVELENGTH_M,
) -> np.ndarray:
    """
    Convert unwrapped interferometric phase to line-of-sight displacement.

    Physical derivation:
    The two-way travel path difference between primary and secondary acquisitions:
        delta_range = r2 - r1 = (lambda / 4*pi) * phi
    where phi is the unwrapped interferometric phase in radians.
    The NEGATIVE sign: positive phase = satellite moved closer to target
    (range decreased) = target moved TOWARD satellite = negative LOS displacement
    in the standard convention (positive LOS = surface moving away from satellite).

    This is the standard convention used by:
    - SNAP / ESA processing chain
    - NASA ISCE2 / HyP3 / ASF products
    - InSAR literature (Hanssen 2001, Bürgmann et al. 2000)

    Formula:
        d_LOS = -(lambda / 4*pi) * phi_unwrapped

    One full fringe (2*pi phase) corresponds to:
        d_LOS = lambda/2 = 2.773 cm (Sentinel-1 C-band)

    Args:
        unwrapped_phase_rad: Unwrapped phase array (radians).
        wavelength_m:        Radar wavelength in metres.

    Returns:
        LOS displacement in metres (positive = away from satellite).
    """
    return -(wavelength_m / (4.0 * np.pi)) * unwrapped_phase_rad


def los_to_vertical_displacement(
    d_los: np.ndarray,
    incidence_angle_deg: float,
    heading_deg: float | None = None,
) -> dict[str, np.ndarray]:
    """
    Decompose LOS displacement into vertical (and optionally horizontal) components.

    IMPORTANT CAVEAT:
    Decomposing a single LOS measurement into full 3D displacement is
    underdetermined. This function implements the standard assumption that
    horizontal motion is negligible (valid for pure vertical subsidence or
    uplift). For tectonic events with significant horizontal motion, use
    ascending + descending geometry simultaneously.

    Reference: Brouwer & Hanssen 2024 (Journal of Geodesy 98:110)
    https://arxiv.org/pdf/2412.10282

    Pure vertical assumption (d_E = d_N = 0):
        d_LOS = d_U * cos(theta)
        => d_U = d_LOS / cos(theta)

    Where theta is the local incidence angle (angle from vertical/zenith).

    Full 3D LOS equation (right-looking geometry):
        d_LOS = -sin(theta)*cos(az)*d_E + sin(theta)*sin(az)*d_N + cos(theta)*d_U
    where az is the satellite heading angle (typically ~-13° for S-1 descending,
    ~193° for ascending in West Africa).

    Incidence angles for Sentinel-1 IW subswaths:
        IW1: 29.1° – 36.9°   IW2: 34.9° – 41.7°   IW3: 38.5° – 46.0°
    Use the mid-swath angle for a scene-wide approximation.
    Per-pixel incidence angle bands from SNAP terrain correction are more accurate.

    Args:
        d_los:               LOS displacement array (metres).
        incidence_angle_deg: Local incidence angle in degrees. Use mid-swath value
                             (e.g. 38.3° for IW2 over Accra) if no per-pixel map.
        heading_deg:         Satellite heading in degrees. Used only for east
                             component estimation. None = skip east component.

    Returns:
        Dict with keys:
        - 'vertical'  (np.ndarray): Vertical displacement (m). Positive = uplift.
        - 'east'      (np.ndarray, optional): East displacement (m). Positive = eastward.
        - 'assumption' (str): Documentation of assumptions made.
    """
    theta_rad = np.deg2rad(incidence_angle_deg)
    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)

    # Vertical component (assuming no horizontal motion)
    d_vertical = d_los / cos_theta

    result = {
        "vertical":   d_vertical,
        "assumption": (
            f"Pure vertical assumption: d_E=d_N=0. "
            f"Valid for vertical subsidence/uplift only. "
            f"Incidence angle: {incidence_angle_deg:.1f}°. "
            f"For tectonic events, combine with ascending orbit geometry."
        ),
    }

    # East component (approximate, assuming d_N=0)
    if heading_deg is not None:
        # Standard right-looking geometry
        # d_LOS ≈ -sin(theta)*cos(heading)*d_E + cos(theta)*d_U
        # If we know d_U from above, solve for d_E:
        heading_rad = np.deg2rad(heading_deg)
        los_due_to_vertical = d_vertical * cos_theta
        residual_los = d_los - los_due_to_vertical
        d_east = -residual_los / (sin_theta * np.cos(heading_rad))
        result["east"] = d_east

    return result


def apply_linear_aps_correction(
    d_los: np.ndarray,
    coherence_arr: np.ndarray,
    width: int,
    dem_path: str,
    coherence_threshold: float,
) -> tuple[np.ndarray, dict]:
    """Fit and remove the elevation-correlated (tropospheric) component
    of LOS displacement — the standard "linear APS" correction.

    Tropospheric delay correlates with elevation (moist troposphere is
    denser at low elevation); this fits d_LOS = a*elevation + b via
    least-squares regression over coherent pixels and removes only the
    elevation-dependent slope term — the constant term isn't atmospheric
    noise, it's whatever reference-point convention the caller already
    uses.

    Args:
        d_los: Flat LOS displacement array (metres), from
            phase_to_los_displacement().
        coherence_arr: Flat coherence array, same length as d_los.
        width: Image width (columns) for reshaping the flat arrays —
            get this from a verified source (e.g. a real ENVI header's
            "samples" field), never a guessed/default value; a wrong
            width silently corrupts the reshape.
        dem_path: DEM GeoTIFF covering the same real-world extent as the
            interferogram (resampled to match its grid if the DEM's
            native resolution differs — a real, if approximate, use of
            the same DEM already used for coregistration upstream).
        coherence_threshold: Minimum coherence for a pixel to be used in
            the regression.

    Returns:
        (corrected_d_los, log) where log = {"applied": bool, "message": str}.
        If correction wasn't possible or failed, corrected_d_los is the
        original, unmodified array and log["applied"] is False.
    """
    try:
        height = len(d_los) // width
        d_los_2d = d_los[: height * width].reshape(height, width)
        coh_2d = coherence_arr[: height * width].reshape(height, width)

        import rasterio as _rasterio
        with _rasterio.open(dem_path) as dem_src:
            dem_arr = dem_src.read(1).astype(np.float32)
        if dem_arr.shape != d_los_2d.shape:
            from scipy.ndimage import zoom
            zoom_factors = (d_los_2d.shape[0] / dem_arr.shape[0],
                            d_los_2d.shape[1] / dem_arr.shape[1])
            dem_arr = zoom(dem_arr, zoom_factors, order=1)

        valid = (coh_2d >= coherence_threshold) & np.isfinite(dem_arr) & np.isfinite(d_los_2d)
        if valid.sum() <= 100:  # need enough points for a meaningful fit
            return d_los, {
                "applied": False,
                "message": (
                    "Linear APS correction skipped — too few coherent "
                    f"pixels ({int(valid.sum())}) for a reliable elevation regression."
                ),
            }

        elev_valid = dem_arr[valid]
        phase_valid = d_los_2d[valid]
        A = np.vstack([elev_valid, np.ones_like(elev_valid)]).T
        slope, _intercept = np.linalg.lstsq(A, phase_valid, rcond=None)[0]
        aps_trend = slope * dem_arr  # elevation-correlated component only
        corrected = (d_los_2d - aps_trend).flatten()
        return corrected, {
            "applied": True,
            "message": (
                f"Linear APS correction applied (slope={slope*1000:.3f} "
                f"mm/m elevation, {int(valid.sum())} valid pixels)"
            ),
        }
    except Exception as exc:
        return d_los, {
            "applied": False,
            "message": f"Linear APS correction failed, using uncorrected phase: {exc}",
        }


def compute_sensitivity_to_noise(
    incidence_angle_deg: float,
    wavelength_m: float = SENTINEL1_WAVELENGTH_M,
    coherence: float = 0.5,
    n_looks: int = 4,
) -> dict[str, float]:
    """
    Compute InSAR phase noise and resulting displacement uncertainty.

    Phase standard deviation (Cramer-Rao lower bound for InSAR):
        sigma_phi = sqrt((1 - coherence^2) / (2 * n_looks * coherence^2))

    Displacement uncertainty:
        sigma_d_LOS    = (lambda / 4*pi) * sigma_phi
        sigma_d_vert   = sigma_d_LOS / cos(theta)

    Reference: Hanssen 2001 "Radar Interferometry" Chapter 4.

    Args:
        incidence_angle_deg: Local incidence angle (degrees).
        wavelength_m:        Radar wavelength (m). Default Sentinel-1 C-band.
        coherence:           Mean scene coherence (0-1).
        n_looks:             Number of looks in interferogram.

    Returns:
        Dict with sigma values in metres.
    """
    # Phase noise (radians)
    gamma = coherence
    sigma_phi = np.sqrt((1 - gamma**2) / (2 * n_looks * gamma**2 + 1e-10))

    # LOS displacement noise
    sigma_los = (wavelength_m / (4.0 * np.pi)) * sigma_phi

    # Vertical displacement noise
    theta = np.deg2rad(incidence_angle_deg)
    sigma_vert = sigma_los / np.cos(theta)

    return {
        "sigma_phi_rad":    float(sigma_phi),
        "sigma_phi_mm":     float(sigma_phi * wavelength_m / (4 * np.pi) * 1000),
        "sigma_los_mm":     float(sigma_los * 1000),
        "sigma_vertical_mm": float(sigma_vert * 1000),
        "coherence":        coherence,
        "n_looks":          n_looks,
        "note": (
            f"At coherence={coherence:.2f}, {n_looks} looks: "
            f"LOS uncertainty ~{sigma_los*1000:.1f} mm, "
            f"vertical uncertainty ~{sigma_vert*1000:.1f} mm"
        ),
    }


# ── Step 5: Atmospheric correction ───────────────────────────────────────────

def estimate_linear_aps(
    unwrapped_phase: np.ndarray,
    elevation:       np.ndarray,
    coherence:       np.ndarray | None = None,
    coherence_min:   float = 0.5,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Estimate and remove linear elevation-phase tropospheric delay (APS).

    The stratified tropospheric delay produces a phase signal proportional
    to elevation: phi_atm ≈ k * height. This linear regression approach
    is the simplest APS correction and is effective when:
    - The scene has significant topographic relief (>300 m)
    - The deformation signal is NOT correlated with elevation
    - More sophisticated corrections (GACOS, ERA5) are unavailable

    For Accra (near-flat, <80 m elevation range) this correction has
    minimal effect but is included for completeness.

    Reference: Multiple studies 2023-2025 comparing GACOS, ERA5, linear methods.
    https://www.mdpi.com/2072-4292/15/4/990

    Args:
        unwrapped_phase: Unwrapped phase array (radians).
        elevation:       DEM elevation array, same shape (metres).
        coherence:       Coherence array. If provided, used to weight regression.
        coherence_min:   Minimum coherence for including pixels in regression.

    Returns:
        (corrected_phase, stats_dict) where corrected_phase has the linear
        elevation-phase correlation removed.
    """

    valid = np.isfinite(unwrapped_phase) & np.isfinite(elevation)
    if coherence is not None:
        valid = valid & (coherence >= coherence_min)

    if valid.sum() < 100:
        logger.warning("APS linear correction: insufficient valid pixels — skipping")
        return unwrapped_phase, {"slope_rad_per_m": 0.0, "r2": 0.0, "skipped": True}

    # Weighted linear regression: phase ~ k * elevation + c
    ph_v  = unwrapped_phase[valid].ravel()
    el_v  = elevation[valid].ravel()
    w_v   = coherence[valid].ravel() if coherence is not None else np.ones(ph_v.shape)

    # Weighted least squares
    W = np.diag(w_v)
    A = np.column_stack([el_v, np.ones_like(el_v)])
    try:
        coeffs = np.linalg.lstsq(W @ A, W @ ph_v, rcond=None)[0]
        slope, intercept = coeffs
    except np.linalg.LinAlgError:
        return unwrapped_phase, {"slope_rad_per_m": 0.0, "r2": 0.0, "skipped": True}

    # R² for quality reporting
    ph_pred = slope * el_v + intercept
    ss_res  = np.sum((ph_v - ph_pred)**2)
    ss_tot  = np.sum((ph_v - ph_v.mean())**2)
    r2      = 1 - ss_res / (ss_tot + 1e-10)

    # Remove the linear APS from the phase field
    aps_model    = slope * elevation + intercept
    corrected    = unwrapped_phase - aps_model

    slope_mm_per_km = slope * (SENTINEL1_WAVELENGTH_M / (4 * np.pi)) * 1000 * 1000
    logger.info(
        f"Linear APS correction: slope={slope_mm_per_km:.2f} mm/km elevation, "
        f"R²={r2:.3f}"
    )

    return corrected, {
        "slope_rad_per_m": float(slope),
        "slope_mm_per_km_elevation": float(slope_mm_per_km),
        "intercept_rad":   float(intercept),
        "r2":              float(r2),
        "n_pixels_used":   int(valid.sum()),
        "note": (
            "Linear APS only. For better results use GACOS "
            "(https://gacos.net) or ERA5 correction."
        ),
    }


# ── Step 6: Coherence analysis ────────────────────────────────────────────────

def analyse_coherence(
    coherence: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Compute coherence statistics for quality assessment and reporting.

    Coherence interpretation for Sentinel-1 C-band:
    - > 0.7:  High coherence. Urban areas, bare rock, arid zones.
              Deformation measurements reliable to ~1 mm.
    - 0.4-0.7: Moderate coherence. Suburban areas, dry agricultural land.
               Measurements reliable, some noise.
    - 0.3-0.4: Low coherence. Sparse vegetation, wet soil.
               Measurements possible but noisy. Phase unwrapping challenging.
    - < 0.3:  Very low coherence. Dense tropical vegetation, water bodies,
               fresh snow. Phase measurements unreliable. Mask before unwrapping.

    Reference: Hanssen 2001 Chapter 4; ESA TM-19.

    Args:
        coherence:  Coherence array (values 0-1).
        valid_mask: Optional mask of pixels to include (True = include).

    Returns:
        Dict of coherence statistics.
    """
    coh = coherence[valid_mask] if valid_mask is not None else coherence
    coh = coh[np.isfinite(coh)]

    if len(coh) == 0:
        return {"error": "No valid coherence pixels"}

    return {
        "mean":          float(np.mean(coh)),
        "median":        float(np.median(coh)),
        "std":           float(np.std(coh)),
        "pct_high":      float((coh > 0.7).mean() * 100),   # > 0.7
        "pct_moderate":  float(((coh >= 0.4) & (coh <= 0.7)).mean() * 100),
        "pct_low":       float(((coh >= 0.3) & (coh < 0.4)).mean() * 100),
        "pct_incoherent": float((coh < 0.3).mean() * 100),  # < 0.3
        "pct_unwrappable": float((coh >= COHERENCE_THRESHOLD_UNWRAP).mean() * 100),
        "n_pixels":      int(len(coh)),
        "interpretation": (
            "Good" if np.mean(coh) > 0.5 else
            "Moderate" if np.mean(coh) > 0.35 else
            "Poor — consider shorter temporal baseline or dry season acquisition"
        ),
    }


# ── Convenience wrapper ───────────────────────────────────────────────────────

class SLCInSARPipeline:
    """
    High-level Sentinel-1 TOPSAR InSAR pipeline orchestrator.

    Implements the complete chain from SLC pair to geocoded displacement map,
    following ESA/NASA state-of-the-art methods (2024).

    Usage:
        pipeline = SLCInSARPipeline(
            output_dir   = './results/insar_accra/',
            subswath     = 'IW2',
            bursts       = [3, 4, 5],
            snap_gpt     = '/usr/local/snap/bin/gpt',
        )
        result = pipeline.run(
            primary_zip   = './data/slc/S1A_IW_SLC_20240115.zip',
            secondary_zip = './data/slc/S1A_IW_SLC_20240208.zip',
            dem_path      = './data/dem/copernicus_dem_30m_accra.tif',
        )
        print(f'Vertical displacement map: {result.vertical_disp_m}')
    """

    def __init__(
        self,
        output_dir:   str,
        subswath:     str   = "IW2",
        bursts:       list[int] | None = None,
        polarisation: str   = "VV",
        snap_gpt:     str   = "gpt",
        java_heap_gb: int   = 16,
        n_processors: int   = 4,
        coherence_threshold: float = COHERENCE_THRESHOLD_UNWRAP,
        goldstein_alpha:     float = GOLDSTEIN_ALPHA,
    ):
        self.output_dir          = Path(output_dir)
        self.subswath            = subswath
        self.bursts              = bursts
        self.polarisation        = polarisation
        self.snap_gpt            = snap_gpt
        self.java_heap_gb        = java_heap_gb
        self.n_processors        = n_processors
        self.coherence_threshold = coherence_threshold
        self.goldstein_alpha     = goldstein_alpha
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_native(
        self,
        primary_zip: str,
        secondary_zip: str,
        bbox: tuple[float, float, float, float],
        dem_path: str | None = None,
        apply_atmospheric_correction: bool = True,
        goldstein_filter: bool = True,
        max_perpendicular_baseline_m: float = 200.0,
    ) -> InSARProcessingResult:
        """Run the InSAR chain using pygeofetch's real, pure-Python InSAR
        suite — no ESA SNAP required at all (unlike `run()`, which shells
        out to SNAP's `gpt` binary for coregistration/interferogram
        formation, and only takes over for the final SNAPHU unwrapping
        step).

        This is an ADDITIVE alternative to `run()`, not a replacement —
        `run()` keeps working exactly as before for anyone with an
        existing SNAP-based workflow. Every pygeofetch method call below
        was verified against the actual installed package's real
        signatures before being written (not assumed from documentation).

        Honest scope: verified structurally (imports, signatures, object
        construction) in this environment, but NOT yet run end-to-end
        against a real Sentinel-1 SLC pair — this sandbox has neither a
        real SLC test fixture nor the `snaphu` binary installed. Treat
        this as a real, carefully-built implementation that still needs
        validation against a known InSAR benchmark scene (e.g. the
        ERS-2 Etna or a real Sentinel-1 ascending-pair test case) before
        it becomes anyone's default path, the same way pygeofetch's own
        L-band module is honestly flagged as "real and tested against a
        synthetic fixture, not yet validated against a real product."

        Args:
            bbox: (min_lon, min_lat, max_lon, max_lat), WGS84 — the AOI
                used to extract only the relevant burst(s)/subswath from
                each full SLC product. Replaces `subswath`/`bursts` (the
                SNAP path's region-selection mechanism) with a plain
                geographic box, since pygeofetch's extractor determines
                the right burst/subswath automatically from it.
        """
        result = InSARProcessingResult(success=False)

        try:
            from pygeofetch.insar.extraction import SLCExtractor
            from pygeofetch.insar.interferogram import InterferogramGenerator
            from pygeofetch.insar.unwrap import PhaseUnwrapper
            from pygeofetch.models.search_query import BoundingBox
        except ImportError as exc:
            result.errors.append(f"pygeofetch InSAR suite unavailable: {exc}")
            return result

        aoi = BoundingBox(min_lon=bbox[0], min_lat=bbox[1], max_lon=bbox[2], max_lat=bbox[3])

        # Stage 1: extract only the relevant burst(s) from each full SLC
        # product — replaces SNAP's TOPSAR-Split + Apply-Orbit-File steps.
        result.processing_log.append("Extracting SLC pair (pygeofetch, SNAP-free)...")
        extractor = SLCExtractor(polarisation=self.polarisation)
        extract_dir = self.output_dir / "native_extract"
        extract_dir.mkdir(parents=True, exist_ok=True)
        try:
            ref_path, sec_path = extractor.extract_pair(
                reference=primary_zip, secondary=secondary_zip,
                aoi=aoi, output_dir=str(extract_dir),
            )
        except Exception as exc:
            result.errors.append(f"SLC extraction failed: {exc}")
            return result
        if ref_path is None or sec_path is None:
            result.errors.append(
                "SLC extraction returned no overlapping burst for this AOI — "
                "check that the AOI genuinely falls within both scenes."
            )
            return result
        result.processing_log.append(f"Extracted: {ref_path.name}, {sec_path.name}")

        # Stage 2: coregistration + interferogram formation + ESD +
        # Goldstein filtering — replaces SNAP's Back-Geocoding, ESD,
        # Interferogram, and Goldstein-Filter steps in one real call.
        result.processing_log.append("Forming interferogram (coregister + ESD + filter)...")
        ifg_gen = InterferogramGenerator(
            esd_enabled=True,
            use_real_burst_processing=True,
            remove_flat_earth_phase=True,
        )
        try:
            ifg_result = ifg_gen.process_pair(
                reference=ref_path, secondary=sec_path, dem=dem_path,
                apply_goldstein_filter=goldstein_filter,
                goldstein_alpha=self.goldstein_alpha,
            )
        except Exception as exc:
            result.errors.append(f"Interferogram formation failed: {exc}")
            return result
        result.processing_log.append(
            f"Interferogram formed (perp. baseline={ifg_result.perpendicular_baseline_m}, "
            f"temporal baseline={ifg_result.temporal_baseline_days}d)"
        )

        # The REAL perpendicular baseline check — this is the actual
        # point in the pipeline where a genuine, computed value exists
        # (from real orbit data, via InterferogramGenerator). Unlike
        # select_insar_pair()'s parameter of the same name, this one is
        # genuinely applied, not just accepted and ignored.
        if (ifg_result.perpendicular_baseline_m is not None
                and abs(ifg_result.perpendicular_baseline_m) > max_perpendicular_baseline_m):
            result.warnings.append(
                f"Perpendicular baseline {ifg_result.perpendicular_baseline_m:.1f}m "
                f"exceeds max_perpendicular_baseline_m={max_perpendicular_baseline_m}m "
                f"— expect reduced topographic/coherence quality for this pair."
            )

        # Stage 3: phase unwrapping via SNAPHU (auto-installed by
        # pygeofetch on first use — see the integration doc's section 8.5
        # — genuinely no separate SNAP or manual SNAPHU install needed).
        result.processing_log.append("Unwrapping phase (SNAPHU, DEFO mode)...")
        unwrapper = PhaseUnwrapper(cost_mode="defo", init_method="mcf")
        try:
            unwrap_result = unwrapper.unwrap_pair(
                interferogram=ifg_result.interferogram,
                coherence=ifg_result.coherence,
                profile=ifg_result.profile,
                reference_date=ifg_result.reference_date,
                secondary_date=ifg_result.secondary_date,
            )
        except Exception as exc:
            result.warnings.append(f"Unwrapping failed — LOS displacement not computed: {exc}")
            result.success = True  # interferogram was still produced successfully
            return result

        result.unwrapping_success = True
        result.processing_log.append("Unwrapping complete")

        def _write_array_geotiff(arr, profile: dict, out_path: Path) -> str:
            """Write an array to a real GeoTIFF, handling complex-valued
            interferogram data correctly (real+imaginary as two bands)
            rather than assuming every array here is real-valued."""
            import numpy as _np
            import rasterio
            out_path.parent.mkdir(parents=True, exist_ok=True)
            prof = dict(profile)
            if _np.iscomplexobj(arr):
                prof.update(count=2, dtype="float32")
                with rasterio.open(out_path, "w", **prof) as dst:
                    dst.write(_np.real(arr).astype("float32"), 1)
                    dst.write(_np.imag(arr).astype("float32"), 2)
                    dst.set_band_description(1, "real")
                    dst.set_band_description(2, "imaginary")
            else:
                prof.update(count=1, dtype="float32")
                with rasterio.open(out_path, "w", **prof) as dst:
                    dst.write(_np.asarray(arr).astype("float32"), 1)
            return str(out_path)

        native_dir = self.output_dir / "native_products"
        result.interferogram = _write_array_geotiff(
            ifg_result.interferogram, ifg_result.profile, native_dir / "interferogram.tif",
        )
        result.coherence = _write_array_geotiff(
            ifg_result.coherence, ifg_result.profile, native_dir / "coherence.tif",
        )

        # Stage 4 (optional): tropospheric phase correction.
        corrected_phase = None
        if apply_atmospheric_correction:
            try:
                from pygeofetch.insar.atmosphere import AtmosphericCorrector
                corrector = AtmosphericCorrector(method="elevation")
                corrected_phase = corrector.correct(
                    phase=unwrap_result.unwrapped_phase,
                    dem=dem_path,
                    reference_datetime=ifg_result.reference_date,
                    secondary_datetime=ifg_result.secondary_date,
                    profile=ifg_result.profile,
                    unwrapped=True,
                )
                result.processing_log.append("Atmospheric (tropospheric) correction applied")
                result.metadata["atmospheric_correction"] = "elevation"
            except Exception as exc:
                result.warnings.append(f"Atmospheric correction failed, using uncorrected phase: {exc}")

        # If atmospheric correction ran successfully, the corrected phase
        # is what should actually be written out — using the raw
        # uncorrected phase regardless would silently discard the
        # correction just computed above.
        final_phase = corrected_phase if corrected_phase is not None else unwrap_result.unwrapped_phase
        result.unwrapped_phase = _write_array_geotiff(
            final_phase, ifg_result.profile, native_dir / "unwrapped_phase.tif",
        )

        result.success = True
        result.metadata["backend"] = "pygeofetch-native"
        return result

    def run(
        self,
        primary_zip:   str,
        secondary_zip: str,
        dem_path:      str | None = None,
        apply_linear_aps: bool = True,
    ) -> InSARProcessingResult:
        """
        Run the complete InSAR chain: SLC pair → geocoded displacement.

        Returns InSARProcessingResult with paths to all output products.
        """
        result = InSARProcessingResult(success=False)

        # STEP 1-8: SNAP processing chain (coregistration → interferogram → filter)
        result.processing_log.append("Generating SNAP processing graph...")
        graph_path = generate_snap_graph(
            primary_zip, secondary_zip,
            output_dir      = str(self.output_dir / "snap_work"),
            subswath        = self.subswath,
            bursts          = self.bursts,
            dem_path        = dem_path,
            polarisation    = self.polarisation,
            goldstein_alpha = self.goldstein_alpha,
        )

        result.processing_log.append("Running SNAP InSAR chain (Steps 1-8)...")
        snap_ok = run_snap_graph(
            graph_path,
            snap_gpt       = self.snap_gpt,
            java_heap_gb   = self.java_heap_gb,
            n_threads      = self.n_processors,
        )
        if not snap_ok:
            result.errors.append("SNAP processing failed — check snap_gpt path and memory")
            return result

        # Locate SNAP outputs
        snap_work = self.output_dir / "snap_work"
        ifg_product = snap_work / "interferogram_goldstein.dim"
        if not ifg_product.exists():
            result.errors.append(f"SNAP output not found: {ifg_product}")
            return result

        result.interferogram = str(ifg_product)
        result.processing_log.append(f"SNAP complete: {ifg_product}")

        # STEP 9: SNAPHU export + unwrapping
        # Export wrapped phase and coherence from SNAP DIMAP format to binary
        result.processing_log.append("Exporting for SNAPHU phase unwrapping...")
        snaphu_dir = self.output_dir / "snaphu_work"
        snaphu_dir.mkdir(exist_ok=True)

        # Note: In production, use SNAP's SnaphuExport operator to convert DIMAP
        # to the flat binary format SNAPHU expects. Here we call it via GPT.
        export_graph = self._generate_snaphu_export_graph(str(ifg_product), str(snaphu_dir))
        run_snap_graph(export_graph, snap_gpt=self.snap_gpt, java_heap_gb=8)

        # Find exported files
        wrapped_files  = list(snaphu_dir.glob("*Phase*.img")) + list(snaphu_dir.glob("*phase*.img"))
        coherence_files = list(snaphu_dir.glob("*coh*.img")) + list(snaphu_dir.glob("*Coh*.img"))

        if not wrapped_files or not coherence_files:
            result.warnings.append(
                "SNAPHU export files not found — skipping phase unwrapping. "
                "Check SNAP SnaphuExport output format."
            )
            result.success = True  # interferogram was produced successfully
            return result

        wrapped_path   = str(wrapped_files[0])
        coherence_path = str(coherence_files[0])

        # Get image width from SNAP header — required for correctly
        # reshaping the raw binary SAR data; there is no safe default
        # to guess if it's missing (see _read_envi_width's docstring).
        hdr_files = list(snaphu_dir.glob("*.hdr"))
        if not hdr_files:
            result.errors.append(
                f"No .hdr file found in {snaphu_dir} — cannot determine "
                f"image width for SNAPHU unwrapping. Check SNAP "
                f"SnaphuExport output."
            )
            result.success = False
            return result
        try:
            width = self._read_envi_width(str(hdr_files[0]))
        except RuntimeError as exc:
            result.errors.append(str(exc))
            result.success = False
            return result

        result.processing_log.append(
            f"Running SNAPHU (DEFO mode, coherence mask >= {self.coherence_threshold})..."
        )
        unwrapped_path = run_snaphu(
            wrapped_path, coherence_path,
            output_dir          = str(snaphu_dir),
            width               = width,
            mode                = "DEFO",
            coherence_threshold = self.coherence_threshold,
            n_processors        = self.n_processors,
        )

        if unwrapped_path:
            result.unwrapped_phase  = unwrapped_path
            result.unwrapping_success = True
            result.processing_log.append("SNAPHU unwrapping complete")
        else:
            result.warnings.append("SNAPHU unwrapping failed — LOS displacement not computed")
            result.success = True
            return result

        # STEP 10: Phase to LOS displacement
        result.processing_log.append("Converting phase to LOS displacement...")
        try:
            unwrapped_arr = np.fromfile(unwrapped_path, dtype="float32")
            coherence_arr = np.fromfile(coherence_path, dtype="float32")

            # Get incidence angle for this subswath (mid-swath value)
            inc_min, inc_max = IW_INCIDENCE_ANGLES.get(self.subswath, (34.9, 41.7))
            incidence_angle_deg = (inc_min + inc_max) / 2

            # Convert phase to LOS displacement
            d_los = phase_to_los_displacement(unwrapped_arr)

            # Linear atmospheric phase screen (APS) correction —
            # previously accepted (apply_linear_aps, DEFAULTING TO TRUE)
            # and never referenced anywhere in this method, meaning every
            # normal call silently skipped tropospheric correction while
            # appearing to have it enabled by default.
            if apply_linear_aps and dem_path:
                d_los, aps_log = apply_linear_aps_correction(
                    d_los, coherence_arr, width, dem_path, self.coherence_threshold,
                )
                if aps_log["applied"]:
                    result.processing_log.append(aps_log["message"])
                else:
                    result.warnings.append(aps_log["message"])

            # LOS → vertical (pure vertical assumption)
            decomp = los_to_vertical_displacement(d_los, incidence_angle_deg)
            d_vert = decomp["vertical"]

            # Save displacement arrays
            los_path  = str(self.output_dir / "los_displacement_m.bin")
            vert_path = str(self.output_dir / "vertical_displacement_m.bin")
            d_los.astype("float32").tofile(los_path)
            d_vert.astype("float32").tofile(vert_path)

            result.los_displacement_m  = los_path
            result.vertical_disp_m     = vert_path

            # Coherence stats
            valid_coh = coherence_arr[coherence_arr > 0]
            if len(valid_coh) > 0:
                result.mean_coherence = float(np.mean(valid_coh))
                result.valid_pixel_fraction = float(
                    (coherence_arr >= self.coherence_threshold).mean()
                )
            result.processing_log.append(
                f"LOS displacement: [{d_los.min()*1000:.1f}, {d_los.max()*1000:.1f}] mm"
            )
            result.processing_log.append(
                f"Vertical displacement: [{d_vert.min()*1000:.1f}, {d_vert.max()*1000:.1f}] mm"
            )

        except Exception as e:
            result.warnings.append(f"Phase-to-displacement conversion failed: {e}")

        # STEP 11: Terrain correction / geocoding via SNAP
        result.processing_log.append("Geocoding to WGS84...")
        geocoded_path = self._run_terrain_correction(str(ifg_product), dem_path)
        if geocoded_path:
            result.geocoded_product = geocoded_path

        result.success = True
        result.processing_log.append("InSAR pipeline complete")
        return result

    def _generate_snaphu_export_graph(self, input_dim: str, output_dir: str) -> str:
        """Generate SNAP GPT graph for SNAPHU export."""
        graph_xml = f"""<graph id="SnaphuExport">
  <version>1.0</version>
  <node id="Read">
    <operator>Read</operator>
    <sources/>
    <parameters><parameter name="file">{input_dim}</parameter></parameters>
  </node>
  <node id="SnaphuExport">
    <operator>SnaphuExport</operator>
    <sources><source refid="Read"/></sources>
    <parameters>
      <parameter name="targetFolder">{output_dir}</parameter>
      <parameter name="statCostMode">DEFO</parameter>
      <parameter name="initMethod">MCF</parameter>
      <parameter name="numberOfProcessors">{self.n_processors}</parameter>
      <parameter name="numberOfTileCols">10</parameter>
      <parameter name="numberOfTileRows">10</parameter>
      <parameter name="rowOverlap">200</parameter>
      <parameter name="colOverlap">200</parameter>
    </parameters>
  </node>
</graph>"""
        graph_path = str(self.output_dir / "snaphu_work" / "snaphu_export.xml")
        Path(graph_path).parent.mkdir(exist_ok=True)
        with open(graph_path, "w") as f:
            f.write(graph_xml)
        return graph_path

    def _run_terrain_correction(
        self,
        input_dim: str,
        dem_path: str | None = None,
    ) -> str | None:
        """Run Range-Doppler terrain correction to geocode to WGS84."""
        output_path = str(self.output_dir / "geocoded_displacement.dim")

        if dem_path and Path(dem_path).exists():
            dem_config = f"""
      <parameter name="demName">External DEM</parameter>
      <parameter name="externalDEMFile">{dem_path}</parameter>
      <parameter name="externalDEMNoDataValue">-32768.0</parameter>"""
        else:
            dem_config = """
      <parameter name="demName">Copernicus 30m Global DEM</parameter>"""

        graph_xml = f"""<graph id="TerrainCorrection">
  <version>1.0</version>
  <node id="Read">
    <operator>Read</operator>
    <sources/>
    <parameters><parameter name="file">{input_dim}</parameter></parameters>
  </node>
  <node id="SnaphuImport">
    <operator>SnaphuImport</operator>
    <sources>
      <source refid="Read"/>
      <source refid="Read"/>
    </sources>
    <parameters>
      <parameter name="unwrappedPhase">{self.output_dir}/snaphu_work/unwrapped_phase.snaphu</parameter>
      <parameter name="doNotKeepWrapped">false</parameter>
    </parameters>
  </node>
  <node id="TerrainCorrection">
    <operator>Terrain-Correction</operator>
    <sources><source refid="SnaphuImport"/></sources>
    <parameters>{dem_config}
      <parameter name="demResamplingMethod">BILINEAR_INTERPOLATION</parameter>
      <parameter name="imgResamplingMethod">BILINEAR_INTERPOLATION</parameter>
      <parameter name="pixelSpacingInMeter">30.0</parameter>
      <parameter name="mapProjection">WGS84(DD)</parameter>
      <parameter name="saveSelectedSourceBand">true</parameter>
      <parameter name="nodataValueAtSea">false</parameter>
    </parameters>
  </node>
  <node id="Write">
    <operator>Write</operator>
    <sources><source refid="TerrainCorrection"/></sources>
    <parameters>
      <parameter name="file">{output_path}</parameter>
      <parameter name="formatName">BEAM-DIMAP</parameter>
    </parameters>
  </node>
</graph>"""

        graph_path = str(self.output_dir / "terrain_correction.xml")
        with open(graph_path, "w") as f:
            f.write(graph_xml)

        success = run_snap_graph(graph_path, snap_gpt=self.snap_gpt, java_heap_gb=8)
        return output_path if success and Path(output_path).exists() else None

    @staticmethod
    def _read_envi_width(hdr_path: str) -> int:
        """Read image width from ENVI .hdr file.

        Raises rather than guessing: this value is used to reshape raw
        binary SAR data (e.g. for SNAPHU phase unwrapping and the linear
        APS correction) — a wrong width doesn't just reduce accuracy, it
        silently corrupts the entire reshape into a meaningless
        interferogram that may not even look obviously broken. There is
        no scientifically valid "default" width to fall back to.
        """
        try:
            with open(hdr_path) as f:
                for line in f:
                    if line.strip().startswith("samples"):
                        return int(line.split("=")[1].strip())
        except Exception as exc:
            raise RuntimeError(
                f"Failed to read ENVI header {hdr_path}: {exc}. Cannot "
                f"determine image width — proceeding with a guessed value "
                f"would silently corrupt the SAR data reshape."
            ) from exc
        raise RuntimeError(
            f"ENVI header {hdr_path} has no 'samples' field — cannot "
            f"determine image width for reshaping raw SAR data."
        )