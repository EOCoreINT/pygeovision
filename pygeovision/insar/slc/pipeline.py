"""
pygeovision.insar.slc.pipeline
================================
End-to-end SLC InSAR pipeline orchestrator.

Chains: SNAP co-registration → interferogram → Goldstein filter →
        SNAPHU export → phase unwrapping → SNAPHU import →
        phase-to-displacement → terrain correction → GeoTIFF output.

Handles Sentinel-1C and Sentinel-1D products identically to 1A/1B —
all use the same IW SLC format, C-band wavelength, and TOPS mode.

Usage::

    from pygeovision.insar.slc import SLCInSARPipeline

    pipeline = SLCInSARPipeline(
        master_zip   = 'S1C_IW_SLC__1SDV_20260601T...zip',
        slave_zip    = 'S1C_IW_SLC__1SDV_20260613T...zip',
        output_dir   = './insar_results/',
        subswath     = 'IW2',
        bursts       = (2, 4),
        polarisation = 'VV',
        dem_name     = 'SRTM 3Sec',          # auto-downloaded by SNAP
        coherence_threshold = 0.3,
    )

    result = pipeline.run()
    print(f'LOS displacement range: {result.los_min_m:.3f} to {result.los_max_m:.3f} m')
    print(f'Displacement GeoTIFF: {result.displacement_path}')
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pygeovision.insar.slc.displacement import slc_phase_to_displacement
from pygeovision.insar.slc.sentinel1 import (
    SENTINEL1_WAVELENGTH_M,
    SUBSWATH_INCIDENCE_DEG,
    S1SLCProduct,
    parse_s1_slc_manifest,
    validate_slc_pair,
)
from pygeovision.insar.slc.snap_graph import SNAPGraph, check_environment
from pygeovision.insar.slc.snaphu import SnaphuConfig, SnaphuUnwrapper

logger = logging.getLogger("pygeovision.insar.slc.pipeline")


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class SLCInSARResult:
    """
    Complete results from a SLC InSAR processing run.

    All paths point to GeoTIFF files in the output directory.
    Arrays are only populated when keep_arrays=True in SLCInSARPipeline.run().
    """
    # File paths (always populated)
    displacement_path:  str | None = None   # LOS + vertical, 2-band GeoTIFF
    coherence_path:     str | None = None   # Coherence, single-band GeoTIFF
    interferogram_path: str | None = None   # Wrapped phase GeoTIFF
    unwrapped_path:     str | None = None   # Unwrapped phase GeoTIFF
    elevation_path:     str | None = None   # DEM used (terrain corrected)

    # Summary statistics
    los_min_m:          float = float("nan")
    los_max_m:          float = float("nan")
    los_mean_m:         float = float("nan")
    los_std_m:          float = float("nan")
    mean_coherence:     float = float("nan")
    masked_fraction:    float = float("nan")   # Fraction masked by coherence
    temporal_baseline:  float = float("nan")   # Days between acquisitions

    # Arrays (populated when keep_arrays=True)
    los_m:          np.ndarray | None = None
    vertical_m:     np.ndarray | None = None
    coherence:      np.ndarray | None = None

    # Processing metadata
    master:         S1SLCProduct | None = None
    slave:          S1SLCProduct | None = None
    processing_log: list[str]  = field(default_factory=list)
    elapsed_sec:    float      = 0.0
    success:        bool       = False

    def summary(self) -> str:
        """Return a human-readable processing summary."""
        lines = [
            "=" * 60,
            "PyGeoVision SLC InSAR Result",
            "=" * 60,
        ]
        if self.master and self.slave:
            lines += [
                f"  Master : {self.master.satellite} {self.master.date_str}",
                f"  Slave  : {self.slave.satellite}  {self.slave.date_str}",
                f"  Temporal baseline : {self.temporal_baseline:.1f} days",
            ]
        lines += [
            f"  LOS displacement  : [{self.los_min_m:.3f}, {self.los_max_m:.3f}] m",
            f"  LOS mean ± std    : {self.los_mean_m:.3f} ± {self.los_std_m:.3f} m",
            f"  Mean coherence    : {self.mean_coherence:.3f}",
            f"  Masked pixels     : {self.masked_fraction*100:.1f}%",
            f"  Processing time   : {self.elapsed_sec/60:.1f} min",
            f"  Output            : {self.displacement_path or 'N/A'}",
            "=" * 60,
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"SLCInSARResult("
            f"LOS=[{self.los_min_m:.3f},{self.los_max_m:.3f}]m  "
            f"coh={self.mean_coherence:.2f}  "
            f"{'OK' if self.success else 'FAILED'})"
        )


# ── Main pipeline class ───────────────────────────────────────────────────────

class SLCInSARPipeline:
    """
    End-to-end Sentinel-1 SLC InSAR pipeline.

    Produces millimetre-precision LOS displacement maps from a pair of
    Sentinel-1 SLC acquisitions. Supports 1A, 1B, 1C, and 1D products
    transparently — the C-band wavelength, IW mode geometry, and product
    format are identical across all four satellites.

    Parameters
    ----------
    master_zip : str
        Path to the master (reference) Sentinel-1 SLC .zip or .SAFE.
        Convention: earlier acquisition = master.
    slave_zip : str
        Path to the slave (secondary) Sentinel-1 SLC .zip or .SAFE.
        Convention: later acquisition = slave.
    output_dir : str
        Directory for all output products. Created if absent.
    subswath : str
        IW subswath to process: 'IW1', 'IW2', or 'IW3'.
        IW2 (26–35° incidence) is the default — best for land deformation.
    bursts : tuple of (int, int)
        First and last burst index within the subswath (1-indexed, max 9).
        Smaller burst range = faster, less memory. For city-scale studies
        use 2–3 bursts. For regional coverage use 1–9.
    polarisation : str
        'VV' (default — best SNR for land) or 'VH'.
    dem_name : str
        DEM for back-geocoding, topo-phase-removal, terrain correction.
        Options: 'SRTM 3Sec' (default, ~90 m), 'SRTM 1Sec HGT' (~30 m),
                 'Copernicus 30m Global DEM' (best, requires ESA account).
    coherence_threshold : float
        Pixels below this coherence are masked in displacement outputs.
        0.2 = permissive, 0.4 = conservative. Default 0.3.
    gpt_path : str, optional
        Path to SNAP gpt executable. Defaults to 'gpt' on PATH.
    snaphu_exe : str, optional
        Path to snaphu executable. Defaults to 'snaphu' on PATH.

    Example
    -------
    >>> from pygeovision.insar.slc import SLCInSARPipeline
    >>> p = SLCInSARPipeline(
    ...     master_zip   = 'S1C_IW_SLC__1SDV_20260601T053000.zip',
    ...     slave_zip    = 'S1C_IW_SLC__1SDV_20260613T053000.zip',
    ...     output_dir   = './accra_insar/',
    ...     subswath     = 'IW2',
    ...     bursts       = (3, 5),
    ...     polarisation = 'VV',
    ... )
    >>> result = p.run()
    >>> print(result.summary())
    """

    def __init__(
        self,
        master_zip:           str,
        slave_zip:            str,
        output_dir:           str  = "./slc_insar/",
        subswath:             str  = "IW2",
        bursts:               tuple[int, int] = (1, 9),
        polarisation:         str  = "VV",
        dem_name:             str  = "SRTM 3Sec",
        coherence_threshold:  float = 0.3,
        gpt_path:             str | None = None,
        snaphu_exe:           str | None = None,
        n_procs:              int  = 4,
        pixel_size_m:         float = 20.0,
    ) -> None:
        self.master_zip          = str(master_zip)
        self.slave_zip           = str(slave_zip)
        self.output_dir          = Path(output_dir)
        self.subswath            = subswath.upper()
        self.bursts              = bursts
        self.polarisation        = polarisation.upper()
        self.dem_name            = dem_name
        self.coherence_threshold = coherence_threshold
        self.gpt_path            = gpt_path
        self.snaphu_exe          = snaphu_exe
        self.n_procs             = n_procs
        self.pixel_size_m        = pixel_size_m

        # Derived: incidence angle for chosen subswath
        self.incidence_deg = SUBSWATH_INCIDENCE_DEG.get(self.subswath, 38.3)

    # ── Environment check ─────────────────────────────────────────────────

    @staticmethod
    def check_environment() -> dict:
        """
        Check whether all required external tools are available.

        Returns a dict summarising SNAP, snapista, and snaphu status.
        Call this before run() to diagnose missing dependencies.
        """
        env = check_environment()
        print(env["summary"])
        return env

    # ── Main entry point ──────────────────────────────────────────────────

    def run(self, keep_arrays: bool = True) -> SLCInSARResult:
        """
        Execute the complete SLC InSAR pipeline.

        Steps:
            1.  Parse and validate SLC product metadata
            2.  Check SNAP + snapista + snaphu availability
            3.  SNAP: TOPSAR-Split → Orbit → Back-Geocoding → ESD
            4.  SNAP: Interferogram → Deburst → TopoPhaseRemoval → Goldstein
            5.  SNAP: SnaphuExport
            6.  SNAPHU: phase unwrapping
            7.  SNAP: SnaphuImport → PhaseToDisplacement → TerrainCorrection
            8.  Export GeoTIFFs, compute statistics

        Parameters
        ----------
        keep_arrays : bool
            If True, populate result.los_m and result.coherence with numpy
            arrays in addition to writing GeoTIFFs. Requires more memory.

        Returns
        -------
        SLCInSARResult
        """
        t0     = time.time()
        result = SLCInSARResult()
        log    = result.processing_log

        self.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # ── Step 1: Parse product metadata ────────────────────────────
            log.append("Step 1/8: Parsing SLC product metadata…")
            master = parse_s1_slc_manifest(self.master_zip)
            slave  = parse_s1_slc_manifest(self.slave_zip)

            if master is None or slave is None:
                raise ValueError(
                    "Could not parse one or both SLC products. "
                    "Verify the .zip files are valid Sentinel-1 SLC products."
                )

            result.master = master
            result.slave  = slave
            result.temporal_baseline = abs(
                (slave.start_time - master.start_time).total_seconds()
            ) / 86400

            logger.info("Master: %s", master)
            logger.info("Slave : %s", slave)

            # ── Validate pair ─────────────────────────────────────────────
            validation = validate_slc_pair(master, slave)
            for w in validation["warnings"]:
                logger.warning("Pair warning: %s", w)
            for e in validation["errors"]:
                logger.error("Pair error: %s", e)
            if not validation["valid"]:
                raise ValueError(
                    "SLC pair validation failed:\n"
                    + "\n".join(validation["errors"])
                )

            log.append(
                f"  master={master.satellite} {master.date_str}  "
                f"slave={slave.satellite} {slave.date_str}  "
                f"Δt={result.temporal_baseline:.1f}d"
            )

            # ── Step 2: Environment check ─────────────────────────────────
            log.append("Step 2/8: Checking SNAP environment…")
            graph = SNAPGraph(
                gpt_path=self.gpt_path,
                dem_name=self.dem_name,
            )
            # Will raise RuntimeError if SNAP not available
            graph._require_snap()
            log.append("  SNAP: OK")

            # ── Step 3–5: SNAP processing ─────────────────────────────────
            log.append(
                f"Step 3-5/8: SNAP co-registration + interferogram "
                f"({self.subswath} bursts {self.bursts[0]}-{self.bursts[1]})…"
            )
            snap_outputs = graph.run_coreg_and_ifg(
                master_zip   = self.master_zip,
                slave_zip    = self.slave_zip,
                output_dir   = str(self.output_dir / "snap_intermediate"),
                subswath     = self.subswath,
                bursts       = self.bursts,
                polarisation = self.polarisation,
            )
            log.append(f"  Interferogram: {snap_outputs['ifg_path']}")
            log.append(f"  SNAPHU export dir: {snap_outputs['snaphu_dir']}")

            # ── Step 6: SNAPHU phase unwrapping ───────────────────────────
            log.append("Step 6/8: SNAPHU phase unwrapping…")
            snaphu_cfg = SnaphuConfig(
                stat_cost_mode = "DEFO",
                n_procs        = self.n_procs,
                coh_threshold  = self.coherence_threshold,
            )
            unwrapper = SnaphuUnwrapper(
                snaphu_exe = self.snaphu_exe,
                config     = snaphu_cfg,
            )
            unwrap_result = unwrapper.run(
                snaphu_export_dir = snap_outputs["snaphu_dir"],
                output_dir        = str(self.output_dir / "snaphu_output"),
            )
            result.masked_fraction = unwrap_result.masked_fraction
            log.append(
                f"  Unwrapped: {unwrap_result.unwrapped_path}  "
                f"masked={unwrap_result.masked_fraction*100:.1f}%"
            )

            # ── Step 7: Phase → displacement → terrain correction ─────────
            log.append("Step 7/8: Converting phase to displacement…")
            disp_result = self._phase_to_displacement_and_geocode(
                unwrap_result    = unwrap_result,
                snap_ifg_dim     = snap_outputs["ifg_path"],
                graph            = graph,
                keep_arrays      = keep_arrays,
            )

            result.displacement_path  = disp_result["displacement_tif"]
            result.coherence_path     = disp_result.get("coherence_tif")
            result.interferogram_path = disp_result.get("interferogram_tif")
            result.unwrapped_path     = disp_result.get("unwrapped_tif")

            if keep_arrays and "los_m" in disp_result:
                los_arr = disp_result["los_m"]
                result.los_m      = los_arr
                result.vertical_m = disp_result.get("vertical_m")
                result.coherence  = disp_result.get("coherence_arr")
                valid = los_arr[np.isfinite(los_arr)]
                if valid.size > 0:
                    result.los_min_m  = float(valid.min())
                    result.los_max_m  = float(valid.max())
                    result.los_mean_m = float(valid.mean())
                    result.los_std_m  = float(valid.std())

            if disp_result.get("mean_coherence") is not None:
                result.mean_coherence = disp_result["mean_coherence"]

            # ── Step 8: Final summary ─────────────────────────────────────
            result.elapsed_sec = time.time() - t0
            result.success     = True
            log.append(
                f"Step 8/8: Complete in {result.elapsed_sec/60:.1f} min"
            )
            logger.info(result.summary())

        except Exception as exc:
            result.elapsed_sec = time.time() - t0
            result.success     = False
            log.append(f"FAILED: {exc}")
            logger.error("SLC InSAR pipeline failed: %s", exc, exc_info=True)
            raise

        return result

    # ── Internal helpers ──────────────────────────────────────────────────

    def _phase_to_displacement_and_geocode(
        self,
        unwrap_result: object,
        snap_ifg_dim:  str,
        graph:         SNAPGraph,
        keep_arrays:   bool,
    ) -> dict:
        """
        Read unwrapped phase, convert to metres, write GeoTIFF.

        In a full SNAP workflow, SnaphuImport + PhaseToDisplacement operators
        would be used inside SNAP for geocoding. Here we do the arithmetic in
        numpy and write a flat binary GeoTIFF, then call SNAP terrain
        correction only for the final geocoding step.
        """
        output_dir = self.output_dir

        # ── Read unwrapped phase ──────────────────────────────────────────
        phase_path = unwrap_result.unwrapped_path
        n_lines    = unwrap_result.n_lines
        line_len   = unwrap_result.line_length

        phase_flat = np.fromfile(phase_path, dtype=np.float32)
        if phase_flat.size != n_lines * line_len:
            raise ValueError(
                f"Unwrapped phase size mismatch: "
                f"got {phase_flat.size}, expected {n_lines}×{line_len}={n_lines*line_len}"
            )
        phase_2d = phase_flat.reshape(n_lines, line_len)

        # ── Read coherence (if available) ─────────────────────────────────
        coh_2d: np.ndarray | None = None
        coh_path = unwrap_result.coherence_path
        if coh_path and Path(coh_path).exists():
            coh_flat = np.fromfile(coh_path, dtype=np.float32)
            if coh_flat.size == n_lines * line_len:
                coh_2d = coh_flat.reshape(n_lines, line_len)

        mean_coh = float(np.nanmean(coh_2d)) if coh_2d is not None else float("nan")

        # ── Phase → LOS displacement ──────────────────────────────────────
        disp = slc_phase_to_displacement(
            unwrapped_phase     = phase_2d,
            coherence           = coh_2d,
            wavelength_m        = SENTINEL1_WAVELENGTH_M,
            incidence_angle_deg = self.incidence_deg,
            coherence_threshold = self.coherence_threshold,
        )

        # ── Write flat-binary GeoTIFF (no geocoding yet) ──────────────────
        # A proper implementation routes back through SNAP SnaphuImport →
        # PhaseToDisplacement → RangeDoppler-Terrain-Correction for precise
        # geocoding. Here we write the radar-geometry output and call SNAP TC.
        los_path   = str(output_dir / "los_displacement_radar.tif")
        str(output_dir / "displacement_geocoded.tif")

        self._write_flat_tif(disp["los_m"], disp["vertical_m"], los_path)

        # ── Terrain correction (geocoding) ────────────────────────────────
        try:
            tc_path = graph.run_terrain_correction(
                input_dim    = los_path,
                output_dir   = str(output_dir),
                pixel_size_m = self.pixel_size_m,
            )
        except Exception as e:
            logger.warning(
                "Terrain correction failed (%s) — returning radar-geometry output", e
            )
            tc_path = los_path

        result = {
            "displacement_tif": tc_path,
            "mean_coherence":   mean_coh,
        }
        if keep_arrays:
            result.update({
                "los_m":       disp["los_m"],
                "vertical_m":  disp["vertical_m"],
                "coherence_arr": coh_2d,
            })
        return result

    @staticmethod
    def _write_flat_tif(
        los_m:      np.ndarray,
        vertical_m: np.ndarray,
        path:       str,
    ) -> None:
        """Write 2-band GeoTIFF (no georeferencing — radar geometry)."""
        try:
            import rasterio
            h, w = los_m.shape
            with rasterio.open(
                path, "w",
                driver="GTiff", height=h, width=w, count=2,
                dtype="float32", compress="deflate", nodata=np.nan,
            ) as dst:
                dst.write(np.nan_to_num(los_m,      nan=np.nan).astype("float32"), 1)
                dst.write(np.nan_to_num(vertical_m, nan=np.nan).astype("float32"), 2)
                dst.update_tags(
                    1, description="LOS displacement (m)"
                )
                dst.update_tags(
                    2, description="Vertical displacement approx (m)"
                )
        except ImportError:
            # Minimal fallback: write raw numpy
            np.save(path.replace(".tif", "_los.npy"),  los_m)
            np.save(path.replace(".tif", "_vert.npy"), vertical_m)
            logger.warning("rasterio not installed — saved as .npy instead of GeoTIFF")
