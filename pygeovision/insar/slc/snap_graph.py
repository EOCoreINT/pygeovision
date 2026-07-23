"""
pygeovision.insar.slc.snap_graph
==================================
SNAP / snapista graph builder for Sentinel-1 SLC InSAR.

Supports two backends:
    1. snapista  (preferred) — Python API that builds and runs SNAP GPT graphs.
    2. GPT XML   (fallback)  — Writes a .xml graph file and calls gpt on the CLI.

Both require ESA SNAP ≥ 9 with S1TBX to be installed.

Installation:
    # 1. Install ESA SNAP from https://step.esa.int/main/download/snap-download/
    # 2. Install snapista:
    pip install snapista
    # 3. Install snaphu (phase unwrapping):
    conda install -c conda-forge snaphu    # Linux/macOS
    # or: sudo apt install snaphu          # Ubuntu/Debian

Graph steps (Sentinel-1 TOPS InSAR):
    TOPSAR-Split → Apply-Orbit-File → Back-Geocoding → ESD →
    Interferogram → TOPSAR-Deburst → Topographic-Phase-Removal →
    Goldstein-Phase-Filtering → Snaphu-Export
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger("pygeovision.insar.slc.snap_graph")

# ── Availability checks ───────────────────────────────────────────────────────

def check_snap(gpt_path: str | None = None) -> dict[str, object]:
    """
    Check whether ESA SNAP's GPT is available.

    Parameters
    ----------
    gpt_path : str, optional
        Explicit path to the `gpt` executable. If None, searches PATH.

    Returns
    -------
    dict with keys:
        available (bool)     : True if gpt is found and runnable
        path (str | None)    : Path to the gpt executable
        version (str | None) : SNAP version string
        message (str)        : Human-readable status message
    """
    exe = gpt_path or shutil.which("gpt") or shutil.which("gpt.sh")

    if exe is None:
        return {
            "available": False,
            "path":      None,
            "version":   None,
            "message": (
                "SNAP GPT not found on PATH.\n"
                "Install ESA SNAP >= 9.0 from: https://step.esa.int/main/download/snap-download/\n"
                "Then add the bin/ directory to your PATH, e.g.:\n"
                "  export PATH=/usr/local/snap/bin:$PATH"
            ),
        }

    try:
        r = subprocess.run(
            [exe, "--version"],
            capture_output=True, text=True, timeout=30,
        )
        version_line = next(
            (l for l in r.stdout.split("\n") if "version" in l.lower()),
            r.stdout.split("\n")[0] if r.stdout else "unknown",
        )
        return {
            "available": True,
            "path":      exe,
            "version":   version_line.strip(),
            "message":   f"SNAP GPT found: {exe}  ({version_line.strip()})",
        }
    except Exception as e:
        return {
            "available": False,
            "path":      exe,
            "version":   None,
            "message":   f"SNAP GPT found at {exe} but failed to run: {e}",
        }


def check_snapista() -> dict[str, object]:
    """
    Check whether snapista is importable.

    Returns
    -------
    dict with keys:
        available (bool)
        version (str | None)
        message (str)
    """
    try:
        import snapista
        version = getattr(snapista, "__version__", "unknown")
        return {
            "available": True,
            "version":   version,
            "message":   f"snapista {version} available",
        }
    except ImportError:
        return {
            "available": False,
            "version":   None,
            "message": (
                "snapista not installed.\n"
                "  pip install snapista\n"
                "snapista also requires ESA SNAP installed and gpt on PATH."
            ),
        }


def check_environment() -> dict[str, object]:
    """Full environment check: SNAP, snapista, snaphu."""
    snap     = check_snap()
    snapista = check_snapista()
    snaphu   = _check_snaphu()
    all_ok   = snap["available"] and snapista["available"] and snaphu["available"]
    return {
        "ready":    all_ok,
        "snap":     snap,
        "snapista": snapista,
        "snaphu":   snaphu,
        "summary": (
            "[OK] Full SLC InSAR environment ready." if all_ok else
            "SLC InSAR not fully configured. Missing:\n" +
            "\n".join(
                f"  - {k}: {v['message']}"
                for k, v in [("SNAP", snap), ("snapista", snapista), ("snaphu", snaphu)]
                if not v["available"]
            )
        ),
    }


def _check_snaphu() -> dict[str, object]:
    exe = shutil.which("snaphu")
    if exe:
        return {"available": True, "path": exe, "message": f"snaphu found: {exe}"}
    return {
        "available": False,
        "path":      None,
        "message": (
            "snaphu not found.\n"
            "  conda install -c conda-forge snaphu\n"
            "  or: sudo apt install snaphu"
        ),
    }


# ── SNAPGraph — the main graph builder ───────────────────────────────────────

class SNAPGraph:
    """
    Build and execute Sentinel-1 SLC InSAR SNAP graphs.

    Wraps snapista with a clean PyGeoVision-style API. Falls back to raw
    GPT XML when snapista is not installed.

    Parameters
    ----------
    gpt_path : str, optional
        Path to the SNAP gpt executable. Defaults to 'gpt' on PATH.
    dem_name : str
        DEM to use for back-geocoding and topo-phase-removal.
        Options: 'SRTM 3Sec', 'SRTM 1Sec HGT', 'Copernicus 30m Global DEM'.
    cache_dir : str, optional
        SNAP tile cache directory. Defaults to ~/.snap/var/cache.
    tile_size_mb : int
        SNAP tile cache size in MB. 4096 MB is a safe default.

    Example
    -------
    >>> g = SNAPGraph()
    >>> result = g.run_coreg_and_ifg(
    ...     master_zip  = 'S1C_IW_SLC_...zip',
    ...     slave_zip   = 'S1C_IW_SLC_...zip',
    ...     output_dir  = './snap_output/',
    ...     subswath    = 'IW2',
    ...     bursts      = (2, 4),
    ...     polarisation= 'VV',
    ... )
    """

    # Sentinel-1 IW mode: 3 subswaths × 9 bursts each
    VALID_SUBSWATHS   = ("IW1", "IW2", "IW3")
    VALID_POLS        = ("VV", "VH", "HH", "HV")
    MAX_BURST_INDEX   = 9
    SENTINEL1_WAVELENGTH = 0.05546576   # metres

    def __init__(
        self,
        gpt_path:    str | None = None,
        dem_name:    str = "SRTM 3Sec",
        cache_dir:   str | None = None,
        tile_size_mb: int = 4096,
    ) -> None:
        self.gpt_path    = gpt_path or shutil.which("gpt") or "gpt"
        self.dem_name    = dem_name
        self.cache_dir   = cache_dir or str(Path.home() / ".snap/var/cache")
        self.tile_size_mb = tile_size_mb
        self._snap_ok    = check_snap(self.gpt_path)["available"]
        self._snapista   = check_snapista()["available"]

    # ── Public interface ──────────────────────────────────────────────────

    def run_coreg_and_ifg(
        self,
        master_zip:   str,
        slave_zip:    str,
        output_dir:   str,
        subswath:     str = "IW2",
        bursts:       tuple[int, int] = (1, 9),
        polarisation: str = "VV",
    ) -> dict[str, str]:
        """
        Run the full co-registration → interferogram → deburst → topo-removal
        → Goldstein-filter → Snaphu-export chain.

        Returns
        -------
        dict with keys:
            snaphu_dir (str)         : Directory for SNAPHU unwrapping input
            ifg_path (str)           : Filtered interferogram .dim product
            coherence_path (str)     : Coherence band path
            intensity_master (str)   : Master intensity (for geocoding)

        Raises
        ------
        RuntimeError
            If SNAP is not available or the graph fails.
        """
        self._require_snap()
        self._validate_inputs(subswath, bursts, polarisation)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "SLC InSAR graph: %s + %s  subswath=%s bursts=%s-%s pol=%s",
            Path(master_zip).stem[:40], Path(slave_zip).stem[:40],
            subswath, bursts[0], bursts[1], polarisation,
        )

        if self._snapista:
            return self._run_snapista(
                master_zip, slave_zip, output_dir, subswath, bursts, polarisation
            )
        else:
            return self._run_gpt_xml(
                master_zip, slave_zip, output_dir, subswath, bursts, polarisation
            )

    def run_terrain_correction(
        self,
        input_dim:   str,
        output_dir:  str,
        pixel_size_m: float = 20.0,
    ) -> str:
        """
        Apply Range-Doppler terrain correction (geocoding) to a SNAP product.

        Parameters
        ----------
        input_dim : str
            Path to the .dim SNAP product (e.g. displacement or coherence).
        output_dir : str
            Output directory.
        pixel_size_m : float
            Output pixel size in metres. Default 20 m matches Sentinel-1 IW GRD.

        Returns
        -------
        str
            Path to terrain-corrected GeoTIFF.
        """
        self._require_snap()
        output_dir  = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem        = Path(input_dim).stem + "_TC"
        output_path = str(output_dir / f"{stem}.tif")

        if self._snapista:
            self._tc_snapista(input_dim, output_path, pixel_size_m)
        else:
            self._tc_gpt_xml(input_dim, output_path, pixel_size_m)
        return output_path

    # ── snapista implementation ───────────────────────────────────────────

    def _run_snapista(
        self,
        master_zip:   str,
        slave_zip:    str,
        output_dir:   Path,
        subswath:     str,
        bursts:       tuple[int, int],
        polarisation: str,
    ) -> dict[str, str]:
        """Build and run the InSAR graph using snapista."""
        from snapista import Graph, Operator  # type: ignore[import]

        g = Graph()
        first_burst, last_burst = bursts

        # ── Read master ────────────────────────────────────────────────────
        g.add_node(
            Operator("Read", file=str(master_zip)),
            node_id="Read_Master",
        )
        # ── Read slave ─────────────────────────────────────────────────────
        g.add_node(
            Operator("Read", file=str(slave_zip)),
            node_id="Read_Slave",
        )

        # ── TOPSAR-Split master ────────────────────────────────────────────
        g.add_node(
            Operator(
                "TOPSAR-Split",
                subswath=subswath,
                selectedPolarisations=polarisation,
                firstBurstIndex=first_burst,
                lastBurstIndex=last_burst,
            ),
            node_id="Split_Master",
            source="Read_Master",
        )
        # ── TOPSAR-Split slave ─────────────────────────────────────────────
        g.add_node(
            Operator(
                "TOPSAR-Split",
                subswath=subswath,
                selectedPolarisations=polarisation,
                firstBurstIndex=first_burst,
                lastBurstIndex=last_burst,
            ),
            node_id="Split_Slave",
            source="Read_Slave",
        )

        # ── Apply precise orbit (master) ───────────────────────────────────
        g.add_node(
            Operator(
                "Apply-Orbit-File",
                orbitType="Sentinel Precise (Auto Download)",
                polyDegree=3,
                continueOnFail=True,
            ),
            node_id="Orb_Master",
            source="Split_Master",
        )
        # ── Apply precise orbit (slave) ────────────────────────────────────
        g.add_node(
            Operator(
                "Apply-Orbit-File",
                orbitType="Sentinel Precise (Auto Download)",
                polyDegree=3,
                continueOnFail=True,
            ),
            node_id="Orb_Slave",
            source="Split_Slave",
        )

        # ── Back-Geocoding (co-registration, needs DEM) ────────────────────
        g.add_node(
            Operator(
                "Back-Geocoding",
                demName=self.dem_name,
                demResamplingMethod="BICUBIC_INTERPOLATION",
                externalDEMNoDataValue=0.0,
                resamplingType="BISINC_5_POINT_INTERPOLATION",
                maskOutAreaWithoutElevation=True,
                disableReramp=False,
            ),
            node_id="BackGeocoding",
            source=["Orb_Master", "Orb_Slave"],
        )

        # ── Enhanced Spectral Diversity (ESD — fine TOPS co-registration) ─
        g.add_node(
            Operator(
                "Enhanced-Spectral-Diversity",
                fineWinWidthStr="512",
                fineWinHeightStr="512",
                fineWinAccAzimuth="16",
                fineWinAccRange="16",
                fineWinOversampling="128",
                xCorrThreshold="0.1",
                cohThreshold="0.3",
                numBlocksPerOverlap="10",
                useSupplementaryRange="true",
            ),
            node_id="ESD",
            source="BackGeocoding",
        )

        # ── Interferogram formation ────────────────────────────────────────
        g.add_node(
            Operator(
                "Interferogram",
                subtractFlatEarthPhase=True,
                srpPolynomialDegree=5,
                srpNumberPoints=501,
                orbitDegree=3,
                includeCoherence=True,
                cohWinAz=10,
                cohWinRg=10,
                squarePixel=True,
                subtractTopographicPhase=False,  # done separately
            ),
            node_id="Interferogram",
            source="ESD",
        )

        # ── TOPSAR-Deburst (merge bursts → continuous image) ───────────────
        g.add_node(
            Operator(
                "TOPSAR-Deburst",
                selectedPolarisations=polarisation,
            ),
            node_id="Deburst",
            source="Interferogram",
        )

        # ── Topographic Phase Removal (subtract terrain contribution) ──────
        g.add_node(
            Operator(
                "TopoPhaseRemoval",
                orbitDegree=3,
                demName=self.dem_name,
                tileExtensionPercent="100",
                outputTopoBandPhase=False,
                outputElevationBand=True,
            ),
            node_id="TopoPhaseRemoval",
            source="Deburst",
        )

        # ── Goldstein Phase Filtering ──────────────────────────────────────
        g.add_node(
            Operator(
                "GoldsteinPhaseFiltering",
                alpha=0.5,
                FFTSizeString="64",
                windowSizeString="3",
                useCoherenceMask=True,
                coherenceThreshold=0.2,
            ),
            node_id="GoldsteinFilter",
            source="TopoPhaseRemoval",
        )

        # ── Snaphu-Export (write SNAPHU-compatible tiles) ──────────────────
        snaphu_dir = str(output_dir / "snaphu_export")
        g.add_node(
            Operator(
                "SnaphuExport",
                targetFolder=snaphu_dir,
                statCostMode="DEFO",    # DEFO mode for deformation mapping
                initMethod="MCF",       # Minimum Cost Flow unwrapping init
                numberOfTileCols=10,
                numberOfTileRows=10,
                numberOfProcessors=4,
                rowOverlap=200,
                colOverlap=200,
                tileCostThreshold=500,
            ),
            node_id="SnaphuExport",
            source="GoldsteinFilter",
        )

        # ── Write intermediate product ─────────────────────────────────────
        ifg_dim = str(output_dir / "interferogram_filtered")
        g.add_node(
            Operator(
                "Write",
                file=ifg_dim,
                formatName="BEAM-DIMAP",
            ),
            node_id="Write",
            source="GoldsteinFilter",
        )

        logger.info("Running SNAP graph via snapista…")
        g.run()

        return {
            "snaphu_dir":        snaphu_dir,
            "ifg_path":          ifg_dim + ".dim",
            "coherence_path":    str(output_dir / "coherence.dim"),
            "intensity_master":  str(output_dir / "intensity_master.dim"),
        }

    def _tc_snapista(
        self, input_dim: str, output_path: str, pixel_size_m: float
    ) -> None:
        from snapista import Graph, Operator  # type: ignore[import]

        g = Graph()
        g.add_node(Operator("Read", file=input_dim), node_id="Read")
        g.add_node(
            Operator(
                "RangeDoppler-Terrain-Correction",
                demName=self.dem_name,
                demResamplingMethod="BILINEAR_INTERPOLATION",
                imgResamplingMethod="BILINEAR_INTERPOLATION",
                pixelSpacingInMeter=str(pixel_size_m),
                nodataValueAtSea=False,
                saveDEM=False,
                saveLatLon=False,
                saveIncidenceAngleFromEllipsoid=False,
                saveProjectedLocalIncidenceAngle=True,
                saveLocalIncidenceAngle=False,
                saveSelectedSourceBand=True,
                outputComplex=False,
            ),
            node_id="TC",
            source="Read",
        )
        g.add_node(
            Operator("Write", file=output_path, formatName="GeoTIFF"),
            node_id="Write",
            source="TC",
        )
        g.run()

    # ── GPT XML fallback ──────────────────────────────────────────────────

    def _run_gpt_xml(
        self,
        master_zip:   str,
        slave_zip:    str,
        output_dir:   Path,
        subswath:     str,
        bursts:       tuple[int, int],
        polarisation: str,
    ) -> dict[str, str]:
        """Build an XML graph file and run it via the gpt CLI."""
        first_burst, last_burst = bursts
        snaphu_dir = str(output_dir / "snaphu_export")
        ifg_dim    = str(output_dir / "interferogram_filtered")

        xml = self._build_insar_xml(
            master_zip, slave_zip, snaphu_dir, ifg_dim,
            subswath, first_burst, last_burst, polarisation,
        )

        with tempfile.NamedTemporaryFile(
            suffix=".xml", mode="w", delete=False, dir=str(output_dir)
        ) as f:
            f.write(xml)
            graph_file = f.name

        logger.info("Running SNAP GPT with graph: %s", graph_file)
        cmd = [
            self.gpt_path, graph_file,
            f"-J-Xmx{self.tile_size_mb}M",
            "-c", f"{self.tile_size_mb // 2}M",
            "-q", "4",
        ]
        r = subprocess.run(cmd, capture_output=False, text=True)
        if r.returncode != 0:
            raise RuntimeError(
                f"SNAP GPT failed (exit {r.returncode}). Check SNAP logs at "
                f"~/.snap/var/log/ for details."
            )

        return {
            "snaphu_dir":       snaphu_dir,
            "ifg_path":         ifg_dim + ".dim",
            "coherence_path":   str(output_dir / "coherence.dim"),
            "intensity_master": str(output_dir / "intensity_master.dim"),
        }

    def _tc_gpt_xml(
        self, input_dim: str, output_path: str, pixel_size_m: float
    ) -> None:
        xml = self._build_tc_xml(input_dim, output_path, pixel_size_m)
        with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as f:
            f.write(xml)
            graph_file = f.name
        cmd = [self.gpt_path, graph_file, f"-J-Xmx{self.tile_size_mb}M"]
        r = subprocess.run(cmd, capture_output=False)
        if r.returncode != 0:
            raise RuntimeError(f"SNAP terrain correction failed (exit {r.returncode}).")

    def _build_insar_xml(
        self,
        master: str, slave: str, snaphu_dir: str, ifg_dim: str,
        subswath: str, first_burst: int, last_burst: int, pol: str,
    ) -> str:
        """Generate the complete SNAP GPT XML graph string."""
        dem = self.dem_name
        return f"""<graph id="InSAR_S1_SLC">
  <version>1.0</version>

  <!-- Read master SLC -->
  <node id="Read_Master">
    <operator>Read</operator>
    <parameters><file>{master}</file></parameters>
  </node>

  <!-- Read slave SLC -->
  <node id="Read_Slave">
    <operator>Read</operator>
    <parameters><file>{slave}</file></parameters>
  </node>

  <!-- TOPSAR-Split master -->
  <node id="Split_Master">
    <operator>TOPSAR-Split</operator>
    <sources><sourceProduct refid="Read_Master"/></sources>
    <parameters>
      <subswath>{subswath}</subswath>
      <selectedPolarisations>{pol}</selectedPolarisations>
      <firstBurstIndex>{first_burst}</firstBurstIndex>
      <lastBurstIndex>{last_burst}</lastBurstIndex>
    </parameters>
  </node>

  <!-- TOPSAR-Split slave -->
  <node id="Split_Slave">
    <operator>TOPSAR-Split</operator>
    <sources><sourceProduct refid="Read_Slave"/></sources>
    <parameters>
      <subswath>{subswath}</subswath>
      <selectedPolarisations>{pol}</selectedPolarisations>
      <firstBurstIndex>{first_burst}</firstBurstIndex>
      <lastBurstIndex>{last_burst}</lastBurstIndex>
    </parameters>
  </node>

  <!-- Apply precise orbit — master -->
  <node id="Orb_Master">
    <operator>Apply-Orbit-File</operator>
    <sources><sourceProduct refid="Split_Master"/></sources>
    <parameters>
      <orbitType>Sentinel Precise (Auto Download)</orbitType>
      <polyDegree>3</polyDegree>
      <continueOnFail>true</continueOnFail>
    </parameters>
  </node>

  <!-- Apply precise orbit — slave -->
  <node id="Orb_Slave">
    <operator>Apply-Orbit-File</operator>
    <sources><sourceProduct refid="Split_Slave"/></sources>
    <parameters>
      <orbitType>Sentinel Precise (Auto Download)</orbitType>
      <polyDegree>3</polyDegree>
      <continueOnFail>true</continueOnFail>
    </parameters>
  </node>

  <!-- Back-Geocoding (co-registration) -->
  <node id="BackGeocoding">
    <operator>Back-Geocoding</operator>
    <sources>
      <sourceProduct refid="Orb_Master"/>
      <sourceProduct refid="Orb_Slave"/>
    </sources>
    <parameters>
      <demName>{dem}</demName>
      <demResamplingMethod>BICUBIC_INTERPOLATION</demResamplingMethod>
      <resamplingType>BISINC_5_POINT_INTERPOLATION</resamplingType>
      <maskOutAreaWithoutElevation>true</maskOutAreaWithoutElevation>
      <disableReramp>false</disableReramp>
    </parameters>
  </node>

  <!-- Enhanced Spectral Diversity -->
  <node id="ESD">
    <operator>Enhanced-Spectral-Diversity</operator>
    <sources><sourceProduct refid="BackGeocoding"/></sources>
    <parameters>
      <fineWinWidthStr>512</fineWinWidthStr>
      <fineWinHeightStr>512</fineWinHeightStr>
      <fineWinAccAzimuth>16</fineWinAccAzimuth>
      <fineWinAccRange>16</fineWinAccRange>
      <fineWinOversampling>128</fineWinOversampling>
      <xCorrThreshold>0.1</xCorrThreshold>
      <cohThreshold>0.3</cohThreshold>
      <numBlocksPerOverlap>10</numBlocksPerOverlap>
    </parameters>
  </node>

  <!-- Interferogram formation -->
  <node id="Interferogram">
    <operator>Interferogram</operator>
    <sources><sourceProduct refid="ESD"/></sources>
    <parameters>
      <subtractFlatEarthPhase>true</subtractFlatEarthPhase>
      <srpPolynomialDegree>5</srpPolynomialDegree>
      <srpNumberPoints>501</srpNumberPoints>
      <orbitDegree>3</orbitDegree>
      <includeCoherence>true</includeCoherence>
      <cohWinAz>10</cohWinAz>
      <cohWinRg>10</cohWinRg>
      <squarePixel>true</squarePixel>
      <subtractTopographicPhase>false</subtractTopographicPhase>
    </parameters>
  </node>

  <!-- TOPSAR-Deburst -->
  <node id="Deburst">
    <operator>TOPSAR-Deburst</operator>
    <sources><sourceProduct refid="Interferogram"/></sources>
    <parameters>
      <selectedPolarisations>{pol}</selectedPolarisations>
    </parameters>
  </node>

  <!-- Topographic Phase Removal -->
  <node id="TopoPhaseRemoval">
    <operator>TopoPhaseRemoval</operator>
    <sources><sourceProduct refid="Deburst"/></sources>
    <parameters>
      <orbitDegree>3</orbitDegree>
      <demName>{dem}</demName>
      <tileExtensionPercent>100</tileExtensionPercent>
      <outputTopoBandPhase>false</outputTopoBandPhase>
      <outputElevationBand>true</outputElevationBand>
    </parameters>
  </node>

  <!-- Goldstein Phase Filtering -->
  <node id="GoldsteinFilter">
    <operator>GoldsteinPhaseFiltering</operator>
    <sources><sourceProduct refid="TopoPhaseRemoval"/></sources>
    <parameters>
      <alpha>0.5</alpha>
      <FFTSizeString>64</FFTSizeString>
      <windowSizeString>3</windowSizeString>
      <useCoherenceMask>true</useCoherenceMask>
      <coherenceThreshold>0.2</coherenceThreshold>
    </parameters>
  </node>

  <!-- Write filtered interferogram -->
  <node id="Write_IFG">
    <operator>Write</operator>
    <sources><sourceProduct refid="GoldsteinFilter"/></sources>
    <parameters>
      <file>{ifg_dim}</file>
      <formatName>BEAM-DIMAP</formatName>
    </parameters>
  </node>

  <!-- Snaphu Export -->
  <node id="SnaphuExport">
    <operator>SnaphuExport</operator>
    <sources><sourceProduct refid="GoldsteinFilter"/></sources>
    <parameters>
      <targetFolder>{snaphu_dir}</targetFolder>
      <statCostMode>DEFO</statCostMode>
      <initMethod>MCF</initMethod>
      <numberOfTileCols>10</numberOfTileCols>
      <numberOfTileRows>10</numberOfTileRows>
      <numberOfProcessors>4</numberOfProcessors>
      <rowOverlap>200</rowOverlap>
      <colOverlap>200</colOverlap>
      <tileCostThreshold>500</tileCostThreshold>
    </parameters>
  </node>

</graph>"""

    def _build_tc_xml(
        self, input_dim: str, output_path: str, pixel_size_m: float
    ) -> str:
        return f"""<graph id="TerrainCorrection">
  <version>1.0</version>
  <node id="Read">
    <operator>Read</operator>
    <parameters><file>{input_dim}</file></parameters>
  </node>
  <node id="TC">
    <operator>RangeDoppler-Terrain-Correction</operator>
    <sources><sourceProduct refid="Read"/></sources>
    <parameters>
      <demName>{self.dem_name}</demName>
      <demResamplingMethod>BILINEAR_INTERPOLATION</demResamplingMethod>
      <imgResamplingMethod>BILINEAR_INTERPOLATION</imgResamplingMethod>
      <pixelSpacingInMeter>{pixel_size_m}</pixelSpacingInMeter>
      <nodataValueAtSea>false</nodataValueAtSea>
      <saveDEM>false</saveDEM>
      <saveLocalIncidenceAngle>false</saveLocalIncidenceAngle>
      <saveProjectedLocalIncidenceAngle>true</saveProjectedLocalIncidenceAngle>
      <saveSelectedSourceBand>true</saveSelectedSourceBand>
    </parameters>
  </node>
  <node id="Write">
    <operator>Write</operator>
    <sources><sourceProduct refid="TC"/></sources>
    <parameters>
      <file>{output_path}</file>
      <formatName>GeoTIFF</formatName>
    </parameters>
  </node>
</graph>"""

    # ── Helpers ───────────────────────────────────────────────────────────

    def _require_snap(self) -> None:
        if not self._snap_ok:
            env = check_environment()
            raise RuntimeError(
                "SNAP is required for SLC InSAR processing but was not found.\n\n"
                + env["summary"]
            )

    @staticmethod
    def _validate_inputs(
        subswath: str, bursts: tuple[int, int], polarisation: str
    ) -> None:
        if subswath.upper() not in SNAPGraph.VALID_SUBSWATHS:
            raise ValueError(
                f"Invalid subswath '{subswath}'. "
                f"Must be one of {SNAPGraph.VALID_SUBSWATHS}."
            )
        if polarisation.upper() not in SNAPGraph.VALID_POLS:
            raise ValueError(
                f"Invalid polarisation '{polarisation}'. "
                f"Must be one of {SNAPGraph.VALID_POLS}."
            )
        first, last = bursts
        if not (1 <= first <= last <= SNAPGraph.MAX_BURST_INDEX):
            raise ValueError(
                f"Invalid burst range ({first}, {last}). "
                f"Must satisfy 1 <= first <= last <= {SNAPGraph.MAX_BURST_INDEX}."
            )
