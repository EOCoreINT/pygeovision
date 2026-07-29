"""
pygeovision.data.ai_prep
=========================
AI readiness layer — the bridge between pygeofetch processed rasters
and pygeovision foundation model inference.

This is the ONLY code in pygeovision that performs raster operations.
Everything else is delegated to pygeofetch.

What this module owns exclusively:
  1. GCP recovery         — fixes pygeofetch's identity-transform bug on GRD downloads
  2. DataValidator gate   — validates every raster before it enters an AI model
  3. SAR → 6-channel HLS  — physics-guided channel mapping for Prithvi/DINOv3
  4. SARSpeckleAug        — correct multiplicative speckle augmentation for training
  5. InSAR AI prep        — displacement map → normalised float32 for change models

What this module does NOT own (delegated to pygeofetch):
  - Atmospheric correction, cloud masking, clipping, resampling, compositing
  - Speckle filtering, radiometric calibration, flood mapping
  - Spectral index computation, time-series analysis
  - Vectorization, zonal statistics, COG conversion
"""
from __future__ import annotations

import logging
import pathlib
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ── 1. GCP Recovery (pygeofetch GRD download bug fix) ─────────────────────────

def fix_georeference(file_path: str) -> str:
    """
    Fix identity-transform CRS corruption on pygeofetch GRD downloads.

    pygeofetch's reproject:EPSG:32630 post-processing sometimes writes
    a pixel-space identity transform (a=1.0) while correctly tagging the
    CRS as EPSG:32630. This function reads embedded GCPs and recovers
    the correct affine transform.

    Args:
        file_path: Path to the downloaded GeoTIFF.

    Returns:
        Path to the repaired file (in _repaired/ subdirectory) if recovery
        was needed, or the original path if the georeference was already valid.

    Example::

        working_path = fix_georeference(downloads[0].path)
        # Always use the returned path, not the original
    """
    from pygeovision.data.validators.georeference import validate_sar_georeference
    geo = validate_sar_georeference(str(file_path))

    if geo.valid:
        return str(file_path)

    if geo.recovered and geo.repaired_path:
        logger.info("GCP recovery: %s → %s (pixel_size=%s m)",
                    pathlib.Path(file_path).name,
                    pathlib.Path(geo.repaired_path).name,
                    f"{geo.transform_a:.2f}" if geo.transform_a else "?")
        return geo.repaired_path

    logger.error("Georeference invalid and unrecoverable for %s: %s",
                 file_path, geo.errors)
    raise ValueError(
        f"Georeference recovery failed for {pathlib.Path(file_path).name}: {geo.errors}"
    )


# ── 2. DataValidator gate ─────────────────────────────────────────────────────

class DataValidator:
    """
    Mandatory AI readiness validation gate.

    Every raster passes through here before entering a foundation model,
    ChangeFormer, or any other AI pipeline. The validator:
      - Checks dtype (must be float32)
      - Checks value range (no Inf, no all-NaN bands)
      - Checks spatial metadata (CRS, valid pixel size, non-zero extent)
      - Checks band count against model expectations
      - Auto-repairs dtype and NaN fill
      - Returns a float32 numpy array (C, H, W) ready for inference

    Usage::

        validator = DataValidator()
        arr, report = validator.validate_for_ai(
            raster_path="scene_ready.tif",
            model_type="segmentation",
            expected_bands=6,
        )
        if report["valid"]:
            prediction = model(arr)
    """

    def validate_for_ai(
        self,
        raster_path: str,
        model_type:  str = "segmentation",
        expected_bands: int | None = None,
        expected_range: tuple[float, float] = (0.0, 1.0),
        auto_fix:    bool = True,
    ) -> tuple[np.ndarray | None, dict[str, Any]]:
        """
        Validate a raster and return an AI-ready float32 array.

        Args:
            raster_path:    Path to processed GeoTIFF.
            model_type:     ``"segmentation"`` | ``"classification"`` | ``"regression"``.
            expected_bands: Expected number of bands. None = no check.
            expected_range: Expected value range ``(min, max)``.
            auto_fix:       Auto-fix dtype and NaN fill.

        Returns:
            ``(array, report)`` where:
              - ``array``  : float32 ndarray (C, H, W) or None on failure
              - ``report`` : dict with ``"valid"``, ``"warnings"``, ``"errors"``
        """
        import rasterio

        report = {"valid": False, "warnings": [], "errors": [], "fixes_applied": []}

        if not pathlib.Path(raster_path).exists():
            report["errors"].append(f"File not found: {raster_path}")
            return None, report

        try:
            with rasterio.open(raster_path) as src:
                data = src.read().astype("float32")
                crs  = src.crs
                t    = src.transform
        except Exception as e:
            report["errors"].append(f"Cannot open file: {e}")
            return None, report

        C, H, W = data.shape

        # Band count check
        if expected_bands and expected_bands != C:
            report["warnings"].append(
                f"Expected {expected_bands} bands, got {C}. "
                f"Check band selection in download step."
            )

        # CRS check
        if crs is None:
            report["errors"].append("No CRS — run fix_georeference() first")
            return None, report

        # Pixel size check
        a = abs(t.a)
        if a < 0.5 or a > 10_000:
            report["errors"].append(
                f"Pixel size {a:.4f} is not physically plausible. "
                f"Run fix_georeference() to recover GCP-based transform."
            )
            return None, report

        # Spatial extent check
        if H < 16 or W < 16:
            report["errors"].append(
                f"Raster too small: {H}×{W} px. "
                f"Check clip bbox — may be mis-specified or in wrong CRS."
            )
            return None, report

        # NaN/Inf check per band
        for b in range(C):
            band = data[b]
            n_nan = np.isnan(band).sum()
            n_inf = np.isinf(band).sum()
            n_tot = band.size

            if n_nan == n_tot:
                report["errors"].append(f"Band {b+1} is entirely NaN.")
                return None, report
            if n_nan > 0.5 * n_tot:
                report["warnings"].append(
                    f"Band {b+1}: {n_nan/n_tot*100:.1f}% NaN pixels — "
                    f"check cloud masking or clip boundary."
                )
            if n_inf > 0:
                if auto_fix:
                    data[b][np.isinf(data[b])] = 0.0
                    report["fixes_applied"].append(f"Band {b+1}: replaced {n_inf} Inf with 0")
                else:
                    report["errors"].append(f"Band {b+1}: {n_inf} Inf pixels")

        # Replace NaN with 0 for inference
        if auto_fix and np.any(np.isnan(data)):
            n_fixed = np.isnan(data).sum()
            data = np.nan_to_num(data, nan=0.0)
            report["fixes_applied"].append(f"Replaced {n_fixed} NaN with 0")

        # Value range check
        lo, hi = expected_range
        actual_min = float(np.nanmin(data))
        actual_max = float(np.nanmax(data))
        if actual_min < lo - 0.1 or actual_max > hi + 0.1:
            report["warnings"].append(
                f"Values [{actual_min:.4f}, {actual_max:.4f}] outside "
                f"expected [{lo}, {hi}]. "
                f"Check normalisation step."
            )

        report.update({
            "valid":        True,
            "shape":        (C, H, W),
            "dtype":        "float32",
            "crs":          str(crs),
            "pixel_size_m": float(a),
            "value_range":  (actual_min, actual_max),
            "model_type":   model_type,
        })
        return data, report


# ── 3. SAR → 6-channel HLS mapping (Prithvi/DINOv3 AI prep) ──────────────────

def prepare_sar_for_ai(
    file_path:       str,
    channel_mapping: str = "physics_guided",
    fix_georeference_first: bool = True,
    validate:        bool = True,
) -> dict[str, Any]:
    """
    Complete SAR → AI pipeline:
      1. GCP recovery (if needed)
      2. Read VV + VH bands
      3. Map to 6-channel pseudo-HLS for Prithvi/DINOv3
      4. Validate for AI inference

    This is the primary entry point for SAR data entering the AI layer.
    pygeofetch handles everything upstream (download, despeckle, calibrate,
    normalise). This function handles the AI-specific channel mapping.

    Args:
        file_path:              Path to normalised SAR GeoTIFF (VV=band1, VH=band2).
                                Values must be in [0, 1] (output of normalise step).
        channel_mapping:        ``"physics_guided"`` (recommended) |
                                ``"mean_repeat"`` | ``"vv_vh_ratio"``.
        fix_georeference_first: Run GCP recovery before reading.
        validate:               Run DataValidator on the 6-channel output.

    Returns:
        Dict with:
          - ``"six_channel"``  : np.ndarray (6, H, W) float32 — feed to model
          - ``"vv"``           : np.ndarray (1, H, W) — VV polarisation
          - ``"vh"``           : np.ndarray (1, H, W) — VH polarisation
          - ``"working_path"`` : path used (original or GCP-repaired)
          - ``"validation"``   : validation report dict
          - ``"ready"``        : bool — True if safe to feed to model

    Example::

        result = prepare_sar_for_ai("s1_normalised.tif")
        if result["ready"]:
            logits = prithvi_model(result["six_channel"])
    """
    import rasterio

    from pygeovision.models.adapters.sar_channel_manager import sar_to_hls_6ch

    # Step 1: GCP recovery
    if fix_georeference_first:
        try:
            working_path = fix_georeference(file_path)
        except ValueError as e:
            return {
                "ready": False, "error": str(e),
                "working_path": file_path,
            }
    else:
        working_path = file_path

    # Step 2: Read VV + VH
    with rasterio.open(working_path) as src:
        data = src.read().astype("float32")
        meta = {
            "crs":       str(src.crs),
            "transform": src.transform,
            "shape":     (src.count, src.height, src.width),
            "pixel_m":   abs(src.transform.a),
        }

    if data.shape[0] < 2:
        return {
            "ready": False,
            "error": f"Expected 2 bands (VV, VH), got {data.shape[0]}",
            "working_path": working_path,
        }

    vv = data[0:1]   # (1, H, W)
    vh = data[1:2]   # (1, H, W)

    # Step 3: 6-channel mapping
    six_ch = sar_to_hls_6ch(vv, vh, mapping=channel_mapping)

    # Step 4: Validate
    validation = {"valid": True, "warnings": [], "errors": []}
    if validate:
        validator = DataValidator()
        # Create a temp file for validation
        import tempfile

        import rasterio
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
            tmp_path = tmp.name
        with rasterio.open(working_path) as ref:
            profile = ref.profile.copy()
        profile.update(count=6, dtype="float32")
        with rasterio.open(tmp_path, "w", **profile) as dst:
            dst.write(six_ch)
        _, validation = validator.validate_for_ai(
            tmp_path, model_type="segmentation", expected_bands=6
        )
        pathlib.Path(tmp_path).unlink(missing_ok=True)

    return {
        "six_channel":   six_ch,
        "vv":            vv,
        "vh":            vh,
        "working_path":  working_path,
        "channel_mapping": channel_mapping,
        "metadata":      meta,
        "validation":    validation,
        "ready":         validation.get("valid", False),
    }


def prepare_optical_for_ai(
    file_path:       str,
    expected_bands:  int = 6,
    expected_range:  tuple[float, float] = (0.0, 1.0),
    fix_georeference_first: bool = False,
) -> dict[str, Any]:
    """
    Validate and prepare an optical raster for AI inference.

    pygeofetch handles atmospheric correction, cloud masking, clipping,
    and normalisation. This function validates the result and returns
    the AI-ready array.

    Args:
        file_path:      Path to preprocessed optical GeoTIFF.
        expected_bands: Expected band count (6 for HLS/Sentinel-2 subset).
        expected_range: Expected value range after normalisation.
        fix_georeference_first: Run GCP recovery (usually not needed for optical).

    Returns:
        Dict with ``"array"`` (C, H, W), ``"validation"``, ``"ready"``.
    """

    working_path = file_path
    if fix_georeference_first:
        try:
            working_path = fix_georeference(file_path)
        except ValueError as exc:
            logger.warning(
                "fix_georeference_first=True was requested but the fix "
                "failed (%s) — proceeding with the ORIGINAL file, which "
                "may still have a broken/corrupted georeference.", exc,
            )

    validator = DataValidator()
    arr, report = validator.validate_for_ai(
        working_path,
        model_type="segmentation",
        expected_bands=expected_bands,
        expected_range=expected_range,
    )

    return {
        "array":        arr,
        "working_path": working_path,
        "validation":   report,
        "ready":        report.get("valid", False) and arr is not None,
    }


# ── 4. Full acquisition → AI pipeline (convenience wrapper) ───────────────────

class EOPipeline:
    """
    End-to-end Earth observation pipeline from search to AI inference.

    Orchestrates pygeofetch (acquisition, preprocessing) and pygeovision
    (AI readiness, model inference) with a clean, fluent interface.

    Usage::

        pipeline = EOPipeline(provider="planetary_computer")
        pipeline.authenticate("copernicus", username="...", password="...")

        # Option A: fluent chain
        result = (
            pipeline
            .search(bbox, date_range, satellite="Sentinel-1")
            .select(n=1)
            .download(output_dir="./data/", bands=["vv","vh"])
            .despeckle()
            .calibrate_db()
            .clip(bbox)
            .normalise("minmax")
            .prepare_for_ai(mapping="physics_guided")
        )
        arr = result.six_channel   # (6, H, W) float32, ready for model

        # Option B: step-by-step (same result)
        scenes    = pipeline.search(bbox, date_range, satellite="Sentinel-1")
        downloads = pipeline.download(scenes[:1], "./data/", bands=["vv","vh"])
        ai_input  = pipeline.prepare_sar(downloads[0].path)
    """

    def __init__(self, provider: str = "planetary_computer"):
        from pygeovision.data.acquire import SatelliteAcquirer
        self._acq      = SatelliteAcquirer()
        self._provider = provider
        self._scenes:    list = []
        self._downloads: list = []
        self._current_path: str | None = None

    def authenticate(self, provider: str, **kwargs) -> EOPipeline:
        """Add credentials for a provider."""
        self._acq.add_credentials(provider, **kwargs)
        return self

    def search(
        self, bbox, date_range=None, satellite=None,
        cloud_cover_max=100, collections=None,
        sar_product_type=None, polarisation=None,
        providers=None,
    ) -> EOPipeline:
        """Search for scenes. Chainable."""
        self._scenes = self._acq.search(
            bbox=bbox, date_range=date_range, satellite=satellite,
            providers=providers or [self._provider],
            cloud_cover_max=cloud_cover_max, collections=collections,
            sar_product_type=sar_product_type, polarisation=polarisation,
        )
        logger.info("EOPipeline.search → %d scenes", len(self._scenes))
        return self

    def select(self, n: int = 1, strategy: str = "best") -> EOPipeline:
        """Select n scenes from search results."""
        self._scenes = self._scenes[:n]
        return self

    def download(
        self, scenes=None, output_dir="./eo_data/",
        bands=None, post_process=None,
    ) -> EOPipeline:
        """Download scenes. Chainable."""
        target = scenes or self._scenes
        self._downloads = self._acq.download(
            target, output_dir=output_dir,
            bands=bands, post_process=post_process,
        )
        successful = [d for d in self._downloads if d.success and d.path]
        if successful:
            self._current_path = str(successful[0].path)
        return self

    def preprocess(self, **kwargs) -> EOPipeline:
        """Apply preprocessing chain via pygeofetch. Chainable."""
        if not self._current_path:
            raise ValueError("No downloaded file — call download() first")
        result = self._acq.preprocess(self._current_path, **kwargs)
        if result.success:
            self._current_path = result.output_path
        return self

    def prepare_sar(self, path: str = None, mapping: str = "physics_guided") -> dict:
        """
        Prepare SAR data for AI inference (GCP fix + 6-channel mapping).
        Terminal operation — returns the AI-ready dict.
        """
        p = path or self._current_path
        if not p:
            raise ValueError("No path to prepare — call download() first")
        return prepare_sar_for_ai(p, channel_mapping=mapping)

    def prepare_optical(self, path: str = None, expected_bands: int = 6) -> dict:
        """
        Prepare optical data for AI inference.
        Terminal operation — returns the AI-ready dict.
        """
        p = path or self._current_path
        if not p:
            raise ValueError("No path to prepare — call download() first")
        return prepare_optical_for_ai(p, expected_bands=expected_bands)

    @property
    def scenes(self):
        return self._scenes

    @property
    def downloads(self):
        return self._downloads

    @property
    def current_path(self):
        return self._current_path