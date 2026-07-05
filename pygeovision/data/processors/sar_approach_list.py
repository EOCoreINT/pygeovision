"""
pygeovision.data.processors.sar_approach_list
===============================================
The definitive approach list for SAR preprocessing in PyGeoVision, updated
to incorporate the three bug fixes and the new domain-adaptation workflow
for Prithvi and DINOv3.

This module is primarily documentation — it defines the canonical steps as
a structured list that notebooks and pipeline code can reference. Each step
includes the rationale, the function to call, and the known failure modes
from real processing logs.

Usage::

    from pygeovision.data.processors.sar_approach_list import SAR_APPROACH_LIST
    for step in SAR_APPROACH_LIST:
        print(f"[{step['id']}] {step['name']}: {step['rationale'][:80]}...")

For the NB02 (Disaster Management / Turkey Earthquake) notebook, see
``get_nb02_approach_context()`` which returns the full context block to
inject as a notebook markdown cell.
"""
from __future__ import annotations
from typing import List, Dict, Any

SAR_APPROACH_LIST: List[Dict[str, Any]] = [
    # ──────────────────────────────────────────────────────────────────────
    # PRE-PROCESSING PHASE
    # ──────────────────────────────────────────────────────────────────────
    {
        "id": "S0",
        "phase": "pre_processing",
        "name": "Download completeness verification",
        "location": "pygeovision.data.processors.sar.verify_sar_downloads",
        "when": "Immediately after every SAR asset download, before any other step",
        "rationale": (
            "Partial downloads (network timeout mid-transfer) leave a file on disk "
            "that passes os.path.exists() but fails when rasterio reads a tile from "
            "the truncated region. This caused 'TIFFFillTile: Read error at row 21504' "
            "in the Turkey earthquake preprocessing logs. The check must happen BEFORE "
            "any processing step attempts to open the file — otherwise the error surfaces "
            "inside despeckle or calibration with an opaque exception instead of a clear "
            "'download incomplete' message."
        ),
        "bug_reference": "BUG 2: Partial download → corrupt TIFF → despeckle failure",
        "inputs": ["vv_path", "vh_path"],
        "outputs": ["verified_paths: dict[pol -> Path | None]"],
        "failure_mode": "vh download failed → fallback to VV-only (logged as warning)",
        "code_example": """
from pygeovision.data.processors.sar import verify_sar_downloads
paths = verify_sar_downloads({"vv": "iw-vv.tiff", "vh": "iw-vh.tiff"},
                              fallback_policy="vv_only")
# paths["vh"] is None if download failed; VV-only pipeline continues
""",
    },
    {
        "id": "S1",
        "phase": "pre_processing",
        "name": "Georeference validation (post-reproject)",
        "location": "pygeovision.data.processors.sar.validate_sar_georeference",
        "when": "Immediately after PyGeoFetch post_process=['reproject:EPSG:32637','cog']",
        "rationale": (
            "The PyGeoFetch reproject step produces output with an identity affine "
            "transform (a=1.0, origin at 0,0) instead of real-world UTM coordinates. "
            "This is the root cause of 'Input shapes do not overlap raster' clip errors. "
            "Detection: pixel_width <= 1.0 m OR abs(origin) < 100 m. "
            "Repair: re-reproject from the raw download file using rasterio directly."
        ),
        "bug_reference": "BUG 1: CRS/transform corruption after reproject",
        "inputs": ["reprojected_path", "raw_path (for repair)"],
        "outputs": ["GeoreferenceResult: valid, repaired_path"],
        "failure_mode": "corrupt + raw_path missing → raise GeoreferenceCorruptError",
        "code_example": """
from pygeovision.data.processors.sar import validate_sar_georeference
result = validate_sar_georeference(
    "iw-vh_EPSG_32637.tiff",
    raw_path="iw-vh.tiff",   # if None, no auto-repair
)
if result.repaired:
    working_path = result.repaired_path   # use this instead
""",
    },
    {
        "id": "S2",
        "phase": "pre_processing",
        "name": "Thermal noise removal",
        "location": "pyroSAR / SNAP (Sentinel-1 Noise Removal operator)",
        "when": "After georeference validation; before radiometric calibration",
        "rationale": (
            "Sentinel-1 GRD products contain thermal noise artefacts (especially in "
            "VH cross-polarisation at the subswath boundaries). Removing the noise "
            "floor improves the signal-to-clutter ratio and prevents speckle filters "
            "from treating noise peaks as real scatter. "
            "PyGeoVision calls pyroSAR's apply_thermal_noise_removal() when available, "
            "or falls back to a simplified version when SNAP is not installed."
        ),
        "inputs": ["sar_path"],
        "outputs": ["noise_removed_path"],
        "notes": "Requires pyroSAR + ESA SNAP installation for full accuracy",
    },
    {
        "id": "S3",
        "phase": "pre_processing",
        "name": "Radiometric calibration → sigma-naught (linear)",
        "location": "pygeovision.data.processors.sar.SARPreprocessor._apply_calibration_if_needed",
        "when": "After thermal noise removal",
        "rationale": (
            "Converts raw Digital Numbers (DN) from the Sentinel-1 measurement to "
            "physically meaningful sigma-naught (σ0) in linear power scale. "
            "Formula: σ0 = DN^2 / CalibrationConstant (from the calibration LUT). "
            "All subsequent processing (despeckle, dB conversion, normalisation) must "
            "operate on physically-calibrated data, not raw DNs."
        ),
        "inputs": ["sar_path"],
        "outputs": ["sigma0_linear_path (float32, values ~3e-4 to 3.0)"],
    },
    {
        "id": "S4",
        "phase": "pre_processing",
        "name": "Terrain correction / RTC geocoding",
        "location": "pyroSAR / SNAP (Range-Doppler Terrain Correction)",
        "when": "After radiometric calibration",
        "rationale": (
            "Sentinel-1 GRD is in radar geometry (slant range) with geometric "
            "distortions (foreshortening, layover) in mountainous terrain. Terrain "
            "correction reprojects to a map grid (UTM) using a DEM. "
            "THIS IS WHERE THE REAL-WORLD UTM COORDINATES ARE EMBEDDED — if this "
            "step fails or is skipped, the file may end up with an identity transform "
            "(the BUG 1 condition). Validate georeference again immediately after."
        ),
        "inputs": ["sigma0_linear_path", "DEM (SRTM or Copernicus DEM)"],
        "outputs": ["RTC_geocoded_path (now in UTM with valid affine transform)"],
        "notes": "Requires DEM access; SRTM-1s (30m) is sufficient for Sentinel-1 IW",
    },
    {
        "id": "S5",
        "phase": "pre_processing",
        "name": "Georeference validation (post-RTC)",
        "location": "pygeovision.data.processors.sar.validate_sar_georeference",
        "when": "Immediately after terrain correction — CRITICAL checkpoint",
        "rationale": (
            "This is the MOST IMPORTANT georeference check because terrain correction "
            "is where real-world coordinates should first appear. If the RTC step "
            "failed silently (e.g. DEM not found, SNAP crash), the output may still "
            "have an identity transform. Catching this HERE — not downstream during "
            "the clip step — produces a clear error message with context."
        ),
        "bug_reference": "BUG 1 (second check): validates that RTC produced valid output",
        "inputs": ["RTC_geocoded_path"],
        "outputs": ["GeoreferenceResult: pixel_width_m should be ~10m for Sentinel-1 IW"],
    },
    {
        "id": "S6",
        "phase": "pre_processing",
        "name": "Despeckle (on LINEAR sigma-naught)",
        "location": "pygeovision.data.processors.sar.despeckle_sar",
        "when": "After RTC geocoding and AFTER the second georeference validation",
        "rationale": (
            "CRITICAL ORDER CONSTRAINT: despeckle MUST be applied to LINEARLY-SCALED "
            "sigma-naught, NOT to dB values. "
            "SAR speckle follows a Gamma/Rayleigh distribution in linear power scale. "
            "The Lee filter's statistical model (estimating local mean and variance) "
            "is derived for this distribution. Despckling dB values first corrupts "
            "the statistical assumptions and produces ring/block artefacts. "
            "Wrong order: download → calibrate → to_dB → despeckle → normalise "
            "Right order: download → calibrate → terrain_correction → DESPECKLE → to_dB → normalise"
        ),
        "inputs": ["sigma0_linear_path"],
        "outputs": ["despeckled_linear_path"],
        "filters": ["enhanced_lee (default)", "refined_lee", "boxcar"],
        "recommended_params": {"window_size": 7, "num_looks": 4},
        "failure_mode": (
            "If despeckle is called on a corrupt/incomplete file (BUG 2), it will "
            "raise TIFFFillTile error — hence S0 (download check) must come first."
        ),
    },
    {
        "id": "S7",
        "phase": "pre_processing",
        "name": "dB conversion (10 × log10(sigma0))",
        "location": "pygeovision.data.processors.sar.linear_to_db",
        "when": "After despeckle, before normalisation",
        "rationale": (
            "Converts linear sigma-naught to decibels. The dB scale is preferred for: "
            "(a) visualisation (more perceptually uniform), "
            "(b) normalisation (the physical range is well-bounded: -35 to +5 dB for S1), "
            "(c) change detection (dB differences are more Gaussian-distributed). "
            "Clip to [-35, +5] dB to remove extreme outliers from specular water "
            "reflection and corner reflectors."
        ),
        "inputs": ["despeckled_linear_path"],
        "outputs": ["db_path (float32, values in [-35, +5] dB)"],
    },
    {
        "id": "S8",
        "phase": "pre_processing",
        "name": "Normalise to [0, 1] for AI input",
        "location": "pygeovision.data.processors.sar.normalise_sar_for_ai",
        "when": "After dB conversion, before clip and model input",
        "rationale": (
            "Maps the dB range to [0, 1] using fixed physical bounds so that "
            "normalisation is scene-independent. Using percentile normalisation "
            "here would cause different normalisation between pre/post scenes in "
            "change detection pipelines, which the model could exploit as a "
            "spurious signal. minmax_db uses the known Sentinel-1 backscatter "
            "range and is consistent across all scenes."
        ),
        "inputs": ["db_path"],
        "outputs": ["normalised_path (float32, [0, 1])"],
        "methods": {
            "minmax_db": "Fixed physical bounds [-35, +5] dB (recommended, scene-independent)",
            "percentile": "2nd–98th percentile (adaptive, but inconsistent across scenes)",
            "zscore": "Zero mean, unit variance (for models expecting standardised input)",
        },
    },
    {
        "id": "S9",
        "phase": "pre_processing",
        "name": "CRS-aware clip to study area bbox",
        "location": "pygeovision.data.processors.sar.clip_sar_to_bbox",
        "when": "After normalisation; final preprocessing step before AI input",
        "rationale": (
            "BUG 3 FIX: clips the normalised SAR raster to the WGS84 study area "
            "bbox, but FIRST reprojects the bbox to the raster's native CRS. "
            "Without this reprojection, passing WGS84 lon/lat values to a UTM raster "
            "causes 'Input shapes do not overlap raster' because the coordinate spaces "
            "don't match — the raster is in UTM metres (e.g. 400000, 4000000) while "
            "the bbox is in degrees (e.g. 36.1, 36.4)."
        ),
        "bug_reference": "BUG 3: WGS84 bbox vs UTM raster → no overlap",
        "inputs": ["normalised_path", "bbox_wgs84 (lon_min, lat_min, lon_max, lat_max)"],
        "outputs": ["clipped_path"],
        "code_example": """
from pygeovision.data.processors.sar import clip_sar_to_bbox
# Pass bbox in WGS84 — function handles CRS reprojection automatically
clip_sar_to_bbox("iw-vh_norm.tif", "iw-vh_clipped.tif",
                  bbox_wgs84=(36.1, 36.2, 36.4, 36.6))
""",
    },

    # ──────────────────────────────────────────────────────────────────────
    # AI INFERENCE PHASE
    # ──────────────────────────────────────────────────────────────────────
    {
        "id": "A0",
        "phase": "ai_inference",
        "name": "Channel mapping / domain adaptation",
        "location": "pygeovision.models.adapters.sar_channel_manager",
        "when": "Before passing SAR arrays to any AI model",
        "rationale": (
            "AI models trained on optical imagery expect channels in a specific order "
            "and range (RGB for DINOv3, 6-band HLS for Prithvi). Feeding raw SAR "
            "channels at position 0 and 1 as if they were Blue and Green causes the "
            "model to 'see' radar backscatter where it expects optical reflectance, "
            "producing confused activations and poor results. "
            "The channel manager builds physically-motivated feature composites that "
            "occupy the expected channel positions with meaningful values."
        ),
        "adapters": {
            "prithvi": "sar_to_hls_6ch(vv, vh, mapping='physics_guided') → (6, H, W)",
            "dinov3": "sar_to_pseudo_rgb(vv, vh, arrangement='vv_vh_ratio') → (3, H, W)",
            "changeformer": "stack(vv, vh) → (2, H, W); co-register pre/post first",
        },
    },
    {
        "id": "A1",
        "phase": "ai_inference",
        "name": "Prithvi inference with SAR channel mapping",
        "location": "pygeovision.models.adapters.sar_prithvi.SARPrithviAdapter",
        "when": "For flood detection, land cover, change detection tasks",
        "rationale": (
            "Prithvi is trained on HLS multispectral data. For SAR, we map dual-pol "
            "features into 6 HLS-like positions using physically-motivated assignments: "
            "VH→Blue (volume scatter / veg proxy), (VV+VH)/2→Green, VV→Red, "
            "1-VH→NIR, 1-VV→SWIR1, VV/VH→SWIR2. "
            "For best results: fine-tune on Sen1Floods11 using TerraTorch. "
            "For quick prototyping: use mode='zero_shot' (VH threshold flood mask)."
        ),
        "modes": {
            "zero_shot": "No training. VH threshold flood mask. Fast, ~0.75 F1 on Sen1Floods11",
            "fine_tune": "TerraTorch + Sen1Floods11. Best accuracy (~0.92 F1). Requires GPU+data",
            "spt": "Scattering Prompt Tuning. Lightweight (~0.88 F1). Few labels needed",
        },
        "code_example": """
from pygeovision.models.adapters.sar_prithvi import SARPrithviAdapter
adapter = SARPrithviAdapter(mode="zero_shot", task="flood_detection")
result = adapter.run(vv_array, vh_array)
flood_mask = result["prediction"]   # (H, W) uint8: 1=flood, 0=non-flood
""",
    },
    {
        "id": "A2",
        "phase": "ai_inference",
        "name": "DINOv3 SAR feature extraction / classification",
        "location": "pygeovision.models.adapters.sar_dinov3.SARDINOv3Adapter",
        "when": "For ATR, terrain classification, vessel detection, general SAR features",
        "rationale": (
            "DINOv3 produces rich patch-level features that transfer well to SAR "
            "after domain adaptation. Key insight: use SARSpeckleAugmentation "
            "(multiplicative Gamma noise) instead of RGB colour jitter during "
            "fine-tuning — speckle is multiplicative in linear power scale, not "
            "additive, and treating it like colour jitter produces wrong augmentations "
            "that hurt feature quality."
        ),
        "preprocessing": "sar_to_pseudo_rgb → normalise_for_dino (ImageNet mean/std)",
        "fine_tuning": "Use SARSpeckleAugmentation, NOT colour jitter or Gaussian noise",
        "code_example": """
from pygeovision.models.adapters.sar_dinov3 import SARDINOv3Adapter
adapter = SARDINOv3Adapter(mode="zero_shot", model_size="vitl14")
features = adapter.extract_features(vv_array, vh_array)
# features["patch_features"] : (H//14, W//14, 1024) patch embeddings
# features["cls_token"]      : (1024,) global image descriptor
""",
    },
    {
        "id": "A3",
        "phase": "ai_inference",
        "name": "Pre/post co-registration for change detection",
        "location": "pygeovision.models.adapters.sar_channel_manager.coregister_sar_pair",
        "when": "Before any change detection model (ChangeFormer, post-flood damage)",
        "rationale": (
            "Change detection models compare pre and post pixel-to-pixel. If the "
            "PRE and POST scenes have slightly different UTM grids (different orbit "
            "pass, different tile edge alignment), the change signal is dominated by "
            "geometric misalignment rather than real change. "
            "The co-registration step reprojects the POST scene onto the PRE scene's "
            "exact grid (same transform, height, width, CRS) before stacking."
        ),
        "code_example": """
from pygeovision.models.adapters.sar_channel_manager import coregister_sar_pair
pre_path, post_path = coregister_sar_pair(pre_sar, post_sar, "./aligned/")
# Now pre_path and post_path are pixel-aligned; safe to subtract or stack
""",
    },
    {
        "id": "A4",
        "phase": "ai_inference",
        "name": "Input validation before every model call",
        "location": "pygeovision.models.adapters.sar_channel_manager.validate_sar_ai_input",
        "when": "After channel mapping, BEFORE passing to any model",
        "rationale": (
            "Silently wrong inputs (wrong dtype, NaN, values outside [0,1], wrong "
            "channel count) produce silently wrong model outputs — the model will "
            "run and produce a result but the result will be meaningless. "
            "Validation raises a clear error at the boundary between preprocessing "
            "and inference instead of propagating errors into the model."
        ),
        "code_example": """
from pygeovision.models.adapters.sar_channel_manager import validate_sar_ai_input
val = validate_sar_ai_input(six_ch_array, model="prithvi")
if not val["valid"]:
    raise ValueError(f"Fix SAR input before calling Prithvi: {val['errors']}")
""",
    },
]


def get_nb02_approach_context() -> str:
    """Return the complete approach context markdown string for NB02.

    This is the cell to inject at the top of the Turkey earthquake disaster
    notebook explaining the corrected SAR processing workflow.
    """
    return """
## NB02 — Turkey Earthquake SAR Analysis: Corrected Approach List

### Three bugs fixed from the original processing logs

#### Bug 1 — CRS/transform corruption after reproject [CRITICAL]
```
SYMPTOM: SAR PRE VH bounds: BoundingBox(left=0.0, bottom=0.0, right=26083.0, top=16700.0)
         Transform: | 1.00, 0.00, 0.00 |  ← identity transform, not UTM metres
         ERROR: clip failed: Input shapes do not overlap raster
FIX:     validate_sar_georeference() after every reproject step; auto-repair from raw
```

#### Bug 2 — Partial download → corrupt TIFF → despeckle failure [CRITICAL]
```
SYMPTOM: WARNING: Failed to download asset 'vh': The read operation timed out
         ERROR: TIFFFillTile: Read error at row 21504, col 16384
FIX:     verify_sar_downloads() before any processing step; VV-only fallback
```

#### Bug 3 — WGS84 bbox vs UTM raster → no overlap [CRITICAL]
```
SYMPTOM: ERROR: clip failed: Input shapes do not overlap raster
         (study area bbox in degrees; raster in UTM metres)
FIX:     clip_sar_to_bbox() auto-reprojects bbox to raster CRS before clipping
```

### Corrected pipeline order

| Step | Name | Key correctness note |
|---|---|---|
| S0 | **Download completeness** | Before ANY other step — catch partial downloads |
| S1 | **Georeference validation** | Immediately after PyGeoFetch reproject |
| S2 | Thermal noise removal | Before calibration |
| S3 | Radiometric calibration → σ0 linear | → sigma-naught in linear power scale |
| S4 | Terrain correction / RTC | Embeds real UTM coordinates |
| S5 | **Georeference validation (post-RTC)** | CRITICAL: check RTC produced valid coords |
| S6 | **Despeckle on LINEAR data** | NOT on dB — speckle is multiplicative in linear |
| S7 | dB conversion (10×log10) | After despeckle; clip to [-35, +5] dB |
| S8 | Normalise [0,1] | Fixed physical bounds, scene-independent |
| S9 | **CRS-aware clip** | Reproject bbox to raster CRS first — Bug 3 fix |
| A0 | Channel mapping | sar_to_hls_6ch for Prithvi; pseudo_rgb for DINOv3 |
| A1-A4 | AI inference | validate_sar_ai_input before every model call |

### SAR-to-AI model domain adaptation

| Model | Adapter | Zero-shot quality | Fine-tune dataset |
|---|---|---|---|
| Prithvi-EO-2.0 | SARPrithviAdapter | Moderate (0.75 F1) | Sen1Floods11 via TerraTorch |
| DINOv3-ViT-L/14 | SARDINOv3Adapter | Good features, poor pixel labels | Unlabelled S1 (SSL) |
| ChangeFormer | Direct (2-channel) | Good if co-registered | xBD, OSCD |
"""


if __name__ == "__main__":
    print(f"SAR approach list: {len(SAR_APPROACH_LIST)} steps")
    for step in SAR_APPROACH_LIST:
        print(f"  [{step['id']}] {step['phase']}: {step['name']}")
