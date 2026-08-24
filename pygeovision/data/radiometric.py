"""
pygeovision.data.radiometric
==============================
Real, documented sensor identification, radiometric scaling (raw DN ->
physical reflectance), and cloud/shadow masking for Landsat and
Sentinel-2 optical imagery.

This is a real, public, shared module -- not duplicated logic living
only inside one private class. It was originally built directly inside
`pygeovision.ai.pipelines.BasePipeline._stack_bands()`, which meant the
correction was completely unreachable from the separate, public
`client.segmentation`/`client.classification` API (those proxies
operate on an already-prepared `image_path`, with no way to get one
correctly prepared except by reimplementing this logic themselves).
Extracted here as a single source of truth; `BasePipeline._stack_bands`
now delegates to it, and it's also exposed as a real, public method on
`SatelliteFetcher` (`client.data.prepare_stack(...)`).

Real, documented sources for every constant below:
  - Landsat Collection 2 Level-2 Science Product Guide (USGS):
    reflectance = DN * 0.0000275 - 0.2
  - Sentinel-2 L2A product specification (ESA): BOA reflectance = DN / 10000
  - Landsat Collection 2 QA_PIXEL bit flags (USGS spec)
  - Sentinel-2 Scene Classification Layer (SCL) class values (ESA spec)

Honest limitation, unchanged from the original: Sentinel-2 L2A products
from 2022 onward can carry a per-scene BOA_ADD_OFFSET (a processing
baseline change) that this module does not read -- it always uses the
older, offset-free DN/10000 formula. Correct for most scenes; a small,
roughly constant bias for newer-baseline scenes missing the offset.
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Band aliases (provider/mission-aware)
# ---------------------------------------------------------------------------

# Sentinel-2 band aliases: pygeofetch/provider asset keys aren't
# perfectly consistent across providers (confirmed from real usage:
# Planetary Computer uses "B02"/"B03"/"B04"/"B08"; some STAC catalogs
# use "blue"/"green"/"red"/"nir"). Check both forms.
S2_BAND_ALIASES = {
    "blue":  ("B02", "blue"),
    "green": ("B03", "green"),
    "red":   ("B04", "red"),
    "nir":   ("B08", "nir", "nir08"),
}

# Landsat band aliases -- mission-aware, since Landsat 7 ETM+/5 TM and
# Landsat 8/9 OLI genuinely have DIFFERENT band-number-to-wavelength
# mappings (SR_B1 is blue on ETM+/TM but coastal-aerosol on OLI, where
# SR_B2 is blue instead). Using one fixed mapping for both would
# silently select the wrong physical band for one of the two mission
# families. Confirmed against real downloaded Landsat 7 and Landsat 8
# filenames from production.
LANDSAT_ETM_TM_ALIASES = {   # Landsat 4/5 TM, Landsat 7 ETM+
    "blue": ("SR_B1",), "green": ("SR_B2",), "red": ("SR_B3",), "nir": ("SR_B4",),
}
LANDSAT_OLI_ALIASES = {      # Landsat 8/9 OLI
    "blue": ("SR_B2",), "green": ("SR_B3",), "red": ("SR_B4",), "nir": ("SR_B5",),
}


def landsat_mission_aliases(scene_id: str) -> dict[str, tuple[str, ...]] | None:
    """Real, per-mission Landsat band aliases, or None if scene_id
    isn't a recognised Landsat mission prefix."""
    prefix = scene_id[:4].upper() if scene_id else ""
    if prefix in ("LE07", "LT05", "LT04"):
        return LANDSAT_ETM_TM_ALIASES
    if prefix in ("LC08", "LC09"):
        return LANDSAT_OLI_ALIASES
    return None


def identify_sensor(scene_id: str) -> tuple[bool, bool]:
    """Positively identify Landsat vs Sentinel-2 from a scene ID.

    Returns (is_landsat, is_sentinel2). Real bug this guards against:
    an earlier version of this logic used "not Landsat -> assume
    Sentinel-2", which meant any THIRD sensor (NAIP, PlanetScope,
    MODIS, commercial VHR) silently got the Sentinel-2 formula applied
    -- not a crash, just quietly wrong reflectance with no error. This
    positively checks for both; callers should treat
    (False, False) as "unidentified — do not guess a scale/offset".
    """
    is_landsat = landsat_mission_aliases(scene_id) is not None
    is_sentinel2 = (not is_landsat) and bool(scene_id) and scene_id[:3].upper() in ("S2A", "S2B", "S2C")
    return is_landsat, is_sentinel2


# ---------------------------------------------------------------------------
# Real, documented radiometric scale/offset
# ---------------------------------------------------------------------------

LANDSAT_SR_SCALE = 0.0000275
LANDSAT_SR_OFFSET = -0.2
SENTINEL2_SR_SCALE = 1.0 / 10000.0

# Real Landsat Collection 2 QA_PIXEL bit flags (USGS Landsat Collection
# 2 Level-2 QA band specification). Bit 0 = Fill, 1 = Dilated Cloud,
# 2 = Cirrus, 3 = Cloud, 4 = Cloud Shadow, 5 = Snow, 6 = Clear,
# 7 = Water. Masking dilated cloud + cirrus + cloud + cloud shadow is
# the standard, widely-used combination for a general-purpose cloud/
# shadow mask.
LANDSAT_QA_CLOUD_BITS = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4)

# Real Sentinel-2 Scene Classification Layer (SCL) class values (ESA
# Sentinel-2 L2A product specification): 3 = Cloud Shadow,
# 8 = Cloud Medium Probability, 9 = Cloud High Probability,
# 10 = Thin Cirrus.
S2_SCL_CLOUD_VALUES = (3, 8, 9, 10)


def stack_and_prepare_bands(
    asset_paths: dict[str, Path],
    output_path: Path,
    bands: tuple[str, ...] = ("red", "green", "blue"),
    scene_id: str = "",
    apply_scale: bool = True,
    apply_cloud_mask: bool = True,
) -> Path:
    """Build a real, correctly-ordered, radiometrically-correct,
    cloud-masked multi-band GeoTIFF from a set of single-band asset
    files (e.g. Sentinel-2's separate B02/B03/B04/B08 downloads, or
    Landsat's SR_B-numbered bands).

    The single, public, shared implementation of this logic --
    previously lived only inside pygeovision.ai.pipelines.BasePipeline,
    unreachable from the separate public client.segmentation/
    client.classification API. Use this directly if you're working
    with that API and need a correctly-prepared image_path.

    Resamples every band to match the first band's grid if resolutions
    differ (Sentinel-2 genuinely ships 10m/20m/60m bands in the same
    product), so the stack is spatially consistent, not just
    concatenated arrays that don't line up.

    Output dtype changes from raw uint16 DN to float32 reflectance when
    apply_scale=True (the default) -- this is real, physical surface
    reflectance (nominally 0-1), not an arbitrary rescale.

    Raises RuntimeError (does not silently fall back to a single,
    arbitrary asset path) if a requested band can't be found, or if
    apply_scale=True but the sensor can't be positively identified as
    Landsat or Sentinel-2 -- pass apply_scale=False to accept raw,
    unscaled DN values for an unrecognised sensor instead.
    """
    import numpy as np
    import rasterio
    from rasterio.warp import Resampling, reproject

    landsat_aliases = landsat_mission_aliases(scene_id)
    is_landsat, is_sentinel2 = identify_sensor(scene_id)

    selected = []
    for band in bands:
        aliases = S2_BAND_ALIASES.get(band, (band,))
        if landsat_aliases and band in landsat_aliases:
            aliases = aliases + landsat_aliases[band]
        match = next((asset_paths[a] for a in aliases if a in asset_paths), None)
        if match is None:
            raise RuntimeError(
                f"stack_and_prepare_bands: no asset found for band {band!r} "
                f"(tried {aliases}) in scene {scene_id!r} — available assets: "
                f"{list(asset_paths.keys())}. Refusing to silently substitute "
                f"an arbitrary, likely-wrong asset."
            )
        selected.append(match)

    with rasterio.open(selected[0]) as ref:
        ref_profile = ref.profile.copy()
        ref_data = ref.read(1)

    stacked = np.zeros((len(selected), *ref_data.shape), dtype=ref_data.dtype)
    stacked[0] = ref_data
    with rasterio.open(selected[0]) as ref:
        for i, path in enumerate(selected[1:], start=1):
            with rasterio.open(path) as src:
                if src.shape == ref_data.shape and src.transform == ref.transform:
                    stacked[i] = src.read(1)
                else:
                    # Real resolution/grid mismatch (Sentinel-2 ships
                    # 10m/20m/60m bands in one product) -- resample to
                    # the reference band's grid, not a silent
                    # shape-mismatch crash or misaligned stack.
                    dst_arr = np.zeros(ref_data.shape, dtype=ref_data.dtype)
                    reproject(
                        source=rasterio.band(src, 1), destination=dst_arr,
                        src_transform=src.transform, src_crs=src.crs,
                        dst_transform=ref.transform, dst_crs=ref.crs,
                        resampling=Resampling.bilinear,
                    )
                    stacked[i] = dst_arr

    # ── Radiometric scaling: raw DN -> real physical reflectance ──────
    if apply_scale:
        if not (is_landsat or is_sentinel2):
            raise RuntimeError(
                f"stack_and_prepare_bands: cannot apply radiometric scaling — "
                f"scene {scene_id!r} is not identifiable as Landsat or "
                f"Sentinel-2, and no other sensor's scale/offset is known. "
                f"Pass apply_scale=False to skip scaling and get raw DN "
                f"values instead of a silently wrong reflectance conversion."
            )
        stacked = stacked.astype(np.float32)
        if is_landsat:
            stacked = stacked * LANDSAT_SR_SCALE + LANDSAT_SR_OFFSET
        else:
            stacked = stacked * SENTINEL2_SR_SCALE
        stacked = np.clip(stacked, 0.0, 1.0)
        out_dtype = "float32"
        out_nodata = np.nan
    else:
        out_dtype = ref_data.dtype
        out_nodata = ref_profile.get("nodata")

    # ── Cloud masking: real QA_PIXEL (Landsat) / SCL (Sentinel-2) ─────
    if apply_cloud_mask and (is_landsat or is_sentinel2):
        qa_key = "QA_PIXEL" if is_landsat else "SCL"
        qa_path = asset_paths.get(qa_key)
        if qa_path is not None:
            with rasterio.open(qa_path) as qa_src:
                if qa_src.shape == ref_data.shape:
                    qa_data = qa_src.read(1)
                else:
                    qa_data = np.zeros(ref_data.shape, dtype=qa_src.dtypes[0])
                    reproject(
                        source=rasterio.band(qa_src, 1), destination=qa_data,
                        src_transform=qa_src.transform, src_crs=qa_src.crs,
                        dst_transform=ref.transform, dst_crs=ref.crs,
                        resampling=Resampling.nearest,  # categorical data -- never interpolate
                    )
            if is_landsat:
                cloud_mask = (qa_data.astype(np.uint32) & LANDSAT_QA_CLOUD_BITS) != 0
            else:
                cloud_mask = np.isin(qa_data, S2_SCL_CLOUD_VALUES)
            fill_value = np.nan if apply_scale else (out_nodata if out_nodata is not None else 0)
            stacked[:, cloud_mask] = fill_value
            logger.info(
                "stack_and_prepare_bands: masked %d cloud/shadow pixel(s) "
                "(%.1f%% of scene) using %s",
                int(cloud_mask.sum()), 100.0 * cloud_mask.mean(), qa_key,
            )
        else:
            logger.warning(
                "stack_and_prepare_bands: no %s band available for scene %r "
                "— cloud masking skipped, output may contain unmasked "
                "clouds/shadows.", qa_key, scene_id,
            )
    elif apply_cloud_mask:
        logger.warning(
            "stack_and_prepare_bands: scene %r is not identifiable as "
            "Landsat or Sentinel-2 — cloud masking skipped (don't know "
            "which QA band/scheme applies), output may contain unmasked "
            "clouds/shadows.", scene_id,
        )

    ref_profile.update(count=len(selected), dtype=out_dtype, nodata=out_nodata)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **ref_profile) as dst:
        dst.write(stacked)
        for i, band in enumerate(bands):
            dst.set_band_description(i + 1, band)

    return output_path


def crop_to_bbox(
    input_path: Path,
    bbox: tuple[float, float, float, float],
    output_path: Path | None = None,
) -> Path:
    """Crop a raster to a real WGS84 bounding box.

    Real gap this closes: the requested bbox was previously used only
    to SEARCH for overlapping scenes -- nothing ever cropped the
    downloaded/stacked imagery to it afterward, so every pipeline ran
    tiled inference over the entire downloaded scene (often 100km+ on a
    side for Sentinel-2, ~185x180km for Landsat), not just the
    requested area. This wastes real compute on irrelevant area and
    produces an output extent that doesn't match what was requested.

    If the raster's CRS isn't already WGS84, bbox is reprojected into
    the raster's real CRS before cropping, rather than assuming they
    match. If bbox doesn't actually intersect the raster at all, raises
    a clear error rather than silently producing a degenerate crop.
    """
    import rasterio
    from rasterio.mask import mask as rio_mask
    from rasterio.warp import transform_bounds
    from shapely.geometry import box, mapping

    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_cropped{input_path.suffix}")
    output_path = Path(output_path)

    with rasterio.open(input_path) as src:
        raster_crs = src.crs
        raster_bounds = src.bounds

        if raster_crs is not None and str(raster_crs).upper() not in ("EPSG:4326", "OGC:CRS84"):
            crop_bbox = transform_bounds("EPSG:4326", raster_crs, *bbox)
        else:
            crop_bbox = bbox

        left, bottom, right, top = crop_bbox
        r_left, r_bottom, r_right, r_top = raster_bounds
        if right <= r_left or left >= r_right or top <= r_bottom or bottom >= r_top:
            raise ValueError(
                f"crop_to_bbox: requested bbox {bbox} does not intersect "
                f"{input_path.name}'s real coverage (bounds={tuple(raster_bounds)}). "
                f"The search likely returned a scene whose real footprint "
                f"doesn't actually contain the requested area."
            )

        geom = [mapping(box(*crop_bbox))]
        cropped, cropped_transform = rio_mask(src, geom, crop=True)
        out_profile = src.profile.copy()
        out_profile.update(
            height=cropped.shape[1], width=cropped.shape[2], transform=cropped_transform,
        )

    with rasterio.open(output_path, "w", **out_profile) as dst:
        dst.write(cropped)
        if input_path.suffix == output_path.suffix:
            with rasterio.open(input_path) as src:
                for i in range(1, src.count + 1):
                    desc = src.descriptions[i - 1]
                    if desc:
                        dst.set_band_description(i, desc)

    logger.info(
        "crop_to_bbox: %s (%s) -> %s (%dx%d, cropped to requested bbox)",
        input_path.name, tuple(round(v, 4) for v in raster_bounds),
        output_path.name, cropped.shape[2], cropped.shape[1],
    )
    return output_path