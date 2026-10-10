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

Sentinel-2 processing baseline 04.00 (products generated from 2022-01-25
on, including the later reprocessing of the 2015-2021 archive) stores
reflectance as DN = 10000 * reflectance + 1000. Reflectance is therefore
(DN - 1000) / 10000 for those scenes and DN / 10000 for older ones; see
``sentinel2_boa_add_offset``. Ignoring the offset adds 0.1 to every band,
which is large for dark targets (water) and flattens every normalised
difference index.
"""
from __future__ import annotations

import logging
import re
from datetime import date
from pathlib import Path

from pygeovision.data._profile import writable_profile

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
    "rededge1": ("B05", "rededge1", "re1"),  # 705nm -- Sentinel-2 only, no Landsat equivalent
    "nir":   ("B08", "nir", "nir08"),
    "swir1": ("B11", "swir1", "swir16"),
    "swir2": ("B12", "swir2", "swir22"),
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
    "swir1": ("SR_B5",), "swir2": ("SR_B7",), "thermal": ("ST_B6",),
}
LANDSAT_OLI_ALIASES = {      # Landsat 8/9 OLI/TIRS
    "blue": ("SR_B2",), "green": ("SR_B3",), "red": ("SR_B4",), "nir": ("SR_B5",),
    "swir1": ("SR_B6",), "swir2": ("SR_B7",), "thermal": ("ST_B10",),
}

# Real Landsat Collection 2 Level-2 Surface Temperature scale/offset
# (USGS Landsat Collection 2 Level-2 Science Product Guide):
# temperature_K = DN * LANDSAT_ST_SCALE + LANDSAT_ST_OFFSET. This is a
# genuinely different formula from the SR (reflectance) scale/offset
# above -- thermal DN encodes Kelvin, not reflectance, so it must never
# go through the same apply_scale path as SR_* bands.
LANDSAT_ST_SCALE = 0.00341802
LANDSAT_ST_OFFSET = 149.0


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

# Sentinel-2 L2A processing baseline 04.00 (operational 2022-01-25) added a
# +1000 DN offset to the stored values (the product's BOA_ADD_OFFSET).
SENTINEL2_BOA_ADD_OFFSET_DN = 1000
SENTINEL2_BASELINE_04_START = date(2022, 1, 25)

_DT_TOKEN = re.compile(r"^(\d{4})(\d{2})(\d{2})T\d{6}$")
_DATE_TOKEN = re.compile(r"^(\d{4})(\d{2})(\d{2})$")


def _token_date(token: str) -> date | None:
    m = _DT_TOKEN.match(token) or _DATE_TOKEN.match(token)
    if not m:
        return None
    try:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except ValueError:
        return None


def sentinel2_boa_add_offset(scene_id: str) -> int:
    """DN offset to subtract before dividing by 10000 for a Sentinel-2 L2A scene.

    Returns 1000 for processing baseline >= 04.00 and 0 for older products. The
    baseline is read from the scene id, best source first:

    1. an ``_N0510_`` baseline token (ESA product names);
    2. the product-generation timestamp (last token of a Planetary Computer id,
       ``S2B_MSIL2A_<sensing>_R094_T30UXC_<generated>``): every product generated
       on or after 2022-01-25 is baseline 04.00 or newer;
    3. the acquisition date (Earth Search ids, ``S2B_30UXC_20240626_0_L2A``):
       acquired on or after 2022-01-25 means baseline 04.00 or newer.

    An older acquisition whose id carries neither a baseline nor a generation
    time is treated as offset-free, as it was before this existed.
    """
    sid = scene_id or ""
    m = re.search(r"_N(\d{2})(\d{2})(?:_|\.|$)", sid)
    if m:
        return SENTINEL2_BOA_ADD_OFFSET_DN if int(m.group(1)) >= 4 else 0

    tokens = [t for t in re.split(r"[_.]", sid) if t]
    stamps = [(i, _token_date(t)) for i, t in enumerate(tokens) if _DT_TOKEN.match(t)]
    if len(stamps) >= 2 and stamps[-1][1] is not None:
        return SENTINEL2_BOA_ADD_OFFSET_DN if stamps[-1][1] >= SENTINEL2_BASELINE_04_START else 0

    for t in tokens[1:]:
        d = _token_date(t)
        if d is not None and d.year >= 2015:
            return SENTINEL2_BOA_ADD_OFFSET_DN if d >= SENTINEL2_BASELINE_04_START else 0
    return 0


def dn_to_reflectance(dn, scene_id: str, *, is_landsat: bool):
    """Raw digital numbers -> surface reflectance (float32, 0-1), NaN where no data.

    ``dn`` is (bands, rows, cols). A pixel with DN 0 in any band is fill (both
    Sentinel-2 and Landsat use 0), so it becomes NaN in every band instead of a
    real-looking reflectance of 0.
    """
    import numpy as np

    raw = np.asarray(dn)
    fill = (raw == 0).any(axis=0)
    out = raw.astype(np.float32)
    if is_landsat:
        out = out * LANDSAT_SR_SCALE + LANDSAT_SR_OFFSET
    else:
        out = (out - sentinel2_boa_add_offset(scene_id)) * SENTINEL2_SR_SCALE
    out = np.clip(out, 0.0, 1.0)
    out[:, fill] = np.nan
    return out

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

    # Where the scene really holds data (DN 0 / nodata is outside the swath or the tile), recorded before any
    # cloud masking so that "no imagery here" can be told from "cloud here". Written next to the stack.
    footprint = (stacked != 0)
    if np.issubdtype(stacked.dtype, np.floating):
        footprint &= np.isfinite(stacked)
    nodata_value = ref_profile.get("nodata")
    if nodata_value is not None and nodata_value == nodata_value and nodata_value != 0:
        footprint &= (stacked != nodata_value)
    footprint = footprint.any(axis=0)
    footprint_profile = writable_profile(ref_profile, count=1, dtype="uint8", nodata=None, compress="lzw")

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
        stacked = dn_to_reflectance(stacked, scene_id, is_landsat=is_landsat)
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

    ref_profile = writable_profile(ref_profile, count=len(selected), dtype=out_dtype, nodata=out_nodata)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    footprint_path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(footprint_path(output_path), "w", **footprint_profile) as fp_dst:
        fp_dst.write(footprint.astype("uint8"), 1)
    with rasterio.open(output_path, "w", **ref_profile) as dst:
        dst.write(stacked)
        for i, band in enumerate(bands):
            dst.set_band_description(i + 1, band)
        if apply_scale and is_sentinel2:
            dst.update_tags(
                reflectance_scale=str(SENTINEL2_SR_SCALE),
                boa_add_offset_dn=str(sentinel2_boa_add_offset(scene_id)),
            )

    return output_path


def footprint_path(stack_path: Path | str) -> Path:
    """The data-footprint raster of a stacked scene, in a ``footprints`` folder beside it (1 = data there)."""
    p = Path(stack_path)
    return p.parent / "footprints" / p.name


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
    import numpy as np
    from rasterio.warp import transform_bounds

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

        # Every pixel the box touches is kept, with its data (a masked crop blanked the edge row/column that the
        # box only partly covers, which showed up as a strip of missing data along the AOI border).
        from rasterio.windows import Window, from_bounds
        win = from_bounds(left, bottom, right, top, transform=src.transform)
        eps = 1e-6
        c0 = max(0, int(np.floor(win.col_off + eps)))
        r0 = max(0, int(np.floor(win.row_off + eps)))
        c1 = min(src.width, int(np.ceil(win.col_off + win.width - eps)))
        r1 = min(src.height, int(np.ceil(win.row_off + win.height - eps)))
        window = Window(c0, r0, max(1, c1 - c0), max(1, r1 - r0))
        cropped = src.read(window=window)
        cropped_transform = src.window_transform(window)
        out_profile = writable_profile(
            src.profile, height=cropped.shape[1], width=cropped.shape[2], transform=cropped_transform,
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

def real_pixel_area_ha(bounds, width: int, height: int, crs) -> float:
    """Real, geodesically-accurate per-pixel area in hectares.

    Real bug this fixes: computing area as abs(res[0]*res[1])/10000
    is only correct when the raster's resolution is genuinely in
    meters. The real, production _search_and_download/crop_to_bbox
    pipeline always outputs EPSG:4326 (degrees) -- treating a degree-
    based pixel width as if it were meters gives an area wrong by
    roughly (111,320)^2 ~= 1.2e10x, confirmed directly (a real test
    scene produced "4.7 billion vehicles per km2").

    For a geographic (degree-based) CRS, uses pyproj.Geod for real
    geodesic polygon area (accounts for Earth's curvature and
    latitude-dependent longitude compression), not a flat-Earth
    approximation. For an already-projected (meters-based) CRS, the
    simple resolution-product formula is genuinely correct and used
    directly.
    """
    is_geographic = False
    if crs is not None:
        try:
            is_geographic = bool(crs.is_geographic)
        except Exception:
            is_geographic = False

    if not is_geographic:
        res_x = abs(bounds.right - bounds.left) / width
        res_y = abs(bounds.top - bounds.bottom) / height
        return (res_x * res_y) / 10000

    from pyproj import Geod
    geod = Geod(ellps="WGS84")
    lons = [bounds.left, bounds.right, bounds.right, bounds.left]
    lats = [bounds.bottom, bounds.bottom, bounds.top, bounds.top]
    total_area_m2, _ = geod.polygon_area_perimeter(lons, lats)
    total_area_m2 = abs(total_area_m2)
    n_pixels = width * height
    if n_pixels == 0:
        return 0.0
    return (total_area_m2 / n_pixels) / 10000