"""Flood mapping that combines Sentinel-1 radar, Sentinel-2 optical imagery and terrain.

``multisensor_flood_mapping`` does not blend the sensors into one score. It keeps what each one says, shows where they
agree and disagree, and removes what the ground cannot hold:

* **Sentinel-1 radar** (``sar_flood_mapping``'s method: one orbit track, per-window median in dB, water threshold,
  change criterion) sees flooding through cloud, but misses floodwater between buildings and under vegetation.
* **Sentinel-2 optical** (MNDWI = (green - SWIR1) / (green + SWIR1), per-window median of the *clear* observations)
  sees water that radar can miss, but only where the sky was clear on a date of the window.
* **Terrain** (Copernicus GLO-30 DEM -> Height Above Nearest Drainage) removes "flooding" on ground that stands high
  above any drainage (radar shadow, dark surfaces on hills).

Output ``flood_extent.tif``:

====  ==============================================================================
0     not flooded
1     flooded, **confirmed** by radar and by a clear optical view
2     flooded, radar only (no clear optical view of the pixel on both windows)
3     flooded, optical only (radar did not flag it)
4     **disputed**: radar says flooded, a clear optical view says not water
5     water already in the reference window (permanent / seasonal water, sea)
255   no data in any sensor
====  ==============================================================================

``stats['flooded_area_ha']`` is classes 1 + 2 + 3 (reported by at least one sensor and contradicted by none);
``flooded_confirmed_ha`` is class 1 and ``flooded_upper_ha`` adds the disputed class. ``stats['sensors_used']`` lists
the data that were actually available; if the optical imagery of the flood window was cloud-covered everywhere the
result is radar + terrain only, and says so.
"""
from __future__ import annotations

import logging
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from pygeovision.ai.pipelines import PipelineResult
from pygeovision.ai.pipelines._raster import write_classes, write_float
from pygeovision.ai.pipelines.sar_flood import (
    DEFAULT_RES_M,
    SAR_NOTE,
    SarFloodMappingPipeline,
    parallel_map,
)
from pygeovision.ai.pipelines.terrain import DrainageAnalysisPipeline, hand_from_dem
from pygeovision.data._dates import date_window
from pygeovision.data.reference import ReferenceDataError, TargetGrid, read_to_grid

logger = logging.getLogger(__name__)

S2_COLLECTION = "sentinel-2-l2a"
#: Scene Classification Layer values that are not usable for water mapping: no data, defective, cloud shadow,
#: medium and high probability cloud, thin cirrus
SCL_BAD = (0, 1, 3, 8, 9, 10)
FLOOD_CLASSES = {0: "not flooded", 1: "flooded: radar and optical", 2: "flooded: radar only", 3: "flooded: optical only",
                 4: "disputed: radar flooded, optical clear and dry", 5: "water in the reference window"}
OPTICAL_NOTE = ("Sentinel-2 L2A surface reflectance (ESA/Copernicus) via the Microsoft Planetary Computer, MNDWI of Xu (2006); "
                "cloud, shadow and cirrus pixels are masked with the product's Scene Classification Layer, so they are "
                "unknown, never dry.")


def _scale_offset(item: dict[str, Any], asset: str) -> tuple[float, float]:
    """Reflectance = DN * scale + offset. From the asset's raster metadata, else from the processing baseline (the
    -0.1 offset applies to baseline 04.00 and later, i.e. scenes from 25 January 2022)."""
    rb = ((item.get("assets") or {}).get(asset) or {}).get("raster:bands") or []
    base = (item.get("properties") or {}).get("s2:processing_baseline")
    try:
        offset_rule = -0.1 if float(str(base)) >= 4.0 else 0.0
    except (TypeError, ValueError):
        offset_rule = 0.0
    if rb and rb[0].get("scale") is not None:
        return float(rb[0]["scale"]), float(rb[0]["offset"]) if rb[0].get("offset") is not None else offset_rule
    return 1e-4, offset_rule


def mndwi(green: np.ndarray, swir: np.ndarray) -> np.ndarray:
    s = green + swir
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(s > 0, (green - swir) / s, np.nan)


def agreement(radar_water: np.ndarray, optical_water: np.ndarray, usable: np.ndarray) -> dict[str, float]:
    """How well radar water matches optical water on the pixels both sensors observed (F1 and IoU of the water class)."""
    r, o = radar_water[usable], optical_water[usable]
    tp = float(np.sum(r & o))
    fp, fn = float(np.sum(r & ~o)), float(np.sum(~r & o))
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0
    return {"f1": f1, "iou": iou, "n": float(r.size)}


def calibrate_water_db(before_db: np.ndarray, optical_water: np.ndarray, usable: np.ndarray, lo: float = -22.0,
                       hi: float = -12.0, step: float = 0.5, min_class: int = 200) -> tuple[float, float] | None:
    """Radar water threshold (dB) that best reproduces the optical water map of the same dry period.

    Both sensors saw the same ground in the reference window, so the optical MNDWI water mask is a reference for where
    radar backscatter should count as water. Candidates between ``lo`` and ``hi`` dB (physically sensible for VV) are
    scored by the F1 of the water class; thresholds within 0.005 of the best score form a plateau and its middle is
    returned with its F1. ``None`` when the usable pixels hold fewer than ``min_class`` of either class (too little to
    calibrate on).
    """
    ow = optical_water[usable]
    if ow.sum() < min_class or (~ow).sum() < min_class:
        return None
    db = before_db[usable]
    cands = np.arange(lo, hi + 1e-9, step)
    scores = np.array([agreement(db < t, ow, np.ones(db.shape, bool))["f1"] for t in cands])
    plateau = cands[scores >= scores.max() - 0.005]
    return float(np.median(plateau)), float(scores.max())


class MultiSensorFloodPipeline(SarFloodMappingPipeline):
    """Flooded area from Sentinel-1 radar, Sentinel-2 optical and a DEM, keeping each sensor's evidence visible.

    The sensors inform each other: the radar dry-reference water mask tells the optical stage what was already water,
    optical imagery calibrates the radar water threshold (``water_db="auto"``, the default; a number or ``"otsu"``
    override it), optical observations confirm or contradict radar floods wherever the sky was clear in the flood
    window, and the radar-versus-optical agreement in the reference window is reported as a quality figure.

    Arguments are those of ``sar_flood_mapping`` plus ``mndwi_threshold`` (default 0), ``max_optical_cloud`` (scene
    cloud-cover limit in %, default 95; clouds are masked per pixel, so a mostly cloudy scene still contributes its clear pixels), ``max_optical_scenes`` (per window,
    default 10), ``max_hand_m`` (flooding on ground more than this above the nearest drainage is removed, default 15;
    ``None`` or 0 switches it off) and ``stream_threshold_km2`` (contributing area that makes a channel, default 1).
    ``use_optical=False`` / ``use_terrain=False`` leave a data source out.
    """
    name = "multisensor_flood_mapping"
    description = ("Flooded area from Sentinel-1 radar + Sentinel-2 optical + terrain (HAND): per-pixel agreement between "
                   "sensors, cloud-independent where radar sees")
    domain = "disaster"
    tags = ["flood", "sar", "optical", "multi-sensor", "sentinel-1", "sentinel-2", "hand"]

    # -- optical ----------------------------------------------------------------------------------------------------
    def _optical_items(self, bbox, window, max_cloud: float, limit: int) -> list[dict[str, Any]]:
        items = self._pc().search(S2_COLLECTION, bbox, datetime=f"{window[0]}T00:00:00Z/{window[1]}T23:59:59Z")
        keep = []
        for it in items:
            day = str((it.get("properties") or {}).get("datetime", ""))[:10]
            cloud = (it.get("properties") or {}).get("eo:cloud_cover")
            assets = it.get("assets") or {}
            if not (window[0] <= day <= window[1]) or not all(a in assets for a in ("B03", "B11", "SCL")):
                continue
            if cloud is not None and float(cloud) > max_cloud:
                continue
            keep.append(it)
        keep.sort(key=lambda i: float((i.get("properties") or {}).get("eo:cloud_cover") or 0.0))
        return keep[:limit]

    def _scene_mndwi(self, item: dict[str, Any], grid: TargetGrid) -> np.ndarray | None:
        pc = self._pc()
        urls = {a: pc.asset_urls(S2_COLLECTION, [item], a) for a in ("B03", "B11", "SCL")}
        if not all(urls.values()):
            return None
        g = read_to_grid(urls["B03"], grid, resampling="average", dtype="float32", invalid_values=[0])
        s = read_to_grid(urls["B11"], grid, resampling="average", dtype="float32", invalid_values=[0])
        c = read_to_grid(urls["SCL"], grid, resampling="nearest", dtype="float32", invalid_values=[0])
        ok = g.valid & s.valid & c.valid & ~np.isin(c.data, SCL_BAD)
        if not ok.any():
            return None
        gs, go = _scale_offset(item, "B03")
        ss, so = _scale_offset(item, "B11")
        out = mndwi(g.data.astype("float64") * gs + go, s.data.astype("float64") * ss + so)
        return np.where(ok, out, np.nan)

    def _optical_composite(self, items, grid: TargetGrid) -> tuple[np.ndarray, int, list[str]]:
        mats = parallel_map(lambda it: (self._scene_mndwi(it, grid), str((it.get("properties") or {}).get("datetime", ""))[:10]),
                            items)
        stack = [m for m, _ in mats if m is not None]
        days = [d for m, d in mats if m is not None]
        if not stack:
            return np.full((grid.height, grid.width), np.nan, dtype="float32"), 0, []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(np.stack(stack), axis=0)
        return med.astype("float32"), len(stack), sorted(set(days))

    # -- run --------------------------------------------------------------------------------------------------------
    def run(self, bbox, output_dir="./output", date_before=None, date_after=None, polarization="vv",
            resolution_m=DEFAULT_RES_M, water_db="auto", change_db=3.0, mndwi_threshold=0.0, max_optical_cloud=95.0,
            max_optical_scenes=10, max_hand_m=15.0, stream_threshold_km2=1.0, min_patch_ha=0.1, speckle_filter=True,
            max_scenes=8, use_optical=True, use_terrain=True, **kwargs) -> PipelineResult:
        t0 = time.time()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        try:
            if not date_before or not date_after:
                raise ReferenceDataError("multisensor_flood_mapping needs date_before (a dry reference) and date_after (the "
                                         "flood), each a day, a month or a range such as 2026-06-27/2026-07-06.")
            pol = str(polarization).lower()
            if pol not in ("vv", "vh", "hh", "hv"):
                raise ReferenceDataError("polarization must be one of vv, vh, hh, hv.")
            if float(change_db) < 0 or float(min_patch_ha) < 0 or int(max_scenes) < 1 or int(max_optical_scenes) < 1:
                raise ReferenceDataError("change_db and min_patch_ha cannot be negative; max_scenes and max_optical_scenes "
                                         "must be at least 1.")
            wb, wa = date_window(str(date_before)), date_window(str(date_after))
            if wb[1] >= wa[0]:
                raise ReferenceDataError(f"the reference window {wb[0]}..{wb[1]} must end before the flood window "
                                         f"{wa[0]}..{wa[1]} starts.")
            wd = str(water_db).lower() if water_db is not None else "auto"
            thr_mode = "otsu" if wd == "otsu" else None
            thr_given = None
            if wd not in ("auto", "otsu", "none", ""):
                try:
                    thr_given = float(water_db)
                except (TypeError, ValueError) as exc:
                    raise ReferenceDataError("water_db must be a number in dB (such as -15), 'otsu' or 'auto'.") from exc
            auto = wd in ("auto", "none", "")

            grid, coarsened = self._grid(bbox, resolution_m)
            shape = (grid.height, grid.width)
            px_ha = grid.pixel_area_ha

            # 1 radar: scenes, composites, a provisional water threshold ----------------------------------------------
            sar = self._sar_stage(bbox, grid, wb, wa, pol, thr_mode, thr_given, bool(speckle_filter), int(max_scenes))
            sar_valid, before_db, after_db = sar["valid"], sar["before_db"], sar["after_db"]
            thr, thr_source = sar["thr"], sar["source"]
            sensors = ["sentinel-1"]
            notes: list[str] = []

            # 2 optical: the flood window first; the reference only if the flood window has a clear view ----------------
            ob = oa = np.full(shape, np.nan, dtype="float32")
            n_ob = n_oa = 0
            days_oa: list[str] = []
            if use_optical:
                try:
                    cloud = float(max_optical_cloud)
                    ia = self._optical_items(bbox, wa, cloud, int(max_optical_scenes))
                    logger.info("Sentinel-2 flood window %s..%s: %d scene(s) under %.0f%% cloud", wa[0], wa[1], len(ia), cloud)
                    if not ia:
                        notes.append(f"no Sentinel-2 scene under {cloud:.0f}% cloud in the flood window {wa[0]}..{wa[1]}: "
                                     "optical imagery of the flood is unavailable")
                    else:
                        oa, n_oa, days_oa = self._optical_composite(ia, grid)
                        if not n_oa:
                            notes.append("the flood-window Sentinel-2 scenes had no clear pixels over the area")
                    if n_oa:                                  # no flood view -> the reference scenes would not be used: skip them
                        ib = self._optical_items(bbox, wb, cloud, int(max_optical_scenes))
                        logger.info("Sentinel-2 reference window %s..%s: %d scene(s)", wb[0], wb[1], len(ib))
                        if ib:
                            ob, n_ob, _ = self._optical_composite(ib, grid)
                        if not n_ob:
                            notes.append("no clear Sentinel-2 view in the reference window: optical-only flooding (class 3) "
                                         "and the radar calibration against optical could not be done")
                except ReferenceDataError as exc:
                    notes.append(f"Sentinel-2 could not be read: {exc}")
                    logger.warning("optical stage skipped: %s", exc)
                    ob = oa = np.full(shape, np.nan, dtype="float32")
                    n_ob = n_oa = 0
            ob_valid, oa_valid = np.isfinite(ob), np.isfinite(oa)
            mt = float(mndwi_threshold)
            ow_b, ow_a = ob_valid & (ob > mt), oa_valid & (oa > mt)
            if oa_valid.any():
                sensors.append("sentinel-2")

            # 3 the sensors talk: radar water threshold calibrated against optical water of the reference window -------
            ref_both = sar_valid & ob_valid
            cal = calibrate_water_db(before_db, ow_b, ref_both) if (auto and ref_both.any()) else None
            if cal is not None:
                thr, f1 = cal
                thr_source = f"calibrated on Sentinel-2 MNDWI water of the reference window (F1 {f1:.2f})"
            elif auto:
                notes.append("the radar water threshold is the conventional default: " + (
                    "Sentinel-2 gave too little clear water/land in the reference window to calibrate it"
                    if ref_both.any() else "there was no optical reference to calibrate it on"))
            sw_a = sar_valid & (after_db < thr)
            sw_b = sar_valid & (before_db < thr)
            agree = agreement(sw_b, ow_b, ref_both) if ref_both.any() else None

            # 4 radar change, with the radiometric offset between the two windows removed ------------------------------
            diff = after_db - before_db
            stable = sar_valid & ~sw_a & ~sw_b
            offset = float(np.nanmedian(diff[stable])) if stable.sum() > 1000 else 0.0
            diff_c = diff - offset                                   # land that did not change defines "no change"
            sar_flood = sw_a & ~sw_b & (diff_c <= -float(change_db))

            # 5 terrain ------------------------------------------------------------------------------------------------
            hand = np.full(shape, np.nan, dtype="float32")
            hand_applied = False
            if use_terrain and max_hand_m:
                try:
                    dem_pipe = DrainageAnalysisPipeline(self.pgv)
                    dem_pipe.pc = self._pc()
                    dem = dem_pipe._elevation(bbox, out, "copernicus", "cop30", grid.res, grid=grid)
                    hand = hand_from_dem(dem.data, dem.valid, grid.res, float(stream_threshold_km2)).astype("float32")
                    hand_applied = True
                    sensors.append("copernicus-dem")
                except (ReferenceDataError, ValueError, RuntimeError) as exc:
                    notes.append(f"terrain (HAND) not applied: {exc}")
                    logger.warning("terrain stage skipped: %s", exc)

            # 6 fusion -------------------------------------------------------------------------------------------------
            # Radar gives the change (flood vs its own dry reference); a clear optical view of the flood window confirms
            # or contradicts it without needing an optical reference. Optical-only flooding needs the optical reference.
            prior_water = sw_b | ow_b
            s_flood = sar_flood & ~prior_water
            o_wet, o_dry = oa_valid & ow_a, oa_valid & ~ow_a
            o_flood = o_wet & ob_valid & ~ow_b & ~prior_water
            cls = np.zeros(shape, dtype=np.uint8)
            cls[s_flood & o_wet] = 1
            cls[s_flood & ~oa_valid] = 2
            cls[o_flood & ~s_flood] = 3
            cls[s_flood & o_dry] = 4
            excluded = np.zeros(shape, dtype=bool)
            if hand_applied:
                excluded = np.isin(cls, (1, 2, 3, 4)) & np.isfinite(hand) & (hand > float(max_hand_m))
                cls[excluded] = 0
            removed = 0
            if min_patch_ha:
                from scipy import ndimage
                lab, n = ndimage.label(np.isin(cls, (1, 2, 3, 4)), structure=np.ones((3, 3), int))
                if n:
                    sizes = np.bincount(lab.ravel())
                    small = np.nonzero(sizes * px_ha < float(min_patch_ha))[0]
                    small = small[small != 0]
                    if small.size:
                        sm = np.isin(lab, small)
                        removed = int(sm.sum())
                        cls[sm] = 0
            cls[prior_water] = 5
            known = sar_valid | oa_valid | ob_valid
            if not known.any():
                raise ReferenceDataError("no sensor has valid data over this area.")

            prof = grid.profile()
            path = write_classes(out / "flood_extent.tif", cls, known, prof,
                                 [f"{k}={v}" for k, v in FLOOD_CLASSES.items()] + ["255=nodata"])
            write_float(out / "change_db.tif", np.where(sar_valid, diff_c, np.nan), prof,
                        ["radar: flood minus reference backscatter after removing the offset between the windows, dB"])
            if oa_valid.any():                                       # an all-empty layer would be a blank map in the report
                write_float(out / "mndwi_after.tif", oa, prof, ["optical: MNDWI of the flood window (NaN = cloud / no scene)"])
            if hand_applied:
                write_float(out / "hand_m.tif", hand, prof, ["height above nearest drainage, m"])

            def ha(mask):
                return float(mask.sum() * px_ha)
            n_known = int(known.sum())
            land = known & ~prior_water                              # the ground that could have flooded (sea and lakes out)
            n_land = int(land.sum())
            flooded = (cls == 1) | (cls == 2) | (cls == 3)
            stats: dict[str, Any] = {
                "flooded_area_ha": ha(flooded), "flooded_confirmed_ha": ha(cls == 1), "flooded_radar_only_ha": ha(cls == 2),
                "flooded_optical_only_ha": ha(cls == 3), "flooded_disputed_ha": ha(cls == 4),
                "flooded_upper_ha": ha(flooded | (cls == 4)), "permanent_water_area_ha": ha(cls == 5),
                "land_area_ha": ha(land), "flooded_fraction": float(flooded.sum() / n_land) if n_land else 0.0,
                "valid_fraction": n_known / known.size, "excluded_by_hand_ha": ha(excluded),
                "removed_small_patch_ha": float(removed * px_ha),
                "optical_coverage": float((oa_valid & land).sum() / n_land) if n_land else 0.0, "sensors_used": sensors,
                "hand_applied": hand_applied, "max_hand_m": float(max_hand_m) if hand_applied else None,
                "polarization": pol, "water_db": float(thr), "water_db_source": thr_source, "change_db": float(change_db),
                "radiometric_offset_db": offset, "otsu_db": sar["otsu_db"], "mndwi_threshold": mt,
                "radar_optical_f1_reference": agree["f1"] if agree else None,
                "radar_optical_iou_reference": agree["iou"] if agree else None,
                "min_patch_ha": float(min_patch_ha), "orbit_state": sar["orbit_state"], "relative_orbit": sar["rel_orbit"],
                "scenes_reference": sar["n_b"], "scenes_flood": sar["n_a"], "reference_dates": sar["days_b"],
                "flood_dates": sar["days_a"], "optical_scenes_reference": n_ob, "optical_scenes_flood": n_oa,
                "optical_flood_dates": days_oa, "data_notes": "; ".join(notes) if notes else "",
                "resolution_m": grid.res, "resolution_coarsened": coarsened, "product_note": SAR_NOTE + " " + OPTICAL_NOTE,
            }

            def span(days, window):
                days = days or list(window)
                return days[0] if days[0] == days[-1] else f"{days[0]} \u2013 {days[-1]}"
            meta = {"date_before": span(sar["days_b"], wb), "date_after": span(sar["days_a"], wa), "crs": grid.crs,
                    "scene_ids": {"reference": sar["ids_b"], "flood": sar["ids_a"]}}
            return self._ok(path, stats, meta, t0)
        except (ReferenceDataError, ValueError) as exc:
            return self._fail(str(exc), t0)


MULTISENSOR_PIPELINES: dict[str, type] = {MultiSensorFloodPipeline.name: MultiSensorFloodPipeline}