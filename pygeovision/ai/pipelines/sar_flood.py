"""Flood mapping from Sentinel-1 radar backscatter (works through cloud, day or night).

``sar_flood_mapping`` reads Sentinel-1 radiometrically terrain-corrected (RTC) backscatter from the Microsoft
Planetary Computer for a *dry reference* window and a *flood* window, and maps open water that appears in the
flood window but not in the reference. Calm open water reflects the radar beam away from the satellite, so it is
dark (low backscatter); land is brighter. It is the standard radar flood-mapping approach (change detection of
open water, e.g. Twele et al. 2016), reduced to its core and without any trained model.

What is computed, and nothing else:

* only scenes of **one orbit track** (same direction and relative orbit) that exist in both windows are compared,
  so a difference in viewing geometry is not mistaken for flooding;
* each window becomes the per-pixel **median** (in dB) of its scenes, which suppresses speckle; a small mean
  filter is applied to each scene first (``speckle_filter``);
* water = backscatter below a threshold in dB (``water_db``; a conventional value per polarisation unless you
  set it, or ``"otsu"``);
* **flooded** = water in the flood window, not water in the reference, and at least ``change_db`` darker;
  **permanent water** = water in both; patches smaller than ``min_patch_ha`` are removed.

Radar maps *open* floodwater well. It largely misses flooding between buildings and under dense vegetation
(double-bounce makes such floodwater bright, not dark) and cannot see flooding that had receded before the
overpass. Both are written into the report.
"""
from __future__ import annotations

import contextlib
import logging
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from pygeovision.ai.pipelines import PipelineResult
from pygeovision.ai.pipelines._raster import write_classes, write_float
from pygeovision.ai.pipelines.reference_products import _Reference
from pygeovision.data._dates import date_window
from pygeovision.data.reference import GridRaster, ReferenceDataError, TargetGrid, read_to_grid

logger = logging.getLogger(__name__)

RTC_NODATA = -32768.0
#: conventional open-water backscatter thresholds (dB) per polarisation -- a starting point, not a calibration
DEFAULT_WATER_DB = {"vv": -15.0, "vh": -20.0, "hh": -15.0, "hv": -20.0}
DEFAULT_RES_M = 20.0

SAR_NOTE = ("Sentinel-1 C-band radar (ESA/Copernicus), radiometrically terrain-corrected gamma-naught at 10 m "
            "(Catalyst, via the Microsoft Planetary Computer). Calm open water is dark in radar; wind-roughened water, "
            "wet soil, smooth tarmac and radar shadow can be dark too, and floodwater among buildings or under "
            "dense vegetation is bright (double bounce) and is therefore not detected.")


def otsu_threshold(values_db: np.ndarray, lo: float = -30.0, hi: float = 5.0, step: float = 0.1) -> tuple[float, float]:
    """Otsu's (1979) histogram threshold of dB values and its separability (between-class / total variance, 0-1)."""
    v = values_db[np.isfinite(values_db)]
    v = v[(v >= lo) & (v <= hi)]
    if v.size < 100:
        raise ReferenceDataError("too few valid pixels to estimate an Otsu threshold.")
    hist, edges = np.histogram(v, bins=np.arange(lo, hi + step, step))
    p = hist / hist.sum()
    centers = (edges[:-1] + edges[1:]) / 2.0
    omega = np.cumsum(p)
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = (mu_t * omega - mu) ** 2 / (omega * (1.0 - omega))
    sb = np.nan_to_num(sb, nan=0.0, posinf=0.0, neginf=0.0)
    k = int(np.argmax(sb))
    total = float(np.sum(p * (centers - mu_t) ** 2))
    return float(edges[k + 1]), (float(sb[k] / total) if total > 0 else 0.0)


def speckle_mean(lin: np.ndarray, valid: np.ndarray, size: int = 3) -> np.ndarray:
    """Mean of the valid linear-power pixels in a ``size`` x ``size`` window (NaN where the centre is invalid)."""
    from scipy import ndimage
    w = valid.astype("float64")
    num = ndimage.uniform_filter(np.where(valid, lin, 0.0).astype("float64"), size=size, mode="nearest")
    den = ndimage.uniform_filter(w, size=size, mode="nearest")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = num / den
    return np.where(valid & (den > 0), out, np.nan)


def to_db(lin: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(lin > 0, 10.0 * np.log10(lin), np.nan)


def _track(item: dict[str, Any]) -> tuple[str, Any]:
    p = item.get("properties") or {}
    return (str(p.get("sat:orbit_state", "unknown")), p.get("sat:relative_orbit", "unknown"))


def parallel_map(fn, items, workers: int = 4) -> list:
    """``[fn(x) for x in items]`` with up to ``workers`` reads in flight (scene reads wait on the network); the
    results keep the order of ``items`` and an exception in any read is raised here."""
    items = list(items)
    if len(items) <= 1 or workers <= 1:
        return [fn(x) for x in items]
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(workers, len(items))) as ex:
        return list(ex.map(fn, items))


def _day(item: dict[str, Any]) -> str:
    p = item.get("properties") or {}
    return str(p.get("datetime") or p.get("start_datetime") or "")[:10]


def _coverage(items: list[dict[str, Any]], bbox) -> float:
    """Share of the box covered by the union of the scenes' footprints."""
    from shapely.geometry import box, shape
    from shapely.ops import unary_union
    aoi = box(*bbox)
    geoms = []
    for it in items:
        if it.get("geometry"):
            geoms.append(shape(it["geometry"]))
        elif it.get("bbox"):
            geoms.append(box(*it["bbox"][:4]))
    if not geoms or aoi.area == 0:
        return 0.0
    return float(unary_union(geoms).intersection(aoi).area / aoi.area)


class SarFloodMappingPipeline(_Reference):
    """Flooded area from Sentinel-1 radar: open water in the flood window that was not water in a dry reference.

    ``date_before`` is the dry reference and ``date_after`` the flood; each is a day, a month (``2026-06``) or a
    range (``2026-06-27/2026-07-03``). Sentinel-1 passes a place only every 6 to 12 days or longer, so give a range
    that is long enough to contain an acquisition (a single day usually has none); the error lists the nearest
    acquisitions if a window is empty. Outputs: ``flood_extent.tif`` (0 dry, 1 flooded, 2 permanent water, 255 no
    data), ``change_db.tif`` (flood minus reference backscatter, dB), ``before_db.tif`` and ``after_db.tif``.

    ``polarization`` is ``vv`` (default) or ``vh``. ``water_db`` is the open-water threshold in dB (default -15 for
    VV, -20 for VH, conventional values -- check them against your scene), or ``"otsu"`` for a histogram threshold
    of the flood image. ``change_db`` (default 3) is how much darker than the reference a flooded pixel must be.
    """
    name = "sar_flood_mapping"
    description = "Flooded area from Sentinel-1 radar (cloud-independent): open water on the flood date that was not water before"
    domain = "disaster"
    tags = ["flood", "sar", "sentinel-1", "water", "bitemporal"]
    collection = "sentinel-1-rtc"
    native_res_m = 10.0

    # -- scene selection --------------------------------------------------------------------------------------------
    def _scenes(self, bbox, window: tuple[str, str], pol: str) -> list[dict[str, Any]]:
        """Scenes with the polarisation whose acquisition day lies inside ``window``.

        The catalogue is asked for the window, and every answer is checked against it again here: a scene outside
        the window must never be used as if it were inside it (a reference and a flood window that share scenes
        would show no change at all).
        """
        items = self._pc().search(self.collection, bbox, datetime=f"{window[0]}T00:00:00Z/{window[1]}T23:59:59Z")
        kept = [it for it in items if (it.get("assets") or {}).get(pol) and window[0] <= _day(it) <= window[1]]
        if len(kept) != len(items):
            logger.info("Sentinel-1 %s..%s: %d scene(s) returned, %d usable (polarisation %s and inside the window)",
                        window[0], window[1], len(items), len(kept), pol.upper())
        return kept

    def _nearby(self, bbox, window: tuple[str, str], pol: str) -> str:
        """Nearest acquisitions around an empty window, for the error message (best effort)."""
        from datetime import date, timedelta
        try:
            lo = (date.fromisoformat(window[0]) - timedelta(days=30)).isoformat()
            hi = (date.fromisoformat(window[1]) + timedelta(days=30)).isoformat()
            items = self._scenes(bbox, (lo, hi), pol)
        except Exception as exc:  # noqa: BLE001 - diagnostics only
            logger.debug("nearby search failed: %s", exc)
            return ""
        seen = sorted({(_day(it), *_track(it)) for it in items})
        if not seen:
            return f" No {pol.upper()} acquisition was found within 30 days either side."
        txt = ", ".join(f"{d} ({s}, orbit {o})" for d, s, o in seen[:10])
        return f" Acquisitions within 30 days: {txt}."

    def _choose_track(self, before: list[dict[str, Any]], after: list[dict[str, Any]], bbox):
        groups_b, groups_a = defaultdict(list), defaultdict(list)
        for it in before:
            groups_b[_track(it)].append(it)
        for it in after:
            groups_a[_track(it)].append(it)
        common = set(groups_b) & set(groups_a)
        if not common:
            fmt = lambda g: ", ".join(f"{s} orbit {o}" for s, o in sorted(g, key=str)) or "none"   # noqa: E731
            raise ReferenceDataError(
                "the two windows have no Sentinel-1 orbit track in common (reference: " + fmt(groups_b) + "; flood: "
                + fmt(groups_a) + "). Scenes from different tracks look at the ground from different angles, so comparing "
                "them would show the viewing geometry, not flooding. Widen the reference window until it holds a pass "
                "of the same track as the flood scene.")
        scored = []
        for key in common:
            cb, ca = _coverage(groups_b[key], bbox), _coverage(groups_a[key], bbox)
            scored.append((min(cb, ca), len(groups_b[key]) + len(groups_a[key]), key, cb, ca))
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        _, _, key, cb, ca = scored[0]
        return key, groups_b[key], groups_a[key], cb, ca

    # -- reading ----------------------------------------------------------------------------------------------------
    def _composite(self, items: list[dict[str, Any]], pol: str, grid: TargetGrid,
                   smooth: bool) -> tuple[np.ndarray, int, list[str]]:
        """Per-pixel median (dB) of the scenes, the number of scenes with data, and their dates."""
        urls = self._pc().asset_urls(self.collection, items, pol)
        days = [_day(it) for it in items]
        def one(url: str) -> np.ndarray | None:
            r: GridRaster = read_to_grid([url], grid, resampling="average", dtype="float32", invalid_values=[RTC_NODATA])
            lin = r.data.astype("float64")
            ok = r.valid & np.isfinite(lin) & (lin > 0)
            if not ok.any():
                return None
            if smooth:
                lin = speckle_mean(lin, ok)
            return to_db(np.where(ok, lin, np.nan))
        stack = [a for a in parallel_map(one, urls) if a is not None]
        if not stack:
            raise ReferenceDataError("the Sentinel-1 scenes found hold no valid backscatter over this area (all nodata).")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(np.stack(stack), axis=0)
        return med.astype("float32"), len(stack), sorted(set(days))

    def _sar_stage(self, bbox, grid: TargetGrid, wb, wa, pol: str, thr_mode, thr_given, smooth: bool, max_scenes: int) -> dict[str, Any]:
        """Select one orbit track present in both windows, build the two dB composites and choose the water threshold."""
        sb, sa = self._scenes(bbox, wb, pol), self._scenes(bbox, wa, pol)
        if not sb:
            raise ReferenceDataError(f"no Sentinel-1 {pol.upper()} scene covers this area between {wb[0]} and {wb[1]} "
                                     f"(reference window).{self._nearby(bbox, wb, pol)}")
        if not sa:
            raise ReferenceDataError(f"no Sentinel-1 {pol.upper()} scene covers this area between {wa[0]} and {wa[1]} "
                                     f"(flood window).{self._nearby(bbox, wa, pol)}")
        (orbit_state, rel_orbit), tb, ta, cov_b, cov_a = self._choose_track(sb, sa, bbox)
        tb = sorted(tb, key=_day)[-int(max_scenes):]          # the passes nearest the event
        ta = sorted(ta, key=_day)[:int(max_scenes)]
        shared = {it.get("id") for it in tb} & {it.get("id") for it in ta}
        if shared:
            raise ReferenceDataError(f"the reference and flood windows share Sentinel-1 scene(s) {sorted(map(str, shared))}; "
                                     "they must not overlap.")
        logger.info("Sentinel-1 %s orbit %s: reference %s | flood %s", orbit_state, rel_orbit,
                    [f"{_day(i)}" for i in tb], [f"{_day(i)}" for i in ta])
        before_db, n_b, days_b = self._composite(tb, pol, grid, smooth)
        after_db, n_a, days_a = self._composite(ta, pol, grid, smooth)
        valid = np.isfinite(before_db) & np.isfinite(after_db)
        if not valid.any():
            raise ReferenceDataError("the reference and flood scenes share no valid pixels over this area.")
        otsu_db = otsu_sep = None
        with contextlib.suppress(ReferenceDataError):
            otsu_db, otsu_sep = otsu_threshold(after_db[valid])
        if thr_mode == "otsu":
            if otsu_db is None:
                raise ReferenceDataError("could not estimate an Otsu threshold; set water_db in dB.")
            thr, source = otsu_db, "otsu"
        elif thr_given is not None:
            thr, source = thr_given, "user"
        else:
            thr, source = DEFAULT_WATER_DB[pol], "conventional default"
        return {"before_db": before_db, "after_db": after_db, "valid": valid, "thr": thr, "source": source,
                "otsu_db": otsu_db, "otsu_sep": otsu_sep, "orbit_state": orbit_state, "rel_orbit": rel_orbit,
                "n_b": n_b, "n_a": n_a, "days_b": days_b, "days_a": days_a, "cov_b": cov_b, "cov_a": cov_a,
                "ids_b": [str(i.get("id")) for i in tb], "ids_a": [str(i.get("id")) for i in ta]}

    # -- run --------------------------------------------------------------------------------------------------------
    def run(self, bbox, output_dir="./output", date_before=None, date_after=None, polarization="vv",
            resolution_m=DEFAULT_RES_M, water_db=None, change_db=3.0, min_patch_ha=0.1, speckle_filter=True,
            max_scenes=8, **kwargs) -> PipelineResult:
        t0 = time.time()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        try:
            if not date_before or not date_after:
                raise ReferenceDataError("sar_flood_mapping needs date_before (a dry reference) and date_after (the flood), "
                                         "each a day, a month or a range such as 2026-06-27/2026-07-03.")
            pol = str(polarization).lower()
            if pol not in DEFAULT_WATER_DB:
                raise ReferenceDataError("polarization must be one of vv, vh, hh, hv.")
            if float(change_db) < 0 or float(min_patch_ha) < 0 or int(max_scenes) < 1:
                raise ReferenceDataError("change_db and min_patch_ha cannot be negative, and max_scenes must be at least 1.")
            wb, wa = date_window(str(date_before)), date_window(str(date_after))
            if wb[1] >= wa[0]:
                raise ReferenceDataError(f"the reference window {wb[0]}..{wb[1]} must end before the flood window "
                                         f"{wa[0]}..{wa[1]} starts.")
            thr_mode = "otsu" if str(water_db).lower() == "otsu" else None
            if thr_mode is None and water_db is not None:
                try:
                    thr_given: float | None = float(water_db)
                except (TypeError, ValueError) as exc:
                    raise ReferenceDataError("water_db must be a number in dB (such as -15) or 'otsu'.") from exc
            else:
                thr_given = None

            grid, coarsened = self._grid(bbox, resolution_m)
            sar = self._sar_stage(bbox, grid, wb, wa, pol, thr_mode, thr_given, bool(speckle_filter), int(max_scenes))
            before_db, after_db, valid, thr, source = sar["before_db"], sar["after_db"], sar["valid"], sar["thr"], sar["source"]
            otsu_db, otsu_sep = sar["otsu_db"], sar["otsu_sep"]
            orbit_state, rel_orbit, n_b, n_a = sar["orbit_state"], sar["rel_orbit"], sar["n_b"], sar["n_a"]
            days_b, days_a, cov_b, cov_a = sar["days_b"], sar["days_a"], sar["cov_b"], sar["cov_a"]
            diff = after_db - before_db
            w_a = valid & (after_db < thr)
            w_b = valid & (before_db < thr)
            flooded = w_a & ~w_b & (diff <= -float(change_db))
            permanent = w_a & w_b
            px_ha = grid.pixel_area_ha
            removed = 0
            if min_patch_ha and flooded.any():
                from scipy import ndimage
                lab, n = ndimage.label(flooded, structure=np.ones((3, 3), int))
                if n:
                    sizes = np.bincount(lab.ravel())
                    small = np.nonzero(sizes * px_ha < float(min_patch_ha))[0]
                    small = small[small != 0]
                    if small.size:
                        removed = int(sizes[small].sum())
                        flooded = flooded & ~np.isin(lab, small)
            classes = np.zeros(valid.shape, dtype=np.uint8)
            classes[flooded] = 1
            classes[permanent] = 2
            prof = grid.profile()
            path = write_classes(out / "flood_extent.tif", classes, valid, prof,
                                 ["0=dry", "1=flooded", "2=permanent water", "255=nodata"])
            write_float(out / "change_db.tif", np.where(valid, diff, np.nan), prof, ["flood minus reference backscatter, dB"])
            write_float(out / "before_db.tif", before_db, prof, [f"reference backscatter {pol.upper()}, dB"])
            write_float(out / "after_db.tif", after_db, prof, [f"flood-window backscatter {pol.upper()}, dB"])
            n_valid = int(valid.sum())
            stats: dict[str, Any] = {
                "flooded_area_ha": float(flooded.sum() * px_ha), "permanent_water_area_ha": float(permanent.sum() * px_ha),
                "flooded_fraction": float(flooded.sum() / n_valid), "valid_fraction": n_valid / valid.size,
                "polarization": pol, "water_db": float(thr), "water_db_source": source, "change_db": float(change_db),
                "otsu_db": otsu_db, "otsu_separability": otsu_sep, "min_patch_ha": float(min_patch_ha),
                "removed_small_patch_ha": float(removed * px_ha), "orbit_state": orbit_state, "relative_orbit": rel_orbit,
                "scenes_reference": n_b, "scenes_flood": n_a, "reference_dates": days_b, "flood_dates": days_a,
                "aoi_coverage_reference": cov_b, "aoi_coverage_flood": cov_a, "speckle_filter": bool(speckle_filter),
                "resolution_m": grid.res, "resolution_coarsened": coarsened, "product_note": SAR_NOTE,
            }
            def span(days, window):
                days = days or list(window)
                return days[0] if days[0] == days[-1] else f"{days[0]} \u2013 {days[-1]}"
            meta = {"date_before": span(days_b, wb), "date_after": span(days_a, wa), "crs": grid.crs,
                    "scene_ids": {"reference": sar["ids_b"], "flood": sar["ids_a"]}}
            return self._ok(path, stats, meta, t0)
        except (ReferenceDataError, ValueError) as exc:
            return self._fail(str(exc), t0)


SAR_PIPELINES: dict[str, type] = {SarFloodMappingPipeline.name: SarFloodMappingPipeline}