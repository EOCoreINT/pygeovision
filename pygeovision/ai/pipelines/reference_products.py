"""Pipelines that map a *published product* for a bounding box -- no imagery search, no network of ours.

Each one reads the window it needs of Cloud-Optimized GeoTIFFs hosted on the Microsoft Planetary Computer
(``pygeovision.data.reference``), puts them on a local-UTM grid and computes only what the product's own definition
supports. Nothing is classified, thresholded or calibrated by PyGeoVision here, so there is no model to validate:
what you get is the product, clipped, measured and (for the change pipelines) differenced.

* ``landcover_change`` / ``urban_expansion`` -- Impact Observatory / Esri 10 m annual land cover (2017-2023)
* ``forest_change``                          -- JAXA PALSAR forest / non-forest, 25 m (2015-2020)
* ``surface_water_dynamics``                 -- JRC Global Surface Water, 30 m (1984-2020)
* ``evapotranspiration``                     -- MODIS MOD16A3GF annual ET and PET, 500 m
* ``active_fire``                            -- MODIS MOD14A1 daily thermal anomalies, 1 km

Every figure carries the product's own caveats in ``stats['product_note']`` and in the report.
"""
from __future__ import annotations

import logging
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from pygeovision.ai.pipelines import BasePipeline, PipelineResult
from pygeovision.ai.pipelines._raster import write_classes, write_float
from pygeovision.data.reference import (
    GridRaster,
    PlanetaryComputer,
    ReferenceDataError,
    TargetGrid,
    item_year,
    make_grid,
    parse_year,
    read_to_grid,
)

logger = logging.getLogger(__name__)

#: pixels above which the resolution is coarsened automatically (and the stats say so)
AUTO_MAX_PIXELS = 60_000_000


class _Reference(BasePipeline):
    """Shared plumbing: the Planetary Computer client, grid choice and tile selection."""

    collection = ""
    native_res_m = 10.0

    def _pc(self) -> PlanetaryComputer:
        pc = getattr(self, "pc", None)
        if pc is None:
            pc = self.pc = PlanetaryComputer()
        return pc

    def _grid(self, bbox, res: float | None) -> tuple[TargetGrid, bool]:
        """Grid at ``res`` (default: the product's native resolution), coarsened by a whole factor if it would
        exceed ``AUTO_MAX_PIXELS``. Returns the grid and whether it was coarsened."""
        res = float(res or self.native_res_m)
        g = make_grid(bbox, res, max_pixels=10 ** 13)
        if g.width * g.height <= AUTO_MAX_PIXELS:
            return g, False
        factor = math.ceil(math.sqrt(g.width * g.height / AUTO_MAX_PIXELS))
        return make_grid(bbox, res * factor, max_pixels=10 ** 13), True

    def _annual_tiles(self, collection: str, asset: str, bbox, year: int, what: str) -> tuple[list[str], list[str]]:
        """Signed URLs of the ``asset`` of every tile of ``year`` covering ``bbox`` (and the item ids)."""
        pc = self._pc()
        items = pc.search(collection, bbox, datetime=f"{year}-03-01/{year}-09-30")
        items = [it for it in items if item_year(it) == year]
        if not items:
            raise ReferenceDataError(f"{what}: no {collection} tile for {year} covers this area "
                                     f"(check that {year} is within the product's years and the box is on land).")
        return pc.asset_urls(collection, items, asset), [str(it.get("id")) for it in items]

    def _years(self, date_before, date_after, lo: int, hi: int) -> tuple[int, int]:
        if date_before is None or date_after is None:
            raise ReferenceDataError(f"{self.name} needs date_before and date_after: two years such as 2018 and 2023.")
        y0, y1 = parse_year(date_before, "date_before"), parse_year(date_after, "date_after")
        if y0 >= y1:
            raise ReferenceDataError(f"date_before ({y0}) must be earlier than date_after ({y1}).")
        for y in (y0, y1):
            if not lo <= y <= hi:
                raise ReferenceDataError(f"{self.name} covers {lo}-{hi}; {y} is outside that range.")
        return y0, y1


# ═══════════════════════════════════════════════════════════════════════════
# Annual land cover (Impact Observatory / Esri, 10 m, 2017-2023)
# ═══════════════════════════════════════════════════════════════════════════

IO_LULC = {1: "Water", 2: "Trees", 4: "Flooded vegetation", 5: "Crops", 7: "Built area", 8: "Bare ground",
           9: "Snow/ice", 11: "Rangeland"}
IO_CLOUDS = 10
IO_NOTE = ("Impact Observatory / Esri 10 m Annual Land Use Land Cover (v02, Sentinel-2 based, CC BY 4.0). Each year is "
           "classified independently, so a year-to-year difference contains classifier disagreement as well as real "
           "change; single-pixel flips are expected.")


class _LulcPair(_Reference):
    collection = "io-lulc-annual-v02"
    native_res_m = 10.0
    YEARS = (2017, 2023)

    def _read_year(self, bbox, year, grid, what) -> GridRaster:
        urls, ids = self._annual_tiles(self.collection, "data", bbox, year, what)
        r = read_to_grid(urls, grid, dtype="float32", invalid_values=[0, IO_CLOUDS])
        r.sources = ids
        return r

    def _pair(self, bbox, date_before, date_after, resolution_m):
        y0, y1 = self._years(date_before, date_after, *self.YEARS)
        grid, coarsened = self._grid(bbox, resolution_m)
        a = self._read_year(bbox, y0, grid, self.name)
        b = self._read_year(bbox, y1, grid, self.name)
        valid = a.valid & b.valid
        if not valid.any():
            raise ReferenceDataError(f"No pixel has a land-cover class in both {y0} and {y1} inside this area.")
        ca = np.where(valid, a.data, 0).astype(np.uint8)
        cb = np.where(valid, b.data, 0).astype(np.uint8)
        return y0, y1, grid, coarsened, ca, cb, valid, (a.sources, b.sources)

    @staticmethod
    def _write_lulc(path: Path, classes: np.ndarray, valid: np.ndarray, grid: TargetGrid, desc: str) -> Path:
        return write_classes(path, classes, valid, grid.profile(), [desc])


def _area_table(classes: np.ndarray, valid: np.ndarray, ha: float) -> dict[str, float]:
    out = {}
    for v, name in IO_LULC.items():
        n = int(((classes == v) & valid).sum())
        if n:
            out[name] = round(n * ha, 3)
    return out


class LandCoverChangePipeline(_LulcPair):
    """What changed between two years of the 10 m annual land-cover map, and into what.

    ``date_before`` / ``date_after`` are years (2017-2023). Both maps are the Impact Observatory / Esri annual
    land-cover product, each classified separately from that year's Sentinel-2 imagery -- so unlike a comparison
    of two copies of one static map, this one can show change. Pixels that are cloud or no-data in either year are
    nodata. ``min_change_ha`` removes connected change patches smaller than that area (0 keeps every pixel).

    Outputs: ``landcover_<year>.tif`` for both years (class codes of the product), ``change_mask.tif``
    (0 unchanged, 1 changed, 255 nodata) and ``transition.tif`` (0 unchanged, otherwise ``1 + 8*from_index +
    to_index`` over the eight product classes). ``stats['transitions']`` lists the largest from->to changes.
    """
    name = "landcover_change"
    description = "Where annual 10 m land cover changed between two years (2017-2023), and from/to which class"
    domain = "land"
    tags = ["land-cover", "change", "annual", "reference-product"]

    def run(self, bbox, output_dir="./output", date_before=None, date_after=None, resolution_m=None,
            min_change_ha=0.0, **kwargs) -> PipelineResult:
        t0 = time.time()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        try:
            y0, y1, grid, coarsened, ca, cb, valid, srcs = self._pair(bbox, date_before, date_after, resolution_m)
            ha = grid.pixel_area_ha
            changed = (ca != cb) & valid
            removed = 0
            if min_change_ha and min_change_ha > 0 and changed.any():
                from scipy import ndimage
                lab, n = ndimage.label(changed, structure=np.ones((3, 3)))
                sizes = np.bincount(lab.ravel())
                small = sizes * ha < min_change_ha
                small[0] = False
                drop = small[lab]
                removed = int(drop.sum())
                changed &= ~drop
            self._write_lulc(out / f"landcover_{y0}.tif", ca, valid, grid, f"{y0} land cover (product codes)")
            self._write_lulc(out / f"landcover_{y1}.tif", cb, valid, grid, f"{y1} land cover (product codes)")
            path = write_classes(out / "change_mask.tif", changed.astype(np.uint8), valid, grid.profile(),
                                 ["0=unchanged", "1=changed", "255=nodata"])
            idx = {v: i for i, v in enumerate(IO_LULC)}
            trans = np.zeros(ca.shape, np.uint8)
            for v0, i0 in idx.items():
                for v1, i1 in idx.items():
                    if v0 != v1:
                        trans[changed & (ca == v0) & (cb == v1)] = 1 + 8 * i0 + i1
            write_classes(out / "transition.tif", trans, valid, grid.profile(),
                          ["0=unchanged; else 1 + 8*from_index + to_index over " + ",".join(map(str, IO_LULC))])
            pairs = Counter(zip(ca[changed].tolist(), cb[changed].tolist()))
            transitions = [{"from": IO_LULC[a], "to": IO_LULC[b], "area_ha": round(n * ha, 3)}
                           for (a, b), n in pairs.most_common(15)]
            n_valid, n_chg = int(valid.sum()), int(changed.sum())
            from scipy import ndimage as ndi
            small_share = None
            if n_chg:
                lab, _ = ndi.label(changed, structure=np.ones((3, 3)))
                sz = np.bincount(lab.ravel())[1:]
                small_share = float(sz[sz <= 3].sum()) / n_chg
            stats = {
                "year_before": y0, "year_after": y1, "resolution_m": grid.res, "resolution_coarsened": coarsened,
                "changed_area_ha": n_chg * ha, "change_fraction": n_chg / n_valid, "valid_fraction": n_valid / valid.size,
                "area_ha_before": _area_table(ca, valid, ha), "area_ha_after": _area_table(cb, valid, ha),
                "transitions": transitions, "min_change_ha": min_change_ha, "pixels_removed_by_min_change": removed,
                "changed_in_patches_of_3px_or_less": small_share, "product_note": IO_NOTE,
            }
            return self._ok(path, stats, {"date_before": str(y0), "date_after": str(y1), "tiles_before": srcs[0],
                                          "tiles_after": srcs[1], "crs": grid.crs}, t0)
        except (ReferenceDataError, ValueError) as exc:
            return self._fail(str(exc), t0)


class UrbanExpansionPipeline(_LulcPair):
    """New and lost built-up area between two years of the 10 m annual land-cover map.

    Built area is class 7 of the Impact Observatory / Esri product. ``date_before`` / ``date_after`` are years
    (2017-2023). Gross gain and gross loss are reported separately because each year is classified on its own:
    "loss" of built area is more often classifier disagreement than demolition, and a net figure would hide that.

    Output ``built_area_change.tif``: 0 not built on either date, 1 built on both, 2 new built area, 3 built area
    lost, 255 nodata (cloud or no data in either year).
    """
    name = "urban_expansion"
    description = "New and lost built-up area between two years from the annual 10 m land-cover product"
    domain = "urban"
    tags = ["urban", "built-up", "change", "annual", "reference-product"]

    def run(self, bbox, output_dir="./output", date_before=None, date_after=None, resolution_m=None,
            **kwargs) -> PipelineResult:
        t0 = time.time()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        try:
            y0, y1, grid, coarsened, ca, cb, valid, srcs = self._pair(bbox, date_before, date_after, resolution_m)
            ha = grid.pixel_area_ha
            b0, b1 = (ca == 7) & valid, (cb == 7) & valid
            cls = np.zeros(ca.shape, np.uint8)
            cls[b0 & b1], cls[~b0 & b1], cls[b0 & ~b1] = 1, 2, 3
            path = write_classes(out / "built_area_change.tif", cls, valid, grid.profile(),
                                 ["0=not built", "1=built both years", "2=new built area", "3=built area lost", "255=nodata"])
            a0, a1 = float(b0.sum()) * ha, float(b1.sum()) * ha
            gain, loss = float((cls == 2).sum()) * ha, float((cls == 3).sum()) * ha
            years = y1 - y0
            stats = {
                "year_before": y0, "year_after": y1, "built_area_ha_before": a0, "built_area_ha_after": a1,
                "gain_ha": gain, "loss_ha": loss, "net_change_ha": a1 - a0,
                "built_share_before": a0 / (valid.sum() * ha), "built_share_after": a1 / (valid.sum() * ha),
                "annual_growth_rate": ((a1 / a0) ** (1.0 / years) - 1.0) if a0 > 0 and a1 > 0 else None,
                "valid_fraction": float(valid.mean()), "resolution_m": grid.res, "resolution_coarsened": coarsened,
                "product_note": IO_NOTE,
            }
            return self._ok(path, stats, {"date_before": str(y0), "date_after": str(y1), "tiles_before": srcs[0],
                                          "tiles_after": srcs[1], "crs": grid.crs}, t0)
        except (ReferenceDataError, ValueError) as exc:
            return self._fail(str(exc), t0)


# ═══════════════════════════════════════════════════════════════════════════
# Forest / non-forest (JAXA PALSAR, 25 m, 2015-2020)
# ═══════════════════════════════════════════════════════════════════════════

FNF_NOTE = ("JAXA global PALSAR/PALSAR-2 forest/non-forest map (25 m, L-band radar, yearly 2015-2020; in this product "
            "'forest' is natural forest with at least 90% canopy cover unless include_open_forest is set). Each year "
            "is mapped separately, so gains and losses include mapping disagreement; plantations count as forest.")


class ForestChangePipeline(_Reference):
    """Forest loss and gain between two years of the JAXA PALSAR forest / non-forest map.

    ``date_before`` / ``date_after`` are years (2015-2020). The product has four classes: forest with >90% canopy
    (1), forest with 10-90% canopy (2), non-forest (3) and water (4). By default "forest" is class 1 only (the
    product's own forest definition); ``include_open_forest=True`` also counts class 2 (the FAO 10% definition).
    Pixels that are water or no-data in either year are nodata.

    Output ``forest_change.tif``: 0 non-forest both years, 1 forest both years, 2 forest loss, 3 forest gain,
    255 nodata.
    """
    name = "forest_change"
    description = "Forest loss and gain between two years (2015-2020) from the JAXA PALSAR forest/non-forest map"
    domain = "forestry"
    tags = ["forest", "change", "radar", "reference-product"]
    collection = "alos-fnf-mosaic"
    native_res_m = 25.0

    def run(self, bbox, output_dir="./output", date_before=None, date_after=None, include_open_forest=False,
            resolution_m=None, **kwargs) -> PipelineResult:
        t0 = time.time()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        try:
            y0, y1 = self._years(date_before, date_after, 2015, 2020)
            grid, coarsened = self._grid(bbox, resolution_m)
            reads = []
            for y in (y0, y1):
                urls, ids = self._annual_tiles(self.collection, "C", bbox, y, self.name)
                r = read_to_grid(urls, grid, dtype="float32", invalid_values=[0])
                r.sources = ids
                reads.append(r)
            a, b = reads
            valid = a.valid & b.valid & (a.data != 4) & (b.data != 4)
            if not valid.any():
                raise ReferenceDataError(f"No pixel is land with data in both {y0} and {y1} inside this area.")
            forest_codes = (1, 2) if include_open_forest else (1,)
            f0, f1 = np.isin(a.data, forest_codes) & valid, np.isin(b.data, forest_codes) & valid
            cls = np.zeros(a.data.shape, np.uint8)
            cls[f0 & f1], cls[f0 & ~f1], cls[~f0 & f1] = 1, 2, 3
            ha = grid.pixel_area_ha
            path = write_classes(out / "forest_change.tif", cls, valid, grid.profile(),
                                 ["0=non-forest", "1=forest both years", "2=forest loss", "3=forest gain", "255=nodata"])
            fa0, fa1 = float(f0.sum()) * ha, float(f1.sum()) * ha
            loss, gain = float((cls == 2).sum()) * ha, float((cls == 3).sum()) * ha
            years = y1 - y0
            stats = {
                "year_before": y0, "year_after": y1, "forest_area_ha_before": fa0, "forest_area_ha_after": fa1,
                "forest_loss_ha": loss, "forest_gain_ha": gain, "net_change_ha": fa1 - fa0,
                "forest_share_before": fa0 / (valid.sum() * ha), "forest_share_after": fa1 / (valid.sum() * ha),
                "loss_ha_per_year": loss / years, "loss_share_of_initial_forest": (loss / fa0) if fa0 > 0 else None,
                "include_open_forest": bool(include_open_forest), "valid_fraction": float(valid.mean()),
                "resolution_m": grid.res, "resolution_coarsened": coarsened, "product_note": FNF_NOTE,
            }
            return self._ok(path, stats, {"date_before": str(y0), "date_after": str(y1), "tiles_before": a.sources,
                                          "tiles_after": b.sources, "crs": grid.crs}, t0)
        except (ReferenceDataError, ValueError) as exc:
            return self._fail(str(exc), t0)


# ═══════════════════════════════════════════════════════════════════════════
# Surface-water history (JRC Global Surface Water, 30 m, 1984-2020)
# ═══════════════════════════════════════════════════════════════════════════

GSW_TRANSITIONS = {1: "Permanent", 2: "New permanent", 3: "Lost permanent", 4: "Seasonal", 5: "New seasonal",
                   6: "Lost seasonal", 7: "Seasonal to permanent", 8: "Permanent to seasonal",
                   9: "Ephemeral permanent", 10: "Ephemeral seasonal"}
GSW_NOTE = ("JRC Global Surface Water v1.3 (Pekel et al. 2016): water detected in every usable Landsat 5/7/8 scene "
            "from March 1984 to December 2020, 30 m. The history ends in 2020 and says nothing about later water; "
            "for a present-day map use water_bodies, and for an event use flood_mapping.")


class SurfaceWaterDynamicsPipeline(_Reference):
    """Where water has been permanent, seasonal, lost or new over 1984-2020 (JRC Global Surface Water).

    Output ``water_transitions.tif`` holds the product's ten transition classes (0 = never water,
    1-10 as listed in ``stats['area_ha_by_class']``; 255 nodata), ``water_occurrence.tif`` the percentage of
    valid observations in which each pixel was water (0-100) and ``water_seasonality.tif`` the number of months of
    water in 2020. Class groups in the stats follow the product's definitions: *permanent now* = classes 1, 2, 7;
    *seasonal now* = 4, 5, 8; *lost* = 3, 6; *ephemeral* = 9, 10.
    """
    name = "surface_water_dynamics"
    description = "Permanent, seasonal, new and lost surface water 1984-2020 from the JRC Global Surface Water history"
    domain = "water"
    tags = ["water", "history", "landsat", "reference-product"]
    collection = "jrc-gsw"
    native_res_m = 30.0

    def _layer(self, items, assets, bbox, name, grid, invalid) -> GridRaster:
        pc = self._pc()
        urls = pc.asset_urls(self.collection, items, name)
        if not urls:
            raise ReferenceDataError(f"the JRC tiles for this area have no '{name}' layer")
        return read_to_grid(urls, grid, dtype="float32", invalid_values=invalid)

    def run(self, bbox, output_dir="./output", resolution_m=None, **kwargs) -> PipelineResult:
        t0 = time.time()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        try:
            grid, coarsened = self._grid(bbox, resolution_m)
            items = self._pc().search(self.collection, bbox)
            if not items:
                raise ReferenceDataError("no JRC Global Surface Water tile covers this area")
            trans = self._layer(items, None, bbox, "transitions", grid, [255])
            occ = self._layer(items, None, bbox, "occurrence", grid, [255])
            sea = self._layer(items, None, bbox, "seasonality", grid, [255])
            valid = trans.valid
            if not valid.any():
                raise ReferenceDataError("the JRC tiles hold no data for this area (open ocean or outside 60N-60S).")
            ha = grid.pixel_area_ha
            cls = np.where(valid, trans.data, 0).astype(np.uint8)
            path = write_classes(out / "water_transitions.tif", cls, valid, grid.profile(),
                                 ["0=never water", *[f"{k}={v}" for k, v in GSW_TRANSITIONS.items()], "255=nodata"])
            write_float(out / "water_occurrence.tif", np.where(occ.valid, occ.data, np.nan), grid.profile(),
                        ["% of valid observations with water, 1984-2020"])
            write_float(out / "water_seasonality.tif", np.where(sea.valid, sea.data, np.nan), grid.profile(),
                        ["months with water in 2020"])
            by_class = {GSW_TRANSITIONS[k]: round(float((cls == k).sum() * ha), 3) for k in GSW_TRANSITIONS if (cls == k).any()}

            def group(codes):
                return float(np.isin(cls, codes).sum() * ha)
            n_valid = int(valid.sum())
            stats = {
                "area_ha_by_class": by_class, "permanent_now_ha": group([1, 2, 7]), "seasonal_now_ha": group([4, 5, 8]),
                "lost_ha": group([3, 6]), "ephemeral_ha": group([9, 10]),
                "ever_water_ha": group(list(GSW_TRANSITIONS)), "ever_water_share": group(list(GSW_TRANSITIONS)) / (n_valid * ha),
                "valid_fraction": n_valid / valid.size, "resolution_m": grid.res, "resolution_coarsened": coarsened,
                "product_note": GSW_NOTE,
            }
            return self._ok(path, stats, {"period": "1984-03 to 2020-12", "tiles": [str(i.get("id")) for i in items],
                                          "crs": grid.crs}, t0)
        except (ReferenceDataError, ValueError) as exc:
            return self._fail(str(exc), t0)


# ═══════════════════════════════════════════════════════════════════════════
# Annual evapotranspiration (MODIS MOD16A3GF, 500 m)
# ═══════════════════════════════════════════════════════════════════════════

ET_NOTE = ("MODIS MOD16A3GF (Running et al. 2019) 500 m annual evapotranspiration and potential evapotranspiration, "
           "modelled with a Penman-Monteith approach from MODIS vegetation data and reanalysis weather. It is a model "
           "product (typical error of tens of percent against flux towers), land only: water, urban and barren "
           "pixels carry fill codes and are nodata here. 1 kg/m2 of water equals 1 mm.")
MOD16_VALID_MAX = 65500           # uint16 codes above this are the product's fill / land-class flags
MODIS_SIN_PIXEL_M = 463.3127165275


def _tile_key(item_id: str) -> str:
    m = re.search(r"h\d{2}v\d{2}", item_id)
    return m.group(0) if m else item_id


class EvapotranspirationPipeline(_Reference):
    """Annual evapotranspiration (ET) and potential ET for one year from MODIS MOD16A3GF.

    ``date`` is the year (2001 onwards). Outputs ``et_mm_per_year.tif``, ``pet_mm_per_year.tif`` and
    ``et_pet_ratio.tif`` (ET / PET, an evaporative-stress indicator: near 1 means water is not limiting, near 0
    means it is). ``stats['total_et_volume_million_m3']`` is ET integrated over the valid area.
    """
    name = "evapotranspiration"
    description = "Annual evapotranspiration, potential ET and their ratio for a year from MODIS MOD16A3GF (500 m)"
    domain = "water"
    tags = ["evapotranspiration", "water-balance", "modis", "reference-product"]
    collection = "modis-16A3GF-061"
    native_res_m = MODIS_SIN_PIXEL_M

    def run(self, bbox, output_dir="./output", date=None, **kwargs) -> PipelineResult:
        t0 = time.time()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        try:
            if date is None:
                raise ReferenceDataError("evapotranspiration needs date: the year, such as 2022.")
            year = parse_year(date, "date")
            if year < 2001:
                raise ReferenceDataError("MOD16A3GF starts in 2001.")
            pc = self._pc()
            items = [it for it in pc.search(self.collection, bbox, datetime=f"{year}-03-01/{year}-09-30")
                     if item_year(it) == year]
            if not items:
                raise ReferenceDataError(f"no MOD16A3GF tile for {year} covers this area (the year may not be released yet).")
            # one item per MODIS tile: Terra (MOD) preferred over Aqua (MYD) when both exist
            best: dict[str, dict] = {}
            for it in sorted(items, key=lambda i: (not str(i.get("id")).startswith("MOD"), str(i.get("id")))):
                best.setdefault(_tile_key(str(it.get("id"))), it)
            items = list(best.values())
            grid, coarsened = self._grid(bbox, kwargs.get("resolution_m"))
            et = read_to_grid(pc.asset_urls(self.collection, items, "ET_500m"), grid, dtype="float32")
            pet = read_to_grid(pc.asset_urls(self.collection, items, "PET_500m"), grid, dtype="float32")
            v_et = et.valid & (et.data <= MOD16_VALID_MAX)
            v_pet = pet.valid & (pet.data <= MOD16_VALID_MAX)
            if not v_et.any():
                raise ReferenceDataError("every pixel in this area is a MOD16 fill/land-class code "
                                         "(open water, urban or barren land): there is no evapotranspiration estimate.")
            etm = np.where(v_et, et.data * 0.1, np.nan).astype("float32")       # kg/m2/year == mm/year
            petm = np.where(v_pet, pet.data * 0.1, np.nan).astype("float32")
            ratio = np.full(etm.shape, np.nan, "float32")
            ok = v_et & v_pet & (petm > 0)
            ratio[ok] = etm[ok] / petm[ok]
            path = write_float(out / "et_mm_per_year.tif", etm, grid.profile(), ["ET, mm/year"])
            write_float(out / "pet_mm_per_year.tif", petm, grid.profile(), ["PET, mm/year"])
            write_float(out / "et_pet_ratio.tif", ratio, grid.profile(), ["ET / PET"])
            area_m2 = grid.res * grid.res
            stats = {
                "year": year, "mean_et_mm": float(np.nanmean(etm)), "mean_pet_mm": float(np.nanmean(petm)) if v_pet.any() else None,
                "mean_et_pet_ratio": float(np.nanmean(ratio)) if ok.any() else None,
                "min_et_mm": float(np.nanmin(etm)), "max_et_mm": float(np.nanmax(etm)),
                "total_et_volume_million_m3": float(np.nansum(etm) / 1000.0 * area_m2 / 1e6),
                "valid_fraction": float(v_et.mean()), "resolution_m": grid.res, "product_note": ET_NOTE,
            }
            return self._ok(path, stats, {"date": str(year), "tiles": [str(i.get("id")) for i in items], "crs": grid.crs}, t0)
        except (ReferenceDataError, ValueError) as exc:
            return self._fail(str(exc), t0)


# ═══════════════════════════════════════════════════════════════════════════
# Active fire (MODIS MOD14A1, 1 km, daily)
# ═══════════════════════════════════════════════════════════════════════════

FIRE_NOTE = ("MODIS MOD14A1 daily thermal-anomaly product (1 km). A flagged pixel means at least one fire was "
             "burning inside that kilometre at the satellite overpass; the burning area is far smaller than the pixel, "
             "so the counts are of detections, not hectares burned. Cloud hides fires, and small or cool fires are "
             "missed. Pixels are resampled to a UTM grid with nearest neighbour, so counts can differ from the raw MODIS "
             "tile by a small fraction.")
FIRE_LEVELS = {"low": 7, "nominal": 8, "high": 9}


class ActiveFirePipeline(_Reference):
    """Days with a MODIS active-fire detection, per 1 km pixel, over a period.

    Give ``date="YYYY-MM"`` for a month, or ``date_before`` / ``date_after`` (``YYYY-MM-DD``) for explicit first
    and last days. A detection is a MOD14A1 FireMask value of at least ``min_confidence`` (``low`` = 7,
    ``nominal`` = 8, ``high`` = 9). Output ``fire_days.tif`` counts the days with a detection; ``observed_days.tif``
    counts the days the pixel could be observed (clear land), so a quiet pixel can be told from a hidden one.
    ``fire_detections.gpkg`` has one point per pixel-day with its date, confidence class and fire radiative power.
    """
    name = "active_fire"
    description = "Days with a MODIS active-fire detection per 1 km pixel for a month or date range, with fire points"
    domain = "disaster"
    tags = ["fire", "thermal", "modis", "reference-product"]
    collection = "modis-14A1-061"
    native_res_m = 926.6254331                # the MOD14A1 'kilometre'

    MAX_DAYS = 366

    def run(self, bbox, output_dir="./output", date=None, date_before=None, date_after=None,
            min_confidence="nominal", **kwargs) -> PipelineResult:
        t0 = time.time()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        try:
            from pygeovision.data._dates import date_window
            if min_confidence not in FIRE_LEVELS:
                raise ReferenceDataError(f"min_confidence must be one of {sorted(FIRE_LEVELS)}.")
            if date_before and date_after:
                start, end = date_window(str(date_before))[0], date_window(str(date_after))[1]
            elif date:
                start, end = date_window(str(date))
            else:
                raise ReferenceDataError("active_fire needs date (YYYY-MM) or both date_before and date_after.")
            if start > end:
                raise ReferenceDataError("date_before must not be later than date_after.")
            pc = self._pc()
            items = pc.search(self.collection, bbox, datetime=f"{start}/{end}")
            if not items:
                raise ReferenceDataError(f"no MOD14A1 granule covers this area between {start} and {end}.")
            by_day: dict[str, list[dict]] = {}
            for it in items:
                p = it.get("properties") or {}
                stamp = p.get("start_datetime") or p.get("datetime")
                if not stamp:
                    raise ReferenceDataError(f"granule {it.get('id')} has no acquisition date")
                by_day.setdefault(str(stamp)[:10], []).append(it)
            days = sorted(by_day)
            if len(days) > self.MAX_DAYS:
                raise ReferenceDataError(f"{len(days)} days requested; the limit is {self.MAX_DAYS} per run.")
            grid, _ = self._grid(bbox, kwargs.get("resolution_m"))
            thr = FIRE_LEVELS[min_confidence]
            fire_days = np.zeros((grid.height, grid.width), np.uint16)
            observed = np.zeros_like(fire_days)
            max_frp = np.full((grid.height, grid.width), np.nan, "float32")
            points: list[dict[str, Any]] = []
            per_day: dict[str, int] = {}
            for d in days:
                group = by_day[d]
                fm = read_to_grid(pc.asset_urls(self.collection, group, "FireMask"), grid, dtype="float32", resampling="nearest")
                frp = read_to_grid(pc.asset_urls(self.collection, group, "MaxFRP"), grid, dtype="float32", resampling="nearest")
                v = fm.valid
                code = np.where(v, fm.data, 0)
                clear_land = v & np.isin(code, (5, 7, 8, 9))                   # 5 = land with no fire; 7-9 = fire
                fire = v & (code >= thr)
                observed += clear_land.astype(np.uint16)
                fire_days += fire.astype(np.uint16)
                per_day[d] = int(fire.sum())
                f = np.where(frp.valid & fire, frp.data * 0.1, np.nan)             # MaxFRP is stored x10, MW
                max_frp = np.fmax(max_frp, f.astype("float32"))
                rr, cc = np.nonzero(fire)
                for r, c in zip(rr.tolist(), cc.tolist()):
                    x, y = grid.transform * (c + 0.5, r + 0.5)
                    points.append({"x": x, "y": y, "date": d, "firemask": int(code[r, c]),
                                   "confidence": {7: "low", 8: "nominal", 9: "high"}[int(code[r, c])],
                                   "frp_mw": None if not np.isfinite(f[r, c]) else float(f[r, c])})
            valid_any = observed > 0
            path = write_float(out / "fire_days.tif", np.where(valid_any | (fire_days > 0), fire_days, np.nan), grid.profile(),
                               ["days with a fire detection"])
            write_float(out / "observed_days.tif", observed.astype("float32"), grid.profile(), ["days the pixel was observed clear"])
            write_float(out / "max_frp_mw.tif", max_frp, grid.profile(), ["maximum fire radiative power, MW"])
            n_det = len(points)
            if points:
                import geopandas as gpd
                gdf = gpd.GeoDataFrame(points, geometry=gpd.points_from_xy([q["x"] for q in points], [q["y"] for q in points]),
                                       crs=grid.crs).drop(columns=["x", "y"]).to_crs("EPSG:4326")
                gdf.to_file(out / "fire_detections.gpkg", driver="GPKG", layer="fire_detections")
            n_pix = int((fire_days > 0).sum())
            stats = {
                "start": start, "end": end, "days_with_data": len(days), "min_confidence": min_confidence,
                "detections_pixel_days": n_det, "pixels_with_fire": n_pix,
                "pixels_with_fire_share": n_pix / (grid.width * grid.height),
                "max_days_with_fire_in_a_pixel": int(fire_days.max()),
                "max_frp_mw": float(np.nanmax(max_frp)) if np.isfinite(max_frp).any() else None,
                "mean_observed_days": float(observed[valid_any].mean()) if valid_any.any() else None,
                "per_day_detections": per_day, "resolution_m": grid.res, "product_note": FIRE_NOTE,
            }
            meta: dict[str, Any] = {"date_before": start, "date_after": end, "crs": grid.crs, "granules": len(items)}
            if points:
                meta["vector_files"] = [{"name": "fire_detections", "path": str(out / "fire_detections.gpkg")}]
            return self._ok(path, stats, meta, t0)
        except (ReferenceDataError, ValueError) as exc:
            return self._fail(str(exc), t0)


REFERENCE_PIPELINES: dict[str, type] = {
    cls.name: cls for cls in (
        LandCoverChangePipeline, UrbanExpansionPipeline, ForestChangePipeline, SurfaceWaterDynamicsPipeline,
        EvapotranspirationPipeline, ActiveFirePipeline,
    )
}
