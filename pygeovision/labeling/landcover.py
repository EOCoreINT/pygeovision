"""
ESA WorldCover and 10 m land-cover auto-labelers.

Both download only the part of the global product that covers your bounding box (Cloud-Optimised
GeoTIFF windows from Microsoft Planetary Computer), mosaic the tiles, and write one ``uint8`` raster
with the product's own class numbers (or yours, via ``remap_classes``). Pass ``reference_raster`` to
get the labels on the exact grid of your imagery, which is what training chips need.

Read ``docs/labeling/landcover.md`` for what these products are -- they are *model outputs at 10 m*,
good for coarse land cover and wrong for small or changing things.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"

# ESA WorldCover (10 m, 11 classes; 0 = no data)
ESA_WORLDCOVER_CLASSES = {
    10: "Tree cover", 20: "Shrubland", 30: "Grassland",
    40: "Cropland", 50: "Built-up", 60: "Bare/sparse vegetation",
    70: "Snow and ice", 80: "Permanent water bodies",
    90: "Herbaceous wetland", 95: "Mangroves", 100: "Moss and lichen",
}

# Google Dynamic World label band (0-8). Only the Earth Engine backend returns these numbers.
DW_CLASSES = {
    0: "water", 1: "trees", 2: "grass", 3: "flooded_vegetation",
    4: "crops", 5: "shrub_and_scrub", 6: "built", 7: "bare", 8: "snow_and_ice",
}

# Impact Observatory / Esri "10m Annual Land Use Land Cover (9-class)" -- what the
# Planetary Computer backend returns. NOT the same numbering as Dynamic World (1-based, no 0, 3, 6; adds
# 10 = clouds and 11 = rangeland).
IO_LULC_CLASSES = {
    1: "water", 2: "trees", 4: "flooded_vegetation", 5: "crops",
    7: "built", 8: "bare", 9: "snow_and_ice", 10: "clouds", 11: "rangeland",
}
#: A partial, lossy bridge from IO LULC numbers to Dynamic World numbers (rangeland has no equal; clouds neither).
IO_TO_DYNAMIC_WORLD = {1: 0, 2: 1, 4: 3, 5: 4, 7: 6, 8: 7, 9: 8}


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _stac_search(collection: str, bbox, datetime: str, limit: int = 20):
    """POST a STAC search. Returns ``(items, error)``; exactly one of them is meaningful."""
    import requests

    params = {"collections": [collection], "bbox": list(bbox), "datetime": datetime, "limit": limit}
    try:
        resp = requests.post(f"{PC_STAC}/search", json=params, timeout=30)
    except Exception as exc:  # noqa: BLE001
        return [], f"STAC search failed: {type(exc).__name__}: {exc}"
    if resp.status_code != 200:
        return [], f"STAC search returned HTTP {resp.status_code}"
    try:
        return resp.json().get("features", []), None
    except Exception as exc:  # noqa: BLE001
        return [], f"STAC search returned an unreadable response: {exc}"


def _sign(href: str) -> str:
    """Sign a Planetary Computer asset URL when the ``planetary-computer`` package is installed."""
    try:
        import planetary_computer as pc
    except ImportError:
        logger.warning("planetary-computer is not installed; unsigned asset URLs usually return 403. "
                       "pip install planetary-computer")
        return href
    return pc.sign(href)


def remap_array(arr, mapping: dict[int, int]):
    """Apply ``mapping`` to every value *at once* (so {10: 20, 20: 30} does not chain 10 -> 30)."""
    out = arr.copy()
    for src_val, dst_val in mapping.items():
        out[arr == src_val] = dst_val
    return out


def mosaic_window(hrefs: list[str], bbox, output: Path, remap: dict[int, int] | None = None,
                  nodata: int = 0) -> dict[str, Any]:
    """Mosaic the part of ``hrefs`` that lies in ``bbox`` (lon/lat) into one ``uint8`` GeoTIFF.

    The output CRS is that of the first source; sources in other CRSs (e.g. neighbouring UTM zones) are
    warped to it with nearest-neighbour. Only the bbox window of each source is read, and each window
    keeps its own georeferencing (a window clipped by a tile edge is not stretched over the whole bbox).
    """
    import numpy as np
    import rasterio
    from rasterio.merge import merge
    from rasterio.vrt import WarpedVRT
    from rasterio.warp import Resampling, transform_bounds

    opened, vrts = [], []
    try:
        for h in hrefs:
            opened.append(rasterio.open(h))
        dst_crs = opened[0].crs
        sources = []
        for ds in opened:
            if ds.crs != dst_crs:
                v = WarpedVRT(ds, crs=dst_crs, resampling=Resampling.nearest)
                vrts.append(v)
                sources.append(v)
            else:
                sources.append(ds)
        bounds = transform_bounds("EPSG:4326", dst_crs, *bbox, densify_pts=21)
        mosaic, transform = merge(sources, bounds=bounds, nodata=nodata, method="first")
        profile = {"driver": "GTiff", "height": mosaic.shape[1], "width": mosaic.shape[2], "count": 1,
                   "dtype": "uint8", "crs": dst_crs, "transform": transform, "nodata": nodata,
                   "compress": "lzw"}
    finally:
        for v in vrts:
            v.close()
        for ds in opened:
            ds.close()

    data = mosaic[0]
    if remap:
        data = remap_array(data, remap)
    if data.max() > 255 or data.min() < 0:
        raise ValueError("remapped class values must fit in 0-255")
    output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(str(output), "w", **profile) as dst:
        dst.write(data.astype(np.uint8)[np.newaxis])
    return {"height": profile["height"], "width": profile["width"], "crs": str(dst_crs)}


def align_to_reference(label_path: str | Path, reference_raster: str | Path, output_path: str | Path | None = None,
                       nodata: int = 0) -> Path:
    """Resample a label raster onto ``reference_raster``'s exact grid and CRS (nearest-neighbour).

    Class values are never blended. Pixels of the reference that the label raster does not cover get
    ``nodata``. Writes in place of ``label_path`` when ``output_path`` is omitted.
    """
    import numpy as np
    import rasterio
    from rasterio.warp import Resampling, reproject

    label_path = Path(label_path)
    out = Path(output_path) if output_path else label_path
    with rasterio.open(str(reference_raster)) as ref:
        ref_crs, ref_tf, h, w = ref.crs, ref.transform, ref.height, ref.width
    with rasterio.open(str(label_path)) as src:
        dst = np.full((h, w), nodata, dtype=np.uint8)
        reproject(rasterio.band(src, 1), dst, src_transform=src.transform, src_crs=src.crs,
                  src_nodata=src.nodata, dst_transform=ref_tf, dst_crs=ref_crs, dst_nodata=nodata,
                  resampling=Resampling.nearest)
        tags = src.tags()
    tmp = out.with_name(out.stem + "._aligned.tif")
    with rasterio.open(str(tmp), "w", driver="GTiff", height=h, width=w, count=1, dtype="uint8",
                       crs=ref_crs, transform=ref_tf, nodata=nodata, compress="lzw") as o:
        o.write(dst[np.newaxis])
        o.update_tags(**tags)
    tmp.replace(out)
    return out


def _class_counts(path: str) -> dict[int, int]:
    import numpy as np
    import rasterio
    with rasterio.open(path) as src:
        data = src.read(1)
    unique, counts = np.unique(data, return_counts=True)
    return {int(k): int(v) for k, v in zip(unique, counts)}


# ── ESA WorldCover ─────────────────────────────────────────────────────────────

class ESAWorldCoverLabeler:
    """Land-cover labels from ESA WorldCover 2020 (v100) or 2021 (v200).

    Free global map at 10 m with 11 classes (see ``ESA_WORLDCOVER_CLASSES``). Data come from the
    Microsoft Planetary Computer STAC; only the bbox window is read.

    Example::

        labeler = ESAWorldCoverLabeler(year=2021)
        result = labeler.label(
            bbox=(-87.7, 41.8, -87.5, 41.9),
            output_path="./labels/chicago_lc.tif",
            reference_raster="./data/chicago_s2.tif",     # labels on the imagery's grid
        )
    """

    COLLECTION = "esa-worldcover"
    PC_STAC = PC_STAC

    def __init__(self, year: int = 2021) -> None:
        if year not in (2020, 2021):
            raise ValueError("ESA WorldCover available for year=2020 or 2021")
        self.year = year

    def label(
        self,
        bbox: tuple[float, ...],
        output_path: str | Path = "./labels/esa_worldcover.tif",
        remap_classes: dict[int, int] | None = None,
        clip_to_bbox: bool = True,
        reference_raster: str | None = None,
    ) -> dict[str, Any]:
        """Download and mosaic WorldCover for the bbox.

        Args:
            bbox: ``(lon_min, lat_min, lon_max, lat_max)`` in degrees.
            output_path: Output label GeoTIFF (``uint8``, 0 = no data).
            remap_classes: ``{esa_value: new_value}`` applied to all pixels at once, e.g.
                ``{50: 1}`` plus ``{v: 0 for v in (10,20,30,40,60,70,80,90,95,100)}`` for "built-up
                vs not". Values you do not list keep their ESA number.
            clip_to_bbox: Kept for compatibility. The output is always the bbox window; the window
                is snapped outward to whole pixels, so it can be up to one pixel larger than the bbox.
            reference_raster: Resample the result onto this raster's grid and CRS (nearest-neighbour).

        Returns:
            ``success``, ``output_path``, ``year``, ``classes_found`` (value -> pixel count),
            ``class_map``, ``n_tiles``, ``error``.
        """
        output_path = Path(output_path)
        items, err = _stac_search(self.COLLECTION, bbox, f"{self.year}-01-01/{self.year}-12-31")
        if err:
            return {"success": False, "error": err}
        if not items:
            return {"success": False, "error": f"No ESA WorldCover {self.year} tiles found for bbox={tuple(bbox)} "
                                               "-- check the bbox (lon_min, lat_min, lon_max, lat_max)."}
        hrefs = [_sign(i["assets"]["map"]["href"]) for i in items if i.get("assets", {}).get("map", {}).get("href")]
        if not hrefs:
            return {"success": False, "error": "STAC items had no 'map' asset"}

        try:
            mosaic_window(hrefs, bbox, output_path, remap_classes, nodata=0)
            if reference_raster:
                align_to_reference(output_path, reference_raster, nodata=0)
        except Exception as exc:  # noqa: BLE001
            logger.error("ESA WorldCover read failed: %s", exc)
            return {"success": False, "error": f"could not read WorldCover tiles: {type(exc).__name__}: {exc}"}

        return {
            "success": True,
            "output_path": str(output_path),
            "year": self.year,
            "classes_found": _class_counts(str(output_path)),
            "class_map": ESA_WORLDCOVER_CLASSES,
            "n_tiles": len(hrefs),
            "remapped": bool(remap_classes),
            "error": None,
        }

    @staticmethod
    def class_map() -> dict[int, str]:
        return ESA_WORLDCOVER_CLASSES


# ── "Dynamic World" ────────────────────────────────────────────────────────────

class DynamicWorldLabeler:
    """10 m, 9-class land-cover labels from one of two backends. **Read which one you get.**

    ``backend="planetary_computer"`` (default, no account): the Impact Observatory / Esri *10 m Annual
    Land Use Land Cover (9-class)* product (collection ``io-lulc-9-class``). It is **not** Google's
    Dynamic World: it is an annual map (one per year, 2017-2023), and its class numbers are
    ``IO_LULC_CLASSES`` (1 water, 2 trees, 4 flooded vegetation, 5 crops, 7 built, 8 bare, 9 snow/ice,
    10 clouds, 11 rangeland), different from Dynamic World's 0-8. The returned ``class_map`` is the one
    that matches the numbers in the file.

    ``backend="earth_engine"``: real Dynamic World (``GOOGLE/DYNAMICWORLD/V1``, numbers ``DW_CLASSES``),
    needs ``pip install earthengine-api`` and an authenticated Earth Engine project. Earth Engine only
    exports to Google Drive, so this call **starts an export task and returns immediately**; the file
    appears in your Drive later.

    Example::

        result = DynamicWorldLabeler().label(
            bbox=(-0.15, 51.47, -0.10, 51.52),
            date_range=("2023-01-01", "2023-12-31"),
            output_path="./labels/london_lc.tif",
            reference_raster="./data/london_s2.tif",
        )
    """

    EE_COLLECTION = "GOOGLE/DYNAMICWORLD/V1"
    PC_COLLECTION = "io-lulc-9-class"

    def __init__(self, backend: str = "planetary_computer") -> None:
        self.backend = backend

    def label(
        self,
        bbox: tuple[float, ...],
        date_range: tuple[str, str] = ("2023-01-01", "2023-12-31"),
        output_path: str | Path = "./labels/dynamic_world.tif",
        label_mode: str = "most_likely",
        remap_classes: dict[int, int] | None = None,
        reference_raster: str | None = None,
    ) -> dict[str, Any]:
        """Produce a land-cover label raster for ``bbox``.

        Args:
            bbox: ``(lon_min, lat_min, lon_max, lat_max)`` in degrees.
            date_range: ``(start, end)``. Planetary Computer: the newest *annual* map whose year
                overlaps this range is used (the collection currently ends in 2023, so a 2024 range
                finds nothing and returns an error). Earth Engine: the per-pixel mode of the label
                band over the range.
            output_path: Output GeoTIFF (``uint8``, 0 = no data on the Planetary Computer backend).
            label_mode: Only ``"most_likely"`` (one class per pixel) is supported. ``"all_probabilities"``
                returns an error rather than being ignored.
            remap_classes: ``{value_in_file: new_value}``. ``IO_TO_DYNAMIC_WORLD`` is a ready-made
                partial mapping to Dynamic World numbers (rangeland and clouds have no equivalent).
            reference_raster: Resample onto this raster's grid and CRS (nearest-neighbour). Planetary
                Computer backend only.
        """
        output_path = Path(output_path)
        if label_mode != "most_likely":
            return {"success": False,
                    "error": f"label_mode={label_mode!r} is not supported: only 'most_likely' is implemented."}
        if self.backend == "planetary_computer":
            return self._label_via_pc(bbox, date_range, output_path, remap_classes, reference_raster)
        if self.backend == "earth_engine":
            if remap_classes or reference_raster:
                return {"success": False, "error": "remap_classes / reference_raster are not supported by the "
                                                   "earth_engine backend (it only starts a Drive export)."}
            return self._label_via_ee(bbox, date_range, output_path)
        return {"success": False, "error": f"Unknown backend: {self.backend!r} "
                                           "(use 'planetary_computer' or 'earth_engine')"}

    def _label_via_pc(self, bbox, date_range, output_path, remap_classes, reference_raster):
        items, err = _stac_search(self.PC_COLLECTION, bbox, f"{date_range[0]}/{date_range[1]}")
        if err:
            return {"success": False, "error": err}
        if not items:
            return {"success": False,
                    "error": (f"No annual land-cover map in collection '{self.PC_COLLECTION}' overlaps "
                              f"{date_range[0]} to {date_range[1]} for bbox={tuple(bbox)} (the collection "
                              "covers 2017-2023). Try an earlier year.")}

        def year_of(item) -> int:
            props = item.get("properties", {})
            stamp = props.get("start_datetime") or props.get("datetime") or ""
            try:
                return int(str(stamp)[:4])
            except ValueError:
                return -1

        year = max(year_of(i) for i in items)
        chosen = [i for i in items if year_of(i) == year] if year > 0 else items
        hrefs = [_sign(i["assets"]["data"]["href"]) for i in chosen if i.get("assets", {}).get("data", {}).get("href")]
        if not hrefs:
            return {"success": False, "error": "No data asset in the STAC items"}

        try:
            mosaic_window(hrefs, bbox, output_path, remap_classes, nodata=0)
            if reference_raster:
                align_to_reference(output_path, reference_raster, nodata=0)
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "error": f"could not read land-cover tiles: {type(exc).__name__}: {exc}"}

        return {
            "success": True,
            "output_path": str(output_path),
            "product": "io-lulc-9-class (Impact Observatory / Esri annual map, NOT Google Dynamic World)",
            "year": year if year > 0 else None,
            "class_map": IO_LULC_CLASSES if not remap_classes else None,
            "classes_found": _class_counts(str(output_path)),
            "n_classes": len(IO_LULC_CLASSES),
            "n_tiles": len(hrefs),
            "remapped": bool(remap_classes),
            "error": None,
        }

    def _label_via_ee(self, bbox, date_range, output_path):
        try:
            import ee
            lon_min, lat_min, lon_max, lat_max = bbox
            region = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])
            col = (ee.ImageCollection(self.EE_COLLECTION)
                   .filterBounds(region)
                   .filterDate(*date_range))
            label = col.select("label").mode().clip(region)
            task = ee.batch.Export.image.toDrive(
                image=label, description="dynamic_world_label",
                scale=10, region=region,
            )
            task.start()
            return {"success": True, "submitted": True, "output_path": None,
                    "note": "Export task started; the GeoTIFF will appear in your Google Drive when it finishes. "
                            "Nothing was written to disk.",
                    "ee_task": str(task.id), "class_map": DW_CLASSES, "n_classes": len(DW_CLASSES)}
        except ImportError:
            return {"success": False, "error": "pip install earthengine-api"}
        except Exception as exc:
            return {"success": False, "error": f"{type(exc).__name__}: {exc}"}

    @staticmethod
    def class_map() -> dict[int, str]:
        """Class numbers of the *Earth Engine* backend. The Planetary Computer backend uses ``IO_LULC_CLASSES``."""
        return DW_CLASSES
