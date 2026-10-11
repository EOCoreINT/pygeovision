"""
OSM Auto-Labeler — generate training labels from OpenStreetMap.

Uses the Overpass API to fetch building footprints, roads, water, land-use polygons and a few
other categories inside a bounding box, then rasterises them to a GeoTIFF label mask (one class
value per category) and, optionally, writes the raw features as GeoJSON.

What the raster contains (see ``docs/labeling/osm.md``):

* **Areas** (buildings, land use, water bodies, parking, sports, solar) are filled polygons.
  ``multipolygon`` relations (large lakes, courtyard buildings) are assembled from their member ways.
* **Lines** (roads, rivers and canals) are buffered to a width in metres chosen from the road /
  waterway class, then burned with ``all_touched`` so a 4 m service road still leaves a connected
  line at 10 m resolution. (An earlier version closed every road into a polygon, which painted
  a filled blob between its two ends.)
* Where features overlap, the more specific one wins: buildings over roads over water over parking/sports
  over vegetation over zoning-type land use such as residential (see ``DRAW_ORDER``). The class numbers are unchanged (buildings = 1 ... sports = 10).
* If ``reference_raster`` is given, the labels are written on exactly its grid, in its CRS.
"""
from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Largest label raster this module will allocate (pixels). A 1 deg x 1 deg box at 1 m would be ~12 billion.
MAX_LABEL_PIXELS = 400_000_000

OSM_CATEGORIES = {
    "buildings": {
        "query": 'way["building"]{bbox};relation["building"]{bbox};',
        "value": 1,
        "description": "All building footprints",
    },
    "roads": {
        "query": 'way["highway"~"^(motorway|trunk|primary|secondary|tertiary|unclassified|residential|service)(_link)?$"]{bbox};',
        "value": 2,
        "description": "Motorway to service-level roads (buffered lines)",
    },
    "water": {
        "query": 'way["natural"="water"]{bbox};relation["natural"="water"]{bbox};way["waterway"]{bbox};',
        "value": 3,
        "description": "Water bodies (polygons) and waterways (buffered lines)",
    },
    "vegetation": {
        "query": 'way["landuse"~"^(forest|meadow|grass|orchard|vineyard)$"]{bbox};relation["landuse"~"^(forest|meadow|grass|orchard|vineyard)$"]{bbox};way["natural"~"^(wood|scrub|heath)$"]{bbox};relation["natural"~"^(wood|scrub|heath)$"]{bbox};',
        "value": 4,
        "description": "Forest, meadow, grass, orchard, vineyard, wood, scrub, heath",
    },
    "agricultural": {
        "query": 'way["landuse"~"^(farmland|farmyard|greenhouse_horticulture)$"]{bbox};relation["landuse"~"^(farmland|farmyard|greenhouse_horticulture)$"]{bbox};',
        "value": 5,
        "description": "Agricultural areas",
    },
    "commercial": {
        "query": 'way["landuse"~"^(commercial|retail|industrial)$"]{bbox};relation["landuse"~"^(commercial|retail|industrial)$"]{bbox};',
        "value": 6,
        "description": "Commercial and industrial zones",
    },
    "residential": {
        "query": 'way["landuse"="residential"]{bbox};relation["landuse"="residential"]{bbox};',
        "value": 7,
        "description": "Residential zones",
    },
    "solar_panels": {
        "query": 'way["generator:source"="solar"]{bbox};',
        "value": 8,
        "description": "Solar energy installations (mapped as areas)",
    },
    "parking": {
        "query": 'way["amenity"="parking"]{bbox};relation["amenity"="parking"]{bbox};',
        "value": 9,
        "description": "Parking lots and structures",
    },
    "sports": {
        "query": 'way["leisure"~"^(pitch|track|sports_centre)$"]{bbox};relation["leisure"~"^(pitch|track|sports_centre)$"]{bbox};',
        "value": 10,
        "description": "Sports facilities",
    },
}

#: Burn order, first = underneath. More specific features are drawn later and win overlaps.
DRAW_ORDER = ["residential", "commercial", "agricultural", "vegetation", "sports",
              "parking", "solar_panels", "water", "roads", "buildings"]

#: Full road width in metres by ``highway`` value (``*_link`` uses the parent class).
ROAD_WIDTH_M = {"motorway": 14.0, "trunk": 12.0, "primary": 10.0, "secondary": 8.0, "tertiary": 7.0,
                "unclassified": 6.0, "residential": 6.0, "service": 4.0}
#: Full width in metres by ``waterway`` value.
WATERWAY_WIDTH_M = {"river": 15.0, "canal": 10.0, "stream": 4.0, "ditch": 3.0, "drain": 3.0}

_VEG_LANDUSE = {"forest", "meadow", "grass", "orchard", "vineyard"}
_VEG_NATURAL = {"wood", "scrub", "heath"}
_AGRI_LANDUSE = {"farmland", "farmyard", "greenhouse_horticulture"}
_COMM_LANDUSE = {"commercial", "retail", "industrial"}
_SPORTS_LEISURE = {"pitch", "track", "sports_centre"}


def classify_tags(tags: dict) -> tuple[str, str] | None:
    """Return ``(category, geometry_kind)`` for an OSM tag set, or ``None`` if it matches no category.

    ``geometry_kind`` is ``"line"`` (roads, waterways) or ``"area"``. When a feature matches several
    categories the first of buildings, water, roads, solar, parking, sports, vegetation, agricultural,
    commercial, residential wins (a building that is also tagged as parking is a building).
    """
    b = tags.get("building")
    if b and b != "no":
        return "buildings", "area"
    if tags.get("natural") == "water" or tags.get("waterway") == "riverbank":
        return "water", "area"
    if tags.get("waterway"):
        return "water", "line"
    hw = str(tags.get("highway", ""))
    if hw:
        base = hw[:-5] if hw.endswith("_link") else hw
        if base in ROAD_WIDTH_M:
            return "roads", "line"
    if tags.get("generator:source") == "solar":
        return "solar_panels", "area"
    if tags.get("amenity") == "parking":
        return "parking", "area"
    if tags.get("leisure") in _SPORTS_LEISURE:
        return "sports", "area"
    if tags.get("landuse") in _VEG_LANDUSE or tags.get("natural") in _VEG_NATURAL:
        return "vegetation", "area"
    if tags.get("landuse") in _AGRI_LANDUSE:
        return "agricultural", "area"
    if tags.get("landuse") in _COMM_LANDUSE:
        return "commercial", "area"
    if tags.get("landuse") == "residential":
        return "residential", "area"
    return None


def _line_width_m(tags: dict, category: str) -> float:
    if category == "roads":
        hw = str(tags.get("highway", ""))
        return ROAD_WIDTH_M.get(hw[:-5] if hw.endswith("_link") else hw, 5.0)
    return WATERWAY_WIDTH_M.get(str(tags.get("waterway", "")), 4.0)


def _relation_polygon(elem: dict):
    """Assemble a ``multipolygon`` relation (``out geom`` members) into a shapely (Multi)Polygon, or None."""
    try:
        from shapely.geometry import LineString
        from shapely.ops import polygonize, unary_union
    except ImportError:  # pragma: no cover
        return None
    outers, inners = [], []
    for m in elem.get("members", []):
        geom = m.get("geometry")
        if m.get("type") != "way" or not geom or len(geom) < 2:
            continue
        line = LineString([(n["lon"], n["lat"]) for n in geom])
        (inners if m.get("role") == "inner" else outers).append(line)
    if not outers:
        return None
    polys = list(polygonize(unary_union(outers)))
    if not polys:
        return None
    result = unary_union(polys)
    if inners:
        holes = list(polygonize(unary_union(inners)))
        if holes:
            result = result.difference(unary_union(holes))
    return None if result.is_empty else result


OVERPASS_ENDPOINTS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.fr/api/interpreter",
]


class OSMLabeler:
    """Generate geospatial training labels from OpenStreetMap via the Overpass API.

    Downloads vector features for the requested categories inside a bounding box and rasterises
    them to a single-band ``uint8`` label GeoTIFF: ``0`` background, then one value per category
    (``buildings`` 1, ``roads`` 2, ``water`` 3, ``vegetation`` 4, ``agricultural`` 5,
    ``commercial`` 6, ``residential`` 7, ``solar_panels`` 8, ``parking`` 9, ``sports`` 10).

    OSM is volunteer-mapped: coverage and completeness vary hugely by country, and a pixel with no
    label means "nobody mapped it", not "there is nothing there". See ``docs/labeling/osm.md``.

    Example::

        labeler = OSMLabeler()
        result = labeler.label(
            bbox=(-74.05, 40.70, -73.95, 40.80),
            categories=["buildings", "roads", "water"],
            output_path="./labels/nyc_osm.tif",
            reference_raster="./data/nyc_sentinel2.tif",  # recommended: labels on the imagery's grid
        )
    """

    def __init__(
        self,
        endpoint: str | None = None,
        timeout: int = 120,
        retry_attempts: int = 3,
        crs: str = "EPSG:4326",
    ) -> None:
        self.endpoint = endpoint or OVERPASS_ENDPOINTS[0]
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.crs = crs
        self._last_error: str | None = None

    # ── Overpass query ────────────────────────────────────────────────────────
    @staticmethod
    def _validate_categories(categories: list[str]) -> None:
        unknown = [c for c in categories if c not in OSM_CATEGORIES]
        if unknown:
            raise ValueError(f"Unknown OSM categories {unknown}. Valid: {sorted(OSM_CATEGORIES)}")
        if not categories:
            raise ValueError("categories is empty")

    def _build_overpass_query(self, bbox: tuple[float, ...], categories: list[str]) -> str:
        self._validate_categories(categories)
        lon_min, lat_min, lon_max, lat_max = bbox
        # Overpass uses lat_min,lon_min,lat_max,lon_max
        bbox_str = f"({lat_min},{lon_min},{lat_max},{lon_max})"
        parts = [OSM_CATEGORIES[cat]["query"].replace("{bbox}", bbox_str) for cat in categories]
        return "[out:json][timeout:{}];\n(\n{}\n);\nout body geom;".format(
            self.timeout, "\n".join(parts)
        )

    def _fetch_overpass(self, query: str) -> dict | None:
        self._last_error = None
        endpoints = [self.endpoint] + [e for e in OVERPASS_ENDPOINTS if e != self.endpoint]
        for attempt in range(1, self.retry_attempts + 1):
            for endpoint in endpoints:
                try:
                    import requests

                    from pygeovision._version import __version__
                    resp = requests.post(endpoint, data={"data": query}, timeout=self.timeout,
                                         headers={"User-Agent": f"PyGeoVision/{__version__} (+https://github.com/pygeovision)",
                                                  "Accept": "*/*"})
                    if resp.status_code == 200:
                        return resp.json()
                    self._last_error = f"HTTP {resp.status_code} from {endpoint}"
                    logger.warning("Overpass %s HTTP %d (attempt %d)", endpoint, resp.status_code, attempt)
                except Exception as exc:
                    self._last_error = f"{type(exc).__name__}: {exc} ({endpoint})"
                    logger.warning("Overpass request failed (%s): %s", endpoint, exc)
            if attempt < self.retry_attempts:
                time.sleep(2 ** attempt)
        return None

    # ── GeoJSON conversion ────────────────────────────────────────────────────
    def _osm_to_geojson(self, data: dict, categories: list[str]) -> dict:
        """Convert Overpass JSON (``out geom``) to a categorised GeoJSON FeatureCollection.

        Areas become ``Polygon``/``MultiPolygon``; roads and waterways stay ``LineString`` with a
        ``width_m`` property (used when rasterising). Elements that match no requested category, are
        open rings where an area is required, or have no geometry are skipped and counted in the
        collection's ``n_skipped`` member.
        """
        from shapely.geometry import mapping

        features = []
        n_skipped = 0
        for elem in data.get("elements", []):
            etype = elem.get("type")
            if etype not in ("way", "relation"):
                continue
            tags = elem.get("tags", {})
            kind = classify_tags(tags)
            if kind is None or kind[0] not in categories:
                continue
            category, geom_kind = kind
            value = OSM_CATEGORIES[category]["value"]
            props = {**tags, "category": category, "label_value": value, "osm_id": elem.get("id"),
                     "osm_type": etype}

            geom = None
            if etype == "way" and elem.get("geometry"):
                coords = [[n["lon"], n["lat"]] for n in elem["geometry"]]
                closed = len(coords) >= 4 and coords[0] == coords[-1]
                if geom_kind == "line" and len(coords) >= 2:
                    geom = {"type": "LineString", "coordinates": coords}
                    props["width_m"] = _line_width_m(tags, category)
                elif geom_kind == "area" and closed:
                    geom = {"type": "Polygon", "coordinates": [coords]}
            elif etype == "relation" and geom_kind == "area":
                poly = _relation_polygon(elem)
                if poly is not None:
                    geom = mapping(poly)
            if geom is None:
                n_skipped += 1
                continue
            features.append({"type": "Feature", "geometry": geom, "properties": props})

        return {"type": "FeatureCollection", "features": features, "n_skipped": n_skipped}

    # ── Rasterisation ─────────────────────────────────────────────────────────
    @staticmethod
    def _grid_from_bbox(bbox, resolution_m: float):
        """(transform, width, height) of a lon/lat grid whose pixels are about ``resolution_m`` square."""
        from rasterio.transform import from_bounds

        lon_min, lat_min, lon_max, lat_max = bbox
        if resolution_m <= 0:
            raise ValueError("resolution_m must be > 0")
        lat_mid = (lat_min + lat_max) / 2.0
        dy = resolution_m / 111_320.0
        dx = dy / max(math.cos(math.radians(lat_mid)), 1e-6)
        width = max(1, math.ceil((lon_max - lon_min) / dx))
        height = max(1, math.ceil((lat_max - lat_min) / dy))
        if width * height > MAX_LABEL_PIXELS:
            raise ValueError(
                f"A {resolution_m} m grid over this bbox would be {width} x {height} = "
                f"{width * height:,} pixels (limit {MAX_LABEL_PIXELS:,}). Use a smaller bbox, a coarser "
                "resolution_m, or label per tile with reference_raster."
            )
        return from_bounds(lon_min, lat_min, lon_max, lat_max, width, height), width, height

    @staticmethod
    def _buffer_line_m(geom, width_m: float):
        """Buffer a lon/lat shapely line by ``width_m`` / 2 metres (local equirectangular scale)."""
        from shapely import affinity

        lat0 = geom.centroid.y
        k = max(math.cos(math.radians(lat0)), 1e-6)
        flat = affinity.scale(geom, xfact=k, yfact=1.0, origin=(0, 0))
        buf = flat.buffer((width_m / 2.0) / 111_320.0)
        return affinity.scale(buf, xfact=1.0 / k, yfact=1.0, origin=(0, 0))

    def _burn(
        self,
        geojson: dict,
        transform: Any,
        width: int,
        height: int,
        crs: Any,
        all_touched: bool = False,
        value_map: dict[str, int] | None = None,
    ) -> tuple[Any, int]:
        """Burn categorised lon/lat features onto a ``height x width`` grid; returns ``(array, n_burned)``.

        ``value_map`` maps category -> raster value (default: ``OSM_CATEGORIES`` values). Categories
        missing from it are not drawn.
        """
        import numpy as np
        import rasterio.crs
        from rasterio.features import rasterize
        from rasterio.warp import transform_geom
        from shapely.geometry import mapping, shape

        to_target = crs is not None and crs != rasterio.crs.CRS.from_epsg(4326)
        values = value_map if value_map is not None else {c: OSM_CATEGORIES[c]["value"] for c in DRAW_ORDER}

        by_cat: dict[str, list] = {c: [] for c in DRAW_ORDER}
        lines: dict[str, list] = {c: [] for c in DRAW_ORDER}
        n_burned = 0
        for f in geojson.get("features", []):
            if not f.get("geometry"):
                continue
            props = f.get("properties", {})
            cat = props.get("category")
            if cat not in by_cat or cat not in values:
                continue
            geom = shape(f["geometry"])
            is_line = geom.geom_type in ("LineString", "MultiLineString")
            if is_line:
                geom = self._buffer_line_m(geom, float(props.get("width_m", 5.0)))
            if geom.is_empty:
                continue
            g = mapping(geom)
            if to_target:
                g = transform_geom("EPSG:4326", crs, g)
            (lines if is_line else by_cat)[cat].append(g)
            n_burned += 1

        label = np.zeros((height, width), dtype=np.uint8)
        for cat in DRAW_ORDER:
            if cat not in values:
                continue
            for geoms, touched in ((by_cat[cat], all_touched), (lines[cat], True)):
                if not geoms:
                    continue
                m = rasterize([(g, 1) for g in geoms], out_shape=(height, width), transform=transform,
                              fill=0, dtype=np.uint8, all_touched=touched)
                label[m == 1] = values[cat]
        return label, n_burned

    def _rasterise(
        self,
        geojson: dict,
        bbox: tuple[float, ...],
        output_path: Path,
        resolution_m: float = 10.0,
        reference_raster: str | None = None,
        all_touched: bool = False,
    ) -> Path:
        """Rasterise categorised features to a labelled GeoTIFF (see the module docstring)."""
        try:
            import numpy as np
            import rasterio
            import rasterio.crs
        except ImportError as exc:
            raise ImportError(f"rasterio + shapely required: {exc}") from exc

        if reference_raster:
            with rasterio.open(reference_raster) as ref:
                transform, width, height, crs = ref.transform, ref.width, ref.height, ref.crs
        else:
            transform, width, height = self._grid_from_bbox(bbox, resolution_m)
            crs = rasterio.crs.CRS.from_epsg(4326)
        label, n_burned = self._burn(geojson, transform, width, height, crs, all_touched)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(
            str(output_path), "w",
            driver="GTiff", height=height, width=width,
            count=1, dtype="uint8",
            crs=crs, transform=transform,
            compress="lzw",
        ) as dst:
            dst.write(label[np.newaxis])
            dst.update_tags(
                source="OpenStreetMap",
                provider="Overpass API",
                n_features=str(n_burned),
                class_values=str({c: OSM_CATEGORIES[c]["value"] for c in DRAW_ORDER}),
            )
        return output_path

    # ── Public API ────────────────────────────────────────────────────────────
    def label(
        self,
        bbox: tuple[float, ...],
        categories: list[str] | None = None,
        output_path: str | Path = "./labels/osm_labels.tif",
        reference_raster: str | None = None,
        resolution_m: float = 10.0,
        save_vector: bool = True,
        vector_path: str | None = None,
        all_touched: bool = False,
    ) -> dict[str, Any]:
        """Generate a labelled GeoTIFF from OSM features.

        Args:
            bbox: ``(lon_min, lat_min, lon_max, lat_max)`` in degrees (EPSG:4326), always.
            categories: Categories to include (default: all ten). Unknown names raise ``ValueError``.
            output_path: Output label raster path.
            reference_raster: Write the labels on this raster's exact grid and CRS (any CRS).
                Strongly recommended when the labels will be paired with that imagery for training.
            resolution_m: Pixel size in metres when there is no ``reference_raster`` (EPSG:4326 grid).
            save_vector: Also save the features as GeoJSON (EPSG:4326).
            vector_path: Where; default ``<output stem>_vectors.geojson`` beside the raster.
            all_touched: Burn every pixel an *area* touches, not just those whose centre it covers.
                Keeps buildings smaller than a pixel at 10 m; fattens everything else. Lines always
                use it.

        Returns:
            ``success`` (False when the request failed **or** found zero features), ``n_features``,
            ``categories`` (counts), ``class_values`` (category -> raster value), ``n_skipped``,
            ``output_path``, ``vector_path``, ``error``.
        """
        categories = list(categories) if categories else list(OSM_CATEGORIES.keys())
        self._validate_categories(categories)
        output_path = Path(output_path)

        logger.info("OSMLabeler: querying Overpass for bbox=%s cats=%s", bbox, categories)
        query = self._build_overpass_query(bbox, categories)
        data = self._fetch_overpass(query)
        if data is None:
            logger.error("Overpass query failed; check connectivity and retry.")
            return {"success": False,
                    "error": "Overpass request failed" + (f" ({self._last_error})" if self._last_error else "")}
        remark = str(data.get("remark", ""))
        if "runtime error" in remark.lower():
            return {"success": False,
                    "error": f"Overpass could not complete the query: {remark}. Use a smaller bbox or fewer categories."}

        geojson = self._osm_to_geojson(data, categories)
        n_features = len(geojson["features"])
        n_skipped = int(geojson.get("n_skipped", 0))
        logger.info("OSMLabeler: %d features fetched (%d skipped)", n_features, n_skipped)

        v_path: Path | None = None
        if save_vector:
            import json
            v_path = Path(vector_path) if vector_path else output_path.with_name(output_path.stem + "_vectors.geojson")
            v_path.parent.mkdir(parents=True, exist_ok=True)
            with open(v_path, "w") as f:
                json.dump({k: v for k, v in geojson.items() if k != "n_skipped"}, f)

        self._rasterise(geojson, bbox, output_path, resolution_m, reference_raster, all_touched=all_touched)
        logger.info("OSMLabeler: label raster -> %s", output_path)

        cat_counts: dict[str, int] = {}
        for feat in geojson["features"]:
            cat = feat["properties"].get("category", "unknown")
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

        if n_features == 0:
            logger.warning(
                "OSMLabeler: 0 features found for bbox=%s categories=%s -- the output raster exists "
                "but contains no labels.", bbox, categories)
        return {
            "success": n_features > 0,
            "n_features": n_features,
            "n_skipped": n_skipped,
            "categories": cat_counts,
            "class_values": {c: OSM_CATEGORIES[c]["value"] for c in categories},
            "output_path": str(output_path),
            "vector_path": str(v_path) if v_path else None,
            "error": None if n_features > 0 else (
                f"0 OSM features found for bbox={bbox} categories={categories} "
                f"-- the output raster is empty (all background)."
            ),
        }

    def list_categories(self) -> dict[str, str]:
        """Return all supported OSM categories and their descriptions."""
        return {k: v["description"] for k, v in OSM_CATEGORIES.items()}

    def preview_query(self, bbox: tuple[float, ...], categories: list[str] | None = None) -> str:
        """Return the Overpass QL query without executing it."""
        return self._build_overpass_query(bbox, categories or list(OSM_CATEGORIES.keys()))
