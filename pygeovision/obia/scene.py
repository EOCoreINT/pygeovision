"""Get real data into the OBIA layout, through PyGeoVision's own search and download tools.

* :func:`fetch_scene` - a cloud-masked, cropped, band-stacked Sentinel-2 image. It uses the same search, download,
  cloud-mask and mosaic code as every pipeline (``BasePipeline._search_and_download``), so scenes behave the same
  in OBIA as in ``pygeovision channel ...``.
* :func:`fetch_worldcover` - ESA WorldCover on the image's grid, through ``client.labeling.esa_worldcover``.
* :func:`fetch_osm_polygons` - OpenStreetMap polygons (mine outlines by default), through the Overpass client of
  ``pygeovision.labeling.osm`` (User-Agent, mirrors, retries).

All three need a network and the credentials the provider uses. When a download does not work they raise
``RuntimeError`` with the reason; nothing is ever substituted.

A download can take minutes, so each call **reports what it is doing** (``verbose=True`` by default): the stage it is in,
the library's own search/download messages, and a heartbeat every few seconds with the elapsed time and the bytes that
arrived. A heartbeat that says "no new data" means the network is slow or stalled, not that Python is frozen.
"""
from __future__ import annotations

import contextlib
import logging
import sys
import threading
import time
from pathlib import Path

import numpy as np

from .io import RasterData, load_raster, save_raster

DEFAULT_BANDS = ("blue", "green", "red", "nir", "swir1", "swir2")

#: Overpass tag filters (one per line of the union) that outline mines and quarries.
MINE_OSM_FILTERS = ('["landuse"="quarry"]', '["industrial"="mine"]')


_LOGGERS = ("pygeovision", "pygeofetch")      # the loggers whose INFO messages describe a search / download
_KEEP = 6                                      # warnings/errors remembered for the failure message


def _stamp() -> str:
    return time.strftime("%H:%M:%S")


def _dir_bytes(path: Path | None) -> int:
    if path is None or not Path(path).exists():
        return 0
    total = 0
    for f in Path(path).rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:                        # a file that vanishes mid-scan (partial download renamed) is not an error
            pass
    return total


class _Echo(logging.Handler):
    """Writes log records to the *current* ``sys.stdout`` (so a notebook cell shows them), flushing every line."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
            tag = "" if record.levelno < logging.WARNING else ("WARNING: " if record.levelno < logging.ERROR else "ERROR: ")
            print(f"[{_stamp()}]   {tag}{msg}", flush=True)
        except Exception:                      # noqa: BLE001 - logging must never break the download
            pass


class _Collect(logging.Handler):
    """Remembers the last warnings and errors so a failure can say *why*."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.problems: list[str] = []
        self.ok = True            # the step's caller sets False when it returned without raising but did not succeed

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.problems.append(record.getMessage())
            del self.problems[:-_KEEP]
        except Exception:                      # noqa: BLE001
            pass


@contextlib.contextmanager
def _progress(label: str, verbose: bool = True, watch_dir: Path | None = None, heartbeat: float = 10.0):
    """Make a long network step visible and explainable.

    * ``verbose``: print a line when the step starts and ends, echo the library's INFO/WARNING/ERROR messages, and print
      a heartbeat every ``heartbeat`` seconds with the elapsed time and the size of ``watch_dir`` (the bytes that
      arrived). Heartbeats say so when nothing new has arrived for a while.
    * always: remember the last warnings and errors; the context yields the collector so the caller can put
      them in its exception.

    Logging configuration of the user is left alone: if the root logger already has a handler at INFO or lower, the
    library's messages already reach the user and are not echoed a second time.
    """
    collect = _Collect()
    echo = _Echo(level=logging.INFO) if verbose else None
    loggers = [logging.getLogger(n) for n in _LOGGERS]
    root = logging.getLogger()
    user_logging = bool(root.handlers) and root.getEffectiveLevel() <= logging.INFO
    saved = [(lg, lg.level) for lg in loggers]
    for lg in loggers:
        lg.addHandler(collect)
        if echo is not None and not user_logging:
            lg.addHandler(echo)
            if lg.getEffectiveLevel() > logging.INFO:
                lg.setLevel(logging.INFO)
    t0 = time.time()
    stop = threading.Event()

    def beat() -> None:
        base = _dir_bytes(watch_dir)           # files already there from an earlier run do not count as received now
        last, last_change = base, time.time()
        while not stop.wait(heartbeat):
            now = time.time()
            msg = f"[{_stamp()}]   ... {label}: still working, {now - t0:.0f} s elapsed"
            if watch_dir is not None:
                size = _dir_bytes(watch_dir)
                if size != last:
                    last, last_change = size, now
                msg += f", {max(0, size - base) / 1e6:.1f} MB received so far"
                if now - last_change >= 3 * heartbeat:
                    msg += f" (nothing new for {now - last_change:.0f} s: slow or stalled network, or a provider not answering)"
            print(msg, flush=True)

    if verbose:
        print(f"[{_stamp()}] {label} ...", flush=True)
        threading.Thread(target=beat, daemon=True, name=f"obia-heartbeat-{label[:20]}").start()
    ok = False
    try:
        yield collect
        ok = True
    finally:
        stop.set()
        for lg, lvl in saved:
            lg.removeHandler(collect)
            if echo is not None:
                lg.removeHandler(echo)
            lg.setLevel(lvl)
        if verbose:
            print(f"[{_stamp()}] {label}: {'done' if ok and collect.ok else 'FAILED'} after {time.time() - t0:.1f} s", flush=True)


def _why(collect: _Collect) -> str:
    """The remembered warnings/errors as a sentence for an exception message."""
    if not collect.problems:
        return ""
    return " Reasons reported: " + " | ".join(collect.problems[-3:])


_FETCHER = None


def _scene_fetcher(client):
    """A throw-away object that carries the pipelines' shared search/download code. It is not a registered pipeline.

    Built on first use so that importing ``pygeovision.obia`` stays light (the pipeline package pulls in a lot).
    """
    global _FETCHER
    if _FETCHER is None:
        from pygeovision.ai.pipelines import BasePipeline

        class _SceneFetcher(BasePipeline):
            name = "obia_scene_fetcher"

            def run(self, bbox, output_dir, **kwargs):  # pragma: no cover - never called
                raise NotImplementedError("_SceneFetcher only provides _search_and_download")

        _FETCHER = _SceneFetcher
    return _FETCHER(client)


def fetch_scene(client, bbox, date: str, out_dir: str | Path, bands=DEFAULT_BANDS,
                cloud_cover_max: float = 30.0, verbose: bool = True, heartbeat: float = 10.0, **kw) -> RasterData:
    """Download a cloud-masked Sentinel-2 L2A image for ``bbox`` and return it as :class:`RasterData`.

    ``date`` is a day, a month (``"2024-01"``) or a ``"start/end"`` window, as everywhere in PyGeoVision. Pixels hidden
    by cloud or shadow are NaN in every band; the OBIA functions treat them as "no data" (object label 0). Reflectance
    is 0-1, band names are the ones asked for, and the image is cropped to ``bbox`` in the scene's UTM zone.

    ``client`` is a :class:`pygeovision.PyGeoVision`. Extra keyword arguments (``providers=``, ``max_results=``,
    ``require_full_coverage=`` ...) go to the shared download code. Raises ``RuntimeError`` when no scene could be
    searched, downloaded or prepared; the message carries the reasons the download code reported.

    A scene download is several hundred megabytes and can take minutes. With ``verbose`` (the default) the call prints
    the stage it is in, the library's search/download messages, and every ``heartbeat`` seconds the elapsed time and
    the megabytes received so far, so a slow network is distinguishable from a hang. ``verbose=False`` is silent.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bbox = tuple(bbox)
    say = (lambda m: print(f"[{_stamp()}] {m}", flush=True)) if verbose else (lambda m: None)
    say(f"Sentinel-2 L2A, bbox={bbox}, date={date!r}, cloud cover <= {cloud_cover_max:g}%, bands={list(bands)}")
    say("(a scene is a few hundred MB: expect minutes on a slow link; progress is reported below)")
    with _progress("search and download", verbose, out_dir, heartbeat) as collect:
        path = _scene_fetcher(client)._search_and_download(
            bbox, date, out_dir, bands=tuple(bands), cloud_cover_max=cloud_cover_max, **kw)
        collect.ok = path is not None
    if path is None:
        raise RuntimeError(f"no usable Sentinel-2 scene for bbox={bbox} date={date!r} "
                           f"(cloud cover <= {cloud_cover_max}%); check the network, credentials and dates."
                           + _why(collect))
    rd = load_raster(str(path), band_names=list(bands))
    rd.meta.update(source=str(path), date=date, bbox=bbox)
    if verbose:
        nodata = 100 * (1 - float(np.isfinite(rd.image).all(axis=0).mean()))
        say(f"scene ready: {rd.image.shape[1]} x {rd.image.shape[2]} px, CRS {rd.crs}, {nodata:.1f}% no data (cloud/shadow) -> {path}")
    return rd


def fetch_worldcover(client, bbox, out_path: str | Path, like: RasterData | None = None,
                     verbose: bool = True, heartbeat: float = 10.0):
    """Download ESA WorldCover for ``bbox``; return ``(array, path)``.

    With ``like`` the map is resampled (nearest neighbour) onto that image's grid and CRS, so the two line up pixel for
    pixel; otherwise it comes on its own 10 m grid in lon/lat. 0 = no data. Class codes are in
    ``pygeovision.labeling.landcover.ESA_WORLDCOVER_CLASSES``.

    The tiles are read remotely; each remote read is bounded (60 s, 3 retries), so a stalled connection ends in a
    ``RuntimeError`` rather than a wait without end. ``verbose`` and ``heartbeat`` work as in :func:`fetch_scene`.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bbox = tuple(bbox)
    ref = None
    if like is not None:
        ref = str(out_path.with_name(out_path.stem + "_grid.tif"))
        save_raster(ref, np.zeros(like.shape, np.uint8), like.transform, like.crs, dtype="uint8")
    import rasterio
    with _progress("ESA WorldCover download", verbose, out_path.parent, heartbeat) as collect:
        with rasterio.Env(GDAL_HTTP_TIMEOUT="60", GDAL_HTTP_MAX_RETRY="3", GDAL_HTTP_RETRY_DELAY="2"):
            res = client.labeling.esa_worldcover(bbox, output_path=str(out_path),
                                                 **({"reference_raster": ref} if ref else {}))
        collect.ok = bool(res.get("success"))
    if not res.get("success"):
        raise RuntimeError(f"WorldCover download failed: {res.get('error')}" + _why(collect))
    got = Path(res.get("output_path", out_path))
    with rasterio.open(got) as src:
        arr = src.read(1)
    if verbose:
        found = {int(k): int(v) for k, v in zip(*np.unique(arr[arr > 0], return_counts=True))}
        print(f"[{_stamp()}] WorldCover ready: {arr.shape[0]} x {arr.shape[1]} px, class codes {found} -> {got}", flush=True)
    return arr, got


def fetch_osm_polygons(bbox, filters=MINE_OSM_FILTERS, timeout: int = 60, retries: int = 2,
                       verbose: bool = True, heartbeat: float = 10.0):
    """OpenStreetMap polygons inside ``bbox`` as a GeoDataFrame in EPSG:4326.

    ``filters`` are Overpass tag filters, one per alternative: the default outlines mines and quarries
    (``landuse=quarry`` and ``industrial=mine``). Closed ways become polygons and ``multipolygon`` relations are
    assembled (holes cut out); open lines are skipped. Columns: ``osm_id``, ``osm_type``, ``name``, the tags
    ``landuse``, ``industrial``, ``resource`` where present, and ``geometry``. An empty result is a valid answer
    (nothing is mapped there), not an error; a failed request raises ``RuntimeError``.

    OSM is volunteer-mapped: absence of a polygon does not mean absence of a mine.

    Each request waits at most ``timeout`` seconds, and the three public Overpass servers are tried in turn, ``retries``
    times, so the longest possible wait is about ``timeout * 3 * retries`` seconds (6 minutes with the defaults), and
    then a ``RuntimeError`` says why. ``verbose`` and ``heartbeat`` work as in :func:`fetch_scene`.
    """
    import geopandas as gpd
    from shapely.geometry import Polygon

    from pygeovision.labeling.osm import OSMLabeler, _relation_polygon

    lon_min, lat_min, lon_max, lat_max = (float(v) for v in bbox)
    if lon_min >= lon_max or lat_min >= lat_max:
        raise ValueError("bbox must be (min_lon, min_lat, max_lon, max_lat) with min < max")
    box = f"({lat_min},{lon_min},{lat_max},{lon_max})"
    body = "".join(f"way{f}{box};relation{f}{box};" for f in filters)
    query = f"[out:json][timeout:{int(timeout)}];\n(\n{body}\n);\nout body geom;"
    labeler = OSMLabeler(timeout=timeout, retry_attempts=retries)
    with _progress("OpenStreetMap (Overpass) request", verbose, None, heartbeat) as collect:
        data = labeler._fetch_overpass(query)
        collect.ok = data is not None
    if data is None:
        raise RuntimeError("Overpass request failed" + (f" ({labeler._last_error})" if labeler._last_error else "")
                           + _why(collect))
    if data.get("remark") and not data.get("elements"):
        raise RuntimeError(f"Overpass could not complete the query: {data['remark']}. Use a smaller bbox.")

    rows, geoms = [], []
    for elem in data.get("elements", []):
        etype, tags = elem.get("type"), elem.get("tags", {})
        geom = None
        if etype == "way" and elem.get("geometry"):
            coords = [(n["lon"], n["lat"]) for n in elem["geometry"]]
            if len(coords) >= 4 and coords[0] == coords[-1]:
                geom = Polygon(coords)
        elif etype == "relation":
            geom = _relation_polygon(elem)
        if geom is None or geom.is_empty:
            continue
        if not geom.is_valid:
            geom = geom.buffer(0)
        rows.append({"osm_id": elem.get("id"), "osm_type": etype, "name": tags.get("name"),
                     **{k: tags.get(k) for k in ("landuse", "industrial", "resource")}})
        geoms.append(geom)
    import pandas as pd
    cols = ["osm_id", "osm_type", "name", "landuse", "industrial", "resource"]
    return gpd.GeoDataFrame(pd.DataFrame(rows, columns=cols), geometry=geoms, crs="EPSG:4326")
