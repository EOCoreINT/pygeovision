"""Published reference products, read for a bounding box only.

The pipelines that map a *published product* (annual land cover, surface-water history, forest/non-forest,
MODIS evapotranspiration, ...) all need the same three things:

1. find the product tiles that cover a bounding box (a STAC search),
2. read **only the window** of each tile that the box needs -- the tiles are Cloud-Optimized GeoTIFFs of up to
   10 000 x 10 000 km^2 each, so downloading whole files is not an option,
3. put the pieces on one grid in a local UTM zone, so that every pixel has the same, known area.

This module does exactly that and nothing else. It never invents a value: pixels no tile covers stay invalid
(``valid == False``), and callers decide what the product's own no-data / fill codes are.

Network access is injectable (``session``) and the raster source may be a local path, which is how the offline
tests exercise the real window/warp/mosaic code on small files.
"""
from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import numpy as np

logger = logging.getLogger(__name__)

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
PC_SAS = "https://planetarycomputer.microsoft.com/api/sas/v1/token"
AWS_COPDEM30 = "https://copernicus-dem-30m.s3.amazonaws.com"   # AWS Open Data, public, no key


def copernicus_dem_aws_urls(bbox: Iterable[float]) -> list[str]:
    """URLs of the 1-degree Copernicus GLO-30 tiles (AWS Open Data copy) that cover ``bbox`` = (west, south, east, north).

    A tile is named for its south-west corner. Tiles over open sea do not exist; read these with
    ``read_to_grid(..., skip_missing=True)``.
    """
    west, south, east, north = (float(v) for v in bbox)
    if not (west < east and south < north):
        raise ReferenceDataError("bbox must be (west, south, east, north) with west < east and south < north")
    eps = 1e-9
    urls: list[str] = []
    for lat in range(math.floor(south), math.ceil(north - eps)):
        for lon in range(math.floor(west), math.ceil(east - eps)):
            ns, ew = ("N" if lat >= 0 else "S"), ("E" if lon >= 0 else "W")
            name = f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"
            urls.append(f"{AWS_COPDEM30}/{name}/{name}.tif")
    return urls
#: refuse to build a grid larger than this many pixels (about 1 GB as float32); the message says how to fix it.
MAX_GRID_PIXELS = 250_000_000


class ReferenceDataError(RuntimeError):
    """A reference product could not be found or read; the message says what to change."""


# ---------------------------------------------------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class TargetGrid:
    """A north-up grid in a metric CRS."""
    crs: str
    transform: Any            # affine.Affine
    width: int
    height: int
    res: float

    @property
    def pixel_area_ha(self) -> float:
        return self.res * self.res / 10_000.0

    def profile(self) -> dict[str, Any]:
        from rasterio.crs import CRS
        return {"driver": "GTiff", "crs": CRS.from_string(self.crs), "transform": self.transform,
                "width": self.width, "height": self.height}

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        t = self.transform
        return (t.c, t.f + t.e * self.height, t.c + t.a * self.width, t.f)


def utm_crs_for(lon: float, lat: float) -> str:
    """EPSG code string of the UTM zone containing the point (WGS84)."""
    zone = int(math.floor((lon + 180.0) / 6.0)) % 60 + 1
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def make_grid(bbox: Sequence[float], res: float, crs: str | None = None, max_pixels: int = MAX_GRID_PIXELS) -> TargetGrid:
    """Grid of ``res``-metre pixels covering ``bbox`` (lon/lat), snapped to whole multiples of ``res``."""
    from affine import Affine
    from rasterio.warp import transform_bounds

    west, south, east, north = (float(v) for v in bbox)
    if not (west < east and south < north):
        raise ReferenceDataError(f"bbox {tuple(bbox)} must be (min_lon, min_lat, max_lon, max_lat) with min < max")
    if res <= 0:
        raise ReferenceDataError(f"resolution must be positive, got {res}")
    crs = crs or utm_crs_for((west + east) / 2.0, (south + north) / 2.0)
    x0, b, r, t = transform_bounds("EPSG:4326", crs, west, south, east, north, densify_pts=21)
    left, top = math.floor(x0 / res) * res, math.ceil(t / res) * res
    width, height = int(math.ceil((r - left) / res)), int(math.ceil((top - b) / res))
    if width * height > max_pixels:
        raise ReferenceDataError(
            f"the area would be {width} x {height} = {width * height / 1e6:.0f} million pixels at {res:g} m "
            f"(limit {max_pixels / 1e6:.0f} million). Use a smaller bounding box or a coarser resolution.")
    return TargetGrid(crs=crs, transform=Affine(res, 0.0, left, 0.0, -res, top), width=width, height=height, res=float(res))


@dataclass
class GridRaster:
    """Pixels of one product on a :class:`TargetGrid`. ``valid`` is False where no source tile had data."""
    data: np.ndarray
    valid: np.ndarray
    grid: TargetGrid
    sources: list[str]


def _redact(href: str) -> str:
    """The URL without its access-token query string (tokens must not end up in logs and bug reports)."""
    return href.split("?", 1)[0] + ("?<token removed>" if "?" in href else "")


def _scrub(text: str) -> str:
    """Remove SAS-style query strings from any text (GDAL puts the full URL into some of its messages)."""
    import re
    return re.sub(r"\?[^\s'\"]*\b(?:sig|sv|se|sp|sr)=[^\s'\"]*", "?<token removed>", text)


def _http_reason(href: str, message: str) -> str:
    """For an HTTP 401/403, ask the server once more and quote its own explanation (Azure and S3 send an XML error body)."""
    if not href.startswith("http") or not any(c in message for c in ("403", "401")):
        return ""
    try:
        import requests
        r = requests.get(href, headers={"Range": "bytes=0-0"}, timeout=20)
    except Exception as exc:  # noqa: BLE001 - diagnostics only; the original error is still raised
        return f" (a second request to find out why also failed: {exc.__class__.__name__})"
    if r.status_code < 400:
        return (f" (the same URL answered HTTP {r.status_code} to a plain request, so the failure is specific to the "
                f"GDAL read: check the GDAL/libcurl build and proxy settings)")
    text = r.text or ""
    code = text.split("<Code>")[1].split("</Code>")[0] if "<Code>" in text else ""
    msg = text.split("<Message>")[1].split("</Message>")[0].splitlines()[0][:200] if "<Message>" in text else text[:200]
    return f" (server says HTTP {r.status_code} {code}: {msg})".replace(" )", ")")


def read_to_grid(sources: Iterable[str | Path], grid: TargetGrid, *, band: int = 1, resampling: str = "nearest",
                 dtype: str = "float32", invalid_values: Iterable[float] = (), fill: float = np.nan,
                 env: dict[str, Any] | None = None, skip_missing: bool = False) -> GridRaster:
    """Read the window of every source that overlaps ``grid`` and mosaic them (first valid pixel wins).

    A pixel is *invalid* when the source's own nodata, one of ``invalid_values``, or "outside every source" applies;
    invalid pixels are set to ``fill``. Sources may be in any CRS and are resampled with ``resampling``
    (``nearest`` for class maps, ``bilinear`` for continuous surfaces). With ``skip_missing`` a source that answers
    HTTP 404 is skipped (a regular tile grid has no file over open sea); every other failure still raises.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT

    try:
        method = Resampling[resampling]
    except KeyError as exc:
        raise ReferenceDataError(f"unknown resampling {resampling!r}") from exc
    srcs = [str(s) for s in sources]
    if not srcs:
        raise ReferenceDataError("no source tiles to read")
    out = np.full((grid.height, grid.width), fill, dtype=dtype)
    valid = np.zeros((grid.height, grid.width), dtype=bool)
    bad = np.asarray(list(invalid_values), dtype="float64")
    used: list[str] = []
    opts = {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff,.TIF",
            "GDAL_HTTP_MAX_RETRY": "4", "GDAL_HTTP_RETRY_DELAY": "1", "GDAL_HTTP_TIMEOUT": "60"}
    opts.update(env or {})
    with rasterio.Env(**opts):
        for href in srcs:
            if valid.all():
                break
            try:
                with rasterio.open(href) as ds:
                    if band > ds.count:
                        raise ReferenceDataError(f"{href} has {ds.count} band(s); band {band} was requested")
                    # add_alpha: the alpha band is 0 outside the tile and where the tile's own nodata applies, which a
                    # plain zero-filled warp cannot tell apart from real zeros.
                    with WarpedVRT(ds, crs=grid.crs, transform=grid.transform, width=grid.width, height=grid.height,
                                   resampling=method, src_nodata=ds.nodata, init_dest_nodata=False,
                                   nodata=ds.nodata if ds.nodata is not None else 0, add_alpha=True) as vrt:
                        vals = vrt.read(band).astype("float64")
                        footprint = vrt.read(vrt.count) > 0
            except rasterio.errors.RasterioIOError as exc:
                if skip_missing and "404" in str(exc):
                    logger.info("no tile at %s (HTTP 404); skipped", _redact(href))
                    continue
                raise ReferenceDataError(f"could not read {_redact(href)}: {_scrub(str(exc))}{_http_reason(href, str(exc))}") from exc
            ok = footprint & np.isfinite(vals)
            if bad.size:
                ok &= ~np.isin(vals, bad)
            take = ok & ~valid
            if take.any():
                out[take] = vals[take].astype(dtype)
                valid |= take
                used.append(href)
    return GridRaster(data=out, valid=valid, grid=grid, sources=used)


# ---------------------------------------------------------------------------------------------------------------------
# Microsoft Planetary Computer (STAC + SAS tokens)
# ---------------------------------------------------------------------------------------------------------------------

class PlanetaryComputer:
    """Minimal STAC search and asset signing for the open Planetary Computer collections.

    ``session`` is anything with ``get(url, params=..., timeout=...)`` returning an object with ``json()`` and
    ``raise_for_status()`` (a ``requests.Session`` by default), which lets tests supply a fake.
    """

    def __init__(self, session: Any = None, stac_url: str = PC_STAC, sas_url: str = PC_SAS,
                 timeout: float = 60.0, retries: int = 3) -> None:
        if session is None:
            import requests
            session = requests.Session()
            key = os.environ.get("PC_SDK_SUBSCRIPTION_KEY")          # optional: lifts the anonymous rate limit
            if key:
                session.headers["Ocp-Apim-Subscription-Key"] = key
        self.session, self.stac_url, self.sas_url = session, stac_url.rstrip("/"), sas_url.rstrip("/")
        self.timeout, self.retries = timeout, retries
        self._tokens: dict[str, tuple[str, float]] = {}
        self._container_tokens: dict[tuple[str, str], tuple[str, float]] = {}
        self.sign_url = self.sas_url.rsplit("/", 1)[0] + "/sign"

    def _get(self, url: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        last: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                resp.raise_for_status()
                return resp.json()
            except Exception as exc:  # noqa: BLE001 - network errors of any library; re-raised below with context
                last = exc
                logger.warning("GET %s failed (attempt %d/%d): %s", url, attempt, self.retries, exc)
                if attempt < self.retries:
                    time.sleep(min(2.0 ** attempt, 8.0))
        raise ReferenceDataError(f"Planetary Computer request failed after {self.retries} attempts ({url}): {last}")

    def search(self, collection: str, bbox: Sequence[float], datetime: str | None = None,
               limit: int = 100, max_items: int = 1000) -> list[dict[str, Any]]:
        """STAC items of ``collection`` intersecting ``bbox`` (and ``datetime`` ``start/end``), all pages.

        The catalogue's ``/search`` endpoint is used: it applies the date window and the box reliably. The older
        per-collection ``/collections/<id>/items`` endpoint is only the fallback -- for Sentinel-1 it was seen to
        answer a ten-day window with scenes from other weeks and a ten-week window with only the oldest weeks.
        """
        base: dict[str, Any] = {"bbox": ",".join(f"{float(v):.6f}" for v in bbox), "limit": limit}
        if datetime:
            base["datetime"] = datetime
        try:
            return self._pages(f"{self.stac_url}/search", {**base, "collections": collection}, max_items)
        except ReferenceDataError as exc:
            logger.warning("Planetary Computer /search failed (%s); trying /collections/%s/items", exc, collection)
        return self._pages(f"{self.stac_url}/collections/{collection}/items", base, max_items)

    def _pages(self, url: str, params: dict[str, Any], max_items: int) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        nxt_url: str | None = url
        query: dict[str, Any] | None = params
        while nxt_url and len(items) < max_items:
            page = self._get(nxt_url, query)
            items.extend(page.get("features", []))
            nxt = next((lk for lk in page.get("links", [])
                        if lk.get("rel") == "next" and lk.get("href") and str(lk.get("method", "GET")).upper() == "GET"), None)
            nxt_url, query = (nxt["href"] if nxt else None), None     # the next link already carries its query
        return items[:max_items]

    def token(self, collection: str) -> str:
        cached = self._tokens.get(collection)
        if cached and cached[1] > time.time() + 120:
            return cached[0]
        body = self._get(f"{self.sas_url}/{collection}")
        tok = body.get("token")
        if not tok:
            raise ReferenceDataError(f"Planetary Computer returned no access token for {collection!r}")
        self._tokens[collection] = (tok, time.time() + 25 * 60)      # tokens last about an hour
        return tok

    def sign(self, href: str, collection: str) -> str:
        """Signed URL for a blob-storage asset (other URLs are returned unchanged).

        The Planetary Computer's ``/sign`` endpoint is asked first: it issues a token for the very storage account and
        container the asset lives in. The per-collection token endpoint is only the fallback, because a collection's
        data can move to another storage account while that token is still issued for the old one, which Azure then
        rejects with HTTP 403 "AuthenticationFailed". Tokens are reused for every asset in the same container.
        """
        if "blob.core.windows.net" not in href or "sig=" in href:
            return href
        parts = urlsplit(href)
        container = (parts.netloc, parts.path.lstrip("/").split("/", 1)[0])
        cached = self._container_tokens.get(container)
        if cached and cached[1] > time.time() + 120:
            return f"{href}{'&' if '?' in href else '?'}{cached[0]}"
        try:
            body = self._get(self.sign_url, params={"href": href})
            signed = body.get("href") or ""
            query = urlsplit(signed).query
            if "sig=" not in query:
                raise ReferenceDataError("the sign endpoint returned no signature")
            if "sr=c" in query:                       # a container-level token covers every asset in the container
                self._container_tokens[container] = (query, time.time() + 25 * 60)
            return f"{href}{'&' if '?' in href else '?'}{query}"
        except ReferenceDataError as exc:
            logger.warning("Planetary Computer /sign failed for %s (%s); using the collection token", container[0], exc)
        return f"{href}{'&' if '?' in href else '?'}{self.token(collection)}"

    def asset_urls(self, collection: str, items: Sequence[dict[str, Any]], asset: str) -> list[str]:
        out = []
        for it in items:
            a = (it.get("assets") or {}).get(asset)
            if a and a.get("href"):
                out.append(self.sign(a["href"], collection))
        return out


def item_year(item: dict[str, Any]) -> int | None:
    """Calendar year an annual-product item stands for (``start_datetime`` first, then ``datetime``)."""
    props = item.get("properties") or {}
    for key in ("start_datetime", "datetime"):
        v = props.get(key)
        if v:
            return int(str(v)[:4])
    return None


def parse_year(value: Any, what: str) -> int:
    """``2021``, ``"2021"``, ``"2021-06"`` or ``"2021-06-30"`` -> 2021."""
    s = str(value).strip()
    if len(s) < 4 or not s[:4].isdigit():
        raise ReferenceDataError(f"{what} must be a year such as 2021 (or YYYY-MM), got {value!r}")
    return int(s[:4])