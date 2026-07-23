"""
pygeovision.utils.acquire
=========================
Download and preprocessing helpers that wrap pygeovision's client API.

Replaces these patterns that appear across every project notebook:

Pattern 1 — pick lowest-cloud scene::

    scenes.sort(key=lambda s: s.cloud_cover or 99)
    SCENE = scenes[0] if scenes else None

Pattern 2 — download + extract path::

    dl = client.download([scene], output_dir=..., bands=..., post_process=...)
    ok = next((d for d in dl if d.success and d.path), None)
    PATH = str(ok.path) if ok else None
    if ok:
        sz = pathlib.Path(PATH).stat().st_size / 1e6
        print(f"  ✓ {label}: {sz:.1f} MB")

Pattern 3 — S1 download + GCP recovery::

    dl  = client.download([scene], ...)
    ok  = next((d for d in dl if d.success and d.path), None)
    geo = validate_sar_georeference(str(ok.path))
    WORKING = geo.repaired_path if geo.recovered else str(ok.path)

Pattern 4 — S2 clip + normalise::

    clipped = client.preprocess.clip_to_bbox(path, bbox=BBOX, ...)
    normed  = client.preprocess.normalise(clipped, method="scale_factor", ...)

After::

    from pygeovision.utils import best_scene, download_ok, download_s1, preprocess_s2

    scene  = best_scene(scenes)
    path   = download_ok(client, [scene], "./data/", bands=["B02","B04","B08"])
    s1path = download_s1(client, scene, "./data/s1/")
    ready  = preprocess_s2(client, path, bbox=BBOX)
"""
from __future__ import annotations

import logging
import pathlib
from typing import Any

logger = logging.getLogger(__name__)


def best_scene(
    scenes: list[Any],
    sort_key: str = "cloud_cover",
) -> Any | None:
    """
    Return the single best scene from a search result list.

    Sorting strategies:
      - ``"cloud_cover"`` : lowest cloud cover first (default, optical)
      - ``"datetime"``    : most recent first
      - ``"score"``       : highest pygeofetch score first

    Args:
        scenes:   List of ``SearchResult`` objects from ``client.search()``.
        sort_key: ``"cloud_cover"`` | ``"datetime"`` | ``"score"``.

    Returns:
        Single best ``SearchResult``, or ``None`` if list is empty.

    Example::

        from pygeovision.utils import best_scene

        scenes = client.search(bbox=BBOX, date_range=("2024-01-01","2024-03-31"),
                               satellite="Sentinel-2", providers=["planetary_computer"])
        scene = best_scene(scenes)
        print(scene)
    """
    if not scenes:
        return None

    if sort_key == "cloud_cover":
        return min(scenes, key=lambda s: getattr(s, "cloud_cover", None) or 99.0)
    elif sort_key == "datetime":
        return max(scenes, key=lambda s: getattr(s, "datetime", "") or "")
    elif sort_key == "score":
        return max(scenes, key=lambda s: getattr(s, "score", None) or 0.0)
    else:
        return scenes[0]


def download_ok(
    client:       Any,
    scenes:       list[Any],
    output_dir:   str | pathlib.Path,
    bands:        list[str] | None = None,
    post_process: list[str] | None = None,
    target_crs:   str = "EPSG:32630",
    label:        str = "",
    verbose:      bool = True,
) -> str | None:
    """
    Download scenes and return the first successful output path.

    Wraps the boilerplate::

        dl  = client.download([scene], output_dir=..., bands=..., post_process=...)
        ok  = next((d for d in dl if d.success and d.path), None)
        PATH = str(ok.path) if ok else None
        if ok:
            sz = pathlib.Path(PATH).stat().st_size / 1e6
            print(f"  ✓ label: {sz:.1f} MB")

    Args:
        client:       ``PyGeoVision`` client instance.
        scenes:       List of ``SearchResult`` objects.
        output_dir:   Download destination directory.
        bands:        Band names (e.g. ``["B02","B04","B08"]``). ``None`` = all.
        post_process: Processing steps (e.g. ``["reproject:EPSG:32630","cog"]``).
                      Defaults to ``["reproject:{target_crs}", "cog"]``.
        target_crs:   CRS for default post_process. Default ``EPSG:32630`` (UTM 30N).
        label:        Label for log output.
        verbose:      Print success/failure line.

    Returns:
        Path string of the downloaded file, or ``None`` on failure.

    Example::

        from pygeovision.utils import download_ok

        s2_path = download_ok(client, [scene], "./data/s2/",
                              bands=["B02","B03","B04","B08"],
                              label="S2 2024-01")
    """
    if not scenes:
        logger.warning("download_ok: no scenes supplied")
        return None

    if post_process is None:
        post_process = [f"reproject:{target_crs}", "cog"]

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    try:
        dl = client.download(
            scenes,
            output_dir   = str(output_dir),
            bands        = bands,
            post_process = post_process,
        )
    except Exception as e:
        logger.error("download_ok [%s]: download failed: %s", label, e)
        if verbose:
            print(f"  ✗ {label}: {e}")
        return None

    ok = next((d for d in (dl or []) if d.success and d.path), None)
    if ok:
        path = str(ok.path)
        if verbose:
            try:
                sz = pathlib.Path(path).stat().st_size / 1e6
                tag = f"({sz:.1f} MB)"
            except OSError:
                tag = ""
            print(f"  ✓ {label}: {pathlib.Path(path).name}  {tag}")
        return path
    else:
        errs = [d.error for d in (dl or []) if not d.success and d.error]
        msg  = errs[0] if errs else "unknown error"
        if verbose:
            print(f"  ✗ {label}: {msg}")
        logger.warning("download_ok [%s]: %s", label, msg)
        return None


def download_s1(
    client:     Any,
    scene:      Any,
    output_dir: str | pathlib.Path,
    bands:      list[str] | None = None,
    target_crs: str = "EPSG:32630",
    label:      str = "S1",
    verbose:    bool = True,
) -> str | None:
    """
    Download a Sentinel-1 scene and apply GCP recovery.

    Combines the download boilerplate + the
    ``validate_sar_georeference()`` call that every SAR notebook repeats::

        dl  = client.download([scene], ...)
        ok  = next((d for d in dl if d.success and d.path), None)
        geo = validate_sar_georeference(str(ok.path))
        WORKING = geo.repaired_path if geo.recovered else str(ok.path)
        if geo.recovered:
            print(f"  GCP recovery: {geo.gcp_count} GCPs")

    Args:
        client:     ``PyGeoVision`` client instance.
        scene:      Single ``SearchResult`` from ``client.search()``.
        output_dir: Download destination.
        bands:      SAR bands. Defaults to ``["vv","vh"]``.
        target_crs: Reproject target. Default ``EPSG:32630`` (UTM Zone 30N).
        label:      Label for log output.
        verbose:    Print recovery details.

    Returns:
        Working path (GCP-repaired if needed), or ``None`` on failure.

    Example::

        from pygeovision.utils import download_s1

        s1_path = download_s1(client, scene, "./data/s1/", label="S1 flood")
        # Always use the returned path — it may be the repaired copy
    """
    from pygeovision.data.validators.georeference import validate_sar_georeference

    if scene is None:
        logger.warning("download_s1 [%s]: no scene supplied", label)
        return None

    if bands is None:
        bands = ["vv", "vh"]

    raw_path = download_ok(
        client, [scene], output_dir,
        bands        = bands,
        post_process = [f"reproject:{target_crs}", "cog"],
        target_crs   = target_crs,
        label        = label,
        verbose      = verbose,
    )
    if not raw_path:
        return None

    # GCP recovery — fixes pygeofetch identity-transform corruption
    try:
        geo = validate_sar_georeference(raw_path)
    except Exception as e:
        logger.warning("download_s1 [%s]: validate_sar_georeference failed: %s", label, e)
        return raw_path

    if geo.recovered and geo.repaired_path:
        if verbose:
            print(f"    GCP recovery: {geo.gcp_count} GCPs → "
                  f"pixel_size≈{geo.transform_a:.1f}m")
        return geo.repaired_path
    elif geo.valid:
        if verbose:
            print(f"    Georeference valid (pixel_size={geo.transform_a:.1f}m)")
        return raw_path
    else:
        logger.error("download_s1 [%s]: georeference invalid: %s", label, geo.errors)
        if verbose:
            print(f"    ✗ Georeference invalid: {geo.errors}")
        return None


def preprocess_s2(
    client:       Any,
    raw_path:     str,
    bbox:         tuple[float, float, float, float],
    output_path:  str | None = None,
    bbox_crs:     str = "EPSG:4326",
    scale_factor: float = 10000.0,
    label:        str = "",
    verbose:      bool = True,
) -> str | None:
    """
    Clip and normalise a Sentinel-2 scene in one call.

    Replaces::

        clipped = client.preprocess.clip_to_bbox(
            path, bbox=BBOX, bbox_crs="EPSG:4326",
            output_path=str(PREP_DIR / f"{label}_clip.tif"))
        normed = client.preprocess.normalise(
            clipped, output_path=str(PREP_DIR / f"{label}_norm.tif"),
            method="scale_factor", scale_factor=10000.0)

    Args:
        client:       ``PyGeoVision`` client instance.
        raw_path:     Downloaded S2 GeoTIFF path.
        bbox:         ``(lon_min, lat_min, lon_max, lat_max)`` WGS84.
        output_path:  Final output path. Auto-generated if ``None``.
        bbox_crs:     CRS of the bbox. Default ``EPSG:4326``.
        scale_factor: Reflectance scale divisor. Default 10000 (S2 L2A).
        label:        Label for log output.
        verbose:      Print result.

    Returns:
        Path of the normalised, clipped GeoTIFF, or ``None`` on failure.

    Example::

        from pygeovision.utils import preprocess_s2

        ready = preprocess_s2(client, s2_raw, bbox=ANKASA_BBOX, label="S2 dry")
    """
    import pathlib as _pl

    if not raw_path or not _pl.Path(raw_path).exists():
        if verbose:
            print(f"  SKIP preprocess_s2 [{label}]: file not found")
        return None

    stem = _pl.Path(raw_path).stem
    parent = _pl.Path(raw_path).parent

    if output_path is None:
        output_path = str(parent / f"{stem}_ready.tif")

    clip_path = str(parent / f"{stem}_clip.tif")

    try:
        clipped = client.preprocess.clip_to_bbox(
            raw_path,
            bbox        = bbox,
            bbox_crs    = bbox_crs,
            output_path = clip_path,
        )
        normed = client.preprocess.normalise(
            clipped,
            output_path  = output_path,
            method       = "scale_factor",
            scale_factor = scale_factor,
        )
        if verbose:
            import rasterio
            with rasterio.open(normed) as src:
                d = src.read().astype("float32")
            import numpy as np
            print(f"  ✓ {label}: shape={d.shape}  "
                  f"range=[{np.nanmin(d):.4f},{np.nanmax(d):.4f}]")
        return normed
    except Exception as e:
        logger.error("preprocess_s2 [%s]: %s", label, e)
        if verbose:
            print(f"  ✗ {label}: preprocess failed — {e}")
        return None


def preprocess_s1(
    client:      Any,
    raw_path:    str,
    bbox:        tuple[float, float, float, float],
    output_path: str | None = None,
    bbox_crs:    str = "EPSG:4326",
    label:       str = "",
    verbose:     bool = True,
) -> str | None:
    """
    Clip and percentile-normalise a Sentinel-1 GeoTIFF.

    Args:
        client:      ``PyGeoVision`` client instance.
        raw_path:    S1 GeoTIFF (after GCP recovery via ``download_s1()``).
        bbox:        ``(lon_min, lat_min, lon_max, lat_max)`` WGS84.
        output_path: Final output path. Auto-generated if ``None``.
        bbox_crs:    CRS of the bbox.
        label:       Label for log output.
        verbose:     Print result.

    Returns:
        Path of the processed GeoTIFF, or ``None`` on failure.
    """
    import pathlib as _pl

    if not raw_path or not _pl.Path(raw_path).exists():
        if verbose:
            print(f"  SKIP preprocess_s1 [{label}]: file not found")
        return None

    stem   = _pl.Path(raw_path).stem
    parent = _pl.Path(raw_path).parent

    if output_path is None:
        output_path = str(parent / f"{stem}_ready.tif")

    clip_path = str(parent / f"{stem}_clip.tif")

    try:
        clipped = client.preprocess.clip_to_bbox(
            raw_path, bbox=bbox, bbox_crs=bbox_crs, output_path=clip_path
        )
        normed = client.preprocess.normalise(
            clipped, output_path=output_path, method="percentile", percentile=2.0
        )
        if verbose:
            import numpy as np
            import rasterio
            with rasterio.open(normed) as src:
                d = src.read().astype("float32")
            print(f"  ✓ {label}: shape={d.shape}  "
                  f"range=[{np.nanmin(d):.4f},{np.nanmax(d):.4f}]")
        return normed
    except Exception as e:
        logger.error("preprocess_s1 [%s]: %s", label, e)
        if verbose:
            print(f"  ✗ {label}: preprocess failed — {e}")
        return None
