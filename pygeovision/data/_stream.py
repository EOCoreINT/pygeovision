"""The one place where PyGeoVision chooses between streaming a scene and downloading it.

Streaming (pygeofetch ``engine.stream``) reads only the pixels of the area of interest from Cloud-Optimized-GeoTIFF
providers. It only works when the stack really is a COG on a streamable provider, so a stream can fail: archive-only
providers (Copernicus ``.SAFE.zip``, USGS ``.tar``), non-COG files, a band that cannot be read, a network error.
That must never end the run. Every PyGeoVision way of getting imagery for an area calls :func:`fetch_scenes`, which

1. streams the scenes when the user wants streaming (``stream=True``) or has not said (``stream=None``) and an area is
   known;
2. downloads in full, with the normal ``engine.download``, every scene the stream could not deliver completely
   (and says why in the log and on the result);
3. downloads everything in full when the user asked for that (``stream=False``) or no area is known.

``stream=True`` is a preference, not a promise: if streaming fails the scene is downloaded instead of raising.
pygeofetch itself is not changed or extended by this module; it only uses ``engine.stream`` and ``engine.download``.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_OFF = ("0", "false", "no", "off")


def streaming_enabled() -> bool:
    """The default for ``stream=None``: on, unless ``PYGEOVISION_STREAM`` is set to 0 / false / no / off."""
    return os.environ.get("PYGEOVISION_STREAM", "1").strip().lower() not in _OFF


def want_stream(stream: bool | None, stream_bbox: Any) -> bool:
    """Should this call try to stream?

    ``stream=False``: never. ``stream=True``: whenever an area is known (an explicit choice also overrides
    ``PYGEOVISION_STREAM=0``). ``stream=None`` (automatic): when an area is known and ``PYGEOVISION_STREAM`` is not off.
    Streaming cannot work without an area (it reads that window), so without one the scenes are downloaded.
    """
    if stream is False:
        return False
    if stream_bbox is None:
        if stream:
            logger.info("stream=True needs an area of interest (stream_bbox); downloading whole scenes instead")
        return False
    return True if stream else streaming_enabled()


@dataclass
class Fetched:
    """One scene's pygeofetch result and how it was acquired."""

    result: Any  #: pygeofetch ``DownloadResult``
    acquisition: str  #: ``"stream"`` (AOI window only) or ``"download"`` (whole scene)
    stream_error: str = ""  #: why a scene was downloaded although streaming was wanted


def _failed(sd: Any, error: str) -> Any:
    from pygeofetch.models.download_task import DownloadResult, DownloadStatus

    return DownloadResult(
        status=DownloadStatus.FAILED,
        data_id=getattr(sd, "id", "?"),
        provider=getattr(sd, "provider", "") or "",
        error=error,
        error_type="streaming_fallback",
    )


def _stream_problem(result: Any) -> str:
    """Why this stream result is not a complete scene; empty when it is fine."""
    if result is None:
        return "no stream result"
    if not getattr(result, "success", False):
        return str(getattr(result, "error", "") or "stream failed")
    err = str(getattr(result, "error", "") or "")
    if err.startswith("partial"):  # pygeofetch: some bands streamed, others failed
        return err
    paths = list(getattr(result, "output_paths", None) or [])
    if not paths or not all(Path(p).is_file() for p in paths):
        return "the streamed files are missing"
    return ""


def _remove(paths: list[Any]) -> None:
    """Drop the small AOI files of a scene that is going to be downloaded in full (never leave a mix behind)."""
    for p in paths:
        try:
            Path(p).unlink(missing_ok=True)
        except OSError as exc:  # pragma: no cover - a locked file; the download overwrites it by name
            logger.debug("could not remove %s: %s", p, exc)


def _stream_all(
    engine: Any,
    items: list[Any],
    bbox: tuple[float, float, float, float],
    output_dir: Path,
    parallel: int,
    overwrite: bool,
    post_process: list[str] | None,
) -> list[tuple[Any, str]]:
    """Try to stream every scene; returns ``(result, problem)`` per scene, ``problem`` empty when it worked."""
    stream = getattr(engine, "stream", None)
    if not callable(stream):
        return [(None, "the installed pygeofetch has no stream()")] * len(items)
    try:
        # No ``bands`` here: the assets of every scene are already narrowed to the bands wanted, and pygeofetch refuses
        # a band list that matches no asset instead of streaming everything.
        results = list(
            stream(items, tuple(float(v) for v in bbox), output_dir, parallel=parallel, overwrite=overwrite)
        )
    except Exception as exc:  # noqa: BLE001 - any failure of the stream layer means "download instead"
        return [(None, f"stream raised {type(exc).__name__}: {exc}")] * len(items)
    results += [None] * (len(items) - len(results))
    out: list[tuple[Any, str]] = []
    for r in results[: len(items)]:
        problem = _stream_problem(r)
        if not problem:
            problem = reproject_streamed(list(getattr(r, "output_paths", None) or []), post_process)
        out.append((r, problem))
    return out


def fetch_scenes(
    engine: Any,
    satellite_data: list[Any],
    output_dir: str | Path,
    options: Any,
    *,
    stream_bbox: tuple[float, float, float, float] | None = None,
    stream: bool | None = None,
    parallel: int = 4,
    overwrite: bool = False,
    post_process: list[str] | None = None,
) -> list[Fetched]:
    """Acquire the scenes, one :class:`Fetched` per scene and in order.

    Args:
        engine: A pygeofetch ``PyGeoFetch`` (anything with ``download`` and, for streaming, ``stream``).
        satellite_data: pygeofetch ``SatelliteData`` records. Their assets must already be narrowed to the bands needed.
        output_dir: Where the files go (streamed and downloaded scenes use the same ``<provider>/`` layout).
        options: pygeofetch ``DownloadOptions`` for every full download, including the fallback ones.
        stream_bbox: ``(west, south, east, north)`` of the area to stream.
        stream: ``True`` to stream (falling back to download if that fails), ``False`` to always download, ``None``
            to stream when an area is known and ``PYGEOVISION_STREAM`` is not off.
        post_process: The post-process steps of ``options`` as text; streamed files get the ``reproject:`` step here,
            because pygeofetch's download post-processing does not run on them.
    """
    items = list(satellite_data)
    out_dir = Path(output_dir)
    if not items:
        return []
    if not want_stream(stream, stream_bbox):
        return [Fetched(r, "download") for r in engine.download(items, destination=out_dir, options=options)]

    attempts = _stream_all(engine, items, stream_bbox, out_dir, parallel, overwrite, post_process)  # type: ignore[arg-type]
    fetched: list[Fetched | None] = [None] * len(items)
    redo: list[int] = []
    for i, (res, problem) in enumerate(attempts):
        if problem:
            redo.append(i)
            if res is not None:  # a half-streamed scene: remove its AOI files before the full download
                _remove(list(getattr(res, "output_paths", None) or []))
            logger.warning(
                "streaming %s failed (%s); downloading the whole scene instead",
                getattr(items[i], "id", "?"),
                problem[:300],
            )
        else:
            fetched[i] = Fetched(res, "stream")

    if redo:
        todo = [items[i] for i in redo]
        try:
            downloaded = list(engine.download(todo, destination=out_dir, options=options))
        except Exception as exc:  # noqa: BLE001 - reported per scene, never raised
            downloaded = [_failed(sd, f"download failed: {type(exc).__name__}: {exc}") for sd in todo]
        downloaded += [None] * (len(todo) - len(downloaded))
        for i, dres in zip(redo, downloaded[: len(todo)]):
            problem = attempts[i][1]
            if dres is None:
                dres = _failed(items[i], "no download result returned")
            if not getattr(dres, "success", False):
                dres = dres.model_copy(
                    update={"error": f"stream failed: {problem}; then download failed: {getattr(dres, 'error', '')}"}
                )
            fetched[i] = Fetched(dres, "download", problem)
    return [f for f in fetched if f is not None]


def reproject_file_inplace(path: Path, target_crs: str) -> None:
    """Warp a small AOI-sized GeoTIFF to ``target_crs`` (the same step the full download applies to every file)."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import calculate_default_transform, reproject

    with rasterio.open(path) as src:
        if src.crs is None or str(src.crs).upper() == str(target_crs).upper():
            return
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )
        meta = src.meta.copy()
        meta.update(crs=target_crs, transform=transform, width=width, height=height)
        tmp = path.with_suffix(".reproj.tif")
        categorical = src.dtypes[0].startswith(("uint8", "int8"))  # scene classification / QA layers
        with rasterio.open(tmp, "w", **meta) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.nearest if categorical else Resampling.bilinear,
                )
    tmp.replace(path)


def reproject_streamed(paths: list[Any], post_process: list[str] | None) -> str:
    """Apply the ``reproject:`` step to streamed AOI files; returns an error text, empty when all went well.

    A streamed file is not touched by pygeofetch's download post-processing; the CRS is the one step that changes what
    downstream code sees, so it is applied here exactly as the full download applies it to every file. A scene whose
    file cannot be reprojected counts as a failed stream and is downloaded instead.
    """
    target_crs = None
    for step in post_process or []:
        if step.strip().lower().startswith("reproject:"):
            target_crs = step.split(":", 1)[1].strip()
    if not target_crs:
        return ""
    try:
        for pth in paths:
            reproject_file_inplace(Path(pth), target_crs)
    except Exception as exc:  # noqa: BLE001 - reported to the caller, never raised
        logger.warning("could not reproject a streamed file to %s (%s)", target_crs, exc)
        return f"could not reproject the streamed file to {target_crs}: {exc}"
    return ""
