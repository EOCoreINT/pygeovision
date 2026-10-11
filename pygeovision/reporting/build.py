"""Turn one finished pipeline run into its deliverables.

``generate_deliverables`` reads the rasters a pipeline wrote, measures them,
draws the map and charts, exports the vector layers and writes the reports
(PDF, Word, Excel). It never changes the analysis outputs, and a part that
cannot be produced (missing optional library, unreadable file) is recorded in
``Deliverables.warnings`` and logged -- the other parts are still delivered.
"""
from __future__ import annotations

import json
import logging
import platform
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from pygeovision._version import __version__
from pygeovision.reporting import fmt as F
from pygeovision.reporting import geo, maps
from pygeovision.reporting.model import (
    Col,
    Report,
    Section,
    Table,
    bullets,
    callout,
    figure_block,
    kv_block,
    para,
    table_block,
)
from pygeovision.reporting.palettes import ClassDef
from pygeovision.reporting.specs import (
    Binning,
    Layer,
    ReportSpec,
    VectorSpec,
    landcover_classes,
    spec_for,
    user_class,
)

logger = logging.getLogger(__name__)

REPORT_FORMATS = ("pdf", "xlsx", "docx")
DEFAULT_VECTOR_FORMATS = ("gpkg", "geojson")
_VECTOR_MAX_PIXELS = 80_000_000
_PLOT_MAX_PX = 2400
_EXCEL_FEATURE_ROWS = 5000
_ALIASES = {"word": "docx", "doc": "docx", "excel": "xlsx", "xls": "xlsx", "geopackage": "gpkg", "shapefile": "shp", "json": "geojson"}

_NDVI_BINS = Binning("fixed", edges=(0.1, 0.3, 0.6),
                     names=("Non-vegetated or water (< 0.1)", "Sparse (0.1-0.3)", "Moderate (0.3-0.6)", "Dense (>= 0.6)"),
                     colors=("#bdbdbd", "#d9f0a3", "#78c679", "#238443"))


class ReportingUnavailable(RuntimeError):
    """A library needed for deliverables is not installed."""


@dataclass
class Deliverables:
    """What was produced for one pipeline run."""
    directory: Path
    reports: dict[str, Path] = field(default_factory=dict)
    maps: list[Path] = field(default_factory=list)
    charts: list[Path] = field(default_factory=list)
    vectors: dict[str, list[Path]] = field(default_factory=dict)
    summary_json: Path | None = None
    warnings: list[str] = field(default_factory=list)

    def files(self) -> list[Path]:
        out = [*self.reports.values(), *self.maps, *self.charts]
        for paths in self.vectors.values():
            out.extend(paths)
        if self.summary_json:
            out.append(self.summary_json)
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "directory": str(self.directory),
            "reports": {k: str(v) for k, v in self.reports.items()},
            "maps": [str(p) for p in self.maps],
            "charts": [str(p) for p in self.charts],
            "vectors": {k: [str(p) for p in v] for k, v in self.vectors.items()},
            "summary_json": str(self.summary_json) if self.summary_json else None,
            "warnings": list(self.warnings),
        }


# ---------------------------------------------------------------------------
# Context passed to the headline sentences
# ---------------------------------------------------------------------------

_CHANGE_NETWORKS = ("disaster_assessment", "deforestation", "urban_growth", "landslide_detection")


class _Ctx:
    def __init__(self, stats: dict[str, Any], meta: dict[str, Any], params: dict[str, Any]) -> None:
        self.s, self.m, self.p = stats, meta, params
        self.timeseries: list[dict[str, Any]] = []
        self.chart_captions: dict[Path, str] = {}
        self._rows: dict[str, list[dict[str, Any]]] = {}

    def f(self, code: str, v: Any) -> str:
        return F.fmt_value(v, code)

    def num(self, v: Any, nd: int = 2) -> str:
        return F.NA if F.is_missing(v) else F._unneg(f"{float(v):,.{nd}f}")

    def class_rows(self, layer_key: str) -> list[dict[str, Any]]:
        return self._rows.get(layer_key, [])

    def ha(self, layer_key: str, value: int) -> float | None:
        for r in self._rows.get(layer_key, []):
            if r["value"] == value:
                return r["area_ha"]
        return None


@dataclass
class _LayerOut:
    layer: Layer
    path: Path
    rows: list[dict[str, Any]] = field(default_factory=list)
    table: geo.ClassTable | None = None
    summary: dict[str, Any] = field(default_factory=dict)
    render: maps.LayerRender | None = None
    vector: Any = None
    values: np.ndarray | None = None          # sample of the values, for the histogram
    notes: list[str] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _norm_formats(formats: Sequence[str] | str | None, allowed: Sequence[str], default: Sequence[str], what: str,
                  warnings: list[str]) -> list[str]:
    if formats is None:
        return list(default)
    if isinstance(formats, str):
        formats = [x for x in formats.replace(";", ",").split(",") if x.strip()]
    out: list[str] = []
    for raw in formats:
        k = _ALIASES.get(str(raw).strip().lower(), str(raw).strip().lower())
        if k in ("all", "*"):
            out.extend(a for a in allowed if a not in out)
        elif k in ("none", ""):
            continue
        elif k in allowed:
            if k not in out:
                out.append(k)
        else:
            warnings.append(f"unknown {what} format '{raw}' ignored (choose from {', '.join(allowed)}).")
    return out


def _period(meta: dict[str, Any], stats: dict[str, Any]) -> str:
    if meta.get("date_before") and meta.get("date_after"):
        before, after = str(meta["date_before"]), str(meta["date_after"])
        if "\u2013" in before or "\u2013" in after:           # a date range on either side: "to" would read ambiguously
            return f"{before} \u2192 {after}"
        return f"{meta['date_before']} to {meta['date_after']}"
    if meta.get("dates"):
        return ", ".join(str(d) for d in meta["dates"])
    if meta.get("date"):
        return str(meta["date"])
    if stats.get("date"):
        return str(stats["date"])
    if meta.get("worldcover_year"):
        return f"WorldCover {meta['worldcover_year']}"
    return ""


def _bbox_text(bbox: Sequence[float]) -> str:
    w, s, e, n = bbox
    return f"{w:.4f}, {s:.4f}, {e:.4f}, {n:.4f}"


def _jsonable(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_jsonable(x) for x in v]
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, float) and not np.isfinite(v):
        return None
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def _resolve_defs(layer: Layer, meta: dict[str, Any], params: dict[str, Any], class_names: Any,
                  present: Sequence[int], pipeline: str) -> dict[int, ClassDef]:
    defs: dict[int, ClassDef] = {}
    if class_names:
        for v in set(present) | {0}:
            d = user_class(v, class_names, "No change" if pipeline in _CHANGE_NETWORKS else "Background")
            defs[v] = d
        return defs
    if layer.classes:
        defs = {d.value: d for d in layer.classes}
    elif layer.classes_hint == "landcover":
        src = str(meta.get("source") or params.get("source") or "worldcover")
        if src in ("worldcover", "dynamic_world"):
            defs = {d.value: d for d in landcover_classes(src)}
    elif layer.classes_hint == "crops":
        names = params.get("crop_classes") or meta.get("crop_classes")
        if names:
            defs = {0: user_class(0, None, "Background")}
            defs.update({i + 1: user_class(i + 1, list(["Background"] + list(names)), "Background") for i in range(len(names))})
    for v in present:
        if v not in defs:
            base0 = "No change" if pipeline in _CHANGE_NETWORKS else "Background"
            defs[v] = user_class(v, None, base0)
    return defs


def _read_plot(path: Path, band: int, kind: str, extra_nodata: tuple[int, ...], bbox: Sequence[float]) -> maps.LayerRender:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import transform_bounds

    with rasterio.open(path) as src:
        h, w = src.height, src.width
        f = min(1.0, _PLOT_MAX_PX / max(h, w))
        shape = (max(1, int(round(h * f))), max(1, int(round(w * f))))
        res = Resampling.mode if kind == "classes" else Resampling.average
        arr = src.read(band, out_shape=shape, resampling=res, masked=True)
        geographic = bool(src.crs is None or src.crs.is_geographic)
        factor = 1.0 if geographic or src.crs is None else float(src.crs.linear_units_factor[1])
        b = src.bounds
        extent = (b.left, b.right, b.bottom, b.top)
        aoi = None
        if geographic:
            aoi = tuple(bbox)
        elif src.crs is not None:
            aoi = tuple(transform_bounds("EPSG:4326", src.crs, *bbox))
    if kind == "classes":
        data = np.where(np.ma.getmaskarray(arr), -1, arr.data.astype(np.int32)).astype(np.int32)
        for nd in extra_nodata:
            data[data == nd] = -1
    else:
        data = np.ma.filled(arr.astype(np.float32), np.nan)
        data[~np.isfinite(data)] = np.nan
    return maps.LayerRender(key="", title="", kind=kind, data=data, extent=extent, geographic=geographic,
                            metres_per_unit=factor, aoi=aoi)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Layer analysis
# ---------------------------------------------------------------------------

def _load_layer(layer: Layer, c: _Ctx, out: Path, bbox: Sequence[float], *, make_vectors: bool, class_names: Any,
                pipeline: str, aoi_area_ha_hint: float, period: str, max_polygons: int, warn: list[str]) -> _LayerOut | None:
    import rasterio

    path = Path(layer.file)
    if not path.is_absolute():
        path = out / layer.file
    if not path.exists():
        if not layer.optional:
            warn.append(f"expected output '{layer.file}' was not found; the '{layer.title}' layer is missing from the report.")
        return None
    with rasterio.open(path) as src:
        if layer.band > src.count:
            warn.append(f"'{layer.file}' has {src.count} band(s); band {layer.band} for '{layer.title}' skipped.")
            return None
    extra_nodata = (0,) if layer.classes_hint == "landcover" else ()
    lo = _LayerOut(layer, path)
    arr, info, decimated = geo.read_band(path, layer.band, max_pixels=_VECTOR_MAX_PIXELS)
    lo.info = info
    for nd in extra_nodata:
        arr[arr == nd] = np.nan
    aoi_ha = aoi_area_ha_hint

    vec_arr: np.ndarray | None = None
    vec_defs: list = []
    vec_keep = layer.vector.keep if layer.vector else None
    if layer.kind == "classes":
        present = sorted(int(v) for v in np.unique(arr[np.isfinite(arr)]))
        defs = _resolve_defs(layer, c.m, c.p, class_names, present, pipeline)
        tbl = geo.class_table(path, layer.band, defs, extra_nodata=extra_nodata)
        lo.table, lo.rows = tbl, tbl.rows
        aoi_ha = tbl.total_area_ha or aoi_area_ha_hint
        vec_arr, vec_defs = arr, list(defs.values())
    else:
        lo.summary = geo.continuous_summary(arr)
        if decimated:
            lo.notes.append("statistics computed on a reduced-resolution read of a very large raster")
        lo.values = arr[np.isfinite(arr)][:5_000_000]
        if layer.binning is not None:
            cls, bdefs = geo.apply_binning(arr, layer.binning)
            tbl = geo.table_from_classes(cls, bdefs, {**info, "bounds": info["bounds"]})
            lo.table, lo.rows = tbl, tbl.rows
            aoi_ha = tbl.total_area_ha or aoi_area_ha_hint
            vec_arr = np.where(cls == 255, np.nan, cls).astype(np.float32)
            vec_defs = bdefs
            if vec_keep is None:
                vec_keep = layer.binning.keep
    c._rows[layer.key] = lo.rows

    r = _read_plot(path, layer.band, layer.kind, extra_nodata, bbox)
    r.key, r.title, r.cmap, r.vmin, r.vmax, r.center, r.label = layer.key, layer.title, layer.cmap, layer.vmin, layer.vmax, layer.center, layer.label
    r.stats = lo.summary
    r.muted = layer.muted
    if layer.kind == "classes":
        r.rows = lo.rows
    lo.render = r

    if make_vectors and layer.vector is not None and vec_arr is not None:
        if decimated:
            warn.append(f"vector layer '{layer.vector.name}' skipped: the raster has more than {_VECTOR_MAX_PIXELS:,} pixels. "
                        f"Polygonise it in tiles with 'pygeovision postprocess vectorise'.")
        else:
            from pygeovision.reporting.vectors import polygonise
            try:
                lo.vector = polygonise(
                    vec_arr, info["transform"], info["crs"], vec_defs, keep=vec_keep, skip=layer.vector.skip,
                    name=layer.vector.name, source_layer=layer.key, aoi_area_ha=aoi_ha, pipeline=pipeline,
                    period=period, max_polygons=max_polygons)
                if lo.vector.note and lo.vector.n_features:
                    warn.append(f"{layer.vector.name}: {lo.vector.note}")
            except Exception as exc:  # noqa: BLE001 - vectors are a deliverable, not the analysis
                logger.warning("vectorising '%s' failed: %s", layer.title, exc, exc_info=True)
                warn.append(f"vector layer '{layer.vector.name}' could not be created: {exc}")
    return lo


def _dynamic_layers(spec: ReportSpec, stats: dict[str, Any], out: Path) -> tuple[list[Layer], list[dict[str, Any]]]:
    """Layers for pipelines whose outputs depend on the run (one raster per date)."""
    layers: list[Layer] = []
    rows: list[dict[str, Any]] = []
    for date, d in (stats.get("per_date") or {}).items():
        p = Path(str(d.get("indices_path", "")))
        rel = p if p.is_absolute() else out / p
        key = F.slug(date)
        layers.append(Layer(f"ndvi_{key}", str(rel), f"NDVI {date}", "continuous", cmap="RdYlGn", vmin=-0.2, vmax=0.9,
                            label="NDVI", binning=_NDVI_BINS, vector=VectorSpec(f"ndvi_classes_{key}"),
                            primary=not layers, optional=True))
        rows.append({"date": date, "mean_ndvi": d.get("mean_ndvi"), "mean_evi": d.get("mean_evi"),
                     "mean_ndwi": d.get("mean_ndwi"), "valid_fraction": d.get("valid_fraction")})
    return layers, rows


def _chart_series(cs: Any, value: Any) -> tuple[list[str], list[Any]]:
    """Labels and values of a ``ChartSpec``'s statistic (a dict, or a list of dicts)."""
    if isinstance(value, dict):
        items = [(str(k), v) for k, v in value.items()]
    elif isinstance(value, (list, tuple)):
        keys = (cs.x,) if isinstance(cs.x, str) else tuple(cs.x)
        items = [(" → ".join(str(r.get(k)) for k in keys), r.get(cs.y)) for r in value if isinstance(r, dict)]
    else:
        return [], []
    return [a for a, _ in items], [b for _, b in items]


def _spec_tables(spec: ReportSpec, stats: dict[str, Any]) -> list[Table]:
    """Tables for the list / dict statistics a spec asks to show."""
    tables: list[Table] = []
    for ts in spec.tables:
        value = stats.get(ts.stat)
        if not value:
            continue
        if isinstance(value, dict):
            cols = [Col(ts.key_header, "text", "left", 3.0), Col(ts.value_header, ts.value_fmt, "right", 1.5)]
            rows = [[k, v] for k, v in value.items()]
        else:
            cols = [Col(head, code, "left" if code == "text" else "right", 2.5 if code == "text" else 1.3)
                    for _, head, code in ts.columns]
            rows = [[r.get(key) for key, _, _ in ts.columns] for r in value[:ts.max_rows]]
        note = f"Showing the first {ts.max_rows} of {len(value)} rows." if len(value) > ts.max_rows and not isinstance(value, dict) else ""
        tables.append(Table(ts.title, cols, rows, note=note, sheet=ts.sheet or ts.title[:31]))
    return tables


def _vector_file_layers(meta: dict[str, Any], out: Path, pipeline: str, period: str, aoi_area_ha: float,
                        max_features: int, warn: list[str]) -> list[Any]:
    """Vector layers a pipeline wrote itself (``metadata['vector_files']``): buildings, roads, streams, fire points."""
    from pygeovision.reporting.vectors import load_vector_file
    layers = []
    for vf in meta.get("vector_files") or []:
        try:
            path = Path(vf["path"])
            if not path.is_absolute() and not path.exists():
                # a relative path is normally relative to where the pipeline ran ("out/run/x.gpkg"), but accept one
                # relative to the output directory ("x.gpkg") too; joining the two blindly doubled the folder
                path = out / path if (out / path).exists() else path
            lyr = load_vector_file(vf["name"], path, pipeline=pipeline, period=period, aoi_area_ha=aoi_area_ha,
                                   max_features=max_features)
        except Exception as exc:  # noqa: BLE001 - a vector deliverable must not lose the rest of the report
            logger.warning("vector file %s failed: %s", vf, exc, exc_info=True)
            warn.append(f"vector layer '{vf.get('name', '?')}' could not be read: {exc}")
            continue
        if lyr.n_features:
            layers.append(lyr)
        elif lyr.note:
            warn.append(f"{lyr.name}: {lyr.note}")
    return layers


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

CLASS_COLS = [Col("Class", "text", "left", 3.0), Col("Value", "int", "right", 0.8), Col("Pixels", "int", "right", 1.3),
              Col("Area (ha)", "num1", "right", 1.4), Col("Share of valid area", "frac", "right", 1.5),
              Col("Share of AOI", "frac", "right", 1.3)]


def _class_table(lo: _LayerOut, sheet_title: str) -> Table:
    rows, colors = [], []
    for r in lo.rows:
        rows.append([r["name"], r["value"], r["pixels"], r["area_ha"], r["share"], r["share_aoi"]])
        colors.append(r["color"])
    note = ""
    if lo.table is not None:
        note = (f"Valid area {lo.table.valid_area_ha:,.1f} ha of {lo.table.total_area_ha:,.1f} ha "
                f"({lo.table.valid_pixels:,} of {lo.table.total_pixels:,} pixels). Areas are measured from the output raster.")
    return Table(sheet_title, CLASS_COLS, rows, colors, note, sheet="Classes", total_row=True, sum_cols=(2, 3))


def _stat_table(summary: dict[str, Any], unit: str, title: str) -> Table:
    labels = [("Minimum", "min"), ("5th percentile", "p5"), ("25th percentile", "p25"), ("Median", "median"), ("Mean", "mean"),
              ("75th percentile", "p75"), ("95th percentile", "p95"), ("Maximum", "max"), ("Standard deviation", "std")]
    rows = [[lab + (f" ({unit})" if unit and k not in ("n_valid",) else ""), summary[k]] for lab, k in labels if k in summary]
    note = f"Over {summary['n_valid']:,} valid pixels." if summary.get("n_valid") else ""
    return Table(title, [Col("Statistic", "text", "left", 3.0), Col("Value", "num3", "right", 1.5)], rows, note=note, sheet="Statistics")


def _metric_rows(spec: ReportSpec, stats: dict[str, Any]) -> tuple[list[list[Any]], set[str]]:
    rows, used = [], set()
    for m in spec.metrics:
        if m.key in stats:
            rows.append([m.label, stats[m.key], m.fmt, m.note])
            used.add(m.key)
    return rows, used


def _extra_stat_rows(stats: dict[str, Any], used: set[str]) -> list[list[Any]]:
    skip = {"note", "per_date", "class_breakdown", "source_note", "product_note"}
    rows = []
    for k, v in stats.items():
        if k in used or k in skip or k.endswith("_path") or isinstance(v, (dict, list, tuple)):
            continue
        rows.append([k.replace("_", " ").capitalize(), v, F.guess_code(k, v), ""])
    return rows


def _fmt_rows_for_table(rows: list[list[Any]]) -> Table:
    """A (label, value, code, note) list as a two-column table where each value keeps its own format."""
    if not any(r[3] for r in rows):
        return Table("", [Col("Measure", "text", "left", 3.0), Col("Value", "text", "right", 1.6)],
                     [[r[0], F.fmt_value(r[1], r[2])] for r in rows])
    return Table("", [Col("Measure", "text", "left", 3.0), Col("Value", "text", "right", 1.6), Col("Note", "text", "left", 3.0)],
                 [[r[0], F.fmt_value(r[1], r[2]), r[3]] for r in rows])


def _versions() -> list[tuple[str, str]]:
    out = [("PyGeoVision", __version__), ("Python", platform.python_version()), ("numpy", np.__version__)]
    try:
        import rasterio
        out.append(("rasterio", f"{rasterio.__version__} (GDAL {rasterio.__gdal_version__})"))
    except Exception as exc:  # noqa: BLE001 - informational only
        logger.debug("rasterio version unavailable: %s", exc)
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_deliverables(
    result: Any, bbox: Sequence[float], output_dir: str | Path, *,
    parameters: dict[str, Any] | None = None,
    report_formats: Sequence[str] | str | None = REPORT_FORMATS,
    vector_formats: Sequence[str] | str | None = DEFAULT_VECTOR_FORMATS,
    make_maps: bool = True, make_vectors: bool = True,
    class_names: Sequence[str] | dict[int, str] | None = None,
    title: str | None = None, max_polygons: int | None = None,
) -> Deliverables:
    """Create the map, vector layers and reports for a successful pipeline run.

    Writes under ``<output_dir>/deliverables/`` (reports at the top, ``maps/``,
    ``vectors/``) and a machine-readable ``run_summary.json`` in ``output_dir``.
    """
    if not getattr(result, "success", False):
        raise ValueError("deliverables are only created for a successful pipeline run")
    t_start = time.time()
    from pygeovision.reporting.vectors import (
        DEFAULT_MAX_POLYGONS,
        VECTOR_FORMATS,
        aoi_frame,
        write_vectors,
    )
    max_polygons = max_polygons or DEFAULT_MAX_POLYGONS
    out = Path(output_dir)
    dest = out / "deliverables"
    dest.mkdir(parents=True, exist_ok=True)
    warn: list[str] = []
    bbox = tuple(float(x) for x in bbox)
    rep_fmts = _norm_formats(report_formats, REPORT_FORMATS, REPORT_FORMATS, "report", warn)
    vec_fmts = _norm_formats(vector_formats, VECTOR_FORMATS, DEFAULT_VECTOR_FORMATS, "vector", warn)
    make_vectors = make_vectors and bool(vec_fmts)

    stats, meta, params = dict(result.stats or {}), dict(result.metadata or {}), dict(parameters or {})
    out_file = Path(result.output_path).name if result.output_path and Path(result.output_path).suffix.lower() in (".tif", ".tiff") else None
    spec = spec_for(result.pipeline, out_file)
    c = _Ctx(stats, meta, params)
    period = _period(meta, stats)
    aoi_km2 = geo.bbox_area_km2(bbox)
    deliv = Deliverables(dest, warnings=warn)

    # -- analyse every layer ------------------------------------------------
    layers: list[Layer] = list(spec.layers)
    if spec.extra_layers is not None:
        layers += list(spec.extra_layers(stats, meta))
    if spec.multi_date:
        layers, c.timeseries = _dynamic_layers(spec, stats, out)
    outs: list[_LayerOut] = []
    for layer in layers:
        try:
            lo = _load_layer(layer, c, out, bbox, make_vectors=make_vectors, class_names=class_names, pipeline=result.pipeline,
                             aoi_area_ha_hint=aoi_km2 * 100.0, period=period, max_polygons=max_polygons, warn=warn)
        except Exception as exc:  # noqa: BLE001 - keep delivering the other layers
            logger.warning("report layer '%s' failed: %s", layer.title, exc, exc_info=True)
            warn.append(f"layer '{layer.title}' could not be analysed: {exc}")
            continue
        if lo is not None:
            outs.append(lo)
    if not outs:
        warn.append("none of the pipeline's output rasters could be read; the report contains statistics only.")

    # -- maps and charts ----------------------------------------------------
    crs_text = "n/a"
    if outs and outs[0].info.get("crs") is not None:
        crs = outs[0].info["crs"]
        crs_text = crs.to_string() if hasattr(crs, "to_string") else str(crs)
    footer = maps.footer_text(crs_text)
    coverage = stats.get("aoi_coverage_fraction")
    valid = stats.get("valid_fraction")
    quality_warn = ""
    if coverage is not None and coverage < 0.95:
        quality_warn = f"Input imagery covers only {coverage * 100:.0f} % of the requested area."
    elif valid is not None and valid < 0.5:
        quality_warn = f"Only {valid * 100:.0f} % of the area has valid (cloud-free) pixels."

    metric_rows, used = _metric_rows(spec, stats)
    kpis_for_map = [(m[0], F.fmt_value(m[1], m[2])) for m in metric_rows if m[2] != "text" and not F.is_missing(m[1])][:4]
    map_paths: list[Path] = []
    chart_paths: list[Path] = []
    fig_for: dict[str, Path] = {}
    if make_maps and outs:
        mdir = dest / "maps"
        try:
            if spec.multi_date:
                renders = [lo.render for lo in outs if lo.render]
                p = maps.render_panel(renders, mdir / f"{result.pipeline}_ndvi_by_date.png", title=spec.title,
                                      subtitle=f"NDVI by date  ·  AOI {_bbox_text(bbox)}", footer=footer)
                map_paths.append(p)
                fig_for["__panel__"] = p
            else:
                first = True
                for lo in outs:
                    suffix = "map" if first else f"map_{lo.layer.key}"
                    p = maps.render_map(
                        lo.render, mdir / f"{result.pipeline}_{suffix}.png", title=spec.title,
                        subtitle=f"{lo.layer.title}  ·  {period + '  ·  ' if period else ''}AOI {_bbox_text(bbox)}",
                        footer=footer, kpis=kpis_for_map if first else (), warning=quality_warn)
                    map_paths.append(p)
                    fig_for[lo.layer.key] = p
                    first = False
        except Exception as exc:  # noqa: BLE001
            logger.warning("map rendering failed: %s", exc, exc_info=True)
            warn.append(f"map could not be drawn: {exc}")
        cdir = dest / "maps"
        try:
            if spec.charts:
                for cs in spec.charts:
                    labels, vals = _chart_series(cs, stats.get(cs.stat))
                    if not any(v is not None for v in vals):
                        continue
                    p = maps.chart_values(labels, vals, cdir / f"{result.pipeline}_chart_{F.slug(cs.stat + '_' + cs.y)}.png",
                                          title=cs.title, ylabel=cs.ylabel, kind=cs.kind, fmt=cs.fmt)
                    chart_paths.append(p)
                    c.chart_captions[p] = cs.title
            elif spec.chart == "timeseries" and c.timeseries:
                chart_paths.append(maps.chart_timeseries(c.timeseries, cdir / f"{result.pipeline}_chart.png", title="Scene means by date"))
            elif spec.chart == "before_after" and not F.is_missing(stats.get("area_ha_before")):
                chart_paths.append(maps.chart_before_after(stats.get("area_ha_before"), stats.get("area_ha_after"),
                                                           (str(meta.get("date_before")), str(meta.get("date_after"))),
                                                           cdir / f"{result.pipeline}_chart.png", title="Snow / ice area on commonly cloud-free pixels"))
            else:
                primary = next((lo for lo in outs if lo.rows), None)
                if primary is not None:
                    chart_paths.append(maps.chart_class_areas(primary.rows, cdir / f"{result.pipeline}_chart.png",
                                                              title=f"Area by class: {primary.layer.title}", muted=primary.layer.muted))
            cont = next((lo for lo in outs if lo.layer.kind == "continuous" and lo.values is not None and lo.values.size
                         and not spec.multi_date), None)
            if cont is not None:
                chart_paths.append(maps.chart_histogram(cont.values, cont.summary, cdir / f"{result.pipeline}_histogram.png",
                                                        title=f"Distribution: {cont.layer.title}", xlabel=cont.layer.label or cont.layer.title))
        except Exception as exc:  # noqa: BLE001
            logger.warning("chart rendering failed: %s", exc, exc_info=True)
            warn.append(f"chart could not be drawn: {exc}")
    deliv.maps, deliv.charts = map_paths, chart_paths

    # -- vectors ------------------------------------------------------------
    vec_layers = [lo.vector for lo in outs if lo.vector is not None]
    if make_vectors and vec_fmts:
        vec_layers += _vector_file_layers(meta, out, result.pipeline, period, aoi_km2 * 100.0, max_polygons, warn)
    if make_vectors and vec_fmts:
        try:
            aoi = aoi_frame(bbox, {"pipeline": result.pipeline, "title": spec.title, "period": period, "area_km2": round(aoi_km2, 3),
                                   "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d")})
            deliv.vectors, vwarn = write_vectors(vec_layers, aoi, dest / "vectors", result.pipeline, vec_fmts)
            warn.extend(vwarn)
        except Exception as exc:  # noqa: BLE001
            logger.warning("vector export failed: %s", exc, exc_info=True)
            warn.append(f"vector export failed: {exc}")
    elif make_vectors and not vec_layers:
        warn.append("this pipeline produced no polygon layers.")

    # -- report model -------------------------------------------------------
    report = _assemble(result, spec, c, outs, bbox, params, period, aoi_km2, metric_rows, used, deliv, fig_for, rep_fmts,
                       vec_layers, coverage, valid, warn, title, class_names)
    for fmt in rep_fmts:
        try:
            path = _write_report(fmt, report, dest / f"{result.pipeline}_report.{fmt}")
            deliv.reports[fmt] = path
        except ReportingUnavailable as exc:
            warn.append(str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s report failed: %s", fmt, exc, exc_info=True)
            warn.append(f"{fmt} report could not be written: {exc}")

    # -- machine-readable summary ------------------------------------------
    deliv.summary_json = _write_summary(out, result, bbox, params, spec, period, aoi_km2, outs, deliv, time.time() - t_start)
    if warn:
        for w in warn:
            logger.warning("deliverables: %s", w)
    return deliv


def _write_report(fmt: str, report: Report, path: Path) -> Path:
    try:
        if fmt == "pdf":
            from pygeovision.reporting.pdf import write_pdf as fn
        elif fmt == "docx":
            from pygeovision.reporting.docx_report import write_docx as fn
        else:
            from pygeovision.reporting.xlsx_report import write_xlsx as fn
    except ImportError as exc:
        raise ReportingUnavailable(
            f"{fmt} report skipped: {exc}. Install the reporting extras with: pip install \"pygeovision[report]\"") from exc
    return fn(report, path)


def _write_summary(out: Path, result: Any, bbox: Sequence[float], params: dict[str, Any], spec: ReportSpec, period: str,
                   aoi_km2: float, outs: list[_LayerOut], deliv: Deliverables, seconds: float) -> Path:
    doc = {
        "pygeovision_version": __version__,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pipeline": result.pipeline,
        "title": spec.title,
        "success": True,
        "bbox": list(bbox),
        "aoi_area_km2": round(aoi_km2, 4),
        "period": period,
        "output_path": str(result.output_path) if result.output_path else None,
        "duration_seconds": result.duration_seconds,
        "parameters": _jsonable(params),
        "metadata": _jsonable(result.metadata),
        "stats": _jsonable(result.stats),
        "layers": [{"key": lo.layer.key, "title": lo.layer.title, "file": str(lo.path), "kind": lo.layer.kind,
                    "classes": _jsonable(lo.rows), "summary": _jsonable(lo.summary)} for lo in outs],
        "deliverables": deliv.as_dict(),
        "deliverables_seconds": round(seconds, 2),
    }
    path = out / "run_summary.json"
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

def _assemble(result, spec, c, outs, bbox, params, period, aoi_km2, metric_rows, used, deliv, fig_for, rep_fmts, vec_layers,
              coverage, valid, warn, title, class_names) -> Report:
    stats, meta = c.s, c.m
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    clat, clon = (bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2

    # headline
    try:
        head = [h for h in spec.headline(c) if h]
    except Exception as exc:  # noqa: BLE001 - a wording failure must not lose the report
        logger.warning("headline for %s failed: %s", result.pipeline, exc, exc_info=True)
        warn.append(f"the summary sentences could not be generated: {exc}")
        head = []
    if not head:
        head = [f"The {spec.title.lower()} analysis completed for an area of {aoi_km2:,.1f} km²."]
    quality: list[tuple[str, str]] = []
    if coverage is not None and coverage < 0.95:
        quality.append(("warn", f"Input imagery covers only {coverage * 100:.0f} % of the requested area; results describe the covered part only."))
    if valid is not None and valid < 0.5:
        quality.append(("warn", f"Only {valid * 100:.0f} % of the area has valid (cloud-free) pixels, so percentages rest on a small part of the area."))
    if any(lo.notes for lo in outs):
        quality.append(("info", "Some statistics were computed on a reduced-resolution read of a very large raster."))
    published_dataset = "building_count" in stats          # building_footprints in dataset mode: no network involved
    if getattr(result, "pipeline", "") and spec.limitations and not published_dataset \
            and any("network you trained" in t for t in spec.limitations):
        quality.append(("warn", "Model output: accuracy is not assessed in this report. Validate against reference data before using the figures."))
    if not class_names and any(lo.layer.classes_hint == "user" for lo in outs):
        quality.append(("info", "Class names are generic. Pass class names (CLI --class-names) to label them in the maps and tables."))

    kpis = []
    for m in spec.metrics:
        if m.key in stats and not F.is_missing(stats[m.key]) and m.fmt not in ("text", "bool"):
            kpis.append((m.label, F.fmt_value(stats[m.key], m.fmt), ""))
        if len(kpis) == 4:
            break

    sections: list[Section] = []
    # 1 executive summary
    blocks = [bullets(head)]
    for lvl, txt in quality:
        blocks.append(callout(txt, lvl))
    note = stats.get("note")
    if note:
        blocks.append(callout(str(note), "info"))
    for key in ("source_note", "product_note"):
        if stats.get(key):
            blocks.append(callout(str(stats[key]), "info"))
    sections.append(Section("Executive summary", blocks))

    # 2 study area and inputs
    inputs_kv = [("Area of interest (west, south, east, north)", _bbox_text(bbox)),
                 ("Centre", f"{clat:.4f}°, {clon:.4f}°"),
                 ("Area of interest", F.fmt_value(aoi_km2, "km2"))]
    if period:
        inputs_kv.append(("Period", period))
    if outs:
        info = outs[0].info
        crs = info.get("crs")
        inputs_kv.append(("Output grid", f"{info['width']:,} × {info['height']:,} pixels, {crs.to_string() if crs else 'no CRS'}"))
        if outs[0].table:
            inputs_kv.append(("Raster extent area", F.fmt_value(outs[0].table.total_area_ha / 100.0, "km2")))
    if coverage is not None:
        inputs_kv.append(("Requested area covered by input imagery", F.fmt_value(coverage, "frac")))
    if valid is not None:
        inputs_kv.append(("Valid (cloud-free) share", F.fmt_value(valid, "frac")))
    blocks = [kv_block(inputs_kv)]
    cov = meta.get("input_coverage") or []
    if cov:
        rows = [[e.get("image", ""), e.get("aoi_coverage_fraction"), "\n".join(e.get("scenes_used", [])) or "-",
                 "\n".join(e.get("scenes_failed", [])) or "-"] for e in cov]
        blocks.append(table_block(Table("Imagery used", [Col("Image", "text", "left", 1.2), Col("Coverage of area", "frac", "right", 1.2),
                                                         Col("Scenes used", "text", "left", 4.0), Col("Scenes failed", "text", "left", 2.0)],
                                        rows, sheet="Inputs")))
    elif result.pipeline in ("solar_potential", "archaeological_site", "wind_farm_siting"):
        blocks.append(para(f"Inputs: a digital elevation model ({stats.get('dem_type', 'SRTM')}) from OpenTopography"
                           + (" and the ESA WorldCover 2021 land-cover map." if stats.get("landcover_exclusion_applied") else ".")))
    elif result.pipeline in ("land_cover", "biodiversity_hotspot") and meta.get("source", "worldcover") == "worldcover":
        blocks.append(para("Input: the ESA WorldCover 2021 10 m land-cover reference map (no satellite scene is used)."))
    sections.append(Section("Study area and inputs", blocks))

    # 3 results
    blocks = []
    main_fig = fig_for.get("__panel__") or next((fig_for[lo.layer.key] for lo in outs if lo.layer.key in fig_for), None)
    if main_fig:
        blocks.append(figure_block(main_fig, f"Map: {spec.title}"))
    for p in deliv.charts[:1]:
        blocks.append(figure_block(p, c.chart_captions.get(p, "Summary chart")))
    if c.timeseries:
        rows = [[r["date"], r["mean_ndvi"], r["mean_evi"], r["mean_ndwi"], r["valid_fraction"]] for r in c.timeseries]
        blocks.append(table_block(Table("Scene means by date", [Col("Date", "text", "left", 1.4), Col("Mean NDVI", "num3", "right"),
                                                               Col("Mean EVI", "num3", "right"), Col("Mean NDWI", "num3", "right"),
                                                               Col("Valid share", "frac", "right")], rows, sheet="Timeseries")))
    for lo in outs:
        if lo.rows and not spec.multi_date:
            blocks.append(table_block(_class_table(lo, f"Class areas: {lo.layer.title}")))
    for t in _spec_tables(spec, stats):
        blocks.append(table_block(t))
    if metric_rows:
        t = _fmt_rows_for_table(metric_rows)
        t.title, t.sheet = "Key statistics (reported by the pipeline)", "Metrics"
        t.cols[1].fmt = "text"
        t.raw = metric_rows
        blocks.append(table_block(t))
    extra = _extra_stat_rows(stats, used)
    if extra:
        t = _fmt_rows_for_table(extra)
        t.title, t.sheet = "Other statistics", "Metrics"
        t.raw = extra
        blocks.append(table_block(t))
    for lo in outs:
        if lo.layer.kind == "continuous" and lo.summary.get("n_valid") and not spec.multi_date:
            blocks.append(table_block(_stat_table(lo.summary, lo.layer.label if lo.layer.label and len(lo.layer.label) < 14 else "",
                                                  f"Value statistics: {lo.layer.title}")))
    for p in deliv.charts[1:]:
        blocks.append(figure_block(p, c.chart_captions.get(p, "Distribution of values")))
    sections.append(Section("Results", blocks))

    # 3b additional maps
    extra_maps = [(lo, fig_for[lo.layer.key]) for lo in outs if lo.layer.key in fig_for and fig_for[lo.layer.key] != main_fig]
    if extra_maps:
        sections.append(Section("Additional layers", [figure_block(p, f"Map: {lo.layer.title}") for lo, p in extra_maps], page_break=True))

    # 4 method
    mblocks = [para(spec.question)] + [para(t) for t in spec.method]
    prm = [("Pipeline", result.pipeline), ("Area of interest", _bbox_text(bbox))]
    for k, v in sorted(params.items()):
        if v is None or k in ("bbox", "output_dir"):
            continue
        prm.append((k, ", ".join(map(str, v)) if isinstance(v, (list, tuple)) else str(v)))
    if "date" not in params and period:
        prm.append(("period", period))
    mblocks.append(kv_block(prm))
    sections.append(Section("Method and parameters", mblocks))

    # 5 interpretation and limitations
    iblocks = []
    if spec.reading:
        iblocks.append(para("How to read the result"))
        iblocks.append(bullets(list(spec.reading)))
    if spec.limitations:
        iblocks.append(para("Limitations"))
        iblocks.append(bullets(list(spec.limitations)))
    if iblocks:
        sections.append(Section("Interpretation and limitations", iblocks))

    # 6 deliverables
    rel = lambda p: str(Path(p).relative_to(deliv.directory.parent)) if Path(p).is_absolute() and deliv.directory.parent in Path(p).parents else str(p)   # noqa: E731
    files: list[list[Any]] = []
    for lo in outs:
        files.append([rel(lo.path), "GeoTIFF raster", lo.layer.title])
    for fmt in rep_fmts:
        files.append([rel(deliv.directory / f"{result.pipeline}_report.{fmt}"), {"pdf": "PDF report", "docx": "Word report", "xlsx": "Excel workbook"}[fmt],
                      "Complete analysis report"])
    for p in deliv.maps + deliv.charts:
        files.append([rel(p), "PNG image", "Map or chart for slides and documents"])
    desc = {"gpkg": "GeoPackage (all layers)", "geojson": "GeoJSON, EPSG:4326", "shp": "Shapefile, EPSG:4326", "kml": "KML (Google Earth)"}
    for fmt, paths in deliv.vectors.items():
        for p in paths:
            files.append([rel(p), desc.get(fmt, fmt), "Vector features with attributes"])
    files.append(["run_summary.json", "JSON", "Machine-readable record of this run"])
    sections.append(Section("Deliverables", [table_block(Table("Files", [Col("File", "text", "left", 4.2), Col("Type", "text", "left", 2.0),
                                                                          Col("Contents", "text", "left", 3.2)], files, sheet="Files"))]))
    if vec_layers:
        def _total_ha(v):
            if v.classes is not None:
                return v.classes["area_ha"].sum()
            return float(v.gdf["area_ha"].sum()) if "area_ha" in v.gdf.columns else None
        vrows = [[v.name, v.n_features, _total_ha(v), v.note or ""] for v in vec_layers]
        sections[-1].blocks.append(table_block(Table("Vector layers", [Col("Layer", "text", "left", 2.4), Col("Features", "int", "right", 1.0),
                                                                       Col("Total area (ha)", "num1", "right", 1.4), Col("Note", "text", "left", 4.0)],
                                                      vrows, sheet="Vectors")))
        sections[-1].blocks.append(para("Vector layers made from rasters carry class_id, class_name, color_hex, area_ha, area_m2 (equal-area projection), "
                                        "pct_aoi, cx, cy, pipeline and period; layers a pipeline wrote itself (buildings, roads, streams, fire points) "
                                        "keep their own attributes. Coordinates are WGS 84 (EPSG:4326). "
                                        "The GeoPackage also holds the 'aoi' outline and, for raster-derived layers, a dissolved '<layer>_classes' summary."))

    # 7 references
    if spec.references:
        sections.append(Section("References", [bullets(list(spec.references))]))

    report = Report(
        title=title or spec.title,
        subtitle=" · ".join(x for x in (period, f"{aoi_km2:,.1f} km²") if x),
        pipeline=result.pipeline,
        meta=[("Area of interest", _bbox_text(bbox)), ("Period", period or "n/a"), ("Generated", generated),
              ("PyGeoVision", __version__), ("Run time", f"{result.duration_seconds:.1f} s")],
        kpis=kpis, sections=sections, warnings=list(warn), generated=generated, version=__version__,
        cover_image=next(iter(deliv.maps), None), timeseries=c.timeseries,
    )
    for v in vec_layers:
        g = v.gdf
        report.feature_totals[v.name] = len(g)
        cols = [x for x in ("patch_id", "class_id", "class_name", "area_ha", "pct_aoi", "cx", "cy") if x in g.columns]
        report.feature_tables[v.name] = g[cols].head(_EXCEL_FEATURE_ROWS).to_dict("records")
    return report


def regenerate(output_dir: str | Path, **kwargs: Any) -> Deliverables:
    """Rebuild the deliverables of an earlier run from its ``run_summary.json``.

    Useful to add another format, or to refresh a report after upgrading, without
    re-downloading imagery or re-running the analysis.
    """
    from types import SimpleNamespace
    out = Path(output_dir)
    path = out / "run_summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found: {out} is not the output directory of a pipeline run "
                                f"(or the run predates deliverables).")
    doc = json.loads(path.read_text(encoding="utf-8"))
    result = SimpleNamespace(success=True, pipeline=doc["pipeline"], output_path=doc.get("output_path"), stats=doc.get("stats") or {},
                             metadata=doc.get("metadata") or {}, duration_seconds=doc.get("duration_seconds") or 0.0)
    params = kwargs.pop("parameters", None) or doc.get("parameters") or {}
    return generate_deliverables(result, doc["bbox"], out, parameters=params, **kwargs)