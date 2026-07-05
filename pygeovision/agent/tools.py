"""
pygeovision.agent.tools
========================
Tool definitions for the GeoAgent. Every tool wraps a PyGeoVision
client capability as a callable with a structured JSON schema so
the LLM planner can reason about inputs, outputs, and constraints.

Adding a new tool
-----------------
1. Subclass ``GeoTool`` and implement ``run(self, **kwargs)``.
2. Add it to ``TOOL_REGISTRY`` at the bottom of this file.
3. The planner will automatically discover it.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("pygeovision.agent.tools")


# ── Tool result ────────────────────────────────────────────────────────────────

@dataclass
class ToolResult:
    tool: str
    success: bool
    output: Any = None           # structured output (dict, list, path …)
    output_path: Optional[str] = None
    error: str = ""
    duration_s: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "tool": self.tool,
            "success": self.success,
            "output": self.output,
            "output_path": self.output_path,
            "error": self.error,
            "duration_s": round(self.duration_s, 2),
            "metadata": self.metadata,
        }


# ── Base tool ──────────────────────────────────────────────────────────────────

class GeoTool:
    """Base class for all GeoAgent tools."""

    name:        str = "base_tool"
    description: str = ""
    category:    str = "general"

    # JSON-schema style parameter definitions
    parameters: List[Dict[str, Any]] = []

    def __init__(self, pgv_client: Any) -> None:
        self._pgv = pgv_client

    def run(self, **kwargs) -> ToolResult:
        raise NotImplementedError

    def _timed(self, fn: Callable, **kwargs) -> ToolResult:
        t0 = time.perf_counter()
        try:
            result = fn(**kwargs)
            return ToolResult(
                tool=self.name, success=True,
                output=result, duration_s=time.perf_counter() - t0,
            )
        except Exception as exc:
            logger.error("[%s] failed: %s", self.name, exc, exc_info=True)
            return ToolResult(
                tool=self.name, success=False,
                error=str(exc), duration_s=time.perf_counter() - t0,
            )

    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "parameters": self.parameters,
        }


# ── Data tools ─────────────────────────────────────────────────────────────────

class SearchTool(GeoTool):
    name        = "search_satellite_data"
    description = (
        "Search 22+ satellite data providers for scenes matching a "
        "bounding box, date range, and cloud cover constraint. "
        "Returns a list of available scenes with metadata."
    )
    category    = "data"
    parameters  = [
        {"name": "bbox",          "type": "list[float]", "required": True,
         "description": "[lon_min, lat_min, lon_max, lat_max] in WGS84"},
        {"name": "date_range",    "type": "tuple[str,str]", "required": True,
         "description": "('YYYY-MM-DD', 'YYYY-MM-DD')"},
        {"name": "providers",     "type": "list[str]", "required": False,
         "default": ["planetary_computer"],
         "description": "Provider IDs. Default: planetary_computer (open access, no key needed)."},
        {"name": "cloud_cover_max", "type": "int", "required": False, "default": 20},
        {"name": "satellites",    "type": "list[str]", "required": False,
         "description": "e.g. ['Sentinel-2', 'Landsat-8']. None = all."},
    ]

    def run(self, bbox, date_range, providers=None, cloud_cover_max=20,
            satellites=None, **_) -> ToolResult:
        t0 = time.perf_counter()
        try:
            results = self._pgv.search(
                bbox=bbox, date_range=date_range,
                providers=providers or ["planetary_computer"],
                cloud_cover_max=cloud_cover_max,
                satellites=satellites,
            )
            scenes = [
                {"id": getattr(r, "scene_id", str(i)),
                 "provider": getattr(r, "provider", "unknown"),
                 "date": str(getattr(r, "date", "")),
                 "cloud_cover": getattr(r, "cloud_cover", None)}
                for i, r in enumerate(results or [])
            ]
            return ToolResult(
                tool=self.name, success=True,
                output={"scenes": scenes, "count": len(scenes)},
                duration_s=time.perf_counter() - t0,
                metadata={"providers": providers, "bbox": bbox},
            )
        except Exception as exc:
            return ToolResult(tool=self.name, success=False, error=str(exc),
                              duration_s=time.perf_counter() - t0)


class DownloadTool(GeoTool):
    name        = "download_satellite_data"
    description = (
        "Download satellite scenes returned by search_satellite_data. "
        "Applies post-processing (reproject, COG) automatically."
    )
    category    = "data"
    parameters  = [
        {"name": "scenes",       "type": "list", "required": True,
         "description": "Scene objects from search_satellite_data output."},
        {"name": "output_dir",   "type": "str",  "required": False, "default": "./data/"},
        {"name": "bands",        "type": "list[str]", "required": False,
         "description": "Band subset e.g. ['B02','B03','B04','B08','B11','B12']"},
        {"name": "max_scenes",   "type": "int",  "required": False, "default": 3},
    ]

    def run(self, scenes, output_dir="./data/", bands=None, max_scenes=3, **_) -> ToolResult:
        t0 = time.perf_counter()
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            results = self._pgv.download(
                (scenes or [])[:max_scenes],
                output_dir=output_dir,
                bands=bands,
                post_process=["reproject:EPSG:4326", "cog"],
            )
            paths = [r.path for r in (results or []) if getattr(r, "success", False)]
            return ToolResult(
                tool=self.name, success=bool(paths),
                output={"downloaded_paths": [str(p) for p in paths], "count": len(paths)},
                output_path=str(paths[0]) if paths else None,
                duration_s=time.perf_counter() - t0,
            )
        except Exception as exc:
            return ToolResult(tool=self.name, success=False, error=str(exc),
                              duration_s=time.perf_counter() - t0)


class PrepareTool(GeoTool):
    name        = "prepare_for_ai"
    description = (
        "Preprocess a downloaded raster for AI model input: "
        "stack bands, clip to bbox, apply cloud mask, normalise, validate. "
        "Returns a normalised float32 array ready for model inference."
    )
    category    = "preprocessing"
    parameters  = [
        {"name": "input_path", "type": "str", "required": True},
        {"name": "bbox",       "type": "list[float]", "required": False},
        {"name": "bands",      "type": "list[str]",   "required": False},
        {"name": "model_type", "type": "str", "required": False, "default": "segmentation",
         "description": "segmentation | foundation | detection | change_detection"},
        {"name": "output_path","type": "str", "required": False},
    ]

    def run(self, input_path, bbox=None, bands=None, model_type="segmentation",
            output_path=None, **_) -> ToolResult:
        t0 = time.perf_counter()
        try:
            result = self._pgv.prepare_for_ai(
                input_path, stack_bands=bands, bbox=bbox,
                normalise="scale_factor", scale_factor=10000.0,
                model_type=model_type, output_path=output_path,
            )
            return ToolResult(
                tool=self.name, success=True,
                output={"shape": result.get("shape"), "range": [
                    float(result["array"].min()), float(result["array"].max())]},
                output_path=output_path,
                duration_s=time.perf_counter() - t0,
                metadata={"model_type": model_type, "bands": bands},
            )
        except Exception as exc:
            return ToolResult(tool=self.name, success=False, error=str(exc),
                              duration_s=time.perf_counter() - t0)


# ── SAR tools ──────────────────────────────────────────────────────────────────

class SARPreprocessTool(GeoTool):
    name        = "sar_preprocess"
    description = (
        "Preprocess a Sentinel-1 GRD SAR raster through the complete "
        "S0–S9 pipeline: verify download → validate georeference → "
        "despeckle (LINEAR) → dB → normalise → CRS-aware clip. "
        "Applies all three production bug-fixes (BUG 1/2/3). "
        "Use for cloud-independent flood mapping, damage detection, or "
        "any task where optical satellites are unavailable."
    )
    category    = "sar"
    parameters  = [
        {"name": "raw_path",    "type": "str", "required": True,
         "description": "Path to raw Sentinel-1 GRD TIFF (pre-reproject or reprojected)."},
        {"name": "output_path", "type": "str", "required": True},
        {"name": "bbox_wgs84",  "type": "list[float]", "required": True,
         "description": "[lon_min, lat_min, lon_max, lat_max] WGS84."},
        {"name": "filter_type", "type": "str", "required": False,
         "default": "enhanced_lee",
         "description": "enhanced_lee | refined_lee | boxcar"},
    ]

    def run(self, raw_path, output_path, bbox_wgs84, filter_type="enhanced_lee", **_) -> ToolResult:
        t0 = time.perf_counter()
        try:
            from pygeovision.data.processors.sar import (
                verify_sar_downloads, validate_sar_georeference,
                despeckle_sar, linear_to_db, normalise_sar_for_ai, clip_sar_to_bbox,
            )
            from pygeovision.data.validators.georeference import check_download_complete
            import pathlib as _pl

            raw = _pl.Path(raw_path)
            out = _pl.Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            work = out.parent / "work_sar"
            work.mkdir(exist_ok=True)

            dl_check = check_download_complete(str(raw))
            if not dl_check["complete"]:
                return ToolResult(tool=self.name, success=False,
                                  error=f"Incomplete download: {dl_check['errors'][:1]}",
                                  duration_s=time.perf_counter() - t0)

            geo = validate_sar_georeference(str(raw))
            working = geo.repaired_path or str(raw)

            steps = []
            desp = str(work / (raw.stem + "_desp.tif"))
            despeckle_sar(working, desp, filter_type=filter_type, window_size=7)
            steps.append("despeckle")

            db = str(work / (raw.stem + "_db.tif"))
            linear_to_db(desp, db)
            steps.append("linear_to_db")

            norm = str(work / (raw.stem + "_norm.tif"))
            normalise_sar_for_ai(db, norm, method="minmax_db")
            steps.append("normalise")

            clip_sar_to_bbox(norm, str(out), tuple(bbox_wgs84))
            steps.append("clip")

            return ToolResult(
                tool=self.name, success=True,
                output={"steps": steps, "georeference_valid": geo.valid,
                        "repaired": geo.repaired},
                output_path=str(out),
                duration_s=time.perf_counter() - t0,
            )
        except Exception as exc:
            return ToolResult(tool=self.name, success=False, error=str(exc),
                              duration_s=time.perf_counter() - t0)


class SARFloodTool(GeoTool):
    name        = "sar_flood_detection"
    description = (
        "Detect flood inundation from a preprocessed Sentinel-1 SAR array "
        "using Prithvi domain adaptation (zero-shot VH threshold). "
        "Cloud-independent — works at night and through storm clouds. "
        "Best used within 6 hours of a new Sentinel-1 pass."
    )
    category    = "sar"
    parameters  = [
        {"name": "sar_ready_path", "type": "str", "required": True,
         "description": "Output of sar_preprocess tool."},
        {"name": "output_path",    "type": "str", "required": True},
        {"name": "mode", "type": "str", "required": False, "default": "zero_shot",
         "description": "zero_shot | fine_tune | spt"},
    ]

    def run(self, sar_ready_path, output_path, mode="zero_shot", **_) -> ToolResult:
        t0 = time.perf_counter()
        try:
            import numpy as np
            import rasterio
            from pygeovision.models.adapters.sar_prithvi import SARPrithviAdapter

            with rasterio.open(sar_ready_path) as src:
                vv = src.read(1).astype("float32")[np.newaxis]
                vh = src.read(2).astype("float32")[np.newaxis] if src.count >= 2 else vv / 2.5
                prof = src.profile.copy()

            adapter = SARPrithviAdapter(mode=mode, task="flood_detection")
            result  = adapter.run(vv, vh)
            mask    = result["prediction"]

            prof.update(count=1, dtype="uint8", nodata=255)
            import pathlib as _pl
            _pl.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(output_path, "w", **prof) as dst:
                dst.write(mask[np.newaxis], 1)

            flooded_pct = float(mask.mean() * 100)
            return ToolResult(
                tool=self.name, success=True,
                output={"flooded_pixels": int(mask.sum()),
                        "flooded_pct": round(flooded_pct, 2),
                        "mode": mode},
                output_path=output_path,
                duration_s=time.perf_counter() - t0,
            )
        except Exception as exc:
            return ToolResult(tool=self.name, success=False, error=str(exc),
                              duration_s=time.perf_counter() - t0)


# ── Inference tools ────────────────────────────────────────────────────────────

class PrithviInferenceTool(GeoTool):
    name        = "prithvi_inference"
    description = (
        "Run Prithvi-EO-2.0 (600M parameter geospatial foundation model) "
        "for land cover, flood detection, crop mapping, or burn scar mapping. "
        "Input must be a 6-band HLS-normalised raster from prepare_for_ai."
    )
    category    = "inference"
    parameters  = [
        {"name": "input_path", "type": "str", "required": True},
        {"name": "task", "type": "str", "required": True,
         "description": "land_cover | flood_detection | crop_mapping | burn_scar"},
        {"name": "output_path", "type": "str", "required": True},
        {"name": "source", "type": "str", "required": False, "default": "sentinel2",
         "description": "sentinel2 | landsat | hls"},
        {"name": "source_bands", "type": "list[str]", "required": False},
    ]

    def run(self, input_path, task, output_path, source="sentinel2",
            source_bands=None, **_) -> ToolResult:
        t0 = time.perf_counter()
        try:
            import numpy as np
            import rasterio
            from pygeovision.models.foundation.prithvi import (
                PrithviTasks, validate_prithvi_input,
            )

            with rasterio.open(input_path) as src:
                arr = src.read().astype("float32")

            val = validate_prithvi_input(arr, source=source, n_prithvi_bands=6)
            if not val.get("valid"):
                return ToolResult(tool=self.name, success=False,
                                  error=f"Input validation failed: {val}",
                                  duration_s=time.perf_counter() - t0)

            tasks  = PrithviTasks("prithvi_eo_2_0")
            method = getattr(tasks, task.replace("-", "_"), None)
            if method is None:
                return ToolResult(tool=self.name, success=False,
                                  error=f"Unknown task '{task}'",
                                  duration_s=time.perf_counter() - t0)

            import pathlib as _pl
            _pl.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            result = method(arr, source=source, source_bands=source_bands,
                            output_path=output_path)

            return ToolResult(
                tool=self.name, success=True,
                output={"task": task, "shape": result.get("shape"),
                        "classes": result.get("class_distribution")},
                output_path=output_path,
                duration_s=time.perf_counter() - t0,
            )
        except Exception as exc:
            return ToolResult(tool=self.name, success=False, error=str(exc),
                              duration_s=time.perf_counter() - t0)


class ChangeDetectionTool(GeoTool):
    name        = "change_detection"
    description = (
        "Detect changes between two co-registered rasters using ChangeFormer "
        "(bi-temporal MiT-B2 transformer). Output is a per-class change mask. "
        "Supports 2-class (changed/unchanged) or 4-class (no/minor/moderate/severe)."
    )
    category    = "inference"
    parameters  = [
        {"name": "pre_path",    "type": "str", "required": True},
        {"name": "post_path",   "type": "str", "required": True},
        {"name": "output_path", "type": "str", "required": True},
        {"name": "num_classes", "type": "int", "required": False, "default": 2},
        {"name": "in_channels", "type": "int", "required": False, "default": 6},
    ]

    def run(self, pre_path, post_path, output_path, num_classes=2,
            in_channels=6, **_) -> ToolResult:
        t0 = time.perf_counter()
        try:
            from pygeovision.models.change_detection.changeformer import ChangeDetection
            from pygeovision.models.adapters.sar_channel_manager import coregister_sar_pair
            import pathlib as _pl, tempfile

            out_dir = _pl.Path(output_path).parent
            out_dir.mkdir(parents=True, exist_ok=True)

            # Co-register to ensure pixel alignment
            with tempfile.TemporaryDirectory() as tmp:
                _, post_aligned = coregister_sar_pair(pre_path, post_path, tmp)
                cd = ChangeDetection(model_variant="changeformer",
                                     in_channels=in_channels, num_classes=num_classes)
                cd.build()
                result = cd.detect(pre_path, post_aligned, output_path=output_path)

            return ToolResult(
                tool=self.name, success=True,
                output={"change_pct": result.get("change_pct"),
                        "num_classes": num_classes},
                output_path=output_path,
                duration_s=time.perf_counter() - t0,
            )
        except Exception as exc:
            return ToolResult(tool=self.name, success=False, error=str(exc),
                              duration_s=time.perf_counter() - t0)


class SpectralIndexTool(GeoTool):
    name        = "compute_spectral_index"
    description = (
        "Compute a spectral index (NDVI, NDWI, NDBI, EVI, NBR, MNDWI) "
        "from a multi-band raster and save the result as a single-band GeoTIFF."
    )
    category    = "analysis"
    parameters  = [
        {"name": "input_path",  "type": "str", "required": True},
        {"name": "index",       "type": "str", "required": True,
         "description": "ndvi | ndwi | ndbi | evi | nbr | mndwi"},
        {"name": "output_path", "type": "str", "required": True},
    ]

    def run(self, input_path, index, output_path, **_) -> ToolResult:
        t0 = time.perf_counter()
        try:
            import pathlib as _pl
            _pl.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fn = getattr(self._pgv.indices, index.lower(), None)
            if fn is None:
                return ToolResult(tool=self.name, success=False,
                                  error=f"Unknown index '{index}'",
                                  duration_s=time.perf_counter() - t0)
            fn(input_path, output_path=output_path)
            import numpy as np, rasterio
            with rasterio.open(output_path) as src:
                data = src.read(1)
                valid = data[np.isfinite(data)]
            return ToolResult(
                tool=self.name, success=True,
                output={"index": index,
                        "min": round(float(valid.min()), 4),
                        "max": round(float(valid.max()), 4),
                        "mean": round(float(valid.mean()), 4)},
                output_path=output_path,
                duration_s=time.perf_counter() - t0,
            )
        except Exception as exc:
            return ToolResult(tool=self.name, success=False, error=str(exc),
                              duration_s=time.perf_counter() - t0)


class PostprocessTool(GeoTool):
    name        = "postprocess"
    description = (
        "Postprocess a prediction raster: sieve (remove small spurious patches), "
        "vectorise (raster → GeoJSON polygons), and export as Cloud-Optimised GeoTIFF."
    )
    category    = "postprocessing"
    parameters  = [
        {"name": "input_path",      "type": "str",  "required": True},
        {"name": "operations",      "type": "list[str]", "required": True,
         "description": "Ordered list of: sieve | vectorise | cog"},
        {"name": "output_dir",      "type": "str",  "required": False, "default": "./outputs/"},
        {"name": "min_pixels",      "type": "int",  "required": False, "default": 40},
        {"name": "min_area_m2",     "type": "float","required": False, "default": 1000.0},
        {"name": "target_class",    "type": "int",  "required": False, "default": 1},
    ]

    def run(self, input_path, operations, output_dir="./outputs/",
            min_pixels=40, min_area_m2=1000.0, target_class=1, **_) -> ToolResult:
        t0 = time.perf_counter()
        try:
            import pathlib as _pl
            out_dir = _pl.Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            stem = _pl.Path(input_path).stem
            outputs = {}
            current = input_path

            for op in operations:
                if op == "sieve":
                    sieved = str(out_dir / f"{stem}_sieved.tif")
                    self._pgv.postprocess.sieve_filter(current, min_pixels=min_pixels,
                                                       output_path=sieved)
                    current = sieved
                    outputs["sieved"] = sieved
                elif op == "vectorise":
                    vec = str(out_dir / f"{stem}.geojson")
                    self._pgv.postprocess.vectorise(current, vec,
                                                    target_class=target_class,
                                                    min_area_m2=min_area_m2)
                    outputs["geojson"] = vec
                elif op == "cog":
                    cog = str(out_dir / f"{stem}_cog.tif")
                    self._pgv.postprocess.to_cog(current, cog)
                    outputs["cog"] = cog

            return ToolResult(
                tool=self.name, success=True,
                output={"operations_applied": operations, "outputs": outputs},
                output_path=outputs.get("geojson") or outputs.get("cog") or current,
                duration_s=time.perf_counter() - t0,
            )
        except Exception as exc:
            return ToolResult(tool=self.name, success=False, error=str(exc),
                              duration_s=time.perf_counter() - t0)


class EndToEndPipelineTool(GeoTool):
    name        = "run_pipeline"
    description = (
        "Run a named end-to-end pipeline (search → download → AI → postprocess). "
        "Fastest path for standard tasks. For custom workflows, chain individual tools."
    )
    category    = "pipeline"
    parameters  = [
        {"name": "pipeline_name", "type": "str", "required": True,
         "description": (
             "One of: building_footprints | change_detection | land_cover | "
             "flood_mapping | crop_mapping | forest_monitoring | road_network | "
             "solar_panels | wildfire_severity | glacier_monitoring"
         )},
        {"name": "bbox",        "type": "list[float]", "required": True},
        {"name": "date",        "type": "str", "required": True,
         "description": "ISO date or YYYY-MM range e.g. '2024-06'"},
        {"name": "output_dir",  "type": "str", "required": False, "default": "./outputs/"},
    ]

    def run(self, pipeline_name, bbox, date, output_dir="./outputs/", **_) -> ToolResult:
        t0 = time.perf_counter()
        try:
            import pathlib as _pl
            _pl.Path(output_dir).mkdir(parents=True, exist_ok=True)
            result = self._pgv.pipeline(
                pipeline_name, bbox=bbox, date=date, output_dir=output_dir,
            )
            return ToolResult(
                tool=self.name, success=getattr(result, "success", True),
                output={"pipeline": pipeline_name,
                        "stats": getattr(result, "stats", {})},
                output_path=str(getattr(result, "output_path", "")),
                duration_s=time.perf_counter() - t0,
            )
        except Exception as exc:
            return ToolResult(tool=self.name, success=False, error=str(exc),
                              duration_s=time.perf_counter() - t0)


# ── Tool registry ──────────────────────────────────────────────────────────────

TOOL_REGISTRY: Dict[str, type] = {
    "search_satellite_data":   SearchTool,
    "download_satellite_data": DownloadTool,
    "prepare_for_ai":          PrepareTool,
    "sar_preprocess":          SARPreprocessTool,
    "sar_flood_detection":     SARFloodTool,
    "prithvi_inference":       PrithviInferenceTool,
    "change_detection":        ChangeDetectionTool,
    "compute_spectral_index":  SpectralIndexTool,
    "postprocess":             PostprocessTool,
    "run_pipeline":            EndToEndPipelineTool,
}


def build_tools(pgv_client: Any) -> Dict[str, GeoTool]:
    """Instantiate all tools bound to a PyGeoVision client."""
    return {name: cls(pgv_client) for name, cls in TOOL_REGISTRY.items()}
