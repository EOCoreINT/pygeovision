"""AutoLabelPipeline -- run several label sources over one area and (when it makes sense) fuse them.

Fusion is only meaningful between rasters that use the *same label scheme*. Two building maps
(OSM ``buildings``, Microsoft, Google) are all "1 = building" and can be voted on. An OSM
multi-class map (1 building, 2 road, 3 water ...) and an ESA WorldCover map (10 tree, 50 built-up ...)
are different alphabets: a vote between them is noise, so they are written side by side and **not**
fused (``fusion_skipped_reason`` says why). SAM instance ids and foundation-model cluster ids are never fused.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

FUSION_MODES = ("majority_vote", "union", "intersection", "priority")
#: Sources whose output values are arbitrary ids, never comparable with another raster.
_NEVER_FUSE = {"sam_auto", "foundation"}
NODATA = 255


class AutoLabelPipeline:
    """Orchestrate several auto-labeling sources and fuse compatible results.

    Example::

        pipeline = AutoLabelPipeline(
            sources=["osm", "microsoft_buildings", "google_buildings"],
            fusion="majority_vote",
            reference_raster="./data/scene.tif",
        )
        result = pipeline.run(
            bbox=(-74.05, 40.70, -73.95, 40.80),
            output_dir="./labels/nyc/",
            categories=["buildings"],
        )
        result["fused_path"]        # the vote, on the grid of scene.tif
    """

    SOURCES = ["osm", "microsoft_buildings", "google_buildings",
               "esa_worldcover", "dynamic_world", "sam_auto", "foundation"]

    def __init__(
        self,
        sources: list[str] | None = None,
        fusion: str = "majority_vote",  # majority_vote | union | intersection | priority
        quality_threshold: float = 0.6,
        reference_raster: str | None = None,
    ) -> None:
        if fusion not in FUSION_MODES:
            raise ValueError(f"fusion must be one of {FUSION_MODES}, got {fusion!r}")
        self.sources = sources or ["osm", "esa_worldcover"]
        unknown = [s for s in self.sources if s not in self.SOURCES]
        if unknown:
            raise ValueError(f"Unknown source(s) {unknown}. Choose from: {self.SOURCES}")
        self.fusion = fusion
        self.quality_threshold = quality_threshold
        self.reference_raster = reference_raster

    # ------------------------------------------------------------------ running
    def _run_source(self, source: str, bbox, output_dir: str, categories) -> dict:
        ref = self.reference_raster
        if source == "osm":
            from pygeovision.labeling.osm import OSMLabeler
            return OSMLabeler().label(
                bbox, categories=categories or ["buildings", "roads", "water"],
                output_path=f"{output_dir}/osm.tif", reference_raster=ref)
        if source == "microsoft_buildings":
            from pygeovision.labeling.buildings import MicrosoftBuildingsLabeler
            return MicrosoftBuildingsLabeler().label(
                bbox, output_path=f"{output_dir}/ms_buildings.tif", reference_raster=ref)
        if source == "google_buildings":
            from pygeovision.labeling.buildings import GoogleBuildingsLabeler
            return GoogleBuildingsLabeler().label(
                bbox, output_path=f"{output_dir}/google_buildings.tif", reference_raster=ref)
        if source == "esa_worldcover":
            from pygeovision.labeling.landcover import ESAWorldCoverLabeler
            return ESAWorldCoverLabeler().label(
                bbox, output_path=f"{output_dir}/esa_worldcover.tif", reference_raster=ref)
        if source == "dynamic_world":
            from pygeovision.labeling.landcover import DynamicWorldLabeler
            return DynamicWorldLabeler().label(
                bbox, output_path=f"{output_dir}/dynamic_world.tif", reference_raster=ref)
        if source in ("sam_auto", "foundation"):
            if not ref:
                return {"success": False,
                        "error": f"{source} requires reference_raster (it segments an actual image, not a bbox query)"}
            if source == "sam_auto":
                from pygeovision.labeling.sam_auto import SAMAutoLabeler
                return SAMAutoLabeler().auto_label(ref, output_path=f"{output_dir}/sam_auto.tif")
            from pygeovision.labeling.foundation import FoundationModelLabeler
            return FoundationModelLabeler().pseudo_label(ref, output_path=f"{output_dir}/foundation.tif")
        return {"success": False, "error": f"Unknown source: '{source}'. Choose from: {self.SOURCES}"}

    def run(
        self,
        bbox: tuple[float, ...],
        output_dir: str = "./labels/",
        categories: list[str] | None = None,
        assess_quality: bool = True,
    ) -> dict[str, Any]:
        """Run every source, fuse the compatible ones, and assess quality.

        Returns:
            ``success`` (at least one source produced a raster), ``sources_succeeded``,
            ``sources_failed`` (each with its error in ``partial_results``), ``label_paths``,
            ``fused_path`` (``None`` when only one source produced a raster, or when the rasters are not
            comparable), ``fusion_skipped_reason``, ``quality`` (for the fused/only raster) and
            ``quality_by_source``.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        partial_results: dict[str, dict] = {}

        for source in self.sources:
            try:
                r = self._run_source(source, bbox, output_dir, categories)
                partial_results[source] = r
                logger.info("Source '%s': %s", source, "OK" if r.get("success") else r.get("error"))
            except Exception as exc:
                partial_results[source] = {"success": False, "error": f"{type(exc).__name__}: {exc}"}
                logger.warning("Source '%s' failed: %s", source, exc)

        # A source "succeeds" only when it left a raster on disk (Earth Engine exports to Drive instead).
        successful, label_paths = [], []
        for k, v in partial_results.items():
            p = v.get("output_path")
            if v.get("success") and p and Path(p).exists():
                successful.append(k)
                label_paths.append(p)
            elif v.get("success"):
                v["success"] = False
                v["error"] = v.get("error") or "no raster was written (asynchronous export?)"

        fused_path = None
        skipped_reason = None
        if len(label_paths) == 1:
            fused_path = label_paths[0]
        elif len(label_paths) > 1:
            skipped_reason = self._incompatibility(successful, label_paths)
            if skipped_reason is None:
                fused_path = self._fuse(label_paths, f"{output_dir}/fused_labels.tif")
                if fused_path is None:
                    skipped_reason = "fusion failed; see the log"
            if skipped_reason:
                logger.warning("Not fusing %s: %s", successful, skipped_reason)

        result: dict[str, Any] = {
            "success": len(successful) > 0,
            "sources_succeeded": successful,
            "sources_failed": [k for k in partial_results if k not in successful],
            "label_paths": label_paths,
            "fused_path": str(fused_path) if fused_path else None,
            "fusion_skipped_reason": skipped_reason,
            "partial_results": partial_results,
        }

        if assess_quality and label_paths:
            from pygeovision.labeling.quality import LabelQualityAssessor
            qa = LabelQualityAssessor(num_classes=None)
            result["quality_by_source"] = {s: qa.assess(p) for s, p in zip(successful, label_paths)}
            if fused_path:
                result["quality"] = qa.assess(fused_path)
        return result

    # ------------------------------------------------------------------ fusion
    @staticmethod
    def _values(path: str):
        import numpy as np
        import rasterio
        with rasterio.open(path) as src:
            data = src.read(1)
            nd = src.nodata
        vals = set(np.unique(data).tolist())
        if nd is not None:
            vals.discard(nd)
        return vals

    def _incompatibility(self, sources: list[str], paths: list[str]) -> str | None:
        """Why these rasters must not be fused, or ``None`` when they can."""
        never = [s for s in sources if s in _NEVER_FUSE]
        if never:
            return (f"{never} produce arbitrary ids (SAM instances / foundation clusters) that mean nothing "
                    "to another raster")
        value_sets = {s: sorted(self._values(p)) for s, p in zip(sources, paths)}
        if all(set(v) <= {0, 1} for v in value_sets.values()):
            return None
        shown = "; ".join(f"{s} has values {v[:8]}{'...' if len(v) > 8 else ''}" for s, v in value_sets.items())
        return (f"these sources use different label alphabets ({shown}). A vote between them would mix unrelated "
                "classes (e.g. OSM 1=building, 2=road vs ESA 10=tree, 50=built-up). Label one target class from "
                "each source (OSM categories=['buildings'] gives 0/1), or remap each source to a common scheme "
                "first. The per-source rasters were still written.")

    def _fuse(self, paths: list[str], output: str) -> str | None:
        """Fuse rasters on one grid. Binary modes: majority_vote, union, intersection, priority."""
        try:
            import numpy as np
            import rasterio
            from rasterio.warp import Resampling, reproject

            grid_path = self.reference_raster or paths[0]
            with rasterio.open(grid_path) as g:
                crs, tf, h, w = g.crs, g.transform, g.height, g.width

            layers = []      # int32, -1 = abstain
            for p in paths:
                with rasterio.open(p) as src:
                    dst = np.full((h, w), -1, dtype=np.int32)
                    reproject(rasterio.band(src, 1), dst, src_transform=src.transform, src_crs=src.crs,
                              src_nodata=src.nodata, dst_transform=tf, dst_crs=crs, dst_nodata=-1,
                              resampling=Resampling.nearest)
                layers.append(dst)
            stack = np.stack(layers)               # (n, H, W)
            n = stack.shape[0]
            voted = stack >= 0
            any_vote = voted.any(0)

            if self.fusion == "majority_vote":
                uniq = [int(v) for v in np.unique(stack) if v >= 0]
                best = np.full((h, w), -1, dtype=np.int64)
                best_score = np.full((h, w), -1, dtype=np.int64)
                for v in uniq:
                    is_v = stack == v
                    # count votes; break ties in favour of the value the earliest-listed source gave
                    score = is_v.sum(0).astype(np.int64) * (n + 1) + (n - np.argmax(is_v, axis=0))
                    score[~is_v.any(0)] = -1
                    better = score > best_score
                    best[better] = v
                    best_score[better] = score[better]
                fused = best
            elif self.fusion == "union":
                fused = (np.where(voted, stack, 0).max(0) > 0).astype(np.int64)
            elif self.fusion == "intersection":
                fused = ((np.where(voted, stack, 0) > 0).all(0)).astype(np.int64)
            else:  # priority: first source that says non-zero wins, else 0
                fused = np.zeros((h, w), dtype=np.int64)
                for layer in layers:
                    take = (fused == 0) & (layer > 0)
                    fused[take] = layer[take]
            fused = np.where(any_vote, fused, NODATA)
            if fused.max() > 255:
                raise ValueError("fused values exceed uint8")

            Path(output).parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(output, "w", driver="GTiff", height=h, width=w, count=1, dtype="uint8",
                               crs=crs, transform=tf, nodata=NODATA, compress="lzw") as dst:
                dst.write(fused.astype(np.uint8)[np.newaxis])
                dst.update_tags(fusion=self.fusion, sources=",".join(Path(p).name for p in paths))
            return output
        except Exception as exc:
            logger.error("Fusion failed: %s", exc)
            return None
