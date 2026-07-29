"""3D Point Cloud processing for geospatial data (G3)."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class PointCloudProcessor:
    """Process 3D point clouds from LiDAR, photogrammetry, and stereo satellite (G3).

    Supports:
        - LAS/LAZ file reading and writing
        - Ground filtering (per-cell classification-based or min-elevation)
        - Canopy height model (CHM) generation, correctly georeferenced
          using the source file's actual CRS
        - Per-point semantic segmentation via classify_points() — real
          PointNet++/RandLA-Net inference (bring your own trained
          checkpoint; neither ships pretrained weights)
        - Building footprint extraction from classified LiDAR via
          extract_buildings()

    Example::

        proc = PointCloudProcessor()
        chm = proc.canopy_height_model("./lidar/forest.las", "./output/chm.tif", resolution=0.5)
        labeled = proc.classify_points("./lidar/urban.las", "./output/classified.laz",
                                        checkpoint_path="./my_trained_model.pt")
        buildings = proc.extract_buildings("./output/classified.laz", "./output/buildings.geojson")
    """

    def __init__(self) -> None:
        pass

    def read(self, path: str) -> dict[str, Any]:
        """Read a LAS/LAZ point cloud file."""
        try:
            import laspy
            import numpy as np
            las = laspy.read(path)
            return {
                "n_points": len(las.x),
                "x": np.array(las.x), "y": np.array(las.y), "z": np.array(las.z),
                "intensity": np.array(las.intensity) if hasattr(las, "intensity") else None,
                "classification": np.array(las.classification),
                "crs": str(las.header.parse_crs()) if hasattr(las.header, "parse_crs") else "unknown",
                "bounds": (float(las.x.min()), float(las.y.min()), float(las.x.max()), float(las.y.max())),
            }
        except ImportError:
            return {"error": "pip install laspy"}
        except FileNotFoundError:
            return {"error": f"File not found: {path}"}
        except Exception as exc:
            return {"error": str(exc)}

    def canopy_height_model(
        self,
        las_path: str,
        output_path: str,
        resolution: float = 1.0,
        filter_ground: bool = True,
    ) -> dict[str, Any]:
        """Generate a Canopy Height Model (CHM = DSM - DTM) from LiDAR.

        Args:
            filter_ground: if True (default), the ground surface (DTM) is
                built only from points classified as ground (LAS class 2).
                If False, the DTM is instead built from the lowest point
                in each grid cell across ALL points — useful for files
                that lack reliable ground classification, at the cost of
                a noisier terrain estimate.
        """
        try:
            import laspy
            import numpy as np
            import rasterio
            from rasterio.transform import from_bounds
        except ImportError:
            return {"error": "pip install laspy rasterio"}

        pc = self.read(las_path)
        if "error" in pc:
            return pc

        x, y, z = pc["x"], pc["y"], pc["z"]
        cls = pc["classification"]

        if filter_ground:
            ground_mask = cls == 2  # LAS standard: class 2 = ground
            if ground_mask.sum() == 0:
                logger.warning(
                    "%s: no LAS class-2 (ground) points found despite "
                    "filter_ground=True — DTM will be empty/zero. Try "
                    "filter_ground=False to use per-cell minimum elevation "
                    "instead, or verify this file has ground classification.",
                    las_path,
                )
        else:
            ground_mask = np.ones_like(cls, dtype=bool)  # use all points for DTM

        xmin, ymin, xmax, ymax = pc["bounds"]
        W = max(1, int((xmax - xmin) / resolution))
        H = max(1, int((ymax - ymin) / resolution))
        transform = from_bounds(xmin, ymin, xmax, ymax, W, H)

        dtm = np.zeros((H, W), dtype=np.float32)
        dsm = np.zeros((H, W), dtype=np.float32)
        dtm_count = np.zeros((H, W), dtype=np.int32)
        dsm_count = np.zeros((H, W), dtype=np.int32)

        def _coords_to_px(xs, ys):
            cols = np.clip(((xs - xmin) / resolution).astype(int), 0, W-1)
            rows = np.clip(((ymax - ys) / resolution).astype(int), 0, H-1)
            return rows, cols

        if ground_mask.sum() > 0:
            if filter_ground:
                # Average ground-classified points per cell.
                gr, gc = _coords_to_px(x[ground_mask], y[ground_mask])
                np.add.at(dtm, (gr, gc), z[ground_mask])
                np.add.at(dtm_count, (gr, gc), 1)
            else:
                # No reliable ground classification — approximate terrain
                # as the per-cell MINIMUM elevation across all points
                # (lowest return is usually closest to true ground).
                dtm[:] = np.inf
                gr, gc = _coords_to_px(x, y)
                np.minimum.at(dtm, (gr, gc), z)
                dtm_count[gr, gc] = 1
                dtm[dtm == np.inf] = 0

        all_r, all_c = _coords_to_px(x, y)
        np.maximum.at(dsm, (all_r, all_c), z)
        dsm_count[all_r, all_c] += 1

        if filter_ground:
            dtm_valid = dtm_count > 0
            dtm[dtm_valid] /= dtm_count[dtm_valid]
        else:
            dtm_valid = dtm_count > 0

        # Cells with DSM data (any point) but no valid DTM estimate (no
        # ground points landed in that cell) must NOT be left at DTM=0 —
        # subtracting a phantom zero from a real ~absolute elevation would
        # produce a wildly wrong "canopy height" (a real bug found via
        # testing on sparse synthetic data, not a hypothetical edge case).
        # Fill sparse DTM cells via nearest-valid-cell lookup, a standard
        # technique for gridding sparse ground points.
        if dtm_valid.any() and not dtm_valid.all():
            from scipy.ndimage import distance_transform_edt
            _, nearest_idx = distance_transform_edt(~dtm_valid, return_indices=True)
            dtm = dtm[tuple(nearest_idx)]
        elif not dtm_valid.any():
            logger.warning(
                "%s: no valid ground elevation estimate anywhere in the "
                "grid — CHM will be all zeros. Check the file's ground "
                "classification, or try filter_ground=False.", las_path,
            )

        chm = np.maximum(0, dsm - dtm)
        chm[dsm_count == 0] = 0  # cells with no points at all: no CHM value, not a false height

        # Use the LiDAR file's ACTUAL CRS — hardcoding EPSG:4326 here would
        # silently mislabel a projected (UTM-like) coordinate system as
        # WGS84 lat/lon degrees, placing the output thousands of km off on
        # any map despite the pixel grid itself being computed correctly.
        crs_str = pc.get("crs", "unknown")
        try:
            import pyproj
            out_crs = pyproj.CRS.from_user_input(crs_str) if crs_str != "unknown" else None
        except Exception:
            out_crs = None
        if out_crs is None:
            logger.warning(
                "%s: could not determine LiDAR CRS (got %r) — output GeoTIFF "
                "will have NO CRS set rather than a guessed/wrong one. Set "
                "it manually once you know the correct EPSG code.",
                las_path, crs_str,
            )

        import pathlib
        pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", driver="GTiff", height=H, width=W,
                            count=1, dtype="float32",
                            crs=out_crs, transform=transform,
                            compress="lzw") as dst:
            dst.write(chm[np.newaxis])
            dst.update_tags(source="LiDAR_CHM", resolution=str(resolution))

        return {
            "success": True, "output_path": output_path,
            "n_points": len(x), "resolution": resolution,
            "crs": str(out_crs) if out_crs else "unset (unknown source CRS)",
            "height_stats": {
                "max_m": round(float(chm.max()), 1),
                "mean_m": round(float(chm[chm > 0].mean()), 1) if (chm > 0).any() else 0,
                "coverage": round(float((chm > 0).mean()), 3),
            },
        }

    def classify_points(
        self,
        las_path: str,
        output_path: str,
        num_classes: int = 3,
        class_names: list[str] | None = None,
        model: str = "randlanet",
        checkpoint_path: str | None = None,
        device: str | None = None,
    ) -> dict[str, Any]:
        """Per-point semantic segmentation (ground / vegetation / building,
        or your own class scheme) using a real PointNet++ or RandLA-Net
        model.

        Note: neither architecture ships pretrained weights (there is no
        standard pretrained checkpoint for airborne LiDAR semantic
        segmentation the way there is for, say, ImageNet classification) —
        pass `checkpoint_path` to a model YOU trained via
        `pygeovision.models.registry.get_model('randlanet'/'pointnet2-ssg', ...)`.
        Without a checkpoint, this runs with randomly-initialized weights
        and the output is not meaningful — useful only for testing the
        pipeline end-to-end, not for real classification.
        """
        try:
            import numpy as np
            import torch
        except ImportError:
            return {"error": "pip install torch"}

        pc = self.read(las_path)
        if "error" in pc:
            return pc

        from pygeovision.models.registry import get_model
        registry_name = "randlanet" if model == "randlanet" else "pointnet2-ssg"
        net = get_model(registry_name, num_classes=num_classes, in_channels=4, pretrained=False)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        net = net.to(device).eval()

        if checkpoint_path:
            state = torch.load(checkpoint_path, map_location=device)
            net.load_state_dict(state)
        else:
            logger.warning(
                "classify_points: no checkpoint_path given — running with "
                "randomly-initialized weights. Output classes are NOT "
                "meaningful; train a model first via "
                "pygeovision.models.registry.get_model(%r, ...) and pass "
                "its checkpoint here.", registry_name,
            )

        x, y, z = pc["x"], pc["y"], pc["z"]
        intensity = pc.get("intensity")
        if intensity is None:
            intensity = np.zeros_like(x)
        # Centre coordinates (models expect local, not absolute, positions)
        cx, cy, cz = x.mean(), y.mean(), z.mean()
        xyz = np.stack([x - cx, y - cy, z - cz, intensity], axis=0).astype(np.float32)
        xyz_t = torch.from_numpy(xyz).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = net(xyz_t)
            pred = logits.argmax(dim=1)[0].cpu().numpy()

        import laspy
        las = laspy.read(las_path)
        las.classification = pred.astype(np.uint8)
        import pathlib
        pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        las.write(output_path)

        names = class_names or [f"class_{i}" for i in range(num_classes)]
        counts = {names[i]: int((pred == i).sum()) for i in range(num_classes)}
        return {
            "success": True, "output_path": output_path,
            "n_points": len(x), "model": registry_name,
            "used_pretrained_checkpoint": checkpoint_path is not None,
            "class_counts": counts,
        }

    def extract_buildings(
        self,
        las_path: str,
        output_path: str,
        building_class: int = 6,
        resolution: float = 0.5,
        min_area_m2: float = 10.0,
        gap_fill_m: float = 1.5,
    ) -> dict[str, Any]:
        """Extract building footprint polygons from LiDAR points already
        classified as buildings (LAS standard class 6, or the class your
        own `classify_points()` model assigns to buildings).

        Rasterizes building-classified points to a binary mask, applies
        morphological closing to bridge natural gaps between individual
        point-derived cells (point rasters are inherently scattered —
        without this step, a single real building typically fragments into
        dozens of disconnected single-cell polygons instead of one
        footprint), then vectorises into polygons.

        Args:
            gap_fill_m: maximum gap (in metres) to bridge between nearby
                building-classified cells. Increase for sparse point
                clouds (e.g. low-density airborne LiDAR), decrease if
                separate nearby buildings are incorrectly merging into one.
        """
        try:
            import numpy as np
            import rasterio
            from rasterio.transform import from_bounds
            from rasterio.features import shapes as rio_shapes
            from shapely.geometry import shape as shp_shape
        except ImportError:
            return {"error": "pip install rasterio shapely"}

        pc = self.read(las_path)
        if "error" in pc:
            return pc

        x, y, cls = pc["x"], pc["y"], pc["classification"]
        mask = cls == building_class
        if mask.sum() == 0:
            return {
                "success": False,
                "error": (
                    f"No points with classification={building_class} found. "
                    f"Run classify_points() first if this file lacks native "
                    f"building classification, or pass the correct class id."
                ),
            }

        xmin, ymin, xmax, ymax = pc["bounds"]
        W = max(1, int((xmax - xmin) / resolution))
        H = max(1, int((ymax - ymin) / resolution))
        transform = from_bounds(xmin, ymin, xmax, ymax, W, H)

        raster = np.zeros((H, W), dtype=np.uint8)
        cols = np.clip(((x[mask] - xmin) / resolution).astype(int), 0, W - 1)
        rows = np.clip(((ymax - y[mask]) / resolution).astype(int), 0, H - 1)
        raster[rows, cols] = 1

        # Point-derived rasters are naturally scattered/gapped — even real
        # building LiDAR returns rarely fill every single grid cell of the
        # true footprint. Without morphological closing, vectorising
        # directly produces dozens of disconnected single-cell fragments
        # instead of one coherent footprint (found via testing on
        # realistic sparse point data, not a hypothetical concern).
        from scipy.ndimage import binary_closing
        close_iterations = max(1, int(round(gap_fill_m / resolution)))
        raster = binary_closing(raster, iterations=close_iterations).astype(np.uint8)

        crs_str = pc.get("crs", "unknown")
        try:
            import pyproj
            out_crs = pyproj.CRS.from_user_input(crs_str) if crs_str != "unknown" else None
        except Exception:
            out_crs = None

        features = []
        for geom, val in rio_shapes(raster, mask=raster.astype(bool), transform=transform):
            if val != 1:
                continue
            poly = shp_shape(geom)
            if poly.area < min_area_m2:
                continue
            features.append({
                "type": "Feature",
                "geometry": geom,
                "properties": {"area_m2": round(poly.area, 1), "source": "LiDAR"},
            })

        import json
        import pathlib
        pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fc = {"type": "FeatureCollection", "features": features}
        if out_crs:
            fc["crs"] = {"type": "name", "properties": {"name": f"urn:ogc:def:crs:EPSG::{out_crs.to_epsg()}"}}
        with open(output_path, "w") as f:
            json.dump(fc, f)

        return {
            "success": True, "output_path": output_path,
            "n_buildings": len(features),
            "n_building_points": int(mask.sum()),
        }


__all__ = ["PointCloudProcessor"]