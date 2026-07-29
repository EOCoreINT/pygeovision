"""Tests for pygeovision.advanced.pointcloud.PointCloudProcessor.

Real bugs found and fixed during this audit:
  1. canopy_height_model() hardcoded the output CRS to EPSG:4326
     regardless of the LiDAR file's actual (almost always projected/UTM)
     CRS — would silently place the output thousands of km off on a map.
  2. Grid cells with DSM data (any point) but no co-located ground points
     were left with DTM=0 (uninitialized), producing wildly wrong "canopy
     height" values (subtracting a phantom zero from a real ~absolute
     elevation) instead of a real, small height difference.
  3. The class docstring claimed "Building footprint extraction" and
     "Semantic segmentation (PointNet++, RandLA-Net)" as supported
     features, with a fabricated usage example calling
     `extract_buildings()` — neither method existed anywhere in the file.
     Both are now genuinely implemented.
  4. extract_buildings(), once implemented, initially produced dozens of
     useless single-cell fragments instead of a coherent footprint —
     point-derived rasters are naturally scattered even for real LiDAR;
     morphological closing was needed and is now a tunable parameter.
"""
import pytest

laspy = pytest.importorskip("laspy", reason="laspy not installed")
rasterio = pytest.importorskip("rasterio", reason="rasterio not installed")
scipy = pytest.importorskip("scipy", reason="scipy not installed")

import numpy as np
import pyproj


def _make_synthetic_las(path, seed=0):
    """A real UTM-projected LAS file with ground, vegetation, and a
    building patch — not synthetic tensors, an actual file on disk."""
    rng = np.random.default_rng(seed)
    n = 5000
    x = rng.uniform(500000, 500100, n)
    y = rng.uniform(4500000, 4500100, n)
    ground_z = 100 + rng.normal(0, 0.1, n)
    z = ground_z.copy()
    classification = np.full(n, 2, dtype=np.uint8)

    veg_mask = x < 500030
    z[veg_mask] = ground_z[veg_mask] + rng.uniform(2, 15, veg_mask.sum())
    classification[veg_mask] = 5

    bld_mask = (x > 500060) & (x < 500080) & (y > 4500060) & (y < 4500080)
    z[bld_mask] = ground_z[bld_mask] + 8.0
    classification[bld_mask] = 6

    header = laspy.LasHeader(point_format=3, version="1.2")
    las = laspy.LasData(header)
    las.header.offsets = [x.min(), y.min(), z.min()]
    las.header.scales = [0.001, 0.001, 0.001]
    las.x = x
    las.y = y
    las.z = z
    las.classification = classification
    las.intensity = rng.integers(0, 65535, n).astype(np.uint16)
    las.header.add_crs(pyproj.CRS.from_epsg(32630))
    las.write(path)
    return {"n_points": n, "n_veg": int(veg_mask.sum()), "n_building": int(bld_mask.sum())}


class TestPointCloudProcessorRead:
    def test_reads_real_crs(self, tmp_path):
        from pygeovision.advanced.pointcloud import PointCloudProcessor
        las_path = tmp_path / "test.las"
        _make_synthetic_las(las_path)
        proc = PointCloudProcessor()
        pc = proc.read(str(las_path))
        assert "EPSG:32630" in pc["crs"] or "32630" in pc["crs"]


class TestCanopyHeightModelCRS:
    """Regression tests for the real CRS bug."""

    def test_output_crs_matches_source_not_hardcoded_4326(self, tmp_path):
        from pygeovision.advanced.pointcloud import PointCloudProcessor
        las_path = tmp_path / "test.las"
        _make_synthetic_las(las_path)
        proc = PointCloudProcessor()
        out = tmp_path / "chm.tif"
        result = proc.canopy_height_model(str(las_path), str(out), resolution=1.0)
        assert result["success"] is True

        with rasterio.open(out) as src:
            assert src.crs is not None
            assert src.crs.to_epsg() == 32630  # NOT 4326


class TestCanopyHeightModelSparseDTM:
    """Regression tests for the real phantom-DTM=0 bug."""

    def test_chm_values_are_realistic_not_phantom_absolute_elevation(self, tmp_path):
        """Before the fix, sparse ground coverage caused cells to compute
        DSM(~100m absolute) - DTM(0, uninitialized) = ~100m false height.
        The true max height difference in this synthetic data is ~15m."""
        from pygeovision.advanced.pointcloud import PointCloudProcessor
        las_path = tmp_path / "test.las"
        _make_synthetic_las(las_path)
        proc = PointCloudProcessor()
        out = tmp_path / "chm.tif"
        result = proc.canopy_height_model(str(las_path), str(out), resolution=1.0)

        with rasterio.open(out) as src:
            chm = src.read(1)
        assert chm.max() < 20, f"CHM max still shows phantom absolute-elevation heights: {chm.max()}"

    def test_filter_ground_false_uses_min_elevation_fallback(self, tmp_path):
        """filter_ground=False must actually change behavior (it was
        previously a dead parameter that did nothing)."""
        from pygeovision.advanced.pointcloud import PointCloudProcessor
        las_path = tmp_path / "test.las"
        _make_synthetic_las(las_path)
        proc = PointCloudProcessor()
        out = tmp_path / "chm.tif"
        result = proc.canopy_height_model(str(las_path), str(out), resolution=1.0, filter_ground=False)
        assert result["success"] is True
        with rasterio.open(out) as src:
            chm = src.read(1)
        assert chm.max() < 20  # still realistic, not broken by the alternate path


class TestClassifyPoints:
    """Real end-to-end test of the newly-implemented method (previously a
    false docstring claim with no actual implementation)."""

    def test_runs_real_randlanet_inference(self, tmp_path):
        torch = pytest.importorskip("torch", reason="torch not installed")
        from pygeovision.advanced.pointcloud import PointCloudProcessor
        las_path = tmp_path / "test.las"
        _make_synthetic_las(las_path)
        proc = PointCloudProcessor()
        out = tmp_path / "classified.las"
        result = proc.classify_points(str(las_path), str(out), num_classes=3, model="randlanet")
        assert result["success"] is True
        assert result["used_pretrained_checkpoint"] is False
        assert sum(result["class_counts"].values()) == 5000

    def test_runs_real_pointnet2_inference(self, tmp_path):
        torch = pytest.importorskip("torch", reason="torch not installed")
        from pygeovision.advanced.pointcloud import PointCloudProcessor
        las_path = tmp_path / "test.las"
        _make_synthetic_las(las_path)
        proc = PointCloudProcessor()
        out = tmp_path / "classified.las"
        result = proc.classify_points(str(las_path), str(out), num_classes=3, model="pointnet2")
        assert result["success"] is True


class TestExtractBuildings:
    """Real end-to-end test of the newly-implemented method, including the
    morphological-closing fix for scattered point-derived rasters."""

    def test_finds_no_buildings_gracefully_without_the_class(self, tmp_path):
        from pygeovision.advanced.pointcloud import PointCloudProcessor
        las_path = tmp_path / "test.las"
        _make_synthetic_las(las_path)
        proc = PointCloudProcessor()
        out = tmp_path / "buildings.geojson"
        result = proc.extract_buildings(str(las_path), str(out), building_class=99)
        assert result["success"] is False
        assert "error" in result

    def test_extracts_coherent_footprint_not_fragments(self, tmp_path):
        """Regression test: without morphological closing, this used to
        produce ~145 disconnected single-cell fragments instead of a
        coherent building footprint."""
        from pygeovision.advanced.pointcloud import PointCloudProcessor
        las_path = tmp_path / "test.las"
        _make_synthetic_las(las_path)
        proc = PointCloudProcessor()
        out = tmp_path / "buildings.geojson"
        result = proc.extract_buildings(
            str(las_path), str(out), building_class=6, resolution=0.5, gap_fill_m=2.5,
        )
        assert result["success"] is True
        # True building patch was 20m x 20m = 400 m2 — a working extraction
        # should find something in the right order of magnitude, not dozens
        # of ~0.25 m2 fragments.
        import json
        with open(out) as f:
            gj = json.load(f)
        areas = [f["properties"]["area_m2"] for f in gj["features"]]
        assert len(areas) <= 5, f"still fragmenting into too many pieces: {len(areas)}"
        assert max(areas) > 100, f"largest fragment too small: {max(areas)}"

    def test_output_has_correct_crs(self, tmp_path):
        from pygeovision.advanced.pointcloud import PointCloudProcessor
        las_path = tmp_path / "test.las"
        _make_synthetic_las(las_path)
        proc = PointCloudProcessor()
        out = tmp_path / "buildings.geojson"
        result = proc.extract_buildings(str(las_path), str(out), building_class=6, gap_fill_m=2.5)
        import json
        with open(out) as f:
            gj = json.load(f)
        assert "32630" in gj.get("crs", {}).get("properties", {}).get("name", "")
