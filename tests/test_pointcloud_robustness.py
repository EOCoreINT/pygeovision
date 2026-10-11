"""Point-cloud processor: grid geometry, failure modes, blockwise classification (real LAS files)."""
import json

import numpy as np
import pytest

laspy = pytest.importorskip("laspy")
rasterio = pytest.importorskip("rasterio")
pyproj = pytest.importorskip("pyproj")
pytest.importorskip("scipy")

from pygeovision.advanced.pointcloud import PointCloudProcessor  # noqa: E402


def write_las(path, x, y, z, cls=None, intensity=None, epsg=32630, point_format=3, version="1.2"):
    n = len(x)
    header = laspy.LasHeader(point_format=point_format, version=version)
    header.offsets = [float(np.min(x)), float(np.min(y)), float(np.min(z))]
    header.scales = [0.001, 0.001, 0.001]
    las = laspy.LasData(header)
    las.x, las.y, las.z = x, y, z
    las.classification = np.full(n, 2, np.uint8) if cls is None else np.asarray(cls, np.uint8)
    las.intensity = np.zeros(n, np.uint16) if intensity is None else np.asarray(intensity, np.uint16)
    if epsg:
        las.header.add_crs(pyproj.CRS.from_epsg(epsg))
    las.write(str(path))
    return str(path)


def flat_ground_with_tree(tmp_path, extent=10.7, ground_z=100.0):
    """Dense ground grid plus one 12 m tree top near (extent*0.8, extent*0.5)."""
    gx, gy = np.meshgrid(np.arange(0, extent, 0.25), np.arange(0, extent, 0.25))
    x, y = gx.ravel() + 500000, gy.ravel() + 4500000
    z = np.full(len(x), ground_z)
    cls = np.full(len(x), 2)
    tx, ty = 500000 + extent * 0.8, 4500000 + extent * 0.5
    x = np.append(x, tx); y = np.append(y, ty); z = np.append(z, ground_z + 12); cls = np.append(cls, 5)
    return write_las(tmp_path / "t.las", x, y, z, cls), (tx, ty)


class TestRead:
    def test_errors_are_exceptions_not_dicts(self, tmp_path):
        p = PointCloudProcessor()
        with pytest.raises(FileNotFoundError):
            p.read(str(tmp_path / "missing.las"))
        bad = tmp_path / "bad.las"
        bad.write_bytes(b"not a las file")
        with pytest.raises(ValueError, match="could not read"):
            p.read(str(bad))

    def test_no_crs_is_unknown_and_empty_file_refused(self, tmp_path):
        f = write_las(tmp_path / "a.las", np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([1.0, 2.0]), epsg=None)
        assert PointCloudProcessor().read(f)["crs"] == "unknown"
        header = laspy.LasHeader(point_format=3, version="1.2")
        laspy.LasData(header).write(str(tmp_path / "e.las"))
        with pytest.raises(ValueError, match="no points"):
            PointCloudProcessor().read(str(tmp_path / "e.las"))


class TestChm:
    def test_cell_size_is_exactly_the_resolution_and_height_lands_on_the_right_cell(self, tmp_path):
        """Extent 10.7 m at 1 m used to give 10 columns of 1.07 m while points were binned at 1 m."""
        f, (tx, ty) = flat_ground_with_tree(tmp_path)
        out = tmp_path / "chm.tif"
        r = PointCloudProcessor().canopy_height_model(f, str(out), resolution=1.0)
        with rasterio.open(out) as src:
            assert src.res == (1.0, 1.0) and src.width == 11 and src.height == 11
            chm = src.read(1)
            row, col = src.index(tx, ty)
            assert chm[row, col] == pytest.approx(12.0, abs=0.01)
            assert src.nodata == -9999
        assert r["grid"] == {"width": 11, "height": 11}
        assert (chm > 5).sum() == 1

    def test_no_ground_raises_instead_of_writing_absolute_elevations(self, tmp_path):
        x = np.random.default_rng(0).uniform(0, 10, 200)
        f = write_las(tmp_path / "ng.las", x, x[::-1].copy(), np.full(200, 250.0), cls=np.full(200, 1))
        with pytest.raises(ValueError, match="no ground points"):
            PointCloudProcessor().canopy_height_model(f, str(tmp_path / "o.tif"))
        r = PointCloudProcessor().canopy_height_model(f, str(tmp_path / "o2.tif"), filter_ground=False)
        assert r["height_stats"]["max_m"] == 0.0

    def test_below_sea_level_terrain(self, tmp_path):
        """DSM started at 0 so every cell below sea level had its top clipped to 0."""
        f, (tx, ty) = flat_ground_with_tree(tmp_path, ground_z=-30.0)
        out = tmp_path / "chm.tif"
        PointCloudProcessor().canopy_height_model(f, str(out), resolution=1.0)
        with rasterio.open(out) as src:
            r, c = src.index(tx, ty)
            assert src.read(1)[r, c] == pytest.approx(12.0, abs=0.01)

    def test_cells_without_points_are_nodata_and_noise_class_ignored(self, tmp_path):
        gx, gy = np.meshgrid(np.arange(0, 4, 0.25), np.arange(0, 4, 0.25))
        x = np.concatenate([gx.ravel(), [9.5]]) + 500000        # an isolated point far east makes empty cells
        y = np.concatenate([gy.ravel(), [1.0]]) + 4500000
        z = np.concatenate([np.full(gx.size, 10.0), [10.0]])
        cls = np.concatenate([np.full(gx.size, 2), [2]])
        x = np.append(x, 500001.0); y = np.append(y, 4500001.0); z = np.append(z, 500.0); cls = np.append(cls, 18)
        f = write_las(tmp_path / "n.las", x, y, z, cls)
        out = tmp_path / "chm.tif"
        r = PointCloudProcessor().canopy_height_model(f, str(out), resolution=1.0)
        assert r["empty_cells_fraction"] > 0.3
        with rasterio.open(out) as src:
            a = src.read(1)
        assert (a == -9999).any() and a.max() < 1.0            # the 500 m spike (class 18) did not make a tree
        r2 = PointCloudProcessor().canopy_height_model(f, str(tmp_path / "c2.tif"), resolution=1.0, exclude_noise=False)
        assert r2["height_stats"]["max_m"] > 400

    def test_bad_resolution(self, tmp_path):
        f, _ = flat_ground_with_tree(tmp_path)
        with pytest.raises(ValueError, match="resolution"):
            PointCloudProcessor().canopy_height_model(f, str(tmp_path / "o.tif"), resolution=0)


def building_las(tmp_path, *, epsg=32630, offset=(500000.0, 4500000.0), at_edge=False):
    """A dense 12 x 8 m roof plus a ground strip; at_edge puts the roof on the tile's west edge."""
    rng = np.random.default_rng(1)
    ox, oy = offset
    rx0 = 0.0 if at_edge else 10.0
    roofx = rng.uniform(rx0, rx0 + 12, 1500) + ox
    roofy = rng.uniform(10, 18, 1500) + oy
    gx = rng.uniform(0, 30, 800) + ox
    gy = rng.uniform(0, 30, 800) + oy
    x = np.concatenate([roofx, gx]); y = np.concatenate([roofy, gy])
    z = np.concatenate([np.full(1500, 108.0), np.full(800, 100.0)])
    cls = np.concatenate([np.full(1500, 6), np.full(800, 2)])
    return write_las(tmp_path / "b.las", x, y, z, cls, epsg=epsg)


class TestBuildings:
    def test_one_footprint_with_about_the_right_area(self, tmp_path):
        out = tmp_path / "b.geojson"
        r = PointCloudProcessor().extract_buildings(building_las(tmp_path), str(out), resolution=0.5, gap_fill_m=1.5)
        assert r["success"] and r["n_buildings"] == 1
        assert 0.85 * 96 <= r["total_area_m2"] <= 1.25 * 96
        gj = json.loads(out.read_text())
        assert gj["crs"]["properties"]["name"].endswith("32630")

    def test_building_on_the_tile_edge_keeps_its_edge(self, tmp_path):
        r = PointCloudProcessor().extract_buildings(building_las(tmp_path, at_edge=True), str(tmp_path / "e.geojson"),
                                                    resolution=0.5, gap_fill_m=1.5)
        assert r["n_buildings"] == 1 and r["total_area_m2"] >= 0.85 * 96

    def test_geographic_crs_refused_and_unknown_crs_has_no_crs_member(self, tmp_path):
        las = building_las(tmp_path, epsg=4326, offset=(10.0, 50.0))
        with pytest.raises(ValueError, match="geographic"):
            PointCloudProcessor().extract_buildings(las, str(tmp_path / "g.geojson"))
        las2 = building_las(tmp_path, epsg=None)
        out = tmp_path / "u.geojson"
        PointCloudProcessor().extract_buildings(las2, str(out))
        assert "crs" not in json.loads(out.read_text())

    def test_simplify_reduces_vertices(self, tmp_path):
        las = building_las(tmp_path)
        a, b = tmp_path / "a.geojson", tmp_path / "b.geojson"
        PointCloudProcessor().extract_buildings(las, str(a))
        PointCloudProcessor().extract_buildings(las, str(b), simplify_m=0.6)
        n = lambda p: len(json.loads(p.read_text())["features"][0]["geometry"]["coordinates"][0])  # noqa: E731
        assert n(b) < n(a)


torch = pytest.importorskip("torch")


class TestClassify:
    def cloud(self, tmp_path, n=6000, intensity=True):
        rng = np.random.default_rng(3)
        x = rng.uniform(0, 120, n) + 500000
        y = rng.uniform(0, 120, n) + 4500000
        z = rng.uniform(0, 5, n)
        return write_las(tmp_path / "c.las", x, y, z, intensity=rng.integers(0, 60000, n) if intensity else None)

    def test_every_point_predicted_once_in_blocks_and_flagged_meaningless(self, tmp_path):
        out = tmp_path / "o.las"
        r = PointCloudProcessor().classify_points(self.cloud(tmp_path), str(out), num_classes=3, model="pointnet2",
                                                  block_size=60.0, max_points=1500)
        assert r["n_chunks"] >= 4 and sum(r["class_counts"].values()) == 6000
        assert r["meaningful"] is False and "random weights" in r["warning"]
        assert len(laspy.read(str(out)).x) == 6000

    def test_same_seed_same_labels(self, tmp_path):
        las = self.cloud(tmp_path)
        outs = []
        for i in range(2):
            o = tmp_path / f"o{i}.las"
            PointCloudProcessor().classify_points(las, str(o), num_classes=3, model="randlanet", max_points=2000, seed=5)
            outs.append(np.asarray(laspy.read(str(o)).classification))
        assert (outs[0] == outs[1]).all()

    def test_checkpoint_roundtrip_and_mismatch(self, tmp_path):
        from pygeovision.models.registry import get_model
        net = get_model("randlanet", num_classes=3, in_channels=4, pretrained=False)
        ck = tmp_path / "ck.pt"
        torch.save({"state_dict": net.state_dict()}, ck)
        las = self.cloud(tmp_path, n=2000)
        r = PointCloudProcessor().classify_points(las, str(tmp_path / "o.las"), num_classes=3, checkpoint_path=str(ck))
        assert r["meaningful"] is True and "warning" not in r
        with pytest.raises(ValueError, match="does not fit"):
            PointCloudProcessor().classify_points(las, str(tmp_path / "o2.las"), num_classes=5, checkpoint_path=str(ck))

    def test_class_codes_written(self, tmp_path):
        las = self.cloud(tmp_path, n=2000)
        out = tmp_path / "o.las"
        PointCloudProcessor().classify_points(las, str(out), num_classes=2, class_codes=[2, 6], max_points=2000)
        assert set(np.unique(laspy.read(str(out)).classification)) <= {2, 6}
        with pytest.raises(ValueError, match="class_codes"):
            PointCloudProcessor().classify_points(las, str(out), num_classes=2, class_codes=[2, 99])
        with pytest.raises(ValueError, match="class_codes has 1"):
            PointCloudProcessor().classify_points(las, str(out), num_classes=2, class_codes=[2])

    def test_argument_validation(self, tmp_path):
        las = self.cloud(tmp_path, n=500)
        with pytest.raises(ValueError, match="model must be"):
            PointCloudProcessor().classify_points(las, str(tmp_path / "o.las"), model="kpconv")
        with pytest.raises(ValueError, match="class_names has"):
            PointCloudProcessor().classify_points(las, str(tmp_path / "o.las"), num_classes=3, class_names=["a"])

    def test_file_without_intensity_dimension_and_tiny_cloud(self, tmp_path):
        x = np.random.default_rng(0).uniform(0, 5, 40) + 500000
        f = write_las(tmp_path / "s.las", x, x.copy() + 4500000 - 500000, np.zeros(40), point_format=0, version="1.2")
        r = PointCloudProcessor().classify_points(f, str(tmp_path / "o.las"), num_classes=3, model="pointnet2")
        assert sum(r["class_counts"].values()) == 40
