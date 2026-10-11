"""Terrain and drainage: the hydrology algorithms on hand-built terrain, and the two pipelines end to end."""
from __future__ import annotations

import math

import numpy as np
import pytest
import rasterio

from pygeovision.ai.pipelines import _hydrology as hy
from pygeovision.ai.pipelines.terrain import DrainageAnalysisPipeline, TerrainAnalysisPipeline
from pygeovision.data.reference import make_grid

BBOX = (0.45, 5.90, 0.4725, 5.9225)       # about 2.5 km square


class FakePC:
    def __init__(self, items):
        self.items = items

    def search(self, collection, bbox, datetime=None, **kw):
        return self.items if collection == "cop-dem-glo-30" else []

    def asset_urls(self, collection, items, asset):
        return [it["assets"][asset]["href"] for it in items]


def expanded(grid, pad=3):
    """The same grid with ``pad`` extra cells on every side (a real DEM tile extends beyond the requested box)."""
    from affine import Affine

    from pygeovision.data.reference import TargetGrid
    tr = grid.transform
    return TargetGrid(crs=grid.crs, transform=Affine(tr.a, 0.0, tr.c - pad * grid.res, 0.0, tr.e, tr.f + pad * grid.res),
                      width=grid.width + 2 * pad, height=grid.height + 2 * pad, res=grid.res)


def write_dem(path, z, grid):
    prof = grid.profile()
    prof.update(count=1, dtype="float32", nodata=None)
    with rasterio.open(path, "w", **prof) as d:
        d.write(z.astype("float32"), 1)
    return {"id": "dem", "properties": {}, "assets": {"data": {"href": str(path)}}}


# ---------------------------------------------------------------------------------------------------------------------
class TestHydrology:
    def test_plane_drains_down_the_columns(self):
        H, W = 30, 12
        z = np.repeat(np.arange(H - 1, -1, -1, dtype=float)[:, None], W, axis=1)       # high at the top
        filled = hy.priority_flood_epsilon(z)
        np.testing.assert_allclose(filled, z, atol=1e-9)                                # nothing to fill
        tgt = hy.d8_targets(filled, 10.0)
        acc = hy.flow_accumulation(filled, tgt)
        assert (acc[-1, :] == H).all()                                                  # every column collects its own H cells
        assert acc[-1, :].sum() == H * W                                                # the bottom row holds everything
        assert acc[0, 5] == 1

    def test_pit_is_filled_and_flow_continues_through_it(self):
        H, W = 30, 9
        z = np.repeat(np.arange(H - 1, -1, -1, dtype=float)[:, None], W, axis=1) * 1.0
        z[10:13, 3:6] -= 5.0                               # a 3x3 pit, deeper than the slope around it
        filled = hy.priority_flood_epsilon(z)
        assert (filled >= z - 1e-12).all()
        assert filled[11, 4] == pytest.approx(16.0, abs=1e-6)       # raised to the spill level: the lowest rim cell
        tgt = hy.d8_targets(filled, 10.0)
        acc = hy.flow_accumulation(filled, tgt)
        # every cell must reach the bottom row: no cell is a sink inside the grid
        sinks = [(i // W, i % W) for i in np.nonzero(tgt == -1)[0]]
        assert all(r in (0, H - 1) or c in (0, W - 1) for r, c in sinks)
        assert acc[-1, :].sum() == H * W

    def test_filled_surface_has_no_flats_so_no_cycles(self):
        rng = np.random.RandomState(3)
        z = rng.rand(25, 25) * 3
        z[8:14, 8:14] = 0.0                                 # a flat basin
        filled = hy.priority_flood_epsilon(z)
        tgt = hy.d8_targets(filled, 1.0)
        # follow every path: it must terminate within the number of cells
        for start in range(0, 625, 7):
            i, steps = start, 0
            while tgt[i] >= 0:
                i, steps = int(tgt[i]), steps + 1
                assert steps < 625
        assert hy.flow_accumulation(filled, tgt).max() <= 625

    def test_nan_cells_are_outlets_not_barriers(self):
        z = np.repeat(np.arange(19, -1, -1, dtype=float)[:, None], 8, axis=1)
        z[-3:, :] = np.nan
        filled = hy.priority_flood_epsilon(z)
        assert np.isnan(filled[-1, 0]) and np.isfinite(filled[0, 0])
        acc = hy.flow_accumulation(filled, hy.d8_targets(filled, 1.0))
        assert np.isnan(acc[-1, 0]) and acc[-4, 3] == 17

    def test_v_valley_hand_equals_distance_times_slope(self):
        H, W, c0 = 40, 21, 10
        rr, cc = np.mgrid[0:H, 0:W]
        z = 0.5 * np.abs(cc - c0) + 0.01 * (H - 1 - rr)
        filled = hy.priority_flood_epsilon(z)
        tgt = hy.d8_targets(filled, 100.0)
        acc = hy.flow_accumulation(filled, tgt)
        stream = (acc * 0.01) >= 0.5                       # 100 m cells = 0.01 km2; threshold 0.5 km2 = 50 cells
        assert stream[5:, c0].all() and not stream[5:, c0 + 1].any() and not stream[:, c0 - 3].any()
        hand = hy.height_above_drainage(filled, tgt, stream.ravel())
        for d in (1, 3, 7):
            assert hand[20, c0 + d] == pytest.approx(0.5 * d, abs=1e-6)
            assert hand[20, c0 - d] == pytest.approx(0.5 * d, abs=1e-6)
        assert hand[20, c0] == 0

    def test_hand_is_nan_where_water_never_reaches_a_stream(self):
        z = np.repeat(np.arange(9, -1, -1, dtype=float)[:, None], 6, axis=1)
        filled = hy.priority_flood_epsilon(z)
        tgt = hy.d8_targets(filled, 1.0)
        none = np.zeros(z.size, bool)
        assert np.isnan(hy.height_above_drainage(filled, tgt, none)).all()

    def test_strahler_order_rules(self):
        # A1,A2 -> A ; B1,B2 -> B ; A,B -> C ; C -> D ; side headwater S -> D
        names = ["A1", "A2", "B1", "B2", "A", "B", "C", "S", "D"]
        ix = {n: i for i, n in enumerate(names)}
        target = np.full(len(names), -1)
        for a, b in (("A1", "A"), ("A2", "A"), ("B1", "B"), ("B2", "B"), ("A", "C"), ("B", "C"), ("C", "D"), ("S", "D")):
            target[ix[a]] = ix[b]
        high_to_low = np.array([ix[n] for n in ["A1", "A2", "B1", "B2", "S", "A", "B", "C", "D"]])
        o = hy.strahler_order(target, np.ones(len(names), bool), high_to_low)
        got = {n: int(o[ix[n]]) for n in names}
        assert got == {"A1": 1, "A2": 1, "B1": 1, "B2": 1, "S": 1, "A": 2, "B": 2, "C": 3, "D": 3}   # a 1 joining a 3 stays 3

    def test_stream_length_counts_diagonals(self):
        # three cells in a row (0,1,2) then a diagonal step to (1, 3) on a 2 x 4 grid
        w = 4
        target = np.full(8, -1)
        target[0], target[1], target[2] = 1, 2, 7           # (0,0)->(0,1)->(0,2)->(1,3)
        stream = np.zeros(8, bool)
        stream[[0, 1, 2, 7]] = True
        assert hy.stream_length_m(target, stream, w, 30.0) == pytest.approx(30 * (1 + 1 + math.sqrt(2)))

    def test_size_guard(self):
        with pytest.raises(ValueError, match="smaller bounding box"):
            hy.check_size((5000, 5000), 6_000_000)


# ---------------------------------------------------------------------------------------------------------------------
class TestTerrainAnalysis:
    def _run(self, tmp_path, z_fn, **kw):
        grid = make_grid(BBOX, 30)
        big = expanded(grid)
        rr, cc = np.mgrid[0:big.height, 0:big.width]
        item = write_dem(tmp_path / "dem.tif", z_fn(rr, cc), big)
        p = TerrainAnalysisPipeline(None)
        p.pc = FakePC([item])
        return p.run(BBOX, tmp_path / "out", **kw), grid

    def test_plane_slope_aspect_hillshade_and_tri(self, tmp_path):
        r, grid = self._run(tmp_path, lambda rr, cc: 3.0 * cc)           # +3 m per 30 m pixel eastward
        assert r.success, r.error
        sl = math.degrees(math.atan(3.0 / 30.0))
        with rasterio.open(tmp_path / "out" / "slope_deg.tif") as s:
            slope = s.read(1)
        with rasterio.open(tmp_path / "out" / "aspect_deg.tif") as s:
            aspect = s.read(1)
        with rasterio.open(tmp_path / "out" / "hillshade.tif") as s:
            hill = s.read(1)
        with rasterio.open(tmp_path / "out" / "tri_m.tif") as s:
            tri = s.read(1)
        c = (grid.height // 2, grid.width // 2)
        assert slope[c] == pytest.approx(sl, abs=1e-3)
        assert aspect[c] == pytest.approx(270.0, abs=1e-3)               # downhill is west
        zen = math.radians(45)
        expect = math.cos(zen) * math.cos(math.radians(sl)) + math.sin(zen) * math.sin(math.radians(sl)) * math.cos(math.radians(315 - 270))
        assert hill[c] == pytest.approx(expect, abs=1e-4)
        assert tri[c] == pytest.approx(math.sqrt(54.0), abs=1e-3)
        assert r.stats["mean_slope_deg"] == pytest.approx(sl, abs=0.01)
        assert r.stats["pct_area_slope_gt_5deg"] == pytest.approx(1.0) and r.stats["pct_area_slope_gt_15deg"] == 0.0

    def test_flat_ground_has_zero_slope_and_no_aspect(self, tmp_path):
        r, grid = self._run(tmp_path, lambda rr, cc: np.full(rr.shape, 100.0))
        assert r.success
        with rasterio.open(tmp_path / "out" / "aspect_deg.tif") as s:
            a = s.read(1)
        assert np.isnan(a[5, 5]) and r.stats["max_slope_deg"] == 0.0 and r.stats["relief_m"] == 0.0

    def test_cells_with_no_dem_stay_nodata(self, tmp_path):
        grid = make_grid(BBOX, 30)
        z = np.full((grid.height, grid.width), 50.0, "float32")
        z[:20, :] = np.nan
        prof = grid.profile(); prof.update(count=1, dtype="float32", nodata=float("nan"))
        with rasterio.open(tmp_path / "dem.tif", "w", **prof) as d:
            d.write(z, 1)
        p = TerrainAnalysisPipeline(None)
        p.pc = FakePC([{"id": "d", "properties": {}, "assets": {"data": {"href": str(tmp_path / "dem.tif")}}}])
        r = p.run(BBOX, tmp_path / "out")
        assert r.success and r.stats["valid_fraction"] == pytest.approx(1 - 20 / grid.height, abs=1e-6)
        with rasterio.open(tmp_path / "out" / "slope_deg.tif") as s:
            sl = s.read(1)
        assert np.isnan(sl[:20]).all() and np.isfinite(sl[25:]).all()

    def test_errors_are_clear(self, tmp_path):
        p = TerrainAnalysisPipeline(None)
        p.pc = FakePC([])
        assert "no Copernicus DEM" in p.run(BBOX, tmp_path).error
        assert "dem_source" in p.run(BBOX, tmp_path, dem_source="srtm").error
        assert "positive" in p.run(BBOX, tmp_path, resolution_m=0).error


class TestDrainageAnalysis:
    def _valley(self, tmp_path, **kw):
        tmp_path.mkdir(parents=True, exist_ok=True)
        grid = make_grid(BBOX, 30)
        H, W = grid.height, grid.width
        rr, cc = np.mgrid[0:H, 0:W]
        c0 = W // 2
        z = 6.0 * np.abs(cc - c0) / 1.0 + 0.1 * (H - 1 - rr)         # 6 m per pixel sideways: steep valley walls
        item = write_dem(tmp_path / "dem.tif", z, grid)
        p = DrainageAnalysisPipeline(None)
        p.pc = FakePC([item])
        kw.setdefault("stream_threshold_km2", 0.5)
        return p.run(BBOX, tmp_path / "out", **kw), grid, c0

    def test_stream_hand_and_statistics(self, tmp_path):
        r, grid, c0 = self._valley(tmp_path)
        assert r.success, r.error
        H, W = grid.height, grid.width
        with rasterio.open(tmp_path / "out" / "stream_order.tif") as s:
            so = s.read(1)
        with rasterio.open(tmp_path / "out" / "hand_m.tif") as s:
            hand = s.read(1)
        with rasterio.open(tmp_path / "out" / "flow_accumulation_km2.tif") as s:
            acc = s.read(1)
        col = so[:, c0]
        assert (col[H // 2:] == 1).all()                              # single stream along the valley axis, order 1
        assert so[:, :c0 - 1].max() == 0 and so[:, c0 + 2:].max() == 0
        assert hand[H - 5, c0 + 4] == pytest.approx(6.0 * 4, abs=1e-3)
        assert acc[H - 1, c0] == pytest.approx(H * W * 0.0009, rel=1e-6)    # everything drains out at the bottom of the valley
        s = r.stats
        assert s["max_strahler_order"] == 1 and s["stream_threshold_km2"] == 0.5
        n_stream = int((so == 1).sum())
        assert s["stream_cells"] == n_stream
        assert s["stream_length_km"] == pytest.approx((n_stream - 1) * 0.03, rel=0.05)   # the last cell leaves the grid
        assert s["drainage_density_km_per_km2"] == pytest.approx(s["stream_length_km"] / s["area_km2"])
        assert s["pct_area_hand_le_5m"] < s["pct_area_hand_le_10m"] <= s["pct_area_hand_le_15m"] <= 1.0
        # a cell within 5 m of the drainage is within <1 pixel of the axis (walls rise 6 m per pixel)
        assert s["pct_area_hand_le_5m"] == pytest.approx(n_stream / (H * W) + 0.0, abs=0.05)

    def test_stream_lines_are_exported_with_their_order(self, tmp_path):
        r, grid, c0 = self._valley(tmp_path)
        import geopandas as gpd
        g = gpd.read_file(tmp_path / "out" / "stream_network.gpkg")
        assert list(g.strahler) == [1] and g.length_km.iloc[0] > 0.5
        assert r.metadata["vector_files"][0]["name"] == "streams"

    def test_higher_threshold_gives_a_shorter_network(self, tmp_path):
        r1, grid, _ = self._valley(tmp_path / "a", stream_threshold_km2=0.3)
        r2, _, _ = self._valley(tmp_path / "b", stream_threshold_km2=3.0)
        assert r1.stats["stream_length_km"] > r2.stats["stream_length_km"] > 0

    def test_depression_is_reported(self, tmp_path):
        grid = make_grid(BBOX, 30)
        H, W = grid.height, grid.width
        rr, cc = np.mgrid[0:H, 0:W]
        z = 0.1 * (H - 1 - rr) + 0.0 * cc
        z[30:34, 30:34] -= 20.0
        item = write_dem(tmp_path / "dem.tif", z, grid)
        p = DrainageAnalysisPipeline(None)
        p.pc = FakePC([item])
        r = p.run(BBOX, tmp_path / "out")
        assert r.success and r.stats["cells_filled"] >= 16 and r.stats["max_fill_depth_m"] > 15

    def test_too_large_an_area_is_refused_before_any_download(self, tmp_path):
        p = DrainageAnalysisPipeline(None)
        p.pc = FakePC([])
        r = p.run(BBOX, tmp_path / "o", max_cells=1000)
        assert not r.success and "smaller bounding box" in r.error

    def test_threshold_must_be_positive(self, tmp_path):
        p = DrainageAnalysisPipeline(None)
        p.pc = FakePC([])
        assert "positive" in p.run(BBOX, tmp_path, stream_threshold_km2=0).error
