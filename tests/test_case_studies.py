# ruff: noqa: E702, N802, C408, E731
"""The case-study pages (docs/case-studies/) against the PDF reports they were written from, with no network.

* ``docs/_data/case_study_numbers.yaml`` holds figures transcribed from the reports; here they are checked for internal
  consistency (class areas add up, shares are shares, a pixel has the area its grid says).
* ``docs/_examples/case_studies/checks.py`` re-derives report figures from other report figures.
* Every ``pygeovision channel`` command on a page is parsed by the real command line and every Python block is run against a
  recording stand-in for the client, and the two must ask for the same thing.
"""
import importlib.util
import re
import textwrap
import types
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
N = yaml.safe_load((DOCS / "_data" / "case_study_numbers.yaml").read_text(encoding="utf-8"))
spec = importlib.util.spec_from_file_location("cs_checks", DOCS / "_examples" / "case_studies" / "checks.py")
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


def close(a, b, tol):
    return abs(a - b) <= tol


# ── the transcribed numbers hold together ─────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("key", ["flood_accra", "flood_nepal"])
def test_flood_class_pixels_match_the_grid_and_the_areas(key):
    d = N[key]
    px = d["pixels"]
    assert sum(px.values()) == d["grid"]["width"] * d["grid"]["height"]
    ha = d["grid"]["pixel_m"] ** 2 / 1e4
    for name, n in px.items():
        assert close(n * ha, d["ha"][name], 0.06), name
    s = checks.flood_summary(px, d["grid"]["pixel_m"])
    assert close(s["flooded_ha"], d["flooded_ha"], 0.05)
    assert close(s["flooded_upper_ha"], d["flooded_upper_ha"], 0.05)
    assert close(sum(px.values()) * ha, d["valid_ha"], 0.06)


def test_accra_land_and_share():
    d = N["flood_accra"]
    s = checks.flood_summary(d["pixels"], 20)
    assert close(s["land_ha"], d["land_ha"], 0.1)
    assert round(s["flooded_share_of_land_pct"], 1) == d["flooded_share_of_land_pct"]
    assert close(d["valid_ha"] - d["ha"]["reference_water"], d["land_ha"], 0.1)


def test_nepal_is_mostly_optical_only_and_the_grid_is_over_the_terrain_limit():
    d = N["flood_nepal"]
    s = checks.flood_summary(d["pixels"], 20)
    assert 90.0 < s["optical_only_share_of_flooded_pct"] < 90.2
    assert s["confirmed_share_of_flooded_pct"] < 3
    assert d["grid"]["width"] * d["grid"]["height"] > d["cell_limit_million"] * 1e6
    assert close(d["grid"]["width"] * d["grid"]["height"] / 1e6, d["cells_million"], 0.05)


def test_nepal_vector_layer_still_holds_the_reference_water_class():
    d = N["flood_nepal"]
    with_ref = sum(d["ha"][k] for k in ("radar_optical", "radar_only", "optical_only", "disputed", "reference_water"))
    assert close(with_ref, d["vector"]["ha"], 3.0)              # the layer total is classes 1 to 5
    a = N["flood_accra"]
    without_ref = sum(a["ha"][k] for k in ("radar_optical", "radar_only", "optical_only", "disputed"))
    assert close(without_ref, a["vector"]["ha"], 0.3)           # the Accra report's layer is classes 1 to 4


def test_the_vector_layer_of_the_current_code_keeps_classes_one_to_four():
    from pygeovision.reporting.specs import SPECS
    layer = next(lay for lay in SPECS["multisensor_flood_mapping"].layers if lay.primary)
    assert tuple(layer.vector.keep) == (1, 2, 3, 4)


def test_the_nepal_box_fits_the_terrain_limit_at_30_m_and_not_at_20_m():
    bbox = tuple(N["flood_nepal"]["bbox"])
    cells20, fits20 = checks.hand_fits(bbox, 20)
    cells30, fits30 = checks.hand_fits(bbox, 30)
    assert (cells20, fits20) == (13477872, False)
    assert (cells30, fits30) == (5991936, True)
    assert checks.hand_fits(tuple(N["flood_accra"]["bbox"]), 20) == (3723544, True)


@pytest.mark.parametrize("key", ["flood_accra", "flood_nepal", "wildfire", "drainage", "water_quality", "biodiversity", "crop_health"])
def test_grids_are_what_make_grid_builds(key):
    from pygeovision.ai.pipelines.reference_products import make_grid
    d = N[key]
    if "pixel_m" not in d["grid"]:
        pytest.skip("a degree grid whose size depends on the scene")
    g = make_grid(tuple(d["bbox"]), float(d["grid"]["pixel_m"]), max_pixels=10 ** 13)
    assert (g.width, g.height, g.crs) == (d["grid"]["width"], d["grid"]["height"], d["grid"]["crs"])


def test_wildfire_classes_add_up_and_the_two_burned_areas_differ_by_the_pixel_area_used():
    d = N["wildfire"]
    px, ha = d["pixels"], d["ha"]
    assert sum(px.values()) == d["valid_pixels"]
    assert d["grid"]["width"] * d["grid"]["height"] == d["total_pixels"]
    assert close(sum(ha.values()), d["valid_ha"], 0.2)
    burned_px = px["low"] + px["moderate"] + px["high"]
    burned_table = ha["low"] + ha["moderate"] + ha["high"]
    mean_px_ha = d["raster_ha"] / d["total_pixels"]
    assert close(burned_px * mean_px_ha, d["burned_ha_stat"], 0.5)       # the headline figure: one pixel area for all
    assert close(burned_table - d["burned_ha_stat"], 20.8, 0.2)           # the table integrates row by row
    assert burned_table > d["burned_ha_stat"]
    # the burned pixels are, on average, larger than the average pixel: they sit south of the middle of the box
    assert burned_table / burned_px > mean_px_ha
    assert close(100 * d["burned_ha_stat"] / d["valid_ha"], 15.6, 0.05)
    assert close(d["burned_ha_vector"], burned_table, 1.0)


def test_rowwise_area_is_larger_in_the_south_of_a_box_and_matches_one_mean_pixel_for_the_whole_box():
    bbox = tuple(N["wildfire"]["bbox"])
    h, w = 400, 560
    whole = np.ones((h, w), bool)
    south = np.zeros((h, w), bool); south[h // 2:] = True
    north = np.zeros((h, w), bool); north[: h // 2] = True
    a_all = checks.rowwise_area_ha(whole, *bbox)
    a_s, a_n = checks.rowwise_area_ha(south, *bbox), checks.rowwise_area_ha(north, *bbox)
    assert close(a_s + a_n, a_all, 1e-6 * a_all)
    assert a_s > a_n                                            # a southern pixel covers more ground
    assert close(a_all, 332096.6, 60)                           # the WGS 84 area of the box, about 3,321 km²
    assert 0.0015 < (a_s - a_n) / a_all < 0.0021                # the southern half covers about 0.36% more ground
    assert checks.rowwise_area_ha(np.zeros((4, 4), bool), *bbox) == 0.0


def test_drainage_numbers():
    d = N["drainage"]
    px = d["pixels"]
    assert sum(px.values()) == d["total_pixels"] == d["grid"]["width"] * d["grid"]["height"]
    assert px["order1"] + px["order2"] + px["order3"] == d["stream_cells"]
    for k, a in (("order1", "order1"), ("order2", "order2"), ("order3", "order3")):
        assert close(px[k] * 0.09, d["ha"][a], 0.06)
    assert close(d["total_pixels"] * 900 / 1e6, d["grid_km2"], 0.01)
    assert round(checks.drainage_density(d["stream_km"], d["grid_km2"]), 2) == d["density"]
    assert round(checks.drainage_density(d["stream_km"], d["aoi_km2"]), 2) != d["density"]     # not the requested box
    assert d["hand_pct"]["le5"] < d["hand_pct"]["le10"] < d["hand_pct"]["le15"] < 100 - 60       # most ground is high above a stream


def test_water_quality_numbers_and_the_ndci_gap():
    d = N["water_quality"]
    assert sum(d["terciles_pixels"]) == d["ndci_pixels"]
    assert close(sum(d["terciles_ha"]), d["ndci_ha"], 0.2)
    missing = d["red_pixels"] - d["ndci_pixels"]
    assert missing == 61377 and round(100 * missing / d["red_pixels"], 1) == 48.0
    valid_px = d["total_pixels"] * d["valid_pct"] / 100
    assert close(100 * d["red_pixels"] / valid_px, d["water_pct"], 0.06)
    assert d["ndci_p5"] == -1.0 and d["ndci_p95"] == 1.0 and d["red_median"] == 0.0


def test_ndci_undefined_counts_water_pixels_without_an_indicator(tmp_path):
    import rasterio
    from rasterio.transform import from_origin
    ndci = np.full((4, 5), np.nan, "float32"); red = np.full((4, 5), np.nan, "float32")
    red[0, :] = [0.0, 0.0, 0.02, 0.03, 0.01]; ndci[0, :] = [np.nan, 1.0, 0.1, -0.2, np.nan]
    red[2, 1] = 0.05; ndci[2, 1] = 0.3
    p = tmp_path / "water_quality.tif"
    with rasterio.open(p, "w", driver="GTiff", width=5, height=4, count=2, dtype="float32", crs="EPSG:4326",
                       transform=from_origin(0, 4, 1, 1), nodata=np.nan) as dst:
        dst.write(ndci, 1); dst.write(red, 2)
    r = checks.ndci_undefined(str(p))
    assert r["water_pixels"] == 6 and r["ndci_pixels"] == 4 and r["missing"] == 2 and r["red_exactly_zero"] == 2
    assert round(r["missing_share_pct"], 1) == 33.3


def test_shannon_matches_the_biodiversity_report():
    d = N["biodiversity"]
    counts = list(d["classes_px"].values())
    assert sum(counts) == d["total_pixels"]
    h, j = checks.shannon(counts)
    assert round(h, 3) == d["regional_shannon"] and round(j, 3) == d["regional_evenness"]
    assert checks.shannon([5, 5]) == pytest.approx((np.log(2), 1.0))
    assert checks.shannon([9]) == (pytest.approx(0.0), 0.0)


def test_land_cover_is_the_same_map_as_the_biodiversity_classes():
    b, lc = N["biodiversity"]["classes_px"], N["land_cover"]
    total = sum(b.values())
    for k in ("tree", "grass", "built", "bare", "water"):
        assert round(100 * b[k] / total, 1) == lc["pct"][k], k
    assert close(sum(lc["ha"].values()), lc["valid_ha"], 0.2)
    areas = checks.class_areas(b, lc["valid_ha"] / total)
    assert close(areas["built"][0], lc["ha"]["built"], 0.15) and areas["built"][1] == lc["pct"]["built"]
    assert close(areas["water"][0], lc["ha"]["water"], 0.15) and areas["water"][1] == lc["pct"]["water"]


def test_buildings_and_land_cover_cross_check():
    b, lc = N["buildings"], N["land_cover"]
    assert round(b["count"] / b["aoi_km2"], 1) == b["per_km2"]
    assert round(100 * b["footprint_ha"] / (b["aoi_km2"] * 100), 1) == b["coverage_pct"]
    assert round(b["footprint_ha"] * 1e4 / b["count"], 1) == b["mean_m2"]
    assert 0.45 < b["footprint_ha"] / lc["ha"]["built"] < 0.46      # footprints cover under half the 'built-up' class
    assert round(100 * b["with_height"] / b["count"], 1) == 93.5


def test_roads_groups_and_classes_add_up():
    r = N["roads"]
    assert close(sum(r["groups_km"].values()), r["total_km"], 0.1)
    assert close(sum(r["highway_km"].values()), r["total_km"], 0.1)
    assert round(r["total_km"] / r["aoi_km2"], 2) == r["density"]
    h = r["highway_km"]
    assert close(sum(h[k] for k in ("footway", "pedestrian", "cycleway", "steps", "path", "track", "bridleway")), r["groups_km"]["tracks_paths"], 0.02)
    assert close(h["corridor"] + h["raceway"] + h["elevator"], r["groups_km"]["other"], 0.02)
    assert close(sum(h[k] for k in ("primary", "trunk", "secondary", "primary_link", "trunk_link", "secondary_link")), r["groups_km"]["major"], 0.05)
    assert r["valid_cells"] < 8 * 13                                   # cells outside the box carry no density


def test_crop_health_numbers():
    d = N["crop_health"]
    assert d["grid"]["width"] * d["grid"]["height"] == d["total_pixels"]
    assert round(100 * d["valid_pixels"] / d["total_pixels"], 1) == d["valid_pct"]
    assert round(100 * d["flagged_pixels"] / d["valid_pixels"], 1) == d["flagged_pct"]
    assert close(d["ndvi_after"] - d["ndvi_before"], d["mean_change"], 0.001)
    assert close(d["flagged_ha"] + d["not_flagged_ha"], d["valid_ha"], 0.1)
    assert round(100 * d["valid_ha"] / d["aoi_ha"], 0) == round(d["valid_pct"], 0)


def test_vector_area_by_class(tmp_path):
    import geopandas as gpd
    from shapely.geometry import box
    g = gpd.GeoDataFrame({"class_name": ["Flooded: radar only", "Flooded: radar only", "Disputed"], "area_ha": [10.0, 8.9, 3.4]},
                         geometry=[box(0, 0, 1, 1), box(2, 0, 3, 1), box(4, 0, 5, 1)], crs="EPSG:4326")
    p = tmp_path / "f.gpkg"; g.to_file(p, layer="flood_extent", driver="GPKG")
    assert checks.vector_area_by_class(str(p), "flood_extent") == {"Disputed": 3.4, "Flooded: radar only": 18.9}


def test_class_areas_of_the_accra_flood_report():
    d = N["flood_accra"]
    a = checks.class_areas(d["pixels"], 0.04)
    assert a["optical_only"] == (58.2, 0.0) and a["reference_water"] == (59501.0, 39.9) and a["not_flooded"] == (89356.0, 60.0)


# ── the explanations on the pages, reproduced on the real pipelines with scenes whose answer is known ──────────────────────

def _scene(tmp_path, name, bands, bounds):
    from pipeline_synth import write_raster
    return write_raster(tmp_path / name, np.stack(bands).astype("float32"), bounds=bounds)


def test_classify_dnbr_is_what_wildfire_severity_writes(tmp_path, monkeypatch):
    import rasterio
    from pipeline_synth import BBOX, const, patch_downloads, read
    from pygeovision.ai.pipelines.domains import WildfireSeverityPipeline
    pre = _scene(tmp_path, "pre.tif", [const(0.4), const(0.1)], BBOX)
    nir, sw = const(0.4), const(0.1)
    for (lo, hi), (n, s) in {(0, 10): (.38, .12), (10, 20): (.30, .15), (20, 30): (.25, .25), (30, 40): (.15, .30)}.items():
        nir[:, lo:hi], sw[:, lo:hi] = n, s
    nir[:5, :] = sw[:5, :] = np.nan
    post = _scene(tmp_path, "post.tif", [nir, sw], BBOX)
    patch_downloads(monkeypatch, pair=(pre, post))
    r = WildfireSeverityPipeline(object()).run(BBOX, tmp_path / "o", date_before="2018-10", date_after="2018-12")
    assert r.success, r.error
    with rasterio.open(r.output_path.parent / "dnbr.tif") as src:
        dnbr = src.read(1)
    assert np.array_equal(checks.classify_dnbr(dnbr), read(r.output_path)[0])


def _burned_stat_and_rowwise(tmp_path, monkeypatch, rows_burned):
    """Run wildfire_severity over the wildfire box with burned pixels only in ``rows_burned`` of a 100 x 140 raster."""
    from pipeline_synth import const, patch_downloads
    from pygeovision.ai.pipelines.domains import WildfireSeverityPipeline
    bbox = tuple(N["wildfire"]["bbox"])
    shape = (100, 140)
    pre = _scene(tmp_path, "pre.tif", [const(0.4, shape), const(0.1, shape)], bbox)
    nir, sw = const(0.4, shape), const(0.1, shape)
    nir[rows_burned], sw[rows_burned] = 0.25, 0.25                    # dNBR 0.6: moderate severity
    post = _scene(tmp_path, "post.tif", [nir, sw], bbox)
    patch_downloads(monkeypatch, pair=(pre, post))
    r = WildfireSeverityPipeline(object()).run(bbox, tmp_path / "o", date_before="2018-10", date_after="2018-12")
    assert r.success, r.error
    mask = np.zeros(shape, bool); mask[rows_burned] = True
    return r.stats["burned_area_ha"], checks.rowwise_area_ha(mask, *bbox)


def test_the_burned_area_stat_uses_one_pixel_area_and_a_row_by_row_sum_does_not(tmp_path, monkeypatch):
    south_stat, south_rows = _burned_stat_and_rowwise(tmp_path / "s", monkeypatch, slice(50, 100))
    north_stat, north_rows = _burned_stat_and_rowwise(tmp_path / "n", monkeypatch, slice(0, 50))
    assert south_rows > south_stat                                     # burned in the south: the table is the larger
    assert north_rows < north_stat                                     # burned in the north: the table is the smaller
    assert north_stat == pytest.approx(south_stat)                     # the stat only counts pixels, so it cannot tell
    assert 0.0015 < (south_rows - south_stat) / south_stat < 0.0021


def test_dark_water_gives_undefined_ndci_and_values_pinned_at_plus_and_minus_one(tmp_path, monkeypatch):
    """Water pixels whose red or red-edge reflectance is zero: NDCI is empty when both are, and +1 or -1 when only one is."""
    import rasterio
    from pipeline_synth import BBOX, const, patch_downloads
    from pygeovision.ai.pipelines.domains import WaterQualityPipeline
    shape = (20, 20)
    green, red, edge, nir = const(0.08, shape), const(0.2, shape), const(0.25, shape), const(0.4, shape)       # land
    water = np.zeros(shape, bool); water[:, :10] = True
    green[water], nir[water] = 0.05, 0.0                                                                       # NDWI = 1
    red[water], edge[water] = 0.03, 0.05                                                                       # NDCI 0.25
    red[0, :10], edge[0, :10] = 0.0, 0.0                                                                       # row 0: both zero
    red[1, :10], edge[1, :10] = 0.0, 0.04                                                                      # row 1: red zero
    edge[2, :10] = 0.0                                                                                         # row 2: red-edge zero
    img = _scene(tmp_path, "s.tif", [green, red, edge, nir], BBOX)
    patch_downloads(monkeypatch, single=img)
    r = WaterQualityPipeline(object()).run(BBOX, tmp_path / "o")
    assert r.success, r.error
    with rasterio.open(r.output_path) as src:
        ndci = src.read(1)
    assert np.isnan(ndci[0, :10]).all() and (ndci[1, :10] == 1.0).all() and (ndci[2, :10] == -1.0).all()
    found = checks.ndci_undefined(str(r.output_path))
    assert found["water_pixels"] == 200 and found["missing"] == 10 and found["red_exactly_zero"] == 20
    assert np.allclose(ndci[3:, :10], 0.25, atol=1e-5)


# ── the grids of the London reports, and the two ways a density raster holds less than its headline ──────────────────────

def test_the_london_report_grids_are_the_ones_the_package_builds():
    from pygeovision.ai.pipelines.vector_products import _cells_inside, _utm, make_grid
    bbox = tuple(N["buildings"]["bbox"])
    for cell_m, shape, inside in ((100.0, (58, 38), 1929), (500.0, (13, 8), N["roads"]["valid_cells"])):
        grid = make_grid(bbox, cell_m, crs=_utm(bbox))
        assert (grid.height, grid.width) == shape
        assert int(_cells_inside(grid, bbox).sum()) == inside        # the cells the density rasters fill


def test_the_road_density_raster_covers_less_than_the_box_and_the_numbers_say_so():
    d = N["roads"]
    in_raster = d["mean_cell_density"] * d["valid_cells"] * 0.25
    assert close(in_raster, 946.2, 0.1)
    assert close(d["total_km"] - in_raster, 83.9, 0.2)
    assert d["valid_cells"] * 0.25 < d["aoi_km2"]
    assert close(d["total_km"] / d["aoi_km2"], d["density"], 0.01)
    assert close(d["total_km"] / d["ways"] * 1000, 36.2, 0.05)
    vehicle = d["groups_km"]["local"] + d["groups_km"]["major"]
    assert close(vehicle / d["aoi_km2"], 22.0, 0.05)
    assert close(100 * d["groups_km"]["tracks_paths"] / d["total_km"], 58.1, 0.05)


def test_the_building_rasters_hold_less_than_the_report_headline():
    d = N["buildings"]
    cells = 1929
    assert close(0.299 * cells, 576.8, 0.5) and close(d["footprint_ha"] - 0.299 * cells, 71.2, 1.0)
    assert close(2.983 * cells, 5754, 1.5) and d["count"] - 2.983 * cells > 140
    assert close(d["footprint_ha"] * 1e4 / d["count"], d["mean_m2"], 0.1)
    assert close(d["count"] / d["aoi_km2"], d["per_km2"], 0.05)
    assert close(100 * d["footprint_ha"] / (d["aoi_km2"] * 100), d["coverage_pct"], 0.05)
    assert d["mean_m2"] / d["median_m2"] > 3


def test_a_building_bigger_than_its_cell_is_capped_and_one_centred_outside_the_filled_cells_is_in_no_raster(tmp_path, monkeypatch):
    """The two effects of the page: on the real pipeline, with three invented rectangles."""
    import geopandas as gpd
    import rasterio
    from pipeline_synth import BBOX
    from pygeovision.ai.pipelines import BuildingFootprintsPipeline
    from pygeovision.ai.pipelines import vector_products as vp
    from pyproj import Transformer
    from shapely.geometry import box

    utm = vp._utm(BBOX)
    to_utm = Transformer.from_crs("EPSG:4326", utm, always_xy=True)
    to_ll = Transformer.from_crs(utm, "EPSG:4326", always_xy=True)
    grid = vp.make_grid(BBOX, 100.0, crs=utm)
    inside = vp._cells_inside(grid, BBOX)

    def in_box(x, y):
        lon, lat = to_ll.transform(x, y)
        return BBOX[0] <= lon <= BBOX[2] and BBOX[1] <= lat <= BBOX[3]

    spot = None                                       # a point inside the box, in a cell whose centre is outside it
    for r, c in zip(*np.nonzero(~inside)):
        x = grid.transform.c + grid.transform.a * (c + 0.5 + 0.4)
        y = grid.transform.f + grid.transform.e * (r + 0.5)
        if all(in_box(x + dx, y + dy) for dx in (-10, 10) for dy in (-10, 10)):
            spot = (x, y, r, c)
            break
    assert spot, "no cell with its centre outside the box holds a point inside it"
    x, y, r, c = spot
    cx, cy = to_utm.transform(-0.125, 51.495)
    shapes = [box(cx - 75, cy - 75, cx + 75, cy + 75),            # 150 x 150 m = 22,500 m2 in one 100 m cell
              box(cx + 400, cy + 400, cx + 420, cy + 420),        # 400 m2, an ordinary building
              box(x - 10, y - 10, x + 10, y + 10)]                # 400 m2, centroid in an unfilled cell
    gdf = gpd.GeoDataFrame({"height_m": [10.0, 5.0, 5.0]}, geometry=gpd.GeoSeries(shapes, crs=utm).to_crs("EPSG:4326").values,
                           crs="EPSG:4326")
    monkeypatch.setattr(vp.fp, "microsoft_footprints", lambda bbox, **kw: gdf)
    r_ = BuildingFootprintsPipeline(object()).run(BBOX, tmp_path, source="microsoft")
    assert r_.success, r_.error
    assert r_.stats["building_count"] == 3
    total_ha = r_.stats["building_area_ha"]
    with rasterio.open(tmp_path / "building_density.tif") as src:
        density = src.read(1)
    with rasterio.open(tmp_path / "building_count.tif") as src:
        count = src.read(1)
    assert np.nanmax(density) == 1.0 and r_.stats["max_cell_coverage"] == 1.0
    assert float(np.nansum(density)) < total_ha - 1.0              # the cap and the unfilled cell both lose area
    assert float(np.nansum(count)) == 2.0                          # the third building is in no count cell
    assert np.isnan(density[r, c]) and np.isnan(count[r, c])


def test_a_road_in_a_cell_whose_centre_is_outside_the_box_counts_in_the_total_but_not_in_the_raster(tmp_path, monkeypatch):
    import geopandas as gpd
    import rasterio
    from pipeline_synth import BBOX
    from pygeovision.ai.pipelines import vector_products as vp
    from pyproj import Transformer
    from shapely.geometry import LineString

    utm = vp._utm(BBOX)
    to_ll = Transformer.from_crs(utm, "EPSG:4326", always_xy=True)
    grid = vp.make_grid(BBOX, 500.0, crs=utm)
    inside = vp._cells_inside(grid, BBOX)

    def in_box(x, y):
        lon, lat = to_ll.transform(x, y)
        return BBOX[0] <= lon <= BBOX[2] and BBOX[1] <= lat <= BBOX[3]

    lines = []
    for r, c in zip(*np.nonzero(~inside)):                          # a short line inside the box, in an unfilled cell
        x0 = grid.transform.c + grid.transform.a * (c + 0.5)
        y = grid.transform.f + grid.transform.e * (r + 0.5)
        pts = [(x0 + 100 + 20 * k, y) for k in range(6)]
        if all(in_box(px, py) for px, py in pts):
            lines.append(LineString(pts))
            break
    assert lines, "no unfilled cell holds a line inside the box"
    centre = Transformer.from_crs("EPSG:4326", utm, always_xy=True).transform(-0.125, 51.495)
    lines.append(LineString([(centre[0] - 300, centre[1]), (centre[0] + 300, centre[1])]))      # 600 m, inside
    gdf = gpd.GeoDataFrame({"highway": ["footway", "residential"], "name": [None, "A Street"]},
                           geometry=gpd.GeoSeries(lines, crs=utm).to_crs("EPSG:4326").values, crs="EPSG:4326")
    monkeypatch.setattr(vp.fp, "osm_features", lambda bbox, key, **kw: gdf)
    res = vp.RoadNetworkPipeline(object()).run(BBOX, tmp_path)
    assert res.success, res.error
    with rasterio.open(tmp_path / "road_density.tif") as src:
        dens = src.read(1)
    in_raster_km = float(np.nansum(dens)) * 0.25
    assert res.stats["total_length_km"] > in_raster_km + 0.05
    assert in_raster_km == pytest.approx(0.6, abs=0.02)


def test_flagged_by_land_cover_counts_each_class_on_the_flag_grid(tmp_path):
    from pipeline_synth import write_raster
    flag = np.zeros((40, 40), "uint8"); flag[:10, :10] = 1; flag[30:, 30:] = 1
    cover = np.full((20, 30), 50, "uint8"); cover[:7, :10] = 10            # another grid, another shape
    f = write_raster(tmp_path / "f.tif", flag, dtype="uint8")
    c = write_raster(tmp_path / "c.tif", cover, dtype="uint8")
    assert checks.flagged_by_land_cover(str(f), str(c)) == {10: 100, 50: 100}


def test_crop_health_london_numbers_fit_together():
    d = N["crop_health"]
    assert close(100 * d["flagged_pixels"] / d["valid_pixels"], d["flagged_pct"], 0.05)
    assert close(100 * d["valid_pixels"] / d["total_pixels"], d["valid_pct"], 0.05)
    assert close(d["flagged_ha"] / d["aoi_ha"] * 100, 2.6, 0.05)
    assert close(d["ndvi_after"] - d["ndvi_before"], d["mean_change"], 0.0006)
    assert close(d["flagged_pixels"] * d["aoi_ha"] / d["total_pixels"], d["flagged_ha"], 0.1)
    from datetime import date
    assert (date(2024, 8, 23) - date(2024, 5, 20)).days == 95


# ── the pages themselves ──────────────────────────────────────────────────────────────────────────────────────────────

PAGES = sorted((DOCS / "case-studies").glob("*.md"))
STUDY_PAGES = [p for p in PAGES if p.name != "index.md"]
FENCE = re.compile(r"^```(\w+)\n(.*?)^```$", re.S | re.M)


def _blocks(page: Path, lang: str) -> list[str]:
    return [body for kind, body in FENCE.findall(page.read_text(encoding="utf-8")) if kind == lang]


def test_there_is_a_page_for_every_study_and_the_hub_lists_them_all():
    expected = {"flood-accra", "flood-nepal", "wildfire", "drainage-tyrol", "water-quality",
                "london-landcover-habitat", "london-buildings-roads", "london-vegetation"}
    assert {p.stem for p in STUDY_PAGES} == expected
    hub = (DOCS / "case-studies" / "index.md").read_text(encoding="utf-8")
    assert "toctree" not in hub                       # like the other section indexes: the root toctree lists the pages
    assert {m for m in re.findall(r"\]\(([a-z-]+)\.md\)", hub)} >= expected
    home = (DOCS / "index.md").read_text(encoding="utf-8")
    assert ":caption: Case studies" in home
    for name in expected:
        assert f"case-studies/{name}\n" in home


@pytest.mark.parametrize("page", STUDY_PAGES, ids=lambda p: p.stem)
def test_every_check_block_runs_and_the_output_under_it_is_what_it_prints(page):
    import contextlib
    import io
    text = page.read_text(encoding="utf-8")
    pattern = re.compile(r"```python\n(# check:.*?)\n```\n\n```text\n(.*?)```\n", re.S)
    found = pattern.findall(text)
    assert len(found) == text.count("```python\n# check:"), "every check block needs the text block with its output right after it"
    scope = {name: getattr(checks, name) for name in dir(checks) if not name.startswith("_")}
    for code, expected in found:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(compile(code, str(page), "exec"), dict(scope))
        assert buf.getvalue() == expected, f"{page.name}: {code.splitlines()[0]}"


class _Recorder:
    """Stands in for ``PyGeoVision`` so a command and a Python block can be compared without running anything."""

    calls: list = []

    def pipeline(self, name, **kwargs):
        type(self).calls.append((name, kwargs))
        return _result()


class _AnyStats(dict):
    def __missing__(self, key):
        return 0


def _result():
    return types.SimpleNamespace(success=True, error="", output_path=None, stats=_AnyStats(), metadata={}, deliverables=None)


def _cli_calls(page: Path, monkeypatch) -> list:
    import shlex

    from click.testing import CliRunner
    from pygeovision import PyGeoVision
    from pygeovision.cli.main import cli
    calls: list = []
    monkeypatch.setattr(PyGeoVision, "pipeline", lambda self, name, **kw: (calls.append((name, kw)), _Recorder().pipeline(name))[1])
    for block in _blocks(page, "bash"):
        for command in re.split(r"\n\s*\n", block.strip()):
            argv = shlex.split(command.replace("\\\n", " "))
            assert argv[:2] == ["pygeovision", "channel"], command
            res = CliRunner().invoke(cli, argv[1:])
            assert res.exit_code == 0, f"{page.name}: {command}\n{res.output}"
    return calls


def _python_calls(page: Path, monkeypatch) -> list:
    import sys
    calls: list = []

    class Client:
        def pipeline(self, name, **kwargs):
            calls.append((name, kwargs))
            return _result()

    monkeypatch.setitem(sys.modules, "pygeovision", types.SimpleNamespace(PyGeoVision=Client))
    scope: dict = {}                                       # the blocks of a page share one session, as they would in a notebook
    for block in _blocks(page, "python"):
        if ".pipeline(" in block or "client = PyGeoVision()" in block:
            exec(compile(textwrap.dedent(block), str(page), "exec"), scope)
    return calls


PIPELINE_PAGES = [p for p in STUDY_PAGES if "pygeovision channel" in p.read_text(encoding="utf-8")]


def test_every_study_page_has_a_run_it_section():
    assert {p.stem for p in PIPELINE_PAGES} == {p.stem for p in STUDY_PAGES}


@pytest.mark.parametrize("page", PIPELINE_PAGES, ids=lambda p: p.stem)
def test_the_command_and_the_python_block_ask_for_the_same_thing(page, monkeypatch):
    cli_calls = _cli_calls(page, monkeypatch)
    py_calls = _python_calls(page, monkeypatch)
    assert cli_calls and len(cli_calls) == len(py_calls), page.name
    for (cname, ckw), (pname, pkw) in zip(cli_calls, py_calls):
        assert cname == pname
        ckw = {k: v for k, v in ckw.items() if k not in ("report_formats", "vector_formats")}      # the command's own defaults
        ckw["bbox"] = tuple(ckw["bbox"])
        pkw = dict(pkw); pkw["bbox"] = tuple(pkw["bbox"])
        assert ckw == pkw, f"{page.name}: {cname}"


@pytest.mark.parametrize("page", PIPELINE_PAGES, ids=lambda p: p.stem)
def test_every_pipeline_and_argument_on_a_page_exists(page, monkeypatch):
    from pygeovision.ai.pipelines import pipeline_kinds
    kinds = pipeline_kinds()
    generic = {"bbox", "output_dir", "date", "date_before", "date_after", "source", "model", "checkpoint"}
    for name, kw in _cli_calls(page, monkeypatch):
        assert name in kinds, name
        catalog = (DOCS / "pipelines" / "catalog" / f"{name}.md").read_text(encoding="utf-8")
        for key in set(kw) - generic - {"report_formats", "vector_formats"}:
            assert f"`{key}`" in catalog, f"{page.name}: --param {key} is not an argument of {name}"


HEADLINES = {
    "flood-accra": ["81.4", "59,501.0", "-13.8"],
    "flood-nepal": ["451.7", "90.1", "13.5 million", "6.0 million"],
    "wildfire": ["49,568.1", "20.8"],
    "drainage-tyrol": ["64.8", "0.72", "19.5", "100,056", "58.80"],
    "water-quality": ["127,989", "66,612", "61,377", "40.7", "2.1", "0.249", "13,781", "696.4"],
    "london-landcover-habitat": ["1,426.4", "348.0", "0.782", "0.437", "193.2", "0.835", "191,891", "1,776"],
    "london-buildings-roads": ["5,898", "648.0", "305.3", "1,098.7", "359.8", "1,030.1", "53.32", "28,454", "598.77", "493.39"],
    "london-vegetation": ["5.8", "2.6", "44.2", "0.287", "0.249", "4,528", "78,301", "436", "-0.872", "95 days"],
}


@pytest.mark.parametrize("stem", sorted(HEADLINES))
def test_each_page_quotes_the_headline_figures_of_its_report(stem):
    text = (DOCS / "case-studies" / f"{stem}.md").read_text(encoding="utf-8")
    missing = [s for s in HEADLINES[stem] if s not in text]
    assert not missing, f"{stem}: {missing}"


@pytest.mark.parametrize("page", STUDY_PAGES, ids=lambda p: p.stem)
def test_every_figure_a_page_uses_exists_and_has_alt_text(page):
    text = page.read_text(encoding="utf-8")
    for target, alt in re.findall(r"```\{figure\} (\S+)\n:alt: ([^\n]+)", text):
        assert (page.parent / target).resolve().exists(), f"{page.name}: {target}"
        assert len(alt.split()) >= 6, f"{page.name}: alt text of {target} is too short"
    for target in re.findall(r"\]\((?!http|#)([^)\s#]+\.md)", text):
        assert (page.parent / target).resolve().exists(), f"{page.name}: {target}"
