#!/usr/bin/env python
"""Run all 40 pipelines through the real ``pygeovision channel`` CLI -- without network access.

Only the things that need the internet are replaced:

* the scene download (``BasePipeline._search_and_download[_pair]``) returns a small synthetic raster;
* the OpenTopography DEM fetch (``_DemPipeline._fetch_dem``) returns a synthetic terrain;
* the ESA WorldCover tile fetch (``ESAWorldCoverLabeler.label_tile``) returns a synthetic land-cover tile;
* the Planetary Computer catalogue and tile reads used by the reference-product, terrain and radar pipelines
  (``PlanetaryComputer`` and ``read_to_grid``) return synthetic land-cover, forest, water, ET, fire and DEM tiles;
* the building-footprint and OpenStreetMap downloads (``pygeovision.data.footprints``) return a few synthetic
  buildings and roads.

Everything else is the real code path: argument parsing, validation, band maths, nodata handling, the
strict ``--checkpoint`` load, file writing and statistics -- and the deliverables (map, GeoPackage/GeoJSON
vectors, PDF/Word/Excel reports), which each run must leave behind. The seven pipelines that run your own network get a
*wiring* checkpoint (a randomly initialised network of the right shape), so they prove the plumbing works;
their output is meaningless and must never be read as a result.

    python tools/offline_pipeline_smoke.py                    # all 40
    python tools/offline_pipeline_smoke.py water_bodies flood_mapping
    python tools/offline_pipeline_smoke.py --commands         # print the equivalent real-data commands
    python tools/offline_pipeline_smoke.py --keep out_dir     # keep the produced files

Exit status is 0 only if every selected pipeline succeeded.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import tempfile
import warnings

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
warnings.filterwarnings("ignore")

BBOX_ARGS = ["--bbox", "-0.15", "51.47", "-0.10", "51.52"]
PAIR = ["--date-before", "2024-05", "--date-after", "2024-08"]

# name -> extra CLI arguments. A ("ckpt", architecture, num_classes, in_channels) tuple stands for a
# --checkpoint argument: here a generated wiring checkpoint, with real data your own trained weights.
SPECS: dict[str, list] = {
    "archaeological_site": [],
    "biodiversity_hotspot": [],
    "active_fire": ["--date", "2024-02"],
    "building_footprints": ["--param", "source=microsoft"],
    "drainage_analysis": ["--param", "resolution_m=60"],
    "evapotranspiration": ["--date", "2022"],
    "forest_change": ["--date-before", "2015", "--date-after", "2020"],
    "landcover_change": ["--date-before", "2018", "--date-after", "2023"],
    "landslide_detection": PAIR + [("ckpt", "siamese_unet", 2, 3)],
    "road_extraction": ["--date", "2024-06", ("ckpt", "unet_resnet50", 2, 3)],
    "road_network": [],
    "surface_water_dynamics": [],
    "terrain_analysis": [],
    "urban_expansion": ["--date-before", "2018", "--date-after", "2023"],
    "vegetation_anomaly": ["--date", "2024-06"],
    "water_extent_timeseries": ["--dates", "2024-01,2024-03,2024-05"],
    "change_detection": ["--date-before", "2022-06", "--date-after", "2024-06"],
    "coastal_monitoring": ["--date-before", "2020-01", "--date-after", "2024-01"],
    "crop_health": PAIR,
    "crop_monitoring": ["--date", "2023-06", "--param", "num_classes=3", ("ckpt", "segformer_b2", 3, 3)],
    "deforestation": PAIR + [("ckpt", "changeformer", 3, 3)],
    "disaster_assessment": PAIR + ["--param", "disaster_type=flood", ("ckpt", "siamese_unet", 4, 3)],
    "flood_mapping": PAIR,
    "sar_flood_mapping": ["--date-before", "2024-05-01/2024-05-31", "--date-after", "2024-06-10/2024-06-20"],
    "multisensor_flood_mapping": ["--date-before", "2024-05-01/2024-05-31", "--date-after", "2024-06-10/2024-06-20"],
    "glacier_monitoring": PAIR,
    "land_cover": ["--date", "2023-06"],
    "land_surface_temperature": ["--date", "2024-07"],
    "snow_cover": ["--date", "2024-01"],
    "solar_detection": ["--date", "2024-06", ("ckpt", "unet_efficientnet_b4", 2, 3)],
    "solar_potential": ["--date", "2024-06-21", "--param", "hour=12"],
    "urban_growth": PAIR + [("ckpt", "siamese_unet", 2, 3)],
    "urban_heat_island": ["--date", "2024-07"],
    "vegetation_indices": ["--dates", "2024-04,2024-06,2024-08"],
    "volcano_monitoring": ["--date", "2024-07"],
    "water_bodies": ["--date", "2024-06"],
    "water_quality": ["--date", "2024-06"],
    "wetland_mapping": ["--date", "2024-06"],
    "wildfire_severity": PAIR,
    "wind_farm_siting": [],
}


def real_command(name: str) -> str:
    """The command for real data (you supply the bbox/dates and, where needed, your checkpoint)."""
    parts = ["pygeovision", "channel", name, *BBOX_ARGS, "-o", f"./out/{name}"]
    for e in SPECS[name]:
        parts += (["--checkpoint", "path/to/your_trained.pth"] if isinstance(e, tuple)
                  else [f"'{e}'" if " " in e or "," in e else e])
    return " ".join(parts)


def install_offline_patches(tmp: pathlib.Path):
    import numpy as np
    import rasterio
    from pipeline_synth import write_raster

    from pygeovision.ai.labeling.base_labeler import LabelingResult
    from pygeovision.ai.labeling.esa_worldcover import ESAWorldCoverLabeler
    from pygeovision.ai.pipelines import BasePipeline
    from pygeovision.ai.pipelines.domains import _DemPipeline

    rng = np.random.default_rng(0)

    def fake_single(self, bbox, date, output_dir, **kw):
        n = len(kw.get("bands") or []) or 4
        arr = np.clip(rng.normal(0.25, 0.1, (n, 60, 60)), 0.02, 0.9).astype("float32")
        if tuple(kw.get("bands") or ()) == ("thermal",):
            # Landsat Collection 2 ST_B10 digital numbers: K = DN x 0.00341802 + 149. A 30 deg C scene with a warm
            # block, so the example output shows plausible temperatures instead of the reflectance noise above.
            yy, xx = np.mgrid[0:60, 0:60]
            kelvin = 303.0 + 0.04 * xx - 0.02 * yy + rng.normal(0, 0.4, (60, 60))
            kelvin[10:25, 30:50] += 7.0
            arr = ((kelvin - 149.0) / 0.00341802).astype("float32")[None]
        return write_raster(tmp / f"{self.name}_{date}_{abs(hash(str(kw))) % 999}.tif", arr, bbox)

    def fake_pair(self, bbox, date_before, date_after, output_dir, **kw):
        n = len(kw.get("bands") or []) or 4
        a = np.clip(rng.normal(0.25, 0.1, (n, 60, 60)), 0.02, 0.9).astype("float32")
        b = np.clip(a + rng.normal(0, 0.05, a.shape), 0.02, 0.9).astype("float32")
        return (write_raster(tmp / f"{self.name}_b.tif", a, bbox), write_raster(tmp / f"{self.name}_a.tif", b, bbox))

    def fake_dem(self, bbox, out, dem_type):
        y, x = np.mgrid[0:80, 0:80]
        dem = (200 + 40 * np.sin(x / 9.0) + 30 * np.cos(y / 7.0) + 0.8 * x).astype("float64")
        path = write_raster(tmp / "dem.tif", dem.astype("float32"), bbox)
        with rasterio.open(path) as src:
            profile, bounds = src.profile.copy(), src.bounds
        return path, dem, profile, bounds

    def fake_label(self, tile_path, meta, out_path):
        with rasterio.open(tile_path) as src:
            shape = (src.height, src.width)
        lc = np.full(shape, 30, np.uint8); lc[:, :15] = 80; lc[20:30, 20:30] = 10
        write_raster(out_path, lc[None], dtype="uint8")
        return LabelingResult(tile_id="t", label_path=out_path, source="esa_worldcover")

    install_reference_patches(tmp)
    BasePipeline._search_and_download = fake_single
    BasePipeline._search_and_download_pair = fake_pair
    _DemPipeline._fetch_dem = fake_dem
    ESAWorldCoverLabeler.label_tile = fake_label


def install_reference_patches(tmp: pathlib.Path) -> None:
    """Synthetic Planetary Computer, footprint and OpenStreetMap data (see the module docstring)."""
    import geopandas as gpd
    import numpy as np
    from shapely.geometry import LineString, box

    from pygeovision.ai.pipelines import multisensor_flood, reference_products, sar_flood, terrain
    from pygeovision.data import footprints as fp
    from pygeovision.data.reference import GridRaster

    class SmokePC:
        def search(self, collection, bbox, datetime=None, **kw):
            lo = (datetime or "2020-01-01/2020-12-31").split("/")[0]
            hi = (datetime or "2020-01-01/2020-12-31").split("/")[-1]
            year = int(lo[:4])
            if collection == "modis-14A1-061":
                days = [lo, f"{lo[:8]}{int(lo[8:10]) + 1:02d}"]
                return [{"id": f"MOD14A1.{d}", "properties": {"start_datetime": f"{d}T00:00:00Z"}, "assets": {}} for d in days]
            if collection == "modis-16A3GF-061":
                return [{"id": f"MOD16A3GF.A{year}001.h18v08.061.x", "properties": {"start_datetime": f"{year}-01-01T00:00:00Z",
                                                                                    "end_datetime": f"{year}-12-31T23:59:59Z"}, "assets": {}}]
            if collection == "sentinel-1-rtc":
                day = lo[:10]
                geom = {"type": "Polygon", "coordinates": [[[-180, -90], [180, -90], [180, 90], [-180, 90], [-180, -90]]]}
                return [{"id": f"S1_{day}", "geometry": geom, "bbox": [-180, -90, 180, 90],
                         "properties": {"datetime": f"{day}T05:30:00Z", "sat:orbit_state": "descending", "sat:relative_orbit": 1},
                         "assets": {"vv": {"href": "x"}, "vh": {"href": "x"}}}]
            if collection == "sentinel-2-l2a":
                day = lo[:10]
                return [{"id": f"S2_{day}", "properties": {"datetime": f"{day}T10:30:00Z", "eo:cloud_cover": 12.0},
                         "assets": {"B03": {"href": "x"}, "B11": {"href": "x"}, "SCL": {"href": "x"}}}]
            if collection in ("jrc-gsw", "cop-dem-glo-30"):
                return [{"id": f"{collection}-tile", "properties": {"datetime": "2020-07-01T00:00:00Z"}, "assets": {}}]
            return [{"id": f"{collection}-{year}", "properties": {"start_datetime": f"{year}-01-01T00:00:00Z"}, "assets": {}}]

        def asset_urls(self, collection, items, asset):
            return [f"synthetic://{collection}/{asset}/{it['id']}" for it in items]

    def synth(url: str, grid, invalid_values=()):
        h, w = grid.height, grid.width
        rr, cc = np.mgrid[0:h, 0:w]
        _, rest = url.split("://")
        collection, asset, ident = rest.split("/")
        year = int("".join(ch for ch in ident.split(".A")[-1][:4] if ch.isdigit()) or 2020) if "MOD16" in ident else \
            int(ident[-4:]) if ident[-4:].isdigit() else 2020
        if collection == "sentinel-1-rtc":
            a = np.full((h, w), 0.1)                                        # land, about -10 dB
            a[rr > h * 0.85] = 0.003                                          # a permanent lake
            if ident[3:] >= "2024-06-01":                                     # the flood scene: a dark block appears
                a[(rr > h * 0.3) & (rr < h * 0.5) & (cc > w * 0.3) & (cc < w * 0.6)] = 0.004
            return a
        if collection == "sentinel-2-l2a":
            water = rr > h * 0.85                                             # the same permanent lake
            if ident[3:] >= "2024-06-01":                                     # the flood scene: the same block is water
                water = water | ((rr > h * 0.3) & (rr < h * 0.5) & (cc > w * 0.3) & (cc < w * 0.6))
            if asset == "SCL":
                return np.full((h, w), 4.0)
            return np.where(water, 1200.0 if asset == "B03" else 300.0, 1500.0 if asset == "B03" else 2500.0)
        if collection == "io-lulc-annual-v02":
            a = np.full((h, w), 2.0)
            a[rr > h * 0.6] = 5.0
            a[(rr < h * 0.3) & (cc < w * (0.2 + 0.03 * (year - 2017)))] = 7.0
            a[(rr > h * 0.45) & (rr < h * 0.5)] = 1.0
            return a
        if collection == "alos-fnf-mosaic":
            a = np.full((h, w), 3.0)
            a[(cc < w * 0.4) & (rr < h * (0.8 - 0.02 * (year - 2015)))] = 1.0
            a[(cc > w * 0.8) & (rr > h * 0.7)] = 4.0
            return a
        if collection == "jrc-gsw":
            if asset == "transitions":
                a = np.zeros((h, w)); a[rr > h * 0.8] = 1.0; a[(rr > h * 0.7) & (rr <= h * 0.8)] = 4.0; a[(rr < h * 0.1) & (cc < w * 0.3)] = 3.0
                return a
            if asset == "occurrence":
                return np.where(rr > h * 0.7, 80.0, 0.0)
            return np.where(rr > h * 0.7, 12.0, 0.0)
        if collection == "modis-16A3GF-061":
            return np.full((h, w), 12000.0 if asset == "ET_500m" else 19000.0) + cc
        if collection == "modis-14A1-061":
            if asset == "FireMask":
                a = np.full((h, w), 5.0); a[h // 2, w // 2] = 9.0; a[h // 3, w // 3] = 8.0; a[0, 0] = 4.0
                return a
            a = np.zeros((h, w)); a[h // 2, w // 2] = 250.0
            return a
        if collection == "cop-dem-glo-30":
            return 120.0 + 25 * np.sin(cc / 9.0) + 20 * np.cos(rr / 7.0) + 0.6 * cc - 0.3 * rr
        raise ValueError(f"smoke test has no synthetic data for {url}")

    def fake_read(urls, grid, *, band=1, resampling="nearest", dtype="float32", invalid_values=(), fill=np.nan, env=None):
        urls = list(urls)
        data = synth(urls[0], grid).astype(dtype)
        valid = np.ones(data.shape, bool)
        for v in invalid_values:
            valid &= data != v
        return GridRaster(np.where(valid, data, np.nan).astype(dtype), valid, grid, [u.rsplit("/", 1)[-1] for u in urls])

    pc = SmokePC()
    reference_products.read_to_grid = fake_read
    terrain.read_to_grid = fake_read
    sar_flood.read_to_grid = fake_read
    multisensor_flood.read_to_grid = fake_read
    reference_products._Reference._pc = lambda self: pc
    terrain._TerrainBase._pc = lambda self: pc
    sar_flood.SarFloodMappingPipeline._pc = lambda self: pc
    multisensor_flood.MultiSensorFloodPipeline._pc = lambda self: pc

    def fake_buildings(bbox, **kw):
        w, s_, e, n = bbox
        geoms = [box(w + (e - w) * (0.1 + 0.07 * i), s_ + (n - s_) * (0.2 + 0.05 * (i % 5)),
                     w + (e - w) * (0.1 + 0.07 * i) + 0.0002, s_ + (n - s_) * (0.2 + 0.05 * (i % 5)) + 0.0002) for i in range(12)]
        return gpd.GeoDataFrame({"id": range(12)}, geometry=geoms, crs="EPSG:4326")

    def fake_osm(bbox, kind="building", **kw):
        if kind == "building":
            return fake_buildings(bbox)
        w, s_, e, n = bbox
        lines = [LineString([(w, (s_ + n) / 2), (e, (s_ + n) / 2)]), LineString([((w + e) / 2, s_), ((w + e) / 2, n)])]
        return gpd.GeoDataFrame({"highway": ["primary", "residential"], "name": ["Main St", None], "surface": [None, None],
                                 "oneway": [None, None], "osm_id": [1, 2]}, geometry=lines, crs="EPSG:4326")

    fp.microsoft_footprints = fake_buildings
    fp.google_footprints = fake_buildings
    fp.osm_features = fake_osm


def wiring_checkpoint(tmp: pathlib.Path, model: str, classes: int, bands: int, name: str) -> str:
    import torch

    from pygeovision.ai.models.hub import ModelHub
    m = ModelHub().load(model, num_classes=classes, in_channels=bands, pretrained=False)
    p = tmp / f"{name}.pth"
    torch.save(m.state_dict(), p)
    return str(p)


def check_deliverables(out: pathlib.Path, name: str) -> tuple[bool, str]:
    """True (= problem) if a pipeline run did not leave its map, vectors and three reports."""
    import json
    summary = out / "run_summary.json"
    if not summary.is_file():
        return True, "no run_summary.json (deliverables were not created)"
    d = json.loads(summary.read_text())["deliverables"]
    missing = [f for f in ("pdf", "docx", "xlsx") if f not in d["reports"]]
    files = [pathlib.Path(p) for p in [*d["reports"].values(), *d["maps"], *sum(d["vectors"].values(), [])]]
    gone = [p.name for p in files if not p.is_file() or p.stat().st_size == 0]
    if missing or gone or not d["maps"] or "gpkg" not in d["vectors"]:
        return True, f"missing reports {missing}, empty/absent files {gone}, maps={len(d['maps'])}, warnings={d['warnings']}"
    return False, (f"{len(d['reports'])} reports, {len(d['maps'])} map(s), "
                   f"{sum(len(v) for v in d['vectors'].values())} vector file(s)"
                   + (f"  [note: {d['warnings'][0][:60]}]" if d["warnings"] else ""))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pipelines", nargs="*", help="pipelines to run (default: all)")
    ap.add_argument("--commands", action="store_true", help="print the real-data command for each and exit")
    ap.add_argument("--keep", metavar="DIR", help="write the outputs under DIR instead of a temp directory")
    args = ap.parse_args(argv)

    unknown = [p for p in args.pipelines if p not in SPECS]
    if unknown:
        ap.error(f"unknown pipeline(s): {unknown}")
    names = args.pipelines or list(SPECS)
    if args.commands:
        for n in names:
            print(real_command(n))
        return 0

    from click.testing import CliRunner

    from pygeovision.ai.pipelines import list_pipelines
    from pygeovision.cli.main import cli
    missing = set(list_pipelines()) ^ set(SPECS)
    if missing and not args.pipelines:
        print(f"WARNING: registry and SPECS disagree on {sorted(missing)}", file=sys.stderr)

    tmp = pathlib.Path(args.keep) if args.keep else pathlib.Path(tempfile.mkdtemp())
    tmp.mkdir(parents=True, exist_ok=True)
    install_offline_patches(tmp)
    runner, failed = CliRunner(), []
    for name in names:
        cmd = ["channel", name, *BBOX_ARGS, "-o", str(tmp / "out" / name)]
        for e in SPECS[name]:
            if isinstance(e, tuple):
                _, model, classes, bands = e
                cmd += ["--checkpoint", wiring_checkpoint(tmp, model, classes, bands, name)]
            else:
                cmd.append(e)
        res = runner.invoke(cli, cmd)
        problem = res.exit_code != 0
        detail = ""
        if not problem:
            problem, detail = check_deliverables(tmp / "out" / name, name)
        tail = detail or " | ".join(line.strip() for line in res.output.strip().splitlines()[-3:])
        print(f"{'OK ' if not problem else 'ERR'} {name:<26} {tail[:150]}")
        if problem:
            failed.append(name)
    print(f"\n{len(names) - len(failed)}/{len(names)} pipelines succeeded"
          + (f"; failed: {failed}" if failed else "") + f"   (outputs: {tmp})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
