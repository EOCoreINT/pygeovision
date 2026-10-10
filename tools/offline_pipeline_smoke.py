#!/usr/bin/env python
"""Run all 25 pipelines through the real ``pygeovision channel`` CLI -- without network access.

Only the three things that need the internet are replaced:

* the scene download (``BasePipeline._search_and_download[_pair]``) returns a small synthetic raster;
* the OpenTopography DEM fetch (``_DemPipeline._fetch_dem``) returns a synthetic terrain;
* the ESA WorldCover tile fetch (``ESAWorldCoverLabeler.label_tile``) returns a synthetic land-cover tile.

Everything else is the real code path: argument parsing, validation, band maths, nodata handling, the
strict ``--checkpoint`` load, file writing and statistics -- and the deliverables (map, GeoPackage/GeoJSON
vectors, PDF/Word/Excel reports), which each run must leave behind. The six pipelines that run your own network get a
*wiring* checkpoint (a randomly initialised network of the right shape), so they prove the plumbing works;
their output is meaningless and must never be read as a result.

    python tools/offline_pipeline_smoke.py                    # all 25
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
    "building_footprints": ["--date", "2024-06", ("ckpt", "unet_resnet50", 2, 3)],
    "change_detection": ["--date-before", "2022-06", "--date-after", "2024-06"],
    "coastal_monitoring": ["--date-before", "2020-01", "--date-after", "2024-01"],
    "crop_health": PAIR,
    "crop_monitoring": ["--date", "2023-06", "--param", "num_classes=3", ("ckpt", "segformer_b2", 3, 3)],
    "deforestation": PAIR + [("ckpt", "changeformer", 3, 3)],
    "disaster_assessment": PAIR + ["--param", "disaster_type=flood", ("ckpt", "siamese_unet", 4, 3)],
    "flood_mapping": PAIR,
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

    BasePipeline._search_and_download = fake_single
    BasePipeline._search_and_download_pair = fake_pair
    _DemPipeline._fetch_dem = fake_dem
    ESAWorldCoverLabeler.label_tile = fake_label


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
