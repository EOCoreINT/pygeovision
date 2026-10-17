"""The task guides (docs/guides) must work as written.

* Every ``pygeovision ...`` command on a guide page names a real command group and only options that command has.
* Every Python block parses; the blocks that can run offline are run for real, in order, as a reader would run them,
  against pipeline runs whose download is replaced by synthetic scenes with a known answer (tests/pipeline_synth.py).
* The "compare two runs" snippet is checked against an independent count of the pixels that changed.

Nothing here downloads imagery or trains a network. The steps that do (``client.search``, ``infer predict``, training)
are checked for their options and signatures only; the page says so where it matters.
"""
from __future__ import annotations

import ast
import contextlib
import inspect
import io
import json
import re
import shlex
from pathlib import Path

import numpy as np
import pytest

from pipeline_synth import BBOX, const, patch_downloads, write_raster

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
GUIDES = sorted((DOCS / "guides").glob("*.md"))
FENCE = re.compile(r"^```(\w+)\n(.*?)^```$", re.S | re.M)

pytestmark = pytest.mark.skipif(not DOCS.exists(), reason="docs not in this checkout")


def blocks(page: Path, lang: str) -> list[str]:
    return [body for kind, body in FENCE.findall(page.read_text(encoding="utf-8")) if kind == lang]


# --------------------------------------------------------------------------- commands
def commands(page: Path) -> list[list[str]]:
    """Every ``pygeovision ...`` command on the page, as an argument list (placeholders such as $1 are kept as text)."""
    found = []
    for kind in ("bash", "text", "console"):
        for body in blocks(page, kind):
            body = body.replace("\\\n", " ")
            for line in body.splitlines():
                m = re.search(r"(?:^|[\s/])pygeovision\s+(?=[a-z])", line)
                if not m:
                    continue
                tail = line[m.end():]
                tail = re.split(r"\s(?:\|\||&&|>>|2>&1|\|)\s?", tail)[0].split("#")[0]
                tail = tail.replace("\\%", "%")
                found.append(shlex.split(tail))
    return found


def resolve(argv: list[str]):
    """The click command an argument list addresses, and the arguments left over."""
    from pygeovision.cli.main import cli

    cmd, rest = cli, list(argv)
    while rest and hasattr(cmd, "commands") and rest[0] in cmd.commands:
        cmd = cmd.commands[rest.pop(0)]
    return cmd, rest


def check_options(argv: list[str]) -> None:
    cmd, rest = resolve(argv)
    assert not hasattr(cmd, "commands"), f"'pygeovision {' '.join(argv)}' stops at a command group"
    known = {o for p in cmd.params for o in (*p.opts, *p.secondary_opts)} | {"--help"}
    for token in rest:
        if token == "--":
            break
        if token.startswith("-") and not re.match(r"^-\d", token):
            assert token in known, f"pygeovision {' '.join(argv[:2])}: unknown option {token}"


def test_there_are_guides():
    assert {p.stem for p in GUIDES} >= {"index", "your-area", "python", "automate", "use-results", "own-model"}


@pytest.mark.parametrize("page", GUIDES, ids=lambda p: p.stem)
def test_every_command_uses_real_options(page):
    for argv in commands(page):
        check_options(argv)


@pytest.mark.parametrize("page", GUIDES, ids=lambda p: p.stem)
def test_every_channel_command_names_a_real_pipeline_and_real_arguments(page):
    from pygeovision.ai.pipelines import pipeline_catalogue

    catalogue = pipeline_catalogue()
    for argv in commands(page):
        if argv[:1] != ["channel"] or len(argv) < 2 or argv[1] == "list":
            continue
        name = argv[1]
        assert name in catalogue, f"{page.name}: '{name}' is not a pipeline"
        params = inspect.signature(catalogue[name].run).parameters
        for key in re.findall(r"--param\s+([A-Za-z_]+)=", " ".join(argv)):
            assert key in params, f"{page.name}: {key} is not an argument of {name}"


@pytest.mark.parametrize("page", GUIDES, ids=lambda p: p.stem)
def test_every_python_block_parses(page):
    for code in blocks(page, "python"):
        ast.parse(code)


# --------------------------------------------------------------------------- a synthetic world
def band_stack(*bands):
    return np.stack(bands).astype(np.float32)


def water_scene(path: Path, water: slice, cloud: slice | None = None) -> Path:
    """Green and near-infrared bands: water (NDWI +0.54) on the columns in ``water``, land (-0.67) elsewhere, optional cloud."""
    green, nir = const(0.08), const(0.4)
    green[:, water], nir[:, water] = 0.10, 0.03
    if cloud is not None:
        green[:, cloud] = nir[:, cloud] = np.nan
    return write_raster(path, band_stack(green, nir))


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A working folder with the files the guides assume, and downloads that return synthetic scenes by date."""
    monkeypatch.chdir(tmp_path)
    scenes = {
        "2024-06": water_scene(tmp_path / "june.tif", slice(0, 10)),
        "2024-08": water_scene(tmp_path / "august.tif", slice(5, 15), cloud=slice(36, 40)),
    }
    patch_downloads(monkeypatch, single=lambda bbox, date, out, **kw: scenes.get(date, scenes["2024-06"]))
    # a four-band reflectance stack, a study-area file and two district polygons, all inside the box
    rng = np.random.default_rng(1)
    write_raster(tmp_path / "scene.tif", rng.uniform(0.05, 0.6, (4, 40, 40)).astype("float32"))
    west, south, east, north = BBOX
    ring = lambda w, s, e, n: {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]}   # noqa: E731
    def collection(*geoms):
        return {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"name": f"d{i}"}, "geometry": g}
                                                          for i, g in enumerate(geoms)]}
    mid = (west + east) / 2
    (tmp_path / "my_study_area.geojson").write_text(json.dumps(collection(ring(west, south, east, north))))
    (tmp_path / "districts.geojson").write_text(json.dumps(collection(ring(west, south, mid, north), ring(mid, south, east, north))))
    (tmp_path / "district.geojson").write_text(json.dumps(collection(ring(west, south, mid, north))))
    return tmp_path


def run_blocks(page: str, needles: list[str], scope: dict | None = None) -> tuple[dict, str]:
    """Run, in order, the one Python block of the page that contains each needle. Returns the shared scope and the output."""
    path = DOCS / "guides" / page
    python = blocks(path, "python")
    scope = {} if scope is None else scope
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        for needle in needles:
            hits = [b for b in python if needle in b]
            assert len(hits) == 1, f"{page}: {len(hits)} blocks contain {needle!r}"
            exec(compile(hits[0], str(path), "exec"), scope)
    return scope, out.getvalue()


# --------------------------------------------------------------------------- the pages, run
def test_your_area_the_box_checks_and_the_run_work(world):
    scope, out = run_blocks("your-area.md", [
        "bbox = (-0.15, 51.47, -0.10, 51.52)",
        "import geopandas as gpd",
        "from pyproj import Geod",
    ])
    assert tuple(scope["bbox"]) == pytest.approx(BBOX)                       # the file's bounds are the box
    assert 15 < scope["area_km2"] < 25                                       # 0.05 deg x 0.05 deg at 51.5 N is about 19 km2
    assert "million pixels at 10 m" in out
    scope["client"] = __import__("pygeovision").PyGeoVision()
    scope, out = run_blocks("your-area.md", ["result = client.pipeline"], scope)
    assert scope["result"].success, scope["result"].error
    assert (world / "out" / "london" / "run_summary.json").exists()


def test_your_area_limits_are_the_ones_in_the_code():
    from pygeovision.ai.pipelines import terrain
    from pygeovision.data import reference

    text = (DOCS / "guides" / "your-area.md").read_text(encoding="utf-8")
    assert reference.MAX_GRID_PIXELS == 250_000_000 and "250 million pixels" in text
    assert "25,000 km²" in text and 250e6 * 100 / 1e6 == 25_000              # 10 m pixels
    assert inspect.signature(terrain.hand_from_dem).parameters["max_cells"].default == 6_000_000 and "6 million cells" in text


def test_python_guide_runs(world):
    scope, out = run_blocks("python.md", [
        "client = PyGeoVision()",
        "result = client.pipeline(",
        "if not result.success:",
        "print(result.output_path)",
        "mask_file = Path(result.output_path)",
        "layers = result.deliverables.vectors",
        "report = client.validator.validate(",
        "print(client.postprocess.class_statistics",
        "status = client.status()",
    ])
    assert scope["result"].success
    assert "class 0:" in out and "class 1:" in out                           # the shares of the valid pixels
    assert (world / "water.geojson").exists() and (world / "ndvi.tif").exists()
    assert scope["status"]["pygeovision_version"]


def test_python_guide_search_and_download_names_real_arguments():
    from pygeovision import PyGeoVision

    code = next(b for b in blocks(DOCS / "guides" / "python.md", "python") if "client.search(" in b)
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("search", "select_covering_scenes", "download"):
            params = inspect.signature(getattr(PyGeoVision, node.func.attr)).parameters
            for kw in node.keywords:
                assert kw.arg in params, f"client.{node.func.attr}() has no argument {kw.arg}"


def test_automate_guide_loops_collect_one_row_per_area(world):
    scope, out = run_blocks("automate.md", ["areas = {", "for month in", "if (out / \"run_summary.json\").exists():"])
    table = scope["table"]
    assert list(table["area"]) == ["central", "east", "west"] and table["ok"].all()
    assert (world / "out" / "water" / "summary.csv").exists()
    assert all((world / "out" / "months" / m / "run_summary.json").exists() for m in ("2024-04", "2024-05", "2024-06", "2024-07"))
    # the "collect afterwards" block rebuilds the table from the folders alone
    scope, out = run_blocks("automate.md", ["records = []"], scope)
    assert len(scope["records"]) == 3 and {r["area"] for r in scope["records"]} == {"central", "east", "west"}
    assert all(r["water_coverage"] == pytest.approx(10 / 40) for r in scope["records"])


def test_automate_guide_failed_run_exits_non_zero_and_writes_no_summary(world, monkeypatch):
    from click.testing import CliRunner
    from pygeovision.cli.main import cli

    patch_downloads(monkeypatch, single=None)                                # nothing found
    res = CliRunner().invoke(cli, ["channel", "water_bodies", "--bbox", *map(str, BBOX), "--date", "2024-06",
                                   "--output", "out/failed", "--no-report"])
    assert res.exit_code != 0
    assert not (world / "out" / "failed" / "run_summary.json").exists()      # the "skip finished runs" test relies on this


def test_use_results_guide_numbers_and_comparison(world):
    from pygeovision import PyGeoVision
    client = PyGeoVision()
    for month, folder in (("2024-06", "june"), ("2024-08", "august")):
        r = client.pipeline("water_bodies", bbox=BBOX, date=month, output_dir=f"out/{folder}", report=False)
        assert r.success, r.error
    import shutil
    shutil.copytree("out/june", "out/london")
    scope, out = run_blocks("use-results.md", [
        "stats = client.postprocess.class_statistics",
        "client.postprocess.zonal_statistics(",
        "client.preprocess.clip_to_polygon(",
        "with rasterio.open(\"out/june/water_mask.tif\")",
    ])
    # June: water in columns 0-9. August: columns 5-14, and columns 36-39 are cloud. 40 rows.
    assert "water lost: 200 pixels, gained: 200 pixels, compared: 1,440" in out
    assert (world / "districts_with_ndwi.geojson").exists()
    features = json.loads((world / "districts_with_ndwi.geojson").read_text())["features"]
    assert len(features) == 2 and all({"stat_mean", "stat_count"} <= set(f["properties"]) for f in features)
    import rasterio
    with rasterio.open(world / "out" / "london" / "water_mask_district.tif") as src:
        clipped = src.read(1)
    assert src.width == 20 and src.height == 40                              # the west half of the box only


def test_use_results_guide_rebuilds_a_report_without_a_download(world):
    """The command on the page works on an earlier run's folder and does not search or download."""
    from click.testing import CliRunner
    from pygeovision import PyGeoVision
    from pygeovision.cli.main import cli

    assert PyGeoVision().pipeline("water_bodies", bbox=BBOX, date="2024-06", output_dir="out/london").success
    from pipeline_synth import no_network
    import pytest as _pytest
    mp = _pytest.MonkeyPatch()
    try:
        hit = no_network(mp)
        res = CliRunner().invoke(cli, ["report", "out/london", "--formats", "pdf,docx", "--title", "Surface water, central London, June 2024"])
    finally:
        mp.undo()
    assert res.exit_code == 0, res.output
    assert hit["n"] == 0
    assert list((world / "out" / "london" / "deliverables").glob("*.docx"))


def test_own_model_guide_chips_and_label_quality_commands_run(world):
    """Steps 2 and 3 of the route run offline on a synthetic scene and its labels."""
    from click.testing import CliRunner
    from pygeovision.cli.main import cli

    rng = np.random.default_rng(3)
    (world / "data").mkdir(); (world / "labels").mkdir()
    write_raster(world / "data" / "scene.tif", rng.uniform(0, 1, (4, 300, 300)).astype("float32"))
    labels = np.zeros((300, 300), "uint8"); labels[50:150, 60:200] = 1
    write_raster(world / "labels" / "scene.tif", labels, dtype="uint8", nodata=255)
    page = DOCS / "guides" / "own-model.md"
    runnable = [a for a in commands(page) if a[:2] in (["ai", "chips"], ["label", "quality"])]
    assert len(runnable) == 2
    for argv in runnable:
        res = CliRunner().invoke(cli, argv)
        assert res.exit_code == 0, f"{argv}\n{res.output}"
    assert any((world / "data" / "train").rglob("*.tif"))
    images = list((world / "data" / "train").rglob("scene1*"))
    assert images, "the chips carry the prefix the page gives"


def test_the_negative_longitude_of_label_osm_needs_the_double_dash():
    """Why the guides write `-- -74.05 ...`: without it click reads the longitude as an option."""
    from click.testing import CliRunner
    from pygeovision.cli.main import cli

    bad = CliRunner().invoke(cli, ["label", "osm", "-74.05", "40.70", "-73.95", "40.80"])
    assert bad.exit_code == 2 and "No such option" in bad.output
    for page in list((DOCS / "guides").glob("*.md")) + [DOCS / "labeling" / "osm.md", DOCS / "labeling" / "index.md"]:
        for argv in commands(page):
            if argv[:2] == ["label", "osm"]:
                tail = argv[argv.index("--") + 1:] if "--" in argv else argv[2:]
                nums = [t for t in tail if re.match(r"^-?\d", t)]
                if any(n.startswith("-") for n in nums):
                    assert "--" in argv, f"{page.name}: a negative longitude needs a '--' before the box"


def test_no_report_skips_the_summary_that_pygeovision_report_needs_and_none_formats_keep_it(world):
    """The pages say: after --no-report there is nothing to rebuild from; --report-formats none keeps the summary."""
    from click.testing import CliRunner
    from pygeovision import PyGeoVision
    from pygeovision.cli.main import cli

    client = PyGeoVision()
    assert client.pipeline("water_bodies", bbox=BBOX, date="2024-06", output_dir="out/off", report=False).success
    assert not (world / "out" / "off" / "run_summary.json").exists()
    res = CliRunner().invoke(cli, ["report", "out/off"])
    assert res.exit_code == 1 and "run_summary.json" in res.output
    assert client.pipeline("water_bodies", bbox=BBOX, date="2024-06", output_dir="out/quick",
                           report_formats="none", vector_formats="none").success
    assert (world / "out" / "quick" / "run_summary.json").exists()
    assert not list((world / "out" / "quick" / "deliverables").glob("*.pdf"))
    assert CliRunner().invoke(cli, ["report", "out/quick", "--formats", "docx"]).exit_code == 0
