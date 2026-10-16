"""Capture one offline example run per pipeline for the documentation.

    PYTHONPATH=. python tools/capture_pipeline_examples.py

Runs every pipeline through the real ``pygeovision channel`` command with the network replaced by the synthetic
data of ``tools/offline_pipeline_smoke.py`` and writes ``docs/_data/pipeline_examples.json``: the command, the
console output, the files written, the statistics and the layer tables, with local paths removed.

The numbers describe the *synthetic* scene, not a place. The pipeline pages say so wherever they show them.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys
import tempfile
import warnings

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import offline_pipeline_smoke as m  # noqa: E402
from click.testing import CliRunner  # noqa: E402

from pygeovision.cli.main import cli  # noqa: E402

DEST = ROOT / "docs" / "_data" / "pipeline_examples.json"


def _clean(obj, work: str):
    """Drop paths: absolute path strings become their file name, keys ending in ``_path`` vanish."""
    if isinstance(obj, dict):
        return {k: _clean(v, work) for k, v in obj.items() if not k.endswith("_path") and k != "mask_path"}
    if isinstance(obj, list):
        return [_clean(v, work) for v in obj]
    if isinstance(obj, str) and obj.startswith("/"):
        return pathlib.Path(obj).name
    if isinstance(obj, float):
        return round(obj, 5)
    return obj


def main() -> int:
    work = pathlib.Path(tempfile.mkdtemp(prefix="pgv_examples_"))
    m.install_offline_patches(work)
    runner = CliRunner()
    shown = {line.split()[2]: line for line in _commands()}
    result: dict[str, dict] = {}
    for name in m.SPECS:
        out = work / "out" / name
        cmd = ["channel", name, *m.BBOX_ARGS, "-o", str(out)]
        cmd += m.cli_args(name, work)
        r = runner.invoke(cli, cmd)
        if r.exit_code != 0:
            print("FAILED", name, r.output[-400:])
            return 1
        console = r.output.replace(str(out), f"./out/{name}")
        console = re.sub(r"\S*pgv_examples_\S*", "<tmp>", console)
        console = "\n".join(ln for ln in console.splitlines() if "INFO" not in ln and "WARNING" not in ln)
        files = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
        summary = json.loads((out / "run_summary.json").read_text())
        layers = []
        for lay in summary.get("layers", []):
            entry = {"key": lay["key"], "title": lay["title"], "kind": lay["kind"],
                     "file": pathlib.Path(lay["file"]).name}
            if lay.get("classes"):
                entry["classes"] = [{k: c.get(k) for k in ("value", "name", "pixels", "area_ha", "share")}
                                    for c in lay["classes"]]
            if lay.get("summary"):
                entry["summary"] = {k: lay["summary"].get(k) for k in ("min", "max", "mean", "median", "valid_fraction")}
            layers.append(entry)
        result[name] = {
            "command": shown.get(name, ""),
            "console": console.strip(),
            "files": files,
            "title": summary.get("title"),
            "period": summary.get("period"),
            "aoi_area_km2": summary.get("aoi_area_km2"),
            "stats": _clean(summary.get("stats", {}), str(work)),
            "layers": _clean(layers, str(work)),
        }
        print("ok", name, flush=True)
    DEST.parent.mkdir(parents=True, exist_ok=True)
    DEST.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print("wrote", DEST)
    return 0


def _commands() -> list[str]:
    """The equivalent real-data commands, as printed by ``offline_pipeline_smoke.py --commands``."""
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    argv = sys.argv
    sys.argv = ["x", "--commands"]
    try:
        with redirect_stdout(buf):
            try:
                m.main()
            except SystemExit:
                pass
    finally:
        sys.argv = argv
    return [ln for ln in buf.getvalue().splitlines() if ln.startswith("pygeovision channel")]


if __name__ == "__main__":
    sys.exit(main())
