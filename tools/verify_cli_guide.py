#!/usr/bin/env python
"""Execute the ``# [offline]`` blocks of CLI_TESTING_GUIDE.md, in order, in a scratch directory.

    python tools/verify_cli_guide.py                # all offline blocks
    python tools/verify_cli_guide.py --keep         # keep the scratch directory for inspection
    python tools/verify_cli_guide.py --from 12      # start at block 12 (earlier outputs must exist)
    python tools/verify_cli_guide.py --coverage     # only check every command is in the guide
    python tools/verify_cli_guide.py --write-checklist   # rewrite the table in section 19, then exit

It also checks that every leaf command of the CLI appears in the guide (a command added without an example
makes the run fail), and ``--write-checklist`` regenerates the pass/fail table in section 19.

Each block is run by ``bash`` with ``set -e`` after a prelude taken from the guide's own helper block
(the ``export`` / ``mkdir`` / ``expect_fail`` lines), so what runs is exactly what the guide shows.
``pip install`` lines are skipped. Exit status is non-zero if any block fails.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "CLI_TESTING_GUIDE.md"
FENCE = re.compile(r"```bash\n(.*?)```", re.S)


def blocks(text: str):
    for m in FENCE.finditer(text):
        body = m.group(1)
        first = body.lstrip().splitlines()[0] if body.strip() else ""
        yield (re.match(r"#\s*\[(\w+)\]", first) or [None, None])[1], body


def leaf_commands() -> list[str]:
    """Space-joined paths of every leaf command of the click tree, e.g. ``ai change``."""
    import click

    from pygeovision.cli.main import cli
    out: list[str] = []

    def walk(cmd, path):
        if isinstance(cmd, click.Group):
            for name, sub in sorted(cmd.commands.items()):
                walk(sub, [*path, name])
        else:
            out.append(" ".join(path))
    walk(cli, [])
    return out


# Commands whose single entry in the CLI tree covers variants that are not all runnable offline.
NOTES = {
    "channel": "33 ready + 7 needs-weights pipelines: all 40 run offline via tools/offline_pipeline_smoke.py; real data needs network",
    "report": "rebuilds PDF/Word/Excel, maps and vectors from a saved run; runs offline on the smoke-test outputs",
    "ai segment": "water, custom offline; buildings, sam need network + weights",
    "ai detect": "generic/ships/cars need `ultralytics` + a YOLO download; custom refusal offline",
    "ai classify": "scene needs CLIP weights; land-cover needs network",
    "ai change": "spectral_diff and the three network methods (with wiring checkpoints) offline",
    "data search": "network", "data download": "network",
    "data auth test": "open providers: network; others: stored credentials",
    "data pipeline run": "network (search step); structure/refusals offline",
    "data pipeline history": "after a run; needs network for a run that finds scenes",
    "label osm": "network (Overpass API)",
    "vlm caption": "Hugging Face download", "vlm query": "Hugging Face download", "vlm search": "Hugging Face download",
    "cloud deploy-aws": "cloud account (offline: clean failure)", "cloud deploy-gcp": "cloud account (offline: clean failure)",
    "models download": "no-weights entry offline; Hugging Face entries need network",
    "pipeline run": "dry-run and refusals offline; a real run needs network + your checkpoint",
}


def coverage(text: str):
    """{command: (offline_success, offline_refusal_only, other_tags)} for every leaf command."""
    lines: list[tuple[str, str]] = []
    for tag, body in blocks(text):
        joined = re.sub(r"\\\n\s*", " ", body)
        lines += [(tag, ln) for ln in joined.splitlines()]
    res = {}
    for cmd in leaf_commands():
        pat = re.compile(rf"\bpygeovision {re.escape(cmd)}(\s|$)")
        hits = [(tag, ln) for tag, ln in lines if pat.search(ln)]
        ok = any(tag == "offline" and "expect_fail" not in ln for tag, ln in hits)
        refuse = any(tag == "offline" and "expect_fail" in ln for tag, ln in hits)
        other = sorted({tag for tag, _ in hits if tag and tag != "offline"})
        res[cmd] = (ok, refuse, other, bool(hits))
    return res


def checklist(text: str) -> str:
    rows = ["| ☐ | Command | Offline | Needs | Notes |", "|---|---|---|---|---|"]
    for cmd, (ok, refuse, other, _seen) in coverage(text).items():
        mark = "●" if ok else ("◐" if refuse else "○")
        needs = ", ".join(other) if other else "—"
        rows.append(f"| ☐ | `pygeovision {cmd}` | {mark} | {needs} | {NOTES.get(cmd, '')} |")
    legend = ("`●` ran successfully offline for this guide · `◐` only its refusal/failure path ran offline "
              "(the real run needs the tag shown) · `○` not run offline.\n")
    return legend + "\n" + "\n".join(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--from", dest="start", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=1200, help="seconds per block")
    ap.add_argument("--show", action="store_true", help="print each block's output")
    ap.add_argument("--coverage", action="store_true", help="only check that every command is in the guide")
    ap.add_argument("--write-checklist", action="store_true", help="rewrite the section-19 table and exit")
    args = ap.parse_args()

    text = GUIDE.read_text()
    missing = [c for c, (_, _, _, seen) in coverage(text).items() if not seen]
    if missing:
        print(f"commands with no example in the guide: {missing}")
    if args.write_checklist:
        new = re.sub(r"(<!-- CHECKLIST:BEGIN -->)\n?.*?\n?(<!-- CHECKLIST:END -->)",
                     lambda m: m.group(1) + "\n" + checklist(text) + "\n" + m.group(2), text, flags=re.S)
        GUIDE.write_text(new)
        print(f"checklist written ({len(leaf_commands())} commands)")
        return 1 if missing else 0
    if args.coverage:
        print(f"{len(leaf_commands()) - len(missing)}/{len(leaf_commands())} commands covered")
        return 1 if missing else 0

    all_blocks = list(blocks(text))
    offline = [(i, b) for i, (tag, b) in enumerate(all_blocks) if tag == "offline"]
    helper = next(b for _, b in offline if "expect_fail()" in b)
    prelude = "\n".join(ln for ln in helper.splitlines()
                        if ln.startswith(("export ", "mkdir ", "expect_fail()")))

    work = Path(tempfile.mkdtemp(prefix="pgv_guide_"))
    for name in ("tools", "pygeovision"):                     # the guide refers to repo-relative paths
        (work / name).symlink_to(ROOT / name)
    print(f"scratch directory: {work}\n")
    failed = []
    for n, (_idx, body) in enumerate(offline):
        if n < args.start:
            continue
        script = "\n".join(ln for ln in body.splitlines() if not ln.startswith("pip install"))
        t0 = time.time()
        proc = subprocess.run(["bash", "-c", f"set -e\n{prelude}\n{script}"], cwd=work,
                              capture_output=True, text=True, timeout=args.timeout,
                              env={**os.environ, "PYTHONWARNINGS": "ignore"})
        ok = proc.returncode == 0
        first = next((ln for ln in script.splitlines() if ln.strip() and not ln.startswith("#")), "")[:70]
        print(f"{'OK  ' if ok else 'FAIL'} block {n:>2}  {time.time() - t0:5.1f}s  {first}")
        if args.show:
            print("      " + "\n      ".join(proc.stdout.strip().splitlines()))
        if not ok:
            failed.append(n)
            print("      " + "\n      ".join((proc.stdout + proc.stderr).strip().splitlines()[-12:]))
    print(f"\n{len(offline) - len(failed)}/{len(offline)} offline blocks passed"
          + (f"; failed: {failed}" if failed else ""))
    failed += ["coverage"] if missing else []
    if not args.keep and not failed:
        import shutil
        shutil.rmtree(work, ignore_errors=True)
    else:
        print(f"kept {work}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())