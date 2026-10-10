#!/usr/bin/env python
"""Check that a working tree matches the files a delivery was tested with.

    python tools/verify_install.py                 # uses MANIFEST.sha256 in the repo root
    python tools/verify_install.py --manifest path/to/MANIFEST.sha256 --root path/to/repo

Prints every file that is missing or differs (stale or locally edited) and exits 1
if there is any. Line endings are ignored (CRLF and LF hash the same), so a
checkout on another OS does not report false differences.

Why it exists: a partly copied update (some files new, some old) behaves like
neither version. One real example: `pygeovision doctor` printed two failed lines
and "5/5 components healthy" because `data/fetch.py` was the older file while
`cli/main.py` was the newer one.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


def digest(path: Path) -> str:
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def read_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        sha, _, rel = line.partition("  ")
        entries[rel.strip()] = sha.strip()
    return entries


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=here, help="repository root to check (default: this repo)")
    ap.add_argument("--manifest", type=Path, default=None, help="manifest file (default: <root>/MANIFEST.sha256)")
    args = ap.parse_args(argv)

    manifest_path = args.manifest or args.root / "MANIFEST.sha256"
    if not manifest_path.is_file():
        print(f"manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    manifest = read_manifest(manifest_path)

    missing, differ, ok = [], [], 0
    for rel, expected in sorted(manifest.items()):
        target = args.root / rel
        if not target.is_file():
            missing.append(rel)
        elif digest(target) != expected:
            differ.append(rel)
        else:
            ok += 1

    for rel in missing:
        print(f"MISSING  {rel}")
    for rel in differ:
        print(f"DIFFERS  {rel}")
    print(f"\n{ok}/{len(manifest)} files match the delivery; {len(differ)} differ, {len(missing)} missing.")
    if differ or missing:
        print("Copy the listed files from the delivery over your tree, then run this again.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
