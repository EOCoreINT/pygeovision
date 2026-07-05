#!/usr/bin/env python
"""
PyGeoVision — Live Integration Smoke Test
==========================================

A real, end-to-end test against live satellite data providers. Run this
after installing PyGeoVision, after pulling new changes, or whenever
you want to confirm search → download → GeoAI works against your actual
network and credentials.

This is NOT the pytest unit suite (see tests/ — 208 tests, fully mocked,
no network, runs in CI). This script makes real HTTP calls and is meant
to be run manually:

    python test_live.py
    python test_live.py --bbox -74.1 40.6 -73.7 40.9 --providers planetary_computer aws_earth
    python test_live.py --skip-download --skip-geoai
    python test_live.py --clear-cache --fail-fast

Each step is independent: a failed search skips download, a failed/empty
download skips the GeoAI step, and so on — you get a full summary at the
end instead of a stack trace on step one.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path
from typing import Any, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────
# Step runner — labeled sections with timing, pass/fail tracking, and
# graceful continuation (one failed step doesn't abort the whole script).
# ─────────────────────────────────────────────────────────────────────────

class Step:
    results: List[Tuple[str, bool, float, str]] = []
    fail_fast: bool = False

    def __init__(self, name: str) -> None:
        self.name = name
        self.start = 0.0

    def __enter__(self) -> "Step":
        print(f"\n{'─' * 70}")
        print(f"▶ {self.name}")
        print("─" * 70)
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        duration = time.time() - self.start
        if exc_type is not None:
            detail = f"{exc_type.__name__}: {exc_val}"
            print(f"✗ FAILED ({duration:.1f}s): {detail}")
            traceback.print_exception(exc_type, exc_val, exc_tb, limit=4)
            Step.results.append((self.name, False, duration, detail))
            return not Step.fail_fast  # True = swallow & continue, False = re-raise
        print(f"✓ PASSED ({duration:.1f}s)")
        Step.results.append((self.name, True, duration, ""))
        return True

    @classmethod
    def summary(cls) -> bool:
        print(f"\n{'═' * 70}")
        print("  TEST SUMMARY")
        print("═" * 70)
        for name, passed, duration, detail in cls.results:
            mark = "✓" if passed else "✗"
            line = f"  {mark}  {name:<38} {duration:>6.1f}s"
            if detail:
                line += f"   {detail}"
            print(line)
        passed_count = sum(1 for _, p, _, _ in cls.results if p)
        total = len(cls.results)
        print("═" * 70)
        print(f"  {passed_count}/{total} steps passed")
        print("═" * 70)
        return passed_count == total


# ─────────────────────────────────────────────────────────────────────────
# Individual test steps — each one validates its own inputs/outputs
# rather than assuming the previous step produced something usable.
# ─────────────────────────────────────────────────────────────────────────

def test_status(client: Any) -> dict:
    print(repr(client))
    status = client.status()
    assert status, "status() returned empty"
    pgf = status.get("pygeofetch", {})
    ga = status.get("geoai", {})
    print(f"  pygeofetch: available={pgf.get('available')}  version={pgf.get('version')}")
    print(f"  geoai:      available={ga.get('available')}  version={ga.get('version', '')}")
    return status


def test_clear_cache(client: Any) -> None:
    client.clear_cache()
    print("  Local search cache cleared.")


def test_search(
    client: Any,
    bbox: Tuple[float, float, float, float],
    date_range: Tuple[str, str],
    providers: List[str],
    cloud_max: float,
    max_results: int,
    use_cache: bool,
) -> list:
    results = client.search(
        bbox=bbox,
        date_range=date_range,
        providers=providers,
        cloud_cover_max=cloud_max,
        max_results=max_results,
        sort_by="cloud_cover",
        use_cache=use_cache,
    )
    assert results, (
        "search() returned zero results — check bbox / date_range / providers. "
        f"Tried bbox={bbox}, date_range={date_range}, providers={providers}"
    )

    print(f"  Found {len(results)} scene(s):")
    for r in results[:10]:
        print(f"    {r}")
    if len(results) > 10:
        print(f"    ... and {len(results) - 10} more")

    # Catch the "no assets, no satellite_data" problem HERE — as a warning —
    # rather than three steps later as a cryptic rasterio traceback.
    undownloadable = [r.id for r in results if not r.assets and r.satellite_data is None]
    if undownloadable:
        print(
            f"  ⚠ {len(undownloadable)}/{len(results)} result(s) have neither "
            f"assets nor satellite_data and will NOT be downloadable: "
            f"{undownloadable[:3]}{'...' if len(undownloadable) > 3 else ''}"
        )

    return results


def test_download(
    client: Any,
    results: list,
    output_dir: str,
    parallel: int,
    post_process: List[str],
    n_items: int,
) -> list:
    candidates = results[:n_items]
    downloads = client.download(
        candidates,
        output_dir=output_dir,
        parallel=parallel,
        verify_checksum=True,
        post_process=post_process,
    )
    assert downloads, "download() returned an empty list"

    for d in downloads:
        print(f"  {d}")

    succeeded = [d for d in downloads if d.success and d.path is not None]
    failed = [d for d in downloads if not d.success]

    if failed:
        print(f"  ⚠ {len(failed)}/{len(downloads)} download(s) failed:")
        for d in failed:
            print(f"      {d.scene_id}: {d.error}")

    assert succeeded, (
        f"All {len(downloads)} download(s) failed — see errors above. "
        "Nothing usable for the GeoAI step."
    )
    print(f"  {len(succeeded)}/{len(downloads)} download(s) succeeded")
    return succeeded


def test_geoai_segment(client: Any, downloads: list) -> None:
    target = next(
        (d for d in downloads if d.success and d.path and Path(d.path).exists()),
        None,
    )
    assert target is not None, (
        "No successfully downloaded file with a valid, existing path to segment"
    )

    print(f"  Segmenting buildings in: {target.path}")
    out_vector = str(Path(target.path).with_suffix("")) + "_buildings.geojson"
    client.geoai.segment.buildings(
        target.path,
        output_vector=out_vector,
        confidence_threshold=0.5,
    )
    assert Path(out_vector).exists(), f"Expected output was not created: {out_vector}"
    print(f"  ✓ Buildings vector written → {out_vector}")


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="PyGeoVision live integration smoke test (real network calls)."
    )
    parser.add_argument(
        "--bbox", nargs=4, type=float, default=[-74.1, 40.6, -73.7, 40.9],
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        help="Bounding box in WGS84. Default: NYC area.",
    )
    parser.add_argument("--start-date", default="2024-06-01")
    parser.add_argument("--end-date", default="2025-06-30")
    parser.add_argument(
        "--providers", nargs="+", default=["planetary_computer", "aws_earth"],
    )
    parser.add_argument("--cloud-max", type=float, default=20.0)
    parser.add_argument("--max-results", type=int, default=20)
    parser.add_argument("--download-count", type=int, default=3,
                         help="How many top search results to download.")
    parser.add_argument("--output-dir", default="./sentinel2_test/")
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument(
        "--post-process", nargs="+", default=["reproject:EPSG:4326", "cog"],
    )
    parser.add_argument(
        "--use-cache", action="store_true",
        help="Allow the 1-hour search cache (off by default for testing — "
             "you almost always want fresh results with assets here).",
    )
    parser.add_argument("--clear-cache", action="store_true",
                         help="Clear the local search cache before running.")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-geoai", action="store_true")
    parser.add_argument(
        "--fail-fast", action="store_true",
        help="Stop at the first failed step instead of continuing and "
             "printing a summary.",
    )
    args = parser.parse_args()

    Step.results.clear()
    Step.fail_fast = args.fail_fast

    import pygeovision as pgv
    client = pgv.PyGeoVision()

    status: dict = {}
    results: list = []
    downloads: list = []

    with Step("System status") as s:
        status = test_status(client)

    if args.clear_cache:
        with Step("Clear search cache") as s:
            test_clear_cache(client)

    with Step("Search satellite providers") as s:
        results = test_search(
            client,
            bbox=tuple(args.bbox),
            date_range=(args.start_date, args.end_date),
            providers=args.providers,
            cloud_max=args.cloud_max,
            max_results=args.max_results,
            use_cache=args.use_cache,
        )

    if args.skip_download:
        print("\n(skipping download — --skip-download)")
    elif not results:
        print("\n(skipping download — no search results)")
    else:
        with Step("Download + post-process") as s:
            downloads = test_download(
                client, results, args.output_dir, args.parallel,
                args.post_process, args.download_count,
            )

    geoai_available = status.get("geoai", {}).get("available", False)
    if args.skip_geoai:
        print("\n(skipping GeoAI — --skip-geoai)")
    elif not downloads:
        print("\n(skipping GeoAI — no successful downloads to segment)")
    elif not geoai_available:
        print('\n(skipping GeoAI — geoai-py not installed: pip install "pygeovision[geoai]")')
    else:
        with Step("GeoAI: segment buildings") as s:
            test_geoai_segment(client, downloads)

    all_passed = Step.summary()
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
