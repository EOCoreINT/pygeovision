# AOI Coverage & Band Selection

Every pipeline downloads through the same real, shared path
(`_search_and_download`), which does two things worth understanding.

## AOI coverage: real mosaicking when one scene isn't enough

If your requested bounding box straddles two scene footprints (a real,
common situation — a Sentinel-2 MGRS tile boundary, a Landsat WRS-2
path/row edge), a single downloaded scene may not fully cover what you
asked for.

`pygeovision` checks this using each candidate scene's real footprint
*before* downloading. If one scene covers the full request, it's used
directly (no extra cost). If not, a real greedy set-cover selects the
minimal set of scenes needed, downloads and radiometrically-corrects
each one **individually** (correction happens before mosaicking, not
after, to avoid a real sensor-calibration discontinuity at the seam),
then mosaics them with `rasterio.merge` before cropping to your exact
requested bbox.

```python
result = pipeline.run(
    bbox=(-1.5, 50.5, 1.5, 52.5),  # spans multiple tiles
    output_dir="./output",
    date="2024-06",
    require_full_coverage=True,  # fail clearly instead of returning less than requested
)
```

If `require_full_coverage=False` (the default) and even every
available real scene can't fully cover the request, `pygeovision`
proceeds with the best real partial coverage and logs a clear warning
— it does not silently return a smaller-than-requested result without
telling you.

## Band selection: a real, confirmed fix

Requesting specific bands (e.g. `bands=("red", "green", "blue")`)
previously downloaded every band in the scene regardless — the
underlying `client.download()` had a real `bands=` parameter that was
never used.

The fix required checking [pygeofetch's real
documentation](https://pygeofetch.readthedocs.io/en/latest/core-features/download.html)
directly: different providers use genuinely different band-key
conventions for the same bands (Earth Search/AWS use semantic names
like `"red"` directly as real asset keys; Planetary Computer uses
`"B04"`-style codes). `pygeovision` now expands each requested generic
band name into every real, known alias across Sentinel-2 and both
Landsat mission families before calling `download()`, so the real
canonical code is always included regardless of which provider you're
using.

## Three layers of defense against a wrong band count

A real production failure — a model built for 3 channels receiving 13
— led to three independent, real checks, so a mismatch is caught at
the earliest possible point rather than crashing deep inside model
inference:

1. Immediately after downloading a single-asset scene, before any
   further processing.
2. A final check right before `_search_and_download` returns anything.
3. An explicit check immediately before model construction in
   `_run_model`, with a real pre-flight dummy forward pass.

```{note}
This same check is duplicated in `pygeovision.inference.tiled` — a
separate, CLI-reachable `TiledInference` class that turned out not to
share code with the one described above. Both are now fixed and
verified independently.
```
