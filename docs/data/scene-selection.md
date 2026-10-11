# Scene selection and mosaicking

When you give a pipeline a bounding box and a date window, PyGeoVision does more than "take the first scene". This
page describes exactly what it does, so you can predict results, read the logs, and decide when to intervene.

## The problem

- A scene is a **fixed tile or swath** (Sentinel-2: 100 × 100 km tiles; Landsat: ~185 km paths/rows). Your box may
  straddle two or four of them.
- A scene's cloud percentage is for the **whole scene**. A scene listed at 5 % can be cloudy over your 10 km box.
- A tilted swath has empty corners, so a scene whose *bounding box* covers your box can still leave a gap.
- The same acquisition is listed by several providers, and not every copy can be downloaded the same way.

## What happens, step by step

1. **Search a wide pool.** Up to 100 candidates are requested (sorted by scene cloud cover, lowest first) from the
   providers, with the cache bypassed.
2. **Remove duplicates.** Listings of one acquisition — identified by *(MGRS tile or Landsat path/row, date)* — are
   merged. When several providers list it, one that can be downloaded without an account (Planetary Computer,
   Copernicus, USGS) is preferred. The other listings are kept as alternates.
3. **Plan coverage.** If the best scene's footprint covers your box, it is used alone. Otherwise a **greedy set cover**
   picks scenes: at each step, the scene adding the most still-uncovered area wins (among scenes within 5 % of the
   best gain, the clearer one wins), up to four scenes. A scene adding less than 0.5 % of the box is not worth
   downloading.
4. **Download only what is needed** — the bands you asked for plus the cloud layer — and prepare **each scene on
   its own** (stack, scale to reflectance, mask cloud; see [Radiometry](radiometry.md)).
5. **Recover from failures.** A scene that cannot be downloaded or prepared is retried from another provider's listing
   of the same scene, then replaced by another scene covering the same part of the box.
6. **Close real gaps.** Footprints are boxes, but data lies in tilted swaths. The data actually present is measured on
   the prepared rasters; while more than 0.2 % of the box has no imagery, the candidate covering most of the gap is
   added (up to four extra scenes). A scene that adds no coverage is dropped.
7. **Mosaic** the corrected scenes (`rasterio.merge`; the first valid pixel wins) and **crop** to your box.
8. **Check clarity.** The share of the box that is neither cloud, shadow nor empty is measured. If it is below 5 %,
   other scenes of the date window are tried (up to three rounds) and the clearer result is kept.
9. **Record the outcome** in the result: `aoi_coverage_fraction`, the scenes used and failed, and the clear fraction.

## Reading the outcome

```python
result = client.pipeline("water_bodies", bbox=bbox, date="2024-06")
print(result.stats["aoi_coverage_fraction"])         # share of your box with imagery, 0–1
for entry in result.metadata["input_coverage"]:
    print(entry["image"], entry["scenes_used"], entry["scenes_failed"], entry.get("clear_fraction"))
```

| Value | Meaning | What to do |
|---|---|---|
| `aoi_coverage_fraction` = 1.0 | Imagery exists over your whole box | Nothing |
| 0.6 – 0.99 | Part of the box has no imagery (swath edge, failed tile) | Quote areas for the *covered* part, or choose a window that has a pass over the missing part |
| < 0.6 | Little of your box was seen | Widen the dates; shrink the box; check the log for `scenes_failed` |
| `clear_fraction` < 0.3 | Mostly cloud, shadow or empty | Widen the date window, or use radar ([`sar_flood_mapping`](../pipelines/catalog/sar_flood_mapping.md)) |

A result built from partial coverage logs a warning ("the *before* image covers only 23 % of the requested bbox").
When the AOI coverage must be complete, run the search layer yourself with `require_full_coverage=True` (an internal
option of the pipeline base class) or check `aoi_coverage_fraction` before using numbers.

## Bi-temporal pairs

For two-date pipelines the *before* and *after* images are prepared independently and then **aligned to a common
grid**: both are cropped to their real geographic intersection and resampled onto one grid at the finer of the two
resolutions.

## Doing it yourself

```python
results = client.search(bbox=bbox, date_range=("2024-06-01", "2024-06-30"),
                        collections=["sentinel-2-l2a"], cloud_cover_max=30, max_results=100)
chosen = client.select_covering_scenes(results, bbox, max_scenes=4)   # greedy cover, fewest & clearest
downloads = client.download(chosen, output_dir="./data", bands=["B02", "B03", "B04", "B08", "SCL"])
```

`select_covering_scenes` is the same set-cover as step 3. Neither it nor a footprint can see cloud over *your* box:
check the clear share of the prepared raster.

## Tuning

| Want | Do |
|---|---|
| Fewer, larger scenes | Smaller box |
| A specific scene | `client.download([scene])` and prepare it with [`prepare_stack`](radiometry.md) |
| Only one provider | `--provider planetary_computer` (CLI) / `providers=[…]` |
| Stricter clouds | `cloud_cover_max=` (pipeline parameter) — but see the tip in [Concepts](../getting-started/concepts.md#dates) |
