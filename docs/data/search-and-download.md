# Search and download

Use these when you want **scenes** rather than a finished product — to inspect what exists, to feed your own
analysis, or to build a training set. Pipelines call the same functions internally.

## Why this layer exists

Imagery is spread over dozens of catalogues with different APIs, collection names and login schemes.
PyGeoVision puts one interface in front of 22 of them (through [pygeofetch](https://pygeofetch.readthedocs.io/)),
and adds the parts that matter for correct analysis: provider fallback, duplicate removal, band-selective download,
and checks that what arrived is complete and georeferenced.

## Searching

### From Python

```python
import pygeovision as pgv

client = pgv.PyGeoVision()
scenes = client.search(
    bbox=(-0.15, 51.47, -0.10, 51.52),           # min_lon, min_lat, max_lon, max_lat
    date_range=("2024-06-01", "2024-06-30"),     # YYYY-MM-DD, inclusive
    collections=["sentinel-2-l2a"],
    cloud_cover_max=20,
    max_results=20,
    sort_by="cloud_cover", sort_order="asc",
)
for s in scenes[:5]:
    print(s.id, s.date, s.provider, s.cloud_cover, s.bbox)
```

`client.search` is `client.data.search`. Every argument:

| Argument | Default | Meaning |
|---|---|---|
| `bbox` | – | Area, `(min_lon, min_lat, max_lon, max_lat)` in EPSG:4326 |
| `date_range` | – | `(start, end)` as `YYYY-MM-DD` strings |
| `collections` | `None` | STAC collection ids, e.g. `sentinel-2-l2a`, `landsat-c2-l2`, `sentinel-1-rtc`, `cop-dem-glo-30`. Also used to *choose providers* when `providers` is not given |
| `providers` | `None` | Provider ids to query (see [Providers](providers.md)); aliases `pc`, `aws`, `cdse`, `asf` are accepted. Naming a provider always keeps it, even without credentials |
| `satellite` | `None` | Shortcut such as `sentinel-2`, `sentinel-1`, `landsat`, `modis`, `naip`, `dem`, `planet`; picks the providers that carry it |
| `cloud_cover_max` | `30.0` | Maximum scene-wide cloud percentage |
| `max_results` | `100` | Cap on returned scenes (after sorting) |
| `sort_by`, `sort_order` | `datetime`, `desc` | `datetime`, `cloud_cover` or `score`; `asc` or `desc` |
| `processing_level` | `None` | e.g. `L2A`; passed to providers that support it |
| `resolution_range` | `None` | `(min_m, max_m)`; scenes whose resolution is known and outside are dropped |
| `cql2_filter` | `None` | A CQL2 text filter for STAC providers, e.g. `"eo:cloud_cover < 10 AND s2:mgrs_tile = '30UXC'"` |
| `on_provider_failure` | `"skip"` | `"skip"` carries on when one provider fails; `"raise"` stops |
| `timeout` | `120` | Seconds per request |
| `use_cache` | `True` | Reuse an identical search from the last hour (see below) |
| `geometry_file` | `None` | GeoJSON polygon for true polygon filtering instead of the box |

### How providers are chosen

1. Providers you list are used as given.
2. Otherwise `satellite=` selects them (`sentinel-2` → Planetary Computer, Copernicus, AWS, Element 84, Sentinel Hub…).
3. Otherwise `collections=` does (`sentinel-2-l2a` → the same set; `landsat-c2-l2` → Planetary Computer, USGS, AWS…).
4. Otherwise the three **open** providers are searched: `planetary_computer`, `aws_earth`, `element84`.

When you do not name providers, those that cannot answer are quietly left out — providers that need credentials you
have not stored, and Copernicus when you ask for a STAC-style collection id its catalogue does not have — so one
unusable provider does not make every search noisy.

### If the main route finds nothing

When the pygeofetch search returns zero results, or fails, the search falls back to a direct STAC query
(`pystac-client`) against Planetary Computer, AWS Earth Search or Copernicus. You get a result either way; the log
says which route found the scenes.

### What you get back: `SearchResult`

| Field | Meaning |
|---|---|
| `id` | Scene id, e.g. `S2B_MSIL2A_20240612T105629_N0510_R094_T30UXC_20240612T133017` |
| `provider`, `satellite`, `collection` | Where it came from and what it is |
| `datetime` / `.date` | Acquisition time (ISO) / its `YYYY-MM-DD` part |
| `cloud_cover` | Scene-wide percentage (or `None`, e.g. radar) |
| `bbox` | Footprint bounding box — used for coverage planning *before* anything is downloaded |
| `assets` | Downloadable files by name (bands, thumbnails, metadata) |
| `properties` | Everything else the catalogue says (`gsd`, tile id, orbit…) |
| `.is_sar` | `True` for Sentinel-1, PALSAR and similar |
| `.resolution_m` | From properties, otherwise a known value for the sensor (Sentinel-2 10 m, Landsat 30 m…) |

**Scene-wide cloud cover is not cloud cover over your box.** A scene listed at 8 % can be fully cloudy over a small AOI.
Pipelines therefore check the clear share over your own area and try other scenes when it is too low
([Scene selection](scene-selection.md)).

### Caching

Searches are cached for one hour in the cache directory (`client.cache_stats()`, `pygeovision data cache stats`).
The key includes the box, dates, providers, cloud limit, collections, `max_results` and sort order, so a different
request never receives another's answer. Change the lifetime with `client.data.set_cache_ttl(seconds)`, prune with
`client.data.prune_cache(max_size_gb=10)`, empty with `pygeovision data cache clear` (`--provider`, `--older-than 7d`,
`--dry-run`). Pipelines always bypass the cache.

### From the command line

```bash
pygeovision data search --bbox -0.15 51.47 -0.10 51.52 --date 2024-06 \
    --collections sentinel-2-l2a --cloud-max 10 --max-results 10
```

`--date` accepts `YYYY-MM` or `YYYY-MM-DD/YYYY-MM-DD`; or use `--start-date` and `--end-date`. `--format table|json|ids`
changes the output; `-o scenes.geojson` saves results for the next step.

:::{list-table} Options of `pygeovision data search`
:header-rows: 1
:widths: 28 72
* - Option
  - Meaning
* - `--bbox MIN_LON MIN_LAT MAX_LON MAX_LAT`
  - Required. Four numbers separated by spaces
* - `--date`, `--start-date`, `--end-date`
  - Month, range, or explicit start and end
* - `--providers`
  - Comma-separated ids, e.g. `planetary_computer,copernicus`
* - `--satellite`, `--collections`
  - Shortcut or STAC collection ids
* - `--cloud-max`
  - Maximum cloud cover, default 30
* - `--max-results`, `--sort-by`
  - Default 50; `datetime`, `cloud_cover`, `score` or `satellite`
* - `-o`, `--format`
  - Save as GeoJSON; print as `table`, `json` or `ids`
:::

## Downloading

```python
downloads = client.download(
    scenes[:2],
    output_dir="./data",
    bands=["B02", "B03", "B04", "B08", "SCL"],      # only these assets
    post_process=["unzip", "reproject:EPSG:4326"],
    parallel=4,
)
for d in downloads:
    print(d.scene_id, d.success, d.size_mb, d.path)
    print(d.asset_paths)          # every file that arrived, by asset name
```

| Argument | Default | Meaning |
|---|---|---|
| `items` | – | One `SearchResult` or a list |
| `output_dir` | `./data` | Destination |
| `bands` | `None` | Download only these assets. For analysis, name the bands you need **and** the cloud layer (`SCL` for Sentinel-2, `QA_PIXEL` for Landsat). A full Sentinel-2 scene is about 1 GB; a handful of bands is a fraction of that |
| `post_process` | `None` | Chain: `unzip`, `reproject:EPSG:4326`, `compress:lzw`, `cog`, `resample:10`, `clip:area.geojson`, `merge`, `pan-sharpen`, `ndvi`, `ndwi`, `atmospheric:sen2cor` |
| `parallel` | `4` | Concurrent downloads |
| `verify_checksum` | `True` | Verify against the catalogue's checksum when it provides one |
| `resume` | `True` | Continue partial files and skip finished scenes |
| `retry_attempts` | `5` | Retries per file |
| `bandwidth_limit_mb` | `None` | Throttle, MB/s |
| `on_failure` | `"skip"` | `skip`, `abort` or `retry` |
| `overwrite` | `False` | Re-download even if present |
| `notify_webhook` | `None` | URL notified when finished |

Each scene returns a `DownloadResult`: `scene_id`, `provider`, `path` (one representative file), `asset_paths`
(**all** files by asset name), `success`, `bytes_downloaded`, `duration_seconds`, `checksum_verified`, `error`,
`post_process_steps`.

```{warning}
`resume=True` trusts a file that has the right name. A download interrupted at the very end can look finished. If a
scene fails later with a "band count" or "cannot read" error, re-download it with `overwrite=True, resume=False`.
Pipelines do this automatically once before giving up.
```

### From the command line

```bash
pygeovision data search --bbox -0.15 51.47 -0.10 51.52 --date 2024-06 -o scenes.geojson
pygeovision data download --from-search scenes.geojson -o ./data --parallel 4 \
    --post-process unzip,reproject:EPSG:4326,compress:lzw,cog --max-items 2
```

or search and download in one step: `pygeovision data download --bbox … --date 2024-06 --providers planetary_computer`.
Options: `--parallel`, `--verify-checksum`, `--resume`, `--retry`, `--bandwidth-limit`, `--on-failure skip|abort|retry`,
`--max-items`.

## What to do when…

| You see | Likely cause | Do this |
|---|---|---|
| Zero results | Box in the wrong order, dates in the future, `cloud_cover_max` too low | Check `min_lon < max_lon`; widen the dates; raise `cloud_cover_max`; try `use_cache=False` |
| Results, but all from one provider | Others need credentials or lack the collection | `pygeovision data providers`; add credentials ([Authentication](authentication.md)) |
| Same scene listed 2–3 times | Several providers carry it | Normal. `client.pipeline` de-duplicates; do it yourself by `(tile, date)` |
| `HTTP 403` on a Landsat file from AWS | Requester-pays bucket | Use Planetary Computer or USGS for Landsat (the default order prefers them) |
| Download "succeeds" but the file is tiny | Provider returned an error page | `d.error`, `d.size_mb`; re-download with `resume=False, overwrite=True` |
