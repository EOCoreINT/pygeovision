# Real-run fixes — what changed and how to apply it

This zip is the **whole repository** (not a partial update). Copy it over your checkout, then check you have
exactly these files:

    unzip -o pygeovision_full_tree.zip -d /path/to/your/pygeovision
    cd /path/to/your/pygeovision
    pip install -e .
    python tools/verify_install.py        # must say: N/N files match the delivery

Why the whole tree: your `doctor` output (`✗ credentials`, `✗ cache`, `5/5 healthy`) and `status` (`0 open-access`)
come from an older `data/fetch.py` and `data/providers.py` than the rest of your tree. Earlier partial zips
were only correct if every file in them was copied. `verify_install.py` lists any file that is missing or differs.

## Defects found in your logs, and the fix

| What the log showed | Cause | Fix |
|---|---|---|
| `flood_mapping`: result covers 23% of the bbox, no warning; mosaic bounds = one tile | every scene was stacked to the same `imagery/stack.tif`, so the next scene overwrote the previous and the mosaic was one file twice | one `stack_<scene>.tif` per scene; mosaic refuses the same file twice |
| 5 search hits = 2 scenes; a tile never seen; a 1 GB scene downloaded for a sliver | result cap of 5 shared by 3 providers listing the same scene; zero minimum gain | pool of 100, repeat listings dropped, a scene must add 0.5% of the AOI |
| `single-asset ... has 1 real band(s), but 2 were requested`, scene silently dropped | pygeofetch `resume` took one leftover file for a finished scene | one automatic re-download without resume; a failed scene is replaced by another covering the same area |
| `Downloading 15 asset(s)` (400 MB - 1 GB per scene) | `PyGeoVision.download(bands=...)` filtered the wrong object and never passed `bands` on | bands forwarded; only the bands needed plus `SCL`/`QA_PIXEL` are fetched (flood_mapping: 3 of 15 assets) |
| `water_coverage: 0.0000` over the Thames | Sentinel-2 scenes from 2022-01-25 have a +1000 DN offset that was never removed (+0.1 reflectance in every band) | offset removed per scene (`sentinel2_boa_add_offset`); DN 0 (no data) is NaN, not 0 reflectance |
| `change_detection` "before" had no cloud mask and a different scaling path | the search cache returns results without a download handle; they took a degraded direct-download route | pipelines never use cached searches; cache key now includes result cap and sort |
| `2024-06-01→2024-06-28` | month end hard-coded to 28 | real last day of the month (`data/_dates.py`) |
| `Search failed for collection 'SENTINEL-2-L2A' ... HTTP 400` and `sentinel_hub` auth error on every search | default provider list included providers that cannot answer | skipped by default when they cannot answer; named in `--provider` they are still used |
| every log line twice, `PyGeoFetch ready` four times | pygeofetch and the root logger both printed it; a throw-away engine was built to probe a capability | one handler wins; the existing client is probed |
| `BLOCKXSIZE can only be used with TILED=YES` (stack, clip, mosaic, align ...) | profile copied from a striped source | `writable_profile()` used by 30 raster writers |
| `reproject: ... (0.0 m/px, origin=(-2, 52))` | metres format for a degree CRS | reports `deg/px` for geographic CRS |
| `checkpoint not found: buildings.pth` | the guide's placeholder name was used | the message says it is a placeholder and which pipelines need your weights |

New pipeline statistic: `aoi_coverage_fraction` (1.0 = the downloaded scenes cover the whole bbox).

## Not done (needs the network to verify or is a larger change)

* Download volume is still whole bands of whole tiles (~65 MB per 10 m band). Reading only the bbox window from
  the cloud-optimised files would cut a 0.05° box to a few MB per scene.
* Each scene is still reprojected to EPSG:4326 over the whole tile (30-60 s) before cropping.
* The live behaviour itself (real downloads, real offsets on your scenes) has only been reproduced with synthetic
  tiles and pygeofetch's real data models, not on the network.
