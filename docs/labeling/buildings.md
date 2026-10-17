# Building labels

**Classes:** `MicrosoftBuildingsLabeler`, `GoogleBuildingsLabeler` in `pygeovision.labeling` ·
**Client:** `client.labeling.microsoft_buildings(...)`, `client.labeling.google_buildings(...)`

## Why use them

Two companies have published building outlines for most of the world, made by running a model over satellite imagery:

| | Microsoft Global ML Buildings | Google Open Buildings v3 |
|---|---|---|
| Coverage | Worldwide (over a billion footprints) | Africa, South and Southeast Asia, Latin America, the Caribbean. **Not** Europe, North America, Australia or China |
| Imagery behind it | Bing Maps (Maxar, Airbus, Vexcel, IGN France) | 2016–2022 satellite imagery |
| Confidence field | Per footprint in many regions; a negative value means "not available" | Always present (0.65–1.0 in the published data) |
| Default filter | `min_confidence=0.5` | `min_confidence=0.7` |
| Downloaded per request | Only the files whose tile touches your box (manifest lookup) | **Whole S2 level-4 cells**, tens to hundreds of MB each, then filtered |
| Extra package | none | `s2sphere` |

Use them when OSM is thin (so most of the Global South), when you want **individual building polygons** to compare with
your own detections, or as an independent second opinion to [vote against OSM](pipeline.md).

Do **not** treat them as surveyed truth. They are model output: they miss small or tree-covered buildings, merge
terraced rows into one footprint, split large compounds, and date from whichever imagery the provider used. In informal
settlements, thatch and tin roofs are the common failures.

## How to use

```python
from pygeovision.labeling import MicrosoftBuildingsLabeler, GoogleBuildingsLabeler

bbox = (3.35, 6.45, 3.45, 6.55)                         # Lagos; lon_min, lat_min, lon_max, lat_max

ms = MicrosoftBuildingsLabeler(min_confidence=0.5).label(
    bbox, output_path="labels/lagos_ms.tif",
    reference_raster="data/lagos_s2.tif",               # labels on the imagery's grid
)
gg = GoogleBuildingsLabeler(min_confidence=0.7).label(
    bbox, output_path="labels/lagos_google.tif", reference_raster="data/lagos_s2.tif",
)
for r in (ms, gg):
    print(r["source"], r["success"], r["n_buildings"], r["error"])
```

### Options

| Argument | Default | Meaning |
|---|---|---|
| `bbox` | required | `(lon_min, lat_min, lon_max, lat_max)` degrees |
| `output_path` | `./labels/ms_buildings.tif` / `google_buildings.tif` | Raster to write |
| `reference_raster` | `None` | Exact grid and CRS of your imagery. Footprints are reprojected to it |
| `resolution_m` | `1.0` | Pixel size when there is no `reference_raster`. Large boxes at 1 m are refused (see below) |
| `save_vector` | `True` | Also write `<stem>_ms_buildings.geojson` / `<stem>_google_buildings.geojson` |
| `all_touched` | `False` | Burn every pixel a footprint touches. Keeps small houses at 10 m; fattens every building by up to a pixel |

Constructor-only options:

| Option | Labeler | Default | Meaning |
|---|---|---|---|
| `min_confidence` | both | 0.5 / 0.7 | Drop footprints below this. Raising it removes doubtful buildings and misses real ones |
| `quadkey_zoom` | Microsoft | 9 | Tile size used to find the right data files; leave alone |
| `MAX_CELLS` | Google | 4 (class attribute) | Most S2 cells one call will download |

### Pixel size and file size

At `resolution_m=1.0`, a 0.1° × 0.1° box (about 11 km square) is 11,000 × 11,000 pixels, 121 million pixels. Anything above
400 million pixels raises `ValueError` rather than exhausting memory. With a `reference_raster` the grid is your imagery's,
so the size is whatever your imagery is.

## What you get

```text
labels/lagos_ms.tif                    uint8: 0 background, 1 building
labels/lagos_ms_ms_buildings.geojson   one Feature per footprint, with confidence
```

(The GeoJSON name is `<raster stem>` + `_ms_buildings.geojson`; a raster called `lagos_ms.tif` therefore gives
`lagos_ms_ms_buildings.geojson`. Pick output names with that in mind.)

Microsoft result keys:

| Key | Meaning |
|---|---|
| `success` | `True` only if at least one footprint lies inside the box |
| `n_buildings` | Footprints written |
| `n_files_matched` / `n_files_read` / `n_files_failed` | How many data files covered the box, loaded, and failed. `failed > 0` means part of the area is **missing** |
| `truncated` | `True` if a limit cut the download short |
| `n_without_confidence` | Footprints with no confidence value (kept) |
| `error` | The reason, in words, when `success` is `False` |

Google result keys: `success`, `n_buildings`, `n_cells_read`, `n_cells_failed`, `truncated`, `output_path`, `vector_path`,
`source`, `error`. If more than four cells cover your box, only the first four are read and `truncated` is `True`.

Each GeoJSON feature has `confidence`, `label_value` (1) and `source`. Because the polygons stay separate in the
GeoJSON, you can load them in QGIS, select by confidence, and inspect individual buildings.

## What to do when…

| You see | Cause | Do this |
|---|---|---|
| `success=False`, `error: "no Microsoft building files cover bbox"` | No coverage there | Try Google or OSM |
| `error: "none of the N S2 cell file(s) … could be downloaded"` | No Google coverage (Europe, North America, Australia, China) or the bucket is unreachable | Use Microsoft or OSM |
| `error: "pip install s2sphere for Google Buildings"` | Missing package | `pip install s2sphere` |
| `n_files_failed > 0` | A data file did not download | Run again; if it persists, split the box. Do not train on the result without checking the gaps |
| `truncated: True` (Google) | More than 4 cells needed | Split the box so each piece touches at most four cells, or request each corner separately |
| First Google call is very slow | It downloads entire cells before filtering | Normal. Keep the GeoJSON and reuse it instead of calling again |
| Buildings drawn on top of obvious bare ground | Old imagery or false positives | Raise `min_confidence`; compare with the other source |
| Many buildings missing | Thatch/tin roofs, tree cover, new construction | Combine with OSM; hand-label a sample |
| Whole rows of houses merged into one blob | The model merged them | Nothing to do per pixel; for counts, do not count polygons |
| Vector file name looks odd (`…_ms_ms_buildings.geojson`) | The suffix is appended to your raster's stem | Name the raster `lagos.tif`, giving `lagos_ms_buildings.geojson` |
| `ValueError … pixels` | Box too large at that resolution | Use `reference_raster`, a coarser `resolution_m`, or a smaller box |

## Comparing the two (or with OSM)

Fusing them is a few lines. See [Combining sources](pipeline.md), which also explains why OSM *multi-class* maps cannot
be voted against a land-cover map. For a quick sanity check, count how often they agree:

```python
import rasterio
a = rasterio.open("labels/lagos_ms.tif").read(1)
b = rasterio.open("labels/lagos_google.tif").read(1)       # same reference_raster, so same shape
both, either = ((a == 1) & (b == 1)).sum(), ((a == 1) | (b == 1)).sum()
print(f"agreement (IoU) = {both / either:.2f}")            # below ~0.6 means the sources disagree a lot
```

```{seealso}
[Footprints in the data layer](../data/footprints.md), which downloads the same datasets as **vector** footprints for
the `building_footprints` pipeline · [OSM](osm.md) · [Quality and cleaning](quality.md)
```
