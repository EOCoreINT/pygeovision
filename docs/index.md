# AI Task Pipelines

PyGeoVision ships **40 pipelines**: a bounding box (and dates) in, a georeferenced
result out. They fall into two kinds, and the difference matters:

| Kind | Count | What you need | Pages |
|---|---|---|---|
| **ready** | 33 | A bounding box and dates (or years). Each is a documented index, a terrain product, a published reference map or a published vector dataset. | [Ready to run](ready.md) |
| **needs_weights** | 7 | A checkpoint you trained. PyGeoVision ships no trained weights, so these refuse to run without `checkpoint=`. | [Need your weights](needs-weights.md) |

Another 21 of the original 49 stay **removed** because they cannot
produce a correct result today (untrained networks, placeholder indices, invented
constants). They are listed with the reason on [Removed pipelines](removed.md);
asking for one raises an error that quotes that reason.

```{toctree}
:maxdepth: 1

ready
needs-weights
removed
```

## Running a pipeline

```python
from pygeovision import PyGeoVision

client = PyGeoVision()
bbox = (-122.6, 39.7, -121.9, 40.2)          # min_lon, min_lat, max_lon, max_lat

result = client.pipeline("wildfire_severity", bbox=bbox, output_dir="out",
                         date_before="2018-10", date_after="2018-12")
if result.success:
    print(result.output_path, result.stats)
else:
    print(result.error)                       # says exactly what went wrong
```

From the shell, every pipeline is reachable:

```bash
pygeovision channel list
pygeovision channel wildfire_severity --bbox -122.6 39.7 -121.9 40.2 \
    --date-before 2018-10 --date-after 2018-12
pygeovision channel crop_health --bbox ... --date-before 2024-05 --date-after 2024-08 \
    --param anomaly_threshold=0.2
```

`list_all_pipelines()`, `pygeovision.ai.pipelines.list_pipelines()` and
`pygeovision channel list` all read the same registry, and
`pygeovision.ai.pipelines.pipeline_kinds()` returns `{name: "ready" | "needs_weights"}`.

## Conventions every pipeline follows

**Dates.** Single-date pipelines take `date="YYYY-MM"`. Two-date pipelines take
`date_before` and `date_after`. For change and event pipelines `date_before` is the
reference (pre-fire, dry season, baseline) and `date_after` the event. A bare year
(`"2020"`) means July (June for Landsat). Older argument names
(`pre_date`/`post_date`, `baseline_year`/`analysis_year`, `start_year`/`end_year`)
still work.

**Clouds are nodata, not a class.** Cloud- and shadow-masked pixels are written as
nodata (255 in class rasters, NaN in float rasters) and excluded from every
percentage and area. Before this was fixed a cloud counted as "not burned",
"not water" or "no change".

**Results.** Every pipeline returns the same object:

```python
@dataclass
class PipelineResult:
    pipeline: str            # also available as .name
    output_path: Path | None
    metadata: dict
    stats: dict              # coverage, areas in hectares, thresholds actually used
    success: bool
    error: str               # set whenever success is False
    duration_seconds: float
```

`success=False` always carries a specific reason. Nothing falls back silently to a
different method.

## Using your own trained model

The six `needs_weights` pipelines (and the `method="model"` branches of
`change_detection` and `water_bodies`, and `land_cover` with a model name) run your
network:

```python
from pygeovision.ai.training.trainer import GeoTrainer   # train once

result = client.pipeline(
    "building_footprints", bbox=bbox, output_dir="out", date="2024-06",
    checkpoint="runs/buildings/checkpoints/best.pth",
)
```

The checkpoint is loaded with `strict=True` and `weights_only=True`. A mismatch in
architecture, class count or band count is an error, and the pipeline fails
**before** downloading any imagery. `checkpoint` may be a full training checkpoint
(`model_state_dict`) or a bare state dict. Without it the call returns
`success=False` with a message saying trained weights are required.

## Things worth knowing

- **Reference-product pipelines** (`landcover_change`, `urban_expansion`, `forest_change`,
  `surface_water_dynamics`, `evapotranspiration`, `active_fire`) read public products from the Microsoft
  Planetary Computer (no account needed), windowed to your box. `road_network` and, for
  `source="osm"`, `building_footprints` use OpenStreetMap through the Overpass API; `source="microsoft"` and
  `"google"` read the published footprint files. Each result states its product's coverage, date range and
  known weaknesses in `stats["product_note"]` / `stats["source_note"]`, and the report repeats them. They are
  *published datasets*, not detections on your own imagery.
- **`building_footprints`** has two modes: published footprints (default, no weights) and your own network when
  you pass `checkpoint=`.
- **`sar_flood_mapping`** maps flooding from Sentinel-1 radar, so cloud does not block it. Give `date_before` (a dry
  reference) and `date_after` (the flood) as days, months or ranges such as `2026-06-27/2026-07-06`: Sentinel-1
  passes a place only every 6 to 12 days or longer, so a single day is usually empty (the error lists the nearest
  acquisitions). It compares scenes of one orbit track only. It maps *open* floodwater; floodwater between
  buildings or under dense vegetation is bright to radar and is not detected.
- **`multisensor_flood_mapping`** combines Sentinel-1 radar, Sentinel-2 optical (MNDWI, cloud-masked with the Scene
  Classification Layer) and terrain (HAND from the Copernicus DEM) without blending them into one score: the result
  says, per pixel, whether radar and optical agree (class 1), only radar saw it (2), only optical saw it (3) or they
  disagree (4), and which sensors were actually available (`sensors_used`, `data_notes`). If the flood-window optical
  imagery is all cloud, it falls back to radar + terrain and says so. The sensors inform each other: optical imagery
  of the dry reference window calibrates the radar water threshold (`water_db="auto"`, the default; a number or `otsu`
  override it), the radar change is measured after removing the brightness offset between the two windows, and a clear
  optical view of the flood window confirms or contradicts each radar flood pixel. Sea and lakes are excluded from the
  flooded area and from the percentages. Same date arguments as `sar_flood_mapping`.
- **Terrain and drainage** (`terrain_analysis`, `drainage_analysis`) use the Copernicus 30 m DEM from the
  Planetary Computer by default (`dem_source="opentopography"` uses OpenTopography and a key). If the Planetary
  Computer read fails (for example an HTTP 403 from your network), the same Copernicus product is read from the
  public AWS Open Data copy and a warning is logged; `dem_source="copernicus_aws"` selects that copy directly.
  `dem_host` in the result statistics says which host the elevations came from. `drainage_analysis`
  is plain NumPy and takes about a minute per 5 million cells; larger boxes are refused with advice.

- **`land_cover` returns the ESA WorldCover 2021 map**, a published reference
  product, clipped to your bounding box. It is the same map whatever `date` you
  pass, and the result metadata says so. Do not use it to measure change.
  (WorldCover 2020 and 2021 come from different algorithm versions and ESA states
  they are not comparable for change detection.)
- **`change_detection` defaults to change vector analysis** with an Otsu threshold
  and a minimum magnitude, so it works without a model; `method="model"` runs your
  siamese network instead.
- **DEM pipelines** (`solar_potential`, `archaeological_site`, `wind_farm_siting`)
  fetch from OpenTopography and need an API key. Choose the product with
  `dem_type` (`srtm90`, `srtm1arc`, `cop30`, `cop90`, `nasadem`, `alos`, `gedtm30`).
- **Optical limits.** Clouds, haze and shadows still limit what optical imagery can
  show. Pick dates in the same season and compare results against local knowledge.