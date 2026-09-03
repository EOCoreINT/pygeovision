# PyGeoVision Client

The main entry point for most PyGeoVision functionality.

**Verification status**: `search()`, `download()`, and construction (`cache_dir`,
`log_level` parameters) are confirmed real and match the code exactly.
Several method calls below use code paths this audit cycle did not reach —
each is flagged individually. Where a method has a namespace conflict with
a separately-audited version elsewhere in the codebase (a real, confirmed
pattern this cycle — see [Architecture](../architecture.md)), that's called
out explicitly rather than assumed to inherit the other path's fixes.

---

## Class: `PyGeoVision`

```python
import pygeovision as pgv

client = pgv.PyGeoVision(
    cache_dir="~/.cache/pygeovision",   # optional, real parameter
    log_level="INFO",                   # optional, real parameter
)
```

`repr(client)` reports real, current counts: `datasets=503 | models=98 |
pipelines=51` — but "51 pipelines" includes pipelines that don't yet do
what their names claim. See [Pipelines](pipelines.md) for the honest
breakdown, or the [Home page](../index.md) verification table.

---

## Data Search

### `search(bbox, date_range, providers=None, **kwargs)`

Confirmed real, matching signature:

| Parameter | Type | Notes |
|---|---|---|
| `bbox` | `tuple[float, float, float, float]` | `(min_lon, min_lat, max_lon, max_lat)`, WGS84 |
| `date_range` | `tuple[str, str]` | `(start, end)` |
| `collections` | `list[str] \| None` | STAC collection IDs |
| `providers` | `list[str] \| None` | Restricts search — see the real `--provider` CLI flag |
| `cloud_cover_max` | `float` | Default `30.0` |
| `max_results` / `limit` | `int` | `limit` is an alias for `max_results` |
| `cql2_filter` | `str \| None` | Advanced CQL2 filter |

```python
results = client.search(
    bbox=(-74.1, 40.6, -73.7, 40.9),
    date_range=("2024-06-01", "2024-08-31"),
    providers=["planetary_computer"],
    cloud_cover_max=10,
)
```

### `download(results, output_dir, **kwargs)`

Confirmed real this cycle — extensively fixed and tested (real unzip
handling, real cache-hit asset recovery, mission-aware Landsat band
aliases, CRS-aware reprojection validation). See [Architecture](../architecture.md)
for the specific bugs found and fixed.

```python
downloads = client.download(results[:3], output_dir="./data/nyc/")
```

---

## Auto-Labeling — `client.labeling`

**Verification status**: `client.labeling.esa_worldcover()` calls
`pygeovision.labeling.landcover.ESAWorldCoverLabeler` — a **separate class**
from `pygeovision.ai.labeling.esa_worldcover.ESAWorldCoverLabeler`, which
is the one with confirmed, fixed bugs this cycle (nodata handling,
class-name mapping, tile-origin int/float promotion). Those fixes do
**not** apply to this code path; it has not been independently audited.

```python
labels = client.labeling.osm(bbox, categories=["buildings", "roads", "water"])
labels = client.labeling.microsoft_buildings(bbox)
labels = client.labeling.google_buildings(bbox)
labels = client.labeling.esa_worldcover(bbox)   # see verification note above
labels = client.labeling.dynamic_world(bbox, date_range=("2024-01-01", "2024-12-31"))
```

---

## Inference — `client.inference`

**Verification status**: `client.inference.tiled()` calls
`pygeovision.inference.tiled.TiledInference` — a **separate class** from
`pygeovision.ai.inference.tiled_inference.TiledInference`, which is the one
with the memory-efficiency fixes confirmed this cycle (windowed reads,
memory-aware automatic batch sizing). Those fixes do **not** apply here;
this class has a different default (`overlap=128` vs. the audited class's
`overlap=64`) confirming it's genuinely separate code, not a re-export.

```python
inf    = client.inference.tiled(model, chip_size=512, overlap=128)
result = inf.infer("scene.tif", "prediction.tif")
```

For the audited, memory-efficient version, use `pygeovision.ai.inference.tiled_inference.TiledInference` directly — see [Inference](inference.md).

---

## Monitoring — `client.monitoring`

**Verification status**: not independently audited this cycle. Signatures below are confirmed real (checked directly against the code).

```python
drift = client.monitoring.drift_detector(model=None)
tracker = client.monitoring.performance_tracker(model_name="my_model")
alerts = client.monitoring.alert_manager(channels=None)
```

---

## Pipelines — `client.pipeline`

**Important**: `client.pipeline(name)` is **not** an AI pipeline runner — it
returns a PyGeoFetch-style chainable *data processing* builder (cloud
masking, clipping, reprojection, spectral indices, vectorization), for
raw imagery, not a way to run a named AI task like "land_cover" or
"building_footprints."

```python
result = (
    client.pipeline("my_chain")
    .cloud_mask(method="scl", scl_band="SCL.tif")
    .clip(bbox=(-74.1, 40.6, -73.7, 40.9))
    .reproject(crs="EPSG:4326")
    .ndvi(red="B04.tif", nir="B08.tif")
    .run(input="scene.tif", output_dir="./processed/")
)
```

To run a real, audited AI pipeline, use the CLI or the pipeline class directly:

```bash
pygeovision channel land_cover --bbox -74.1 40.6 -73.7 40.9 --date 2024-01
```

```python
from pygeovision.ai.pipelines import LandCoverPipeline
result = LandCoverPipeline(client).run(bbox=(-74.1, 40.6, -73.7, 40.9), output_dir="./out", date="2024-01")
```

---

## SAR / InSAR — removed

`client.sar` has been removed. This functionality is handled directly by
[pygeofetch](https://pypi.org/project/pygeofetch/):

```python
from pygeofetch.processing.sar import SARProcessor

sar = SARProcessor()
despeckled = sar.despeckle("s1_raw.tif", filter="lee")
calibrated = sar.calibrate(despeckled.output_path, output_type="sigma0", in_db=True)
```

See [Architecture](../architecture.md) for the full reasoning behind this removal.

---

## Complete, auto-generated reference — `PyGeoVision` class

The main client class's real, current methods and signatures,
generated directly from source (the top-level module also defines
several internal proxy classes for `client.detection`,
`client.labeling`, etc. — not shown here individually; see their
respective pages).

::: pygeovision.PyGeoVision
    options:
      show_source: false
