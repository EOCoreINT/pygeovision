# OpenStreetMap labels

**Class:** `pygeovision.labeling.OSMLabeler` · **CLI:** `pygeovision label osm` · **Client:** `client.labeling.osm(...)`

## Why use it

OpenStreetMap (OSM) is a free world map drawn by volunteers. In many cities it holds almost every building, every road
and every river, with outlines traced from aerial photos or surveyed on foot. That makes it the fastest way to get
labels for **buildings, roads, water and land use** with no annotation work: you give a box, it returns a raster the
size of your imagery.

OSM is the right first choice when:

- the area is well mapped (most of Europe, much of North America, many cities elsewhere);
- you need several classes at once (buildings **and** roads **and** water in one raster);
- you want vector outlines too (it writes a GeoJSON beside the raster).

It is the wrong choice when the area is thinly mapped. A pixel with no label means **"nobody drew it"**, not "there is
nothing there". A model trained on gaps in the map learns to ignore real buildings. See
[When OSM is incomplete](#when-osm-is-incomplete).

## How it works

1. Your box and categories become an **Overpass API** query (the public read interface to OSM). Three public servers are
   tried in turn if one is busy.
2. The returned features are classified by their tags (`building=*` is a building, `highway=secondary` is a road, and so on).
3. **Areas** (buildings, land use, lakes) are filled polygons. Large lakes and courtyard buildings, which OSM stores as
   *relations*, are assembled from their pieces, with islands cut out.
4. **Lines** (roads, rivers, canals) are widened to a real width in metres chosen from the road or waterway class, so a
   4 m service road is still a connected line at 10 m resolution.
5. Everything is burned into a `uint8` raster. Where features overlap, the more specific one wins.

### Categories and class values

| Category | Value | What it contains | Drawn as |
|---|---:|---|---|
| *(background)* | 0 | everything not matched | |
| `buildings` | 1 | anything with a `building=*` tag (except `building=no`) | filled polygon |
| `roads` | 2 | motorway, trunk, primary, secondary, tertiary, unclassified, residential, service (and `*_link`) | line, widened |
| `water` | 3 | `natural=water`, riverbanks, any `waterway=*` | polygon, or widened line |
| `vegetation` | 4 | forest, meadow, grass, orchard, vineyard, wood, scrub, heath | polygon |
| `agricultural` | 5 | farmland, farmyard, greenhouse horticulture | polygon |
| `commercial` | 6 | commercial, retail, industrial | polygon |
| `residential` | 7 | `landuse=residential` | polygon |
| `solar_panels` | 8 | `generator:source=solar` | polygon |
| `parking` | 9 | `amenity=parking` | polygon |
| `sports` | 10 | pitch, track, sports centre | polygon |

The values are **fixed**, whatever subset you ask for: asking only for `["roads", "water"]` gives a raster with values
0, 2 and 3, not 0, 1 and 2. If you train on a subset, [renumber the classes](../training/data.md#rules-for-labels) to `0…N-1`
first, or use `reference_raster` plus a remap step.

### Widths used for lines

| Road class | Width (m) | Waterway | Width (m) |
|---|---:|---|---:|
| motorway | 14 | river | 15 |
| trunk | 12 | canal | 10 |
| primary | 10 | stream | 4 |
| secondary | 8 | ditch, drain | 3 |
| tertiary | 7 | other | 4 |
| unclassified, residential | 6 | | |
| service | 4 | | |

These are typical full widths, not surveyed ones. A highway drawn 14 m wide where the real road is 22 m gives a label
narrower than the road in the imagery.

### Overlaps: what wins

Drawn from underneath to on top: residential → commercial → agricultural → vegetation → sports → parking → solar →
water → roads → buildings. So a house inside a residential zone is still a house, a wood inside a residential zone
stays wood, and a road over a river (a bridge) is a road.

## How to use it

### Python

```python
from pygeovision.labeling import OSMLabeler

labeler = OSMLabeler()
result = labeler.label(
    bbox=(-74.05, 40.70, -73.95, 40.80),          # lon_min, lat_min, lon_max, lat_max
    categories=["buildings", "roads", "water"],
    output_path="labels/nyc_osm.tif",
    reference_raster="data/nyc_sentinel2.tif",    # strongly recommended
)
if not result["success"]:
    raise SystemExit(result["error"])
print(result["n_features"], result["categories"])
```

### Command line

```console
$ pygeovision label osm -74.05 40.70 -73.95 40.80 \
      --reference data/nyc_sentinel2.tif -c buildings -c roads -c water -o labels/nyc_osm.tif
```

### Through the client

```python
from pygeovision import PyGeoVision
client = PyGeoVision()
client.labeling.osm(bbox, categories=["buildings"], output_path="labels/b.tif", reference_raster="data/scene.tif")
```

### Options

| Argument | Default | Meaning |
|---|---|---|
| `bbox` | required | `(lon_min, lat_min, lon_max, lat_max)` in degrees |
| `categories` | all ten | Which to include. A misspelled name raises `ValueError` listing the valid ones |
| `output_path` | `./labels/osm_labels.tif` | Where the raster goes |
| `reference_raster` | `None` | Write onto this raster's exact grid and CRS (any CRS, including UTM) |
| `resolution_m` | `10.0` | Pixel size in metres when there is no `reference_raster` (lon/lat grid) |
| `save_vector` | `True` | Also write the features as GeoJSON |
| `vector_path` | `<output>_vectors.geojson` | Where the GeoJSON goes. Never the same file as the raster |
| `all_touched` | `False` | Burn every pixel an area touches, not only pixels whose centre it covers |

Constructor: `OSMLabeler(endpoint=None, timeout=120, retry_attempts=3, crs="EPSG:4326")`. Use `endpoint` to point at your
own Overpass server for large jobs.

`labeler.preview_query(bbox, categories)` returns the Overpass query text without sending it.

## What you get

```text
labels/nyc_osm.tif              uint8 GeoTIFF, nodata unset, values 0..10
labels/nyc_osm_vectors.geojson  one Feature per OSM object (EPSG:4326)
```

The result dict (real output from a small offline test with a mocked Overpass server; see the note below):

```json
{
 "success": true,
 "n_features": 5,
 "n_skipped": 0,
 "categories": {"residential": 1, "buildings": 2, "roads": 1, "vegetation": 1},
 "class_values": {"buildings": 1, "roads": 2, "vegetation": 4, "residential": 7},
 "output_path": "labels/osm.tif",
 "vector_path": "labels/osm_vectors.geojson",
 "error": null
}
```

| Key | Meaning |
|---|---|
| `success` | `False` if the request failed **or** zero features were found |
| `n_features` | Features written |
| `n_skipped` | OSM objects dropped because their outline was broken (an area that does not close, for example) |
| `categories` | Feature count per category |
| `class_values` | Raster value per category requested |
| `error` | Why, when `success` is `False` |

Each GeoJSON feature carries `category`, `label_value`, `osm_id`, `osm_type` and the original OSM tags, for example
`{"landuse": "residential", "category": "residential", "label_value": 7, "osm_id": 1, "osm_type": "way"}`. Open it in
QGIS to see individual building outlines rather than one merged mask.

The raster from that small test (a 0.01° box at 10 m, 112 × 72 pixels, lon/lat grid) counted:

| Value | Pixels | Share |
|---:|---:|---:|
| 0 background | 2,691 | 33.4% |
| 1 building | 101 | 1.3% |
| 2 road | 300 | 3.7% |
| 4 vegetation | 96 | 1.2% |
| 7 residential | 4,876 | 60.5% |

```{note}
The test data are five hand-drawn objects, not a real city, and Overpass was replaced by a stand-in so the numbers
are reproducible. They show the **shape** of the output, not what your area will look like.
```

## What to do when…

| You see | Cause | Do this |
|---|---|---|
| `"Overpass request failed (HTTP 429)"` or `(HTTP 406)` | The public server is rate-limiting or refusing your client | Wait a minute and retry; reduce the box; set `endpoint=` to another Overpass server or your own; run once and reuse the file |
| `"Overpass could not complete the query: … timed out"` | Box or category set too big | Split the box into tiles, or ask for fewer categories. Buildings alone in a big city is a heavy query |
| `success=False`, `"0 features"` | The area is not mapped for those categories, or the box is the wrong way round | Check the box order is **lon first**; look at the area on openstreetmap.org |
| Roads look thin compared with the imagery | Width is a class average | Pass `all_touched=True`, or buffer the GeoJSON yourself and rasterize it |
| Small buildings missing at 10 m | A 6 m shed covers less than a pixel centre | `all_touched=True`; or label at the imagery's native resolution with `reference_raster` |
| Labels shifted from the imagery | The imagery or OSM is offset, or the grid was not shared | Use `reference_raster`; inspect an offset by overlaying the GeoJSON. Offsets of 5–10 m in OSM are common |
| `ValueError: Unknown OSM categories` | A typo (`building` instead of `buildings`) | Use the names in the table above |
| `ValueError … pixels` | The grid would be huge (1° at 1 m) | Use a smaller box or coarser `resolution_m` |
| Empty raster on a UTM reference | (older versions) CRS mismatch | Fixed: the reference's CRS is honoured. Update if you see this |

### When OSM is incomplete

If you train on OSM labels for an area where half the buildings are unmapped, the unmapped half is taught as
"background". Three ways to cope:

1. **Restrict the training area** to places you have checked are mapped, and mask the rest with 255:

   ```python
   import rasterio
   from rasterio.features import geometry_mask

   well_mapped = [...]        # GeoJSON-style geometries of the ground you checked, in the raster's CRS
   with rasterio.open("labels/osm.tif") as src:
       lab, prof = src.read(1), src.profile
       inside = ~geometry_mask(well_mapped, out_shape=lab.shape, transform=src.transform)
   lab[~inside] = 255                                  # outside: "no information", not "background"
   prof.update(nodata=255)
   with rasterio.open("labels/osm_masked.tif", "w", **prof) as dst:
       dst.write(lab, 1)
   ```
2. **Combine with a footprint dataset** for the same area ([building footprints](buildings.md), [fusion](pipeline.md))
   and keep only pixels where they agree, or treat disagreement as 255.
3. **Fix by hand** a sample of chips and train with those, using OSM only to pre-fill.

The honest check is to overlay the labels on imagery for a few dozen random places before you trust them.

## Limits

- Coverage is uneven; completeness differs street by street.
- Tags are inconsistent between countries (`landuse=residential` may cover an entire town in one place and a single plot in another).
- Roads are lines with assumed widths; building heights, roof types and vegetation density are not part of the label.
- OSM changes daily; label rasters are a snapshot of the day you ran the query. Keep the GeoJSON and the date.
- The data are © OpenStreetMap contributors, ODbL licence. Keep attribution if you publish a derived dataset.

```{seealso}
[Combining sources](pipeline.md) · [Quality and cleaning](quality.md) · [Reference data and terrain](../data/reference-data.md)
```
