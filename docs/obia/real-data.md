# Real imagery

**Functions:** `obia.fetch_scene`, `obia.fetch_worldcover`, `obia.fetch_osm_polygons`, `obia.valid_mask`,
`obia.samples_from_raster`, `obia.samples_from_geometries(..., crs=)` · **Client:** `client.obia.scene / worldcover / osm_polygons` ·
**CLI:** `pygeovision obia fetch`, `pygeovision obia classify --weak-labels`

## Why

OBIA in PyGeoVision runs on real imagery, fetched with the same search and download tools as every pipeline. A real
Sentinel-2 image has holes and its labels have errors. Three things go wrong first, and each has a defined behaviour here.

| Real-world problem | What OBIA does |
|---|---|
| Cloud and shadow masked out (NaN in every band) | Those pixels get **label 0** (no data). No object is built over them and they are left out of every statistic. |
| Training data is a *map* (WorldCover), not polygons you drew | `samples_from_raster` turns the map into object samples, with a purity threshold. |
| Reference polygons or points in another coordinate system | `crs=` reprojects them; a reference that lies wholly outside the image raises an error instead of returning nothing. |

## No-data pixels

```python
rd = obia.load_raster("s2.tif")            # cloud pixels are NaN
ok = obia.valid_mask(rd)                   # True where the image holds data
seg = obia.segment(rd, "multires", scale=16)
assert ((seg.labels == 0) == ~ok).all()    # label 0 is exactly the invalid pixels
seg.valid                                  # the same mask, or None if there was nothing to mask
```

`valid_mask` calls a pixel invalid when **any** band is NaN or infinite, or when **every** band equals the raster's
declared `nodata`. All six segmentation methods, `segment_levels` and the scale tools follow this rule. Object ids run
`1…n` and each id is one connected piece, as before. An image with no valid pixel raises
`ValueError: the image has no valid pixels`.

What this changes elsewhere:

- `extract_features`: statistics use the object's valid pixels only (an object cannot contain a NaN pixel, so no mean is
  poisoned). Texture (`texture_layers=`) can still be NaN for a very small object; classifiers fill those with the median.
  A texture layer with no finite pixel at all raises `ValueError`.
- `class_context`: label 0 is not a neighbour. A strip of no data between two objects does not make them neighbours; its
  border counts as an open edge.
- `segmentation_quality`: reference objects that fall wholly under cloud have no segment to match and drop out of the table.
- `samples_from_geometries`: label 0 is never a sample.

Rescaling for segmentation (`rescale=True`) takes its percentiles from the finite pixels, so a cloud mask does not distort the
stretch.

```{warning}
A cloud mask hides ground. An area map built from the objects excludes those hectares; report the no-data area next to
the class areas, as the tutorial does.
```

## Getting the data: `fetch_scene`, `fetch_worldcover`, `fetch_osm_polygons`

```python
import pygeovision as pgv
from pygeovision import obia

client = pgv.PyGeoVision()
bbox = (-1.700, 6.180, -1.655, 6.225)                               # lon/lat, about 5 km square
rd = obia.fetch_scene(client, bbox, "2024-01", "./download", cloud_cover_max=30)
wc, path = obia.fetch_worldcover(client, bbox, "./download/worldcover.tif", like=rd)   # on rd's grid
mines = obia.fetch_osm_polygons(bbox)                               # OpenStreetMap quarry / mine outlines
```

The same three calls hang off the client, which is how the rest of PyGeoVision is used: `client.obia.scene(bbox, date, out_dir)`,
`client.obia.worldcover(bbox, out_path, like=rd)` and `client.obia.osm_polygons(bbox)`.

**`fetch_scene(client, bbox, date, out_dir, bands=("blue","green","red","nir","swir1","swir2"), cloud_cover_max=30.0, **kw)`**
uses the same search, download, cloud-mask and mosaic code as the pipelines (`BasePipeline._search_and_download`) and returns a
`RasterData`: reflectance 0-1, cloud and shadow as NaN, band names as requested, cropped to `bbox` in the scene's UTM zone.
`date` is anything the pipelines accept (`"2024-01"`, a day, a `"start/end"` window). Extra keyword arguments such as
`providers=["planetary_computer"]` go to the shared download code. If no scene can be searched, downloaded or prepared it raises
`RuntimeError` with the reason. Nothing is ever substituted for a failed download.

**`fetch_worldcover(client, bbox, out_path, like=None)`** returns `(array, path)` through `client.labeling.esa_worldcover`.
With `like` the map is resampled onto that image's grid and CRS (nearest neighbour), so the two align pixel for pixel. 0 is no
data. A failed download raises `RuntimeError`.

**`fetch_osm_polygons(bbox, filters=MINE_OSM_FILTERS, timeout=120, retries=3)`** asks Overpass (through the labeller's client,
with its User-Agent, mirrors and retries) for polygons and returns a GeoDataFrame in EPSG:4326 with `osm_id`, `osm_type`, `name`,
`landuse`, `industrial`, `resource` and `geometry`. The default filters outline mines and quarries
(`landuse=quarry`, `industrial=mine`); pass your own Overpass tag filters for other features, e.g. `['["natural"="water"]']`.
Closed ways become polygons, multipolygon relations are assembled with their holes, open lines are skipped. An empty result is a
valid answer. OSM is volunteer-mapped, so a missing polygon does not mean there is no mine. A failed request raises
`RuntimeError`.

```{note}
All three need a network, and the credentials the satellite provider uses (Planetary Computer and AWS Earth Search need none).
The multiresolution merging is pure Python: a 5 km x 5 km window (500 x 500 pixels) is comfortable, a full tile is not. Tile
large areas, or use `slic` / `felzenszwalb`.
```

### Progress, and how to tell slow from stuck

A scene is a few hundred megabytes and the three downloads can take minutes, so each call **reports as it goes**
(`verbose=True`, the default). You see the stage it is in, the search and download messages of the library, and every
`heartbeat` seconds (default 10) a line such as

```text
[10:42:31]   ... search and download: still working, 40 s elapsed, 85.3 MB received so far
[10:43:01]   ... search and download: still working, 70 s elapsed, 85.3 MB received so far (nothing new for 30 s: slow or stalled network, or a provider not answering)
```

If the megabytes keep growing, it is a slow link. If the line says nothing new has arrived, the network or a provider is the
problem, not Python. Pass `verbose=False` for silence or `heartbeat=5` for faster updates; in a notebook the lines appear in the
cell's output while it runs.

Waits are bounded: the pygeofetch command-line search has a time limit, WorldCover remote reads time out after 60 s (3 retries),
and Overpass waits at most `timeout` seconds per request (60 by default) across three public servers and `retries` rounds, so at
most about six minutes with the defaults. When a step fails, the `RuntimeError` carries the reasons the download code logged
(for example "the search found no scene for bbox=..., cloud cover <= 30%").

### On the command line

```console
$ pygeovision obia fetch --bbox -1.70 6.18 -1.655 6.225 --date 2024-01 -o scene.tif --worldcover wc.tif
```

writes `scene.tif` (bands named, cloud as NaN) and, with `--worldcover`, WorldCover on the same grid. `--bands`, `--cloud-max`
and `--providers` are passed on; `--heartbeat N` sets the seconds between "still working" lines and `--quiet` turns the progress off. It prints the share of no-data pixels and exits with status 1 and the reason if the download
fails. `scene.tif` goes straight into `obia segment`.

## Weak labels from a map: `samples_from_raster`

```python
wl = obia.samples_from_raster(seg.labels, wc,
                              {10: "forest", 30: "farmland", 40: "farmland", 50: "built_up", 60: "mine_disturbed", 80: "water"},
                              min_purity=0.7)
wl["class"]       # class by object id
wl["purity"]      # share of the object's pixels that carry the winning class
```

`samples_from_raster(labels, class_raster, mapping=None, min_purity=0.6, nodata=0)` gives each object the class most of
its pixels carry **if** that class covers at least `min_purity` of them. Rules worth knowing:

- Codes missing from `mapping` (shrub, wetland, …) never win but **do count against purity**, so an object that is mostly
  "something else" gets no label rather than a wrong one.
- Pixels equal to `nodata` are ignored in numerator and denominator; label 0 is never an object.
- Without `mapping` the raster's own codes are used as class names.
- `labels` and `class_raster` must be on the same grid (`ValueError` otherwise); `min_purity` must be in (0, 1].

On the command line the same step is `pygeovision obia classify scene.tif labels.tif --weak-labels wc.tif --class-map
"10=forest,30=farmland,40=farmland,50=built_up,80=water" --min-purity 0.7 --output classified.gpkg`. Give exactly one of
`--samples`, `--weak-labels` or `--rules`.

These labels carry every error of the source map. Train with them; **do not measure accuracy with them**. Accuracy needs a
reference that is independent of the map and was not used for training: leave its objects out of the training set
(`train = samples.drop(index=ref_objects)`), as the tutorial does.

## Reference data in another CRS

```python
samples = obia.samples_from_geometries(seg.labels, ref_gdf, "class", rd.transform, crs=rd.crs)
report  = obia.assess_classification(result, seg.labels, ref_gdf, "class", rd.transform, crs=rd.crs)
```

With `crs=` the geometries are reprojected to the image's system. If they have no CRS at all, `ValueError` asks you to set
one. If they lie wholly outside the image extent, `ValueError` says so and points at the likely cause (degrees against
metres). Without `crs=` nothing is reprojected, as before, but the extent check still runs whenever a `transform` is given.

```{seealso}
[Data in and out](data-io.md) · [Classification](classification.md) · [Accuracy](accuracy.md) · [Mining workflows](../examples/mining.md)
```
