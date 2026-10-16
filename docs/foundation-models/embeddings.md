# Embeddings: TESSERA and AlphaEarth

Two products give you a vector for **every pixel** that someone else already computed from a whole year of satellite data, so you never run
a large encoder yourself. You fetch the numbers for your area, then use them as the input of a very simple model. With a few hundred labelled
points that is often enough for a land-cover or crop map.

```{figure} ../_static/img/flow-embeddings.svg
:alt: Working with ready-made embeddings: fetch the embedding stack for a box, label points, sample the vectors at them, train a simple classifier, predict every pixel, check on points held back
:width: 100%
:class: pgv-figure

The whole method fits in six steps and needs no GPU.
```

```{admonition} What was tested
:class: note
Run here, against stand-in `geotessera` and `ee` modules that return known arrays (no network): the shape and axis order of what
`TesseraGeo` returns, the GeoTIFF it writes (128 bands, the right bounds, tags), the PCA picture, AlphaEarth's point-sampling order and
`NaN` rows, the export task it starts, the missing-package messages, and the classifier and similarity code below on synthetic embeddings
(`tests/test_docs_embedding_workflow.py`).
**Not run here:** a real download from TESSERA or a real Earth Engine call (no network or account). What the real embeddings
*mean*, and how well a classifier does on them, depends on your area and labels and was not measured.
```

## The two products

| | **TESSERA** | **AlphaEarth** (Satellite Embedding) |
|---|---|---|
| Made by | University of Cambridge | Google DeepMind |
| Channels per pixel | 128 | 64 (bands `A00` … `A63`) |
| Resolution | 10 m | 10 m |
| Built from | A year of Sentinel-1 and Sentinel-2 time series | A year of multi-sensor observations |
| Years | 2017–2024 (as published when this wrapper was written; check the project) | Annual, from 2017 |
| Reached through | The `geotessera` Python package, which downloads tiles to a local cache | Google Earth Engine, collection `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` |
| You need | `pip install geotessera` | `pip install earthengine-api`, an Earth Engine account and project |
| A region comes back as | A NumPy array, immediately | An **export task** you wait for; points are immediate |
| Class | `TesseraGeo` | `AlphaEarthGeo` |

An embedding is not a spectral composite: its values have no unit and no band meaning. The vector at a pixel is a compact description of how
that place behaved over the year, learned so that similar places get similar vectors. That is why a simple classifier works, and also why
you cannot read a "band 17" the way you read a near-infrared band.

## TESSERA

```python
from pygeovision.models.foundation.tessera import TesseraGeo

t = TesseraGeo()                                                    # dataset_version="v1"; cache_dir=None uses the library default
res = t.embeddings_for_bbox((0.20, 6.70, 0.45, 6.95), year=2024,    # (lon_min, lat_min, lon_max, lat_max), WGS 84
                            output_path="tessera.tif")
print(res["shape"], res["crs"])        # (128, H, W) EPSG:4326
embedding = res["embedding"]           # the same array, channel first
```

| Method | Arguments | Returns |
|---|---|---|
| `embeddings_for_bbox` | `bbox`, `year=2024`, `output_path=None`, `target_crs="EPSG:4326"` | `{"success": True, "shape", "year", "n_channels", "crs", "embedding"[, "output_path"]}`, or `{"success": False, "error": ...}` |
| `embeddings_at_points` | `points` as `[(lon, lat), ...]`, `year=2024` | An array `(N, 128)`, one row per point |
| `pca_visualisation` | `bbox`, `output_path`, `year=2024`, `n_components=3` | `{"success": True, "output_path", "explained_variance_ratio"}` |

What is verified about them:

- The library returns height × width × channels; `embeddings_for_bbox` turns it into **channels first** `(128, H, W)`, the order the rest of
  PyGeoVision uses, and the array is in the result (an earlier version described the data without returning it).
- With `output_path` the stack is written as a LZW-compressed GeoTIFF with 128 bands, the box's georeferencing, and the tags `source=TESSERA`,
  `year` and `dataset_version`.
- A fetch that fails (no network, a year or area that is not covered) comes back as `{"success": False, "error": ...}`; it does not raise.
  A missing package **does** raise: `ImportError: geotessera required for TESSERA embeddings: pip install geotessera`.
- `pca_visualisation` runs a PCA on the valid pixels and writes the first three components as a `uint8` picture, each stretched between its
  2nd and 98th percentile, so you can *look* at what the embedding separates. It needs `scikit-learn`
  (`{"success": False, "error": "pip install scikit-learn"}` otherwise). The returned `explained_variance_ratio` tells you how much of the
  variation the three bands carry.

Start with a small box (a few kilometres): the tiles are downloaded and cached on your machine, and a large box means a large download.

## AlphaEarth

```python
from pygeovision.models.foundation.alphaearth_geo import AlphaEarthGeo

ae = AlphaEarthGeo()                    # after ee.Authenticate() and ee.Initialize(project="...") once
vectors = ae.embeddings_at_points([(-0.22, 5.55), (-0.21, 5.56)], year=2024)   # (2, 64), immediately
task = ae.embeddings_for_bbox((-0.25, 5.52, -0.20, 5.60), year=2024, drive_folder="pgv")
print(task["ee_task_id"], task["status"])                                      # started
```

| Method | Behaviour |
|---|---|
| `embeddings_at_points(points, year=2024)` | A synchronous Earth Engine query at 10 m. Returns `(N, 64)` in the **same order** as `points`; a point with no data (open sea, a missing year) is a row of `NaN` |
| `embeddings_for_bbox(bbox, year=2024, description="alphaearth_export", drive_folder=None)` | Earth Engine has no instant bulk download, so this **starts an export to Google Drive** and returns `{"success": True, "ee_task_id", "status": "started", "n_bands": 64, "year"}`. Watch the Tasks tab in the Earth Engine code editor, then download the GeoTIFF from Drive |
| `similarity_map(reference_point, bbox, year, description)` | Starts an export of a cosine-similarity map. **Not run against Earth Engine, and the multiplication it builds is the first thing to check.** Prefer the local function below on a raster you exported |

Authentication is done once, outside the wrapper, as for the other Earth Engine features here. A missing package gives
`{"success": False, "error": "pip install earthengine-api"}` from the export methods and an `ImportError` from `embeddings_at_points`.
The export is limited by Earth Engine's own quotas and by the size of the area.

## A worked recipe: classify from a few points

The functions below take a plain array `(channels, H, W)`. They are a documented example you can copy
({download}`embedding_workflow.py <../_examples/foundation/embedding_workflow.py>`), not part of the package.

```{literalinclude} ../_examples/foundation/embedding_workflow.py
:language: python
:caption: embedding_workflow.py
:linenos:
```

Use them on the GeoTIFF that `embeddings_for_bbox` wrote:

```python
import numpy as np, rasterio
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from embedding_workflow import classify_pixels, similarity_map

with rasterio.open("tessera.tif") as src:
    embedding = src.read()                                        # (128, H, W)
    # labelled field points: (lon, lat, class id); src.index turns coordinates into (row, col)
    rows_cols = [src.index(lon, lat) for lon, lat, _ in field_points]
    labels = [cls for _, _, cls in field_points]

train, test = train_test_split(range(len(labels)), test_size=0.3, random_state=0)   # hold some points back
class_map = classify_pixels(embedding, [rows_cols[i] for i in train], [labels[i] for i in train])
print("accuracy on held-back points:",
      accuracy_score([labels[i] for i in test], [class_map[rows_cols[i]] for i in test]))
```

Verified on synthetic embeddings with two well separated regions: six labelled points recovered the two regions at **above 99 %** of pixels,
pixels without an embedding were marked `-1`, and a labelled point on a `NaN` pixel was refused with a message. Real data is harder
than two clean clusters, so **the held-back score is the number to trust, not this**. Keep points that were not used to train, and spread them
(a held-back point next to a training point is nearly the same pixel and flatters the score).

Practical advice that follows from the method:

- **Many more points than classes.** A few hundred well spread points is a sensible start; a rare class needs enough of its own.
- **Spatial split.** Hold back whole areas rather than random neighbours when you can.
- **Start with the simplest model.** Logistic regression, then k-nearest-neighbours, then a small gradient-boosted model. If a simple model
  cannot separate your classes, a larger one on the same embeddings usually will not either; look at the labels.
- **Same year as the labels.** An embedding describes one year. Labels from 2021 and embeddings from 2024 describe different things.

## Similarity search

`similarity_map(embedding, row, col)` returns the cosine similarity of every pixel to the pixel at `(row, col)`, between -1 and 1, `NaN` where
there is no embedding. In the synthetic test the reference pixel scored 1.0 and its own region scored far above the other
region. Use it to find "more like this": a known field of a crop, a known quarry, a known informal settlement.

```python
sim = similarity_map(embedding, *src.index(0.31, 6.82))           # a reference location
candidates = sim > 0.9                                            # choose the threshold by looking at it, then check on the ground
```

The threshold has no universal value; map `sim`, look at it next to imagery, and pick one for your area.

## Comparing images through other models

`GeoEmbeddings` puts several extractors behind one convention (a flat vector per image or box) so you can compare things with
`cosine_similarity` and `nearest`:

```python
from pygeovision.models.embeddings import GeoEmbeddings, cosine_similarity, nearest

e = GeoEmbeddings(device="cpu")
vectors = {p: e.clip(p) for p in ["a.tif", "b.tif", "c.tif"]}      # RemoteCLIP (remoteclip-l14) by default; weights are downloaded
nearest(e.clip("query.tif"), vectors, top_k=2)                     # [(path, similarity), ...], best first
```

| Method | Input | Returns |
|---|---|---|
| `dofa(image, sensor=None, wavelengths=None, model_name=..., allow_random_init=False)` | An array and a sensor name or the band wavelengths | One vector. Refuses random weights unless allowed ([DOFA](dofa.md)) |
| `prithvi(image_path, source="hls", variant="prithvi_eo_2_0")` | A file path | The class-token vector ([Prithvi](prithvi.md)) |
| `dinov3(image, model_name="dinov3_vitl16_sat")` | A path or an array | One vector; the weights are gated ([DINOv3](dinov3.md)) |
| `clip(image_path, model="remoteclip-l14")` | A path | An L2-normalised vector ([CLIP](clip.md)) |
| `tessera(bbox, year=2024)` | A box | The `(128, H, W)` stack **averaged over all pixels** to one `(128,)` vector |

`cosine_similarity(a, b)` works offline on any two vectors and **refuses vectors of different length** (`ValueError: Cannot compare
embeddings of different real dimensions (128 vs 64)`): vectors from different models live in different spaces, so comparing them would
return a number that means nothing. It returns `0.0` if either vector is all zeros. `nearest` ranks a dictionary of named vectors by
cosine similarity, best first, and returns at most `top_k`. Each extractor keeps its own input requirements (DOFA an array plus a sensor,
Prithvi a file path, TESSERA a box); the class does not hide that.

## What to do when…

| Symptom | Cause | Fix |
|---|---|---|
| `ImportError: geotessera required ...` | Package missing | `pip install geotessera` |
| `{"success": False, "error": ...}` from `embeddings_for_bbox` | No network, or the area or year is not covered | Read `error`; try a different year or a smaller area |
| `earthengine-api required for AlphaEarth` | Package missing | `pip install earthengine-api`, then authenticate once |
| `embeddings_at_points` rows are all `NaN` | The points have no coverage for that year (sea, edge of the data) | Check the points are `(lon, lat)` and not swapped |
| `embeddings_for_bbox` returns a task id and no array | Earth Engine exports are asynchronous | Wait for the task, then download the GeoTIFF from Drive |
| Points come out in the wrong place | Coordinates given as `(lat, lon)` | Everything here is `(lon, lat)` |
| The classifier scores well on training points and badly on held-back ones | Points too close together, or too few per class | Spread them; hold back whole areas |
| `ValueError: Cannot compare embeddings of different real dimensions` | Two different models | Compare vectors from one model only |
| `some labelled points fall on pixels without an embedding` | A labelled point is on a masked pixel | Move or drop it |

## Related

- [Foundation models overview](index.md), [Prithvi-EO](prithvi.md), [DOFA](dofa.md), [CLIP and RemoteCLIP](clip.md)
- [Few-shot classification](../classification/few-shot.md)
- Python API: [`TesseraGeo` and `AlphaEarthGeo`](../reference/api/builders-foundation.md), [`GeoEmbeddings`](../reference/api/models.md)
