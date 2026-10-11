# Object features

**Functions:** `obia.extract_features`, `obia.glcm_texture`, `obia.spectral_layers` · **CLI:** `pygeovision obia features`

## Why

A classifier, whether it is a rule you wrote or a random forest, can only use what you hand it. Pixels give you four or six
numbers. An **object** gives you dozens, and many of them are the very things people use to tell land cover apart by eye:
*how big* is it, *how round*, *how rough*, *what does it touch*. `extract_features` builds that table: one row per object,
one column per measurement.

## How to use

```python
from pygeovision import obia

feats = obia.extract_features(
    seg.labels, scene.raster,
    extra_layers=obia.spectral_layers(scene.raster),     # adds ndvi, ndwi
    texture_layers=["nir"],                              # optional: GLCM texture of the NIR band
)
print(feats.shape)                                        # (45, 52): 45 objects x 52 features
feats.loc[3, ["area_m2", "compactness", "mean_ndvi"]]
```

```console
$ pygeovision obia features scene.tif labels.tif --output features.csv --texture nir
  ✓ 45 objects × 52 features → features.csv
```

`--output` ending in `.csv` writes the table; a `.gpkg` or `.geojson` writes **polygons with the features as attributes**,
which you can colour by any column in QGIS.

### `extract_features(labels, image, band_names=None, *, extra_layers=None, transform=None, pixel_area=None, spectral=True, geometry=True, texture_layers=None, texture_levels=16, context=True)`

| Argument | Meaning |
|---|---|
| `labels` | 2-D integer raster of object ids (`SegmentationResult.labels`, or read from a file) |
| `image` | `RasterData` or `(bands, rows, cols)` array. Must be on the same grid as `labels` |
| `extra_layers` | `{name: 2-D array}` derived layers (NDVI, a height model, slope). Each gets the same spectral and context statistics as the bands |
| `transform` / `pixel_area` | Give either to get `area_m2` and `centroid_x/y`. Taken from the `RasterData` if present |
| `spectral`, `geometry`, `context` | Switch whole groups off to save time and columns |
| `texture_layers` | Names of layers to compute GLCM texture for (slow; see below) |
| `texture_levels` | Grey levels for texture (default 16) |

Raises `ValueError` if `labels` is not 2-D or an extra layer has the wrong shape, and `KeyError` if a texture layer does not
exist (the message lists the layers you have).

## The columns

Names are built from your band and layer names, so with bands `blue, green, red, nir` plus `ndvi`/`ndwi` you get
`mean_nir`, `std_ndvi`, and so on. Punctuation in names becomes `_`, and a name starting with a digit gets a leading `_`.
Use these names directly in rules.

### Spectral (per band and per extra layer)

| Column | Meaning |
|---|---|
| `mean_<layer>` | Average value inside the object |
| `std_<layer>` | Standard deviation inside: low for a flat field, high for forest |
| `min_<layer>`, `max_<layer>` | Extremes inside the object |
| `brightness` | Mean of the object's mean values over the **bands only** (not the extra layers) |

### Geometry

| Column | Meaning | Typical use |
|---|---|---|
| `area_px` | Pixels | Size rules; ignore tiny objects |
| `area_m2` | Area in map units². **Square metres only for a projected CRS** | Compare with real-world sizes |
| `perimeter_px` | Boundary length in **pixel edges**, including edges on the image frame | |
| `compactness` | `4πA/P²`. A perfect disc is 1; a pixel-aligned square is 0.785; staircase edges lower both. Lower = more ragged or elongated | Buildings are compact, roads are not |
| `shape_index` | `P / (4√A)`: exactly 1 for a square, larger for ragged or long shapes | Alternative to compactness |
| `elongation` | Major axis ÷ minor axis of the best-fit ellipse | Roads, rivers: large |
| `solidity` | Area ÷ area of the convex hull. Low = concave, with holes or arms | Separating L-shaped or branching objects |
| `extent` | Area ÷ area of the bounding box | Axis-aligned rectangles score high |
| `orientation` | Angle of the major axis in degrees | Row crops, parcels |
| `centroid_row`, `centroid_col` | Object centre in pixels | |
| `centroid_x`, `centroid_y` | Centre in map coordinates (with a transform) | |

### Texture (opt-in, `texture_layers=[…]`)

`glcm_<property>_<layer>` for `contrast`, `dissimilarity`, `homogeneity`, `energy`, `entropy`, `correlation`. A GLCM (grey-level
co-occurrence matrix) counts how often each pair of grey levels sits side by side inside the object. The layer is first
quantised to 16 levels using the 2nd–98th percentiles of the **whole image**. Only pixel pairs where *both* pixels belong to
the object are counted (pixels outside are masked, not treated as a grey level), pooled over four directions at distance 1.
Objects with fewer than 9 pixels get `NaN`.

### Context (neighbours)

| Column | Meaning |
|---|---|
| `n_neighbors` | Number of touching objects (4-connectivity) |
| `rel_border_image_edge` | Share of the boundary lying on the image frame. High values flag objects that are cut off by the scene edge, whose shape is not real |
| `diff_to_neighbors_<layer>` | Border-length-weighted mean of the absolute difference between this object's mean and its neighbours' means. Large = stands out from its surroundings |

Class-dependent context ("touches water") and hierarchy features need a first classification or a second segmentation level;
see [Hierarchy and context](hierarchy-context.md).

## What the features look like by class

Median over the objects of each true class on the demo scene (scale 20). This is what the classifier "sees":

| Class | `area_px` | `compactness` | `elongation` | `mean_nir` | `std_nir` | `mean_ndvi` | `glcm_contrast_nir` |
|---|---:|---:|---:|---:|---:|---:|---:|
| water | 1443 | 0.46 | 2.2 | 14 | 2.8 | −0.44 | 0.00 |
| forest | 1559 | 0.53 | 1.4 | 129 | **13.9** | 0.69 | **2.21** |
| cropland | 172 | 0.35 | 1.7 | **169** | 5.4 | 0.60 | 0.38 |
| bare_field | 158 | 0.33 | 2.0 | 127 | 5.0 | −0.02 | 0.44 |
| building | 62 | **0.67** | 1.5 | 117 | 6.3 | −0.08 | 0.52 |
| road | 310 | **0.15** | **19.8** | 90 | 6.1 | −0.07 | 0.50 |

Read the bold cells. Forest and cropland have similar NDVI, but forest is far more variable (`std_nir`, `glcm_contrast_nir`).
Bare fields and buildings are close spectrally (NIR 127 vs 117), but buildings are compact and small. Roads are
spectrally unremarkable but have an elongation of about 20. That is the case for object features: **shape and texture
separate what colour alone cannot**.

## Performance

Spectral, geometry and context features are vectorised and take milliseconds for hundreds of objects. **Texture is the
slow part**, one GLCM per object per layer, so ask for it only for the layers you need (usually one band such as NIR).

## What to do when…

| You see | Cause | Do this |
|---|---|---|
| `KeyError: texture layer 'x' not found; have [...]` | Name mismatch | Use a name from the list. Names are cleaned (`B 4` becomes `B_4`) |
| `ValueError: extra layer 'ndvi' has shape …` | Layer not on the image grid | Compute it from the same arrays |
| No `mean_ndvi` column | Bands not named `red`/`nir` | Name the bands (or pass `band_names=`), or build the index yourself and pass it in `extra_layers` |
| `NaN` in `glcm_*` | Object smaller than 9 pixels | Merge small objects (raise `scale`), or drop them; `ObjectClassifier` fills `NaN` with the median |
| `area_m2` looks like 4e-10 | The CRS is geographic (degrees) | Reproject the image to a metric CRS before segmenting |
| `compactness` never reaches 1 | Raster discs have staircase edges | Expected; compare objects with each other, not with 1 |
| Odd shape values for objects at the scene edge | Cut off by the frame | Check `rel_border_image_edge` and exclude high values |
| Features differ between runs | Different segmentation | Features describe the objects you give them; fix the segmentation first |
| `ValueError: labels must be 2-D` | A 3-D array | Pass the 2-D label raster |

```{seealso}
[Classification](classification.md) · [Hierarchy and context](hierarchy-context.md)
```
