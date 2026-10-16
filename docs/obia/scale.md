# Choosing a scale

**Functions:** `obia.scale_diagnostics`, `obia.scale_vs_reference`, `obia.viz.plot_scale` · **CLI:** `pygeovision obia scale`

## Why

For `multires` segmentation, `scale` is the single most important choice, and there is no universally right value. It
depends on how large the things you care about are. Too small and every field is shredded into pieces; too large and a
building is swallowed by its lawn. Instead of guessing, run **one** merging pass, pause it at each scale in a list and
compare the results. Two tools do that:

| Tool | Needs reference objects? | Answers |
|---|---|---|
| `scale_diagnostics` | No | At which scales does the segmentation change character? (a heuristic) |
| `scale_vs_reference` | Yes | Which scale best matches the objects *you* drew? (a measurement) |

If you can digitise even 20–30 reference objects, use `scale_vs_reference`: it measures what you actually want.

## Without reference: `scale_diagnostics`

```python
df = obia.scale_diagnostics("scene.tif", scales=[8, 12, 16, 20, 30, 40])
```

```console
$ pygeovision obia scale scene.tif --scales 8,12,16,20,30,40
```

It prints a table with one row per scale and the candidate scales underneath. The numbers depend entirely on your image.

| Column | Meaning |
|---|---|
| `n_objects` | Objects at that scale |
| `lv` | **Local variance**: the mean, over objects and bands, of the within-object standard deviation. Small = objects are uniform |
| `roc_lv` | Percent change of `lv` from the previous scale (the *rate of change*) |
| `candidate` | `True` at a local maximum of `roc_lv` |

The idea (Drăguţ et al. 2010, behind the ESP tool): as `scale` grows, objects first absorb pixels of the *same* thing
(variance rises quickly, then slowly) and then start swallowing *different* things. A peak in the rate of change marks a
scale where the character of the objects shifts. It is a **hint, not a verdict**.

Treat a candidate as a place to look, not a result. On a smooth curve a "peak" is weak evidence and can sit at a scale
that is too coarse. Always open the objects at the candidate scale (and the ones either side) and look.

Needs at least three scales (`ValueError` otherwise).

## With reference objects: `scale_vs_reference`

```python
from pygeovision import obia
import geopandas as gpd

ref_gdf = gpd.read_file("reference_objects.gpkg").to_crs(raster.crs)
ref_ids, _ = obia.rasterize_geometries(ref_gdf, raster.shape, raster.transform)
df = obia.scale_vs_reference(raster, [8, 12, 16, 20, 30, 40], ref_ids)
print(df["mean_D"].idxmin())
```

```console
$ pygeovision obia scale scene.tif --scales 8,12,16,20,30,40 --reference reference_objects.gpkg
```

It prints the table and `best scale by D (lower = better)`.

| Column | Meaning |
|---|---|
| `mean_over_seg` | How much of each reference object is *not* in its best segment (0 = reference lies inside one segment). Falls as scale rises |
| `mean_under_seg` | How much of each best segment lies *outside* the reference (0 = segment lies inside the reference). Rises with scale |
| `mean_D` | Combined distance `sqrt((over² + under²)/2)`. **0 = perfect.** Pick the minimum |
| `mean_QR` | Jaccard overlap, 1 = perfect |

Read the two columns together. Small scales over-segment, large scales under-segment, and `D` is lowest where they cross.
More detail on each measure is in
[Accuracy assessment](accuracy.md#segmentation-quality).

Plot it:

```python
from pygeovision.obia.viz import plot_scale
fig = plot_scale(df)                  # the table from either function; marks the best scale / candidates
fig.savefig("scale.png", dpi=150)
```

### Common options (both functions)

`rescale=True`, `color_weight=0.9`, `compactness_weight=0.5`, `init="auto"`, `init_segments=None`, `band_weights=None`, the same
as `segment`. Use the same values here as when you finally segment, or the chosen scale will not transfer.

## What to do when…

| You see | Cause | Do this |
|---|---|---|
| `ValueError: give at least 3 scales` | Too short a list | Give 3 or more; 6–10 is better |
| `candidate` column all `False` | The curve has no interior peak | Widen the range of scales, or use the reference-based table |
| Candidate scale looks wrong when you open the objects | The heuristic is weak on this scene | Trust your eyes and the reference-based table |
| Best `D` is at the smallest or largest scale you tried | The optimum is outside the range | Extend the list |
| `D` never gets small (stays above 0.3) | Reference objects are not well described by colour, or the reference is crude | Try other `color_weight`/`band_weights`, other methods, or accept that one scale cannot fit every object |
| Best scale differs between object types (small buildings vs big fields) | One scale cannot suit both | Segment at two scales with [`segment_levels`](segmentation.md#several-scales-at-once-segment_levels) and use [hierarchy](hierarchy-context.md) |
| Reference gives "no object" errors in the CLI | Reference CRS or extent mismatch | The CLI reprojects the reference to the image CRS; check they overlap |
| Run takes long | Many scales on a big image | One merging pass serves all scales, so cost is about one segmentation at the largest scale; reduce the image or use fewer scales |

```{seealso}
[Segmentation](segmentation.md) · [Accuracy assessment](accuracy.md)
```
