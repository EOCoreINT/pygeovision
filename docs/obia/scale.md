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
df = obia.scale_diagnostics("scene.tif", scales=[5, 10, 15, 20, 30, 40, 60, 80])
```

```console
$ pygeovision obia scale scene.tif
       n_objects     lv  roc_lv  candidate
scale
5.0          895  10.18     NaN      False
10.0         176  16.04   57.48      False
15.0           66  20.14   25.57      False
20.0          45  20.41    1.35      False
30.0          38  19.73   -3.36      False
40.0          29  19.63   -0.50       True
60.0          22  18.14   -7.59      False
80.0          20  16.82   -7.27      False

  ROC-LV candidate scales (heuristic — verify visually): [40.0]
```

(real output on the synthetic demo scene)

| Column | Meaning |
|---|---|
| `n_objects` | Objects at that scale |
| `lv` | **Local variance**: the mean, over objects and bands, of the within-object standard deviation. Small = objects are uniform |
| `roc_lv` | Percent change of `lv` from the previous scale (the *rate of change*) |
| `candidate` | `True` at a local maximum of `roc_lv` |

The idea (Drăguţ et al. 2010, behind the ESP tool): as `scale` grows, objects first absorb pixels of the *same* thing
(variance rises quickly, then slowly) and then start swallowing *different* things. A peak in the rate of change marks a
scale where the character of the objects shifts. It is a **hint, not a verdict**.

Look at the example. The candidate is 40, with 29 objects. The reference-based table below shows that 20 is the best match
to the true objects, and 40 already merges things that should stay apart (under-segmentation 0.26). **The heuristic picked a
scale that is too coarse.** That is not a bug: with a clean synthetic scene and a gently changing curve, a "peak" is weak
evidence. Always open the objects at the candidate scale and look.

Needs at least three scales (`ValueError` otherwise).

## With reference objects: `scale_vs_reference`

```python
from pygeovision import obia
import geopandas as gpd

ref_gdf = gpd.read_file("reference.gpkg").to_crs(raster.crs)
ref_ids, _ = obia.rasterize_geometries(ref_gdf, raster.shape, raster.transform)
df = obia.scale_vs_reference(raster, [5, 10, 15, 20, 30, 40, 60, 80], ref_ids)
print(df["mean_D"].idxmin())
```

```console
$ pygeovision obia scale scene.tif --reference reference.gpkg
       n_objects  mean_over_seg  mean_under_seg  mean_D  mean_QR
scale
5.0          895          0.655           0.010   0.464    0.340
10.0         176          0.349           0.054   0.273    0.616
15.0          66          0.136           0.062   0.129    0.819
20.0          45          0.079           0.095   0.109    0.845
30.0          38          0.076           0.155   0.149    0.788
40.0          29          0.057           0.261   0.211    0.702
60.0          22          0.051           0.403   0.306    0.567
80.0          20          0.050           0.439   0.331    0.531

  best scale by D (lower = better): 20
```

| Column | Meaning |
|---|---|
| `mean_over_seg` | How much of each reference object is *not* in its best segment (0 = reference lies inside one segment). Falls as scale rises |
| `mean_under_seg` | How much of each best segment lies *outside* the reference (0 = segment lies inside the reference). Rises with scale |
| `mean_D` | Combined distance `sqrt((over² + under²)/2)`. **0 = perfect.** Pick the minimum |
| `mean_QR` | Jaccard overlap, 1 = perfect |

Read the two columns together. Small scales over-segment (0.655 at scale 5), large scales under-segment (0.439 at 80), and
`D` is lowest where they cross, here at 20. More detail on each measure is in
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
