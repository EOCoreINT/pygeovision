# Few-shot classification

**Package:** `pygeovision.advanced.few_shot` · **Class:** `FewShotLearner` · **Needs:** `transformers`, `torch`, `rasterio`
(a DINOv2 backbone is downloaded from Hugging Face on first use)

## Why

Sometimes you must map a class that nobody has labelled: mangroves in a new region, a particular crop, informal
settlements. Collecting hundreds of labelled chips is exactly what you cannot afford. `FewShotLearner` needs **a handful of
example images per class** (the *support set*, typically 2–10).

How it works, with nothing hidden: a frozen foundation model (DINOv2) turns every picture into a vector of numbers that
describes what it looks like. Each class is represented by the **average vector** of its examples (its *prototype*). A new
image gets the class whose prototype is nearest. **No weights are trained**, so there is nothing to over-fit and nothing to
wait for; the price is that the result is only as good as DINOv2's idea of "looks similar" for your imagery.

```{admonition} What it is not
:class: warning
* **Not a trained classifier.** With 200+ examples per class a normally trained model ([Training](../training/index.md))
  will beat it.
* **Not able to say "none of these".** Every query gets the nearest of the classes *you gave*, even an image of something
  else entirely. Include a "background / other" class with examples if you need that outcome.
* **Not a pixel-accurate mapper.** `predict_geotiff` labels whole chips and blends them; edges are as sharp as the chip
  spacing.
* **Not designed for multispectral data.** DINOv2 was trained on ordinary colour photographs. The learner shows it **three
  bands** as an RGB picture. Pick the three that look natural, or accept that a false-colour composite is out of its
  training distribution.
* **Only prototypical networks.** `method="prototypical"` is the only method; MAML and matching networks are not
  implemented (asking for them raises `ValueError`).
```

## How to use

```{literalinclude} _examples/fewshot_demo.py
:language: python
:caption: fewshot_demo.py
:linenos:
```

{download}`Download fewshot_demo.py <_examples/fewshot_demo.py>`.

```{admonition} Read this about the example
:class: important
To run without a network the script builds a **tiny, randomly initialised** DINOv2-shaped model and passes its folder as
`backbone`. A random network knows nothing about the world; it can only separate images that differ in simple colour and
grain, which is why three synthetic land-cover colours work. The output shows the **API and the file formats**, not
the accuracy you will see with the real `dinov2-base`. This is also how to use a local copy of the real model: pass its
directory instead of a name.
```

### In four lines

```python
from pygeovision.advanced.few_shot import FewShotLearner

learner = FewShotLearner(backbone="dinov2-base", bands=(3, 2, 1), value_range=(0, 3000))
learner.fit_support({"mangrove": ["m1.tif", "m2.tif", "m3.tif"], "urban": ["u1.tif", "u2.tif", "u3.tif"]})
learner.predict(["query1.tif", "query2.tif"])
learner.predict_geotiff("scene.tif", "classes.tif", chip_size=224)
```

### Constructor arguments

| Argument | Default | What it does and when to change it |
|---|---|---|
| `backbone` | `"dinov2-base"` | `dinov2-small`, `-base`, `-large`, `-giant`, or any Hugging Face model id or **local folder** whose output has a class token first (ViT-style). Larger = better features, slower, more memory |
| `method` | `"prototypical"` | Only value |
| `device` | auto (CUDA if available) | `"cpu"` to force CPU |
| `metric` | `"euclidean"` | `"euclidean"` or `"cosine"`. Cosine ignores the vector length; try it if euclidean confuses bright/dark versions of the same class |
| `temperature` | auto | How sharply distances become scores. Default: a quarter of the median distance between prototypes (euclidean) or 0.05 (cosine). Lower = more extreme scores |
| `bands` | `(1, 2, 3)` | 1-based file bands shown as R, G, B. A Sentinel-2 file written B2, B3, B4, B8 needs `(3, 2, 1)` |
| `stretch` | `"joint"` | `"joint"`: one percentile pair for all three bands (keeps colour balance). `"per_band"`: each band separately (more contrast, colours shift) |
| `value_range` | `None` | `(low, high)`: scale **every** image linearly from this range instead of by percentiles. See below |

### Choosing `value_range` (important for consistent results)

By default each picture is stretched between *its own* 2nd and 98th percentiles. That has a consequence worth knowing:
**a dim uniform scene and a bright one look identical** (brightness is relative), and a chip containing only water is
stretched until its tiny variations look like strong texture. For `predict_geotiff`, where chips of very different content
sit side by side, this matters. If your data have a known physical scale, fix it: `value_range=(0, 3000)` for Sentinel-2
surface reflectance ×10 000, `(0, 255)` for 8-bit imagery. Every image and chip then uses the same scale, and brightness
differences between classes are preserved. Pixels equal to the file's nodata value (or NaN) are shown black.

### Preparing the support set

* 2–10 images per class, **cropped to the class**: a "forest" example that is half field teaches the prototype about fields.
* Show the **variation** you expect: different seasons, sensors, regions. The prototype is an average; examples from one
  place make a prototype that only fits that place.
* Same size and resolution as what you will classify. DINOv2 resizes pictures to about 224 px, so a 1 km chip and a 100 m
  chip are different things to it.
* At least two classes. A class whose images all fail to read raises `ValueError`; individual unreadable files are skipped
  with a warning (`class 'forest': using 2 of 3 images`).

## Expected output

### `fit_support` and `predict`

```text
classes: ['bare', 'forest', 'water']

== predict ==
q_water.tif  -> water   confidence 0.92  {'bare': 0.01, 'forest': 0.07, 'water': 0.92}
q_forest.tif -> forest  confidence 0.90  {'bare': 0.02, 'forest': 0.9, 'water': 0.08}
q_bare.tif   -> bare    confidence 0.97  {'bare': 0.97, 'forest': 0.02, 'water': 0.01}
```

`predict` returns one dictionary per query, in the order given:

```python
{"class": "water", "confidence": 0.9231, "probabilities": {"bare": ..., "forest": ..., "water": 0.9231}}
# with predict(..., return_distances=True) also "distances": {class: float} (smaller = closer)
```

* `class_names` (property) lists the classes alphabetically; the order in `probabilities` follows it.
* **`confidence` is a relative score, not a probability.** It is a softmax over distances to the prototypes of *your*
  classes. 0.92 does not mean "right 92 % of the time". Two classes that are both far from the query can still give 0.9 to
  the nearer one. Use it to **rank** results and to spot ties (two classes near 0.5), and validate on labelled images before
  trusting any threshold.

### An unreadable query does not stop the batch

```text
{'class': 'water', 'confidence': 0.9231}
{'class': None, 'confidence': 0.0, 'error': "RasterioIOError: '/tmp/.../broken.tif' n..."}
```

(a warning line is also logged). Check for `result["class"] is None` before using a result.

### `predict_geotiff`

```text
{'success': True, 'n_classes': 3, 'class_map': {'0': 'bare', '1': 'forest', '2': 'water'}, 'n_chips': 9, 'n_skipped_chips': 0}
dtype uint8 | nodata 255.0 | tags {'0': 'bare', '1': 'forest', '2': 'water'}
  code 0 = bare   : left half 0.00   right half 0.00
  code 1 = forest : left half 0.00   right half 1.00
  code 2 = water  : left half 1.00   right half 0.00
```

A 128×128 scene, left half water and right half forest, was labelled correctly with 64-pixel chips every 32 pixels
(9 chips). The output is a single-band `uint8` GeoTIFF on the **same grid and CRS** as the input:

* pixel value *i* = `class_map[str(i)]` (classes alphabetical); the same mapping is stored in the file's `class_map` tag
  as JSON;
* **255 = nodata**: pixels no chip covered (only possible where chips were skipped for being entirely nodata/NaN);
* how it is built: chips of `chip_size` pixels (default 224) are placed every `stride` pixels (default half a chip). The
  last row and column of chips are moved back so every chip has full size; an image smaller than one chip is one chip.
  Each pixel takes the class with the largest total of **confidence-weighted votes** from the chips covering it.
* `n_skipped_chips` counts chips with no valid pixel at all.

Choose `chip_size` to match the scale of your support images, and `stride` smaller for smoother boundaries (cost grows with
the square of the overlap).

## What to do when…

| Situation | What to do |
|---|---|
| `RuntimeError: could not load backbone 'facebook/dinov2-base'` | No network or no cache. Download the model once on a connected machine (`AutoModel.save_pretrained(folder)` and `AutoImageProcessor.save_pretrained(folder)`) and pass the **folder** as `backbone` |
| `ValueError: support must be a dict with at least two classes` | Prototype classification compares classes; give at least two |
| `ValueError: class 'x': none of its N images could be read` | Check the paths and that the files open in QGIS/rasterio; the message includes the first error |
| `RuntimeError: Call fit_support() before predict()` | Fit first. `predict_geotiff` instead returns `{"success": False, "error": ...}` (it does not raise) |
| Every query gets the same class | The support images are not distinguishable to the model (wrong `bands`, very different scales), or one class has non-typical examples. View the RGB pictures: `learner._read_rgb(path)` gives the exact array the backbone sees |
| Colours look wrong / vegetation not green | Wrong `bands` for your file order, or a false-colour composite. Use natural-colour bands; for B2,B3,B4,B8 files `bands=(3, 2, 1)` |
| Water and shadow get mixed up | Add a "shadow" class with examples, or use a different band combination. Foundation features do not know physics |
| Results change when you add a dim and a bright version of the same class | Percentile stretching removes absolute brightness. Set `value_range` |
| The map is blocky | Expected: it labels chips. Reduce `chip_size` and `stride`, or use the few-shot labels as *training labels* for a real segmentation model |
| `predict_geotiff` is slow | Cost is (image area / stride²) chips through the backbone. Use `dinov2-small`, a larger `stride`, a GPU, or classify a clipped area |
| You need a calibrated probability | This method cannot give one. Evaluate on held-out labelled chips and set a threshold from that |

## Verified here, and what is not

* **Verified (tests and the script above):** prototype building, nearest-prototype scoring and its graded confidence at any
  feature scale, support validation, unreadable-file handling, nodata handling, the GeoTIFF writer, windowed reading, the
  full embedding path through a real `transformers` `Dinov2Model` loaded from a local folder.
* **Not verified:** accuracy with the real pretrained DINOv2 on real imagery (the weights cannot be downloaded in the test
  environment). Treat the first run on your own data as an experiment and check it against a few labelled chips.
