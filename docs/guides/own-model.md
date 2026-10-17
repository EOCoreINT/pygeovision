# Train a model and use it

The ready pipelines answer questions that have a published method: where is the water, how severe was the burn, how did the
vegetation change. When your question is about **your own classes** (a crop type, a roof material, a habitat), no published method
exists and you train a network on examples. This guide is the whole route in order, with the commands, what you have at the end of
each step, and what usually goes wrong. Each step links to the page that covers it in depth.

## Do you need to train?

Training costs labelled examples (the expensive part), time and care, and an unchecked network produces maps that look right and are
not. Try the cheaper options first:

| If… | Then use… |
|---|---|
| A [pipeline](../pipelines/choosing.md) already answers the question | The pipeline. No training. |
| You have a handful of labelled examples per class | [Few-shot classification](../classification/few-shot.md) or [object-based analysis](../obia/index.md), which work from tens of samples |
| You want a map of "things like this example" | [Foundation-model embeddings](../foundation-models/embeddings.md) |
| You have hundreds of labelled scenes or chips and a custom class list | Train a model: continue below |

## The route

| Step | What you do | What you have afterwards |
|---|---|---|
| 1. Imagery | Download or collect the scenes | GeoTIFFs of your area |
| 2. Labels | Make or import a raster with one class number per pixel | A label raster on the **same grid** as each scene |
| 3. Chips | Cut scenes and labels into tiles | `images/` and `labels/` folders |
| 4. Train | Fit a network and keep the best epoch | A checkpoint file (`best.pth`) |
| 5. Check | Predict a scene the model never saw, and look | Evidence the model works, or why not |
| 6. Use | Run it on new scenes, or inside a pipeline | Maps from your own model |
| 7. Keep it honest | Explain, compare and watch it | A model you can defend |

### 1. Get imagery

Any GeoTIFF will do. To fetch Sentinel-2 for your box, run a search and download ([Search and download](../data/search-and-download.md))
or let a pipeline fetch and prepare a scene for you. Keep **the same sensor and the same preparation** for training and for later
use: a network trained on reflectance scaled 0 to 1 gives wrong answers on raw counts.

### 2. Make labels

Labels are a raster with a class number in every pixel (0 for background). The quickest source is OpenStreetMap, burned onto the
exact grid of your scene:

```bash
pygeovision label osm --reference data/scene.tif --categories buildings --output labels/scene.tif \
    -- -74.05 40.70 -73.95 40.80
```

The four numbers at the end are the box (west, south, east, north). They come after `--` because the first number is negative; leave
the `--` out for a box east of Greenwich. Then check the labels before you train on them:

```bash
pygeovision label quality labels/scene.tif
```

Open the label raster over the imagery in your GIS and look at a few places. **Labels from automatic sources are approximate**:
OpenStreetMap is incomplete in many regions, and a model learns the labels' mistakes as faithfully as their strengths. Other sources
(WorldCover, building footprints, Segment Anything, your own digitising) and ways to combine them are in
[Labeling](../labeling/index.md); [From a box to training chips](../labeling/workflow.md) is a full worked example.

### 3. Cut chips

Networks train on small tiles, not whole scenes. Cut each scene and its label into aligned chips:

```bash
pygeovision ai chips --image data/scene.tif --label labels/scene.tif --output data/train --chip-size 256 --prefix scene1
```

Use a different `--prefix` for each scene when several go into the same folder, or later scenes overwrite earlier chips. Keep at
least one scene (or one area) **out of training**: it is the only honest test of the model later.
[Preparing training data](../training/data.md) covers splitting, class balance and scaling.

### 4. Train

Training is done from Python, with `GeoTrainer`. The [Training page](../training/index.md) has a complete working example, a real
training log and how to read it. In outline:

1. Open the chips with `GeoSegDataset` (one for training, one for validation).
2. Build a network with `get_model(name, num_classes=..., in_channels=...)`. Both numbers must match your labels and bands.
3. Describe the run in a `TrainingConfig` and call `GeoTrainer(model, cfg).fit(train, val)`.

It writes checkpoints under the output folder, and `best.pth` is the one to keep. Install the training tools first:

```bash
pip install "pygeovision[train]"
```

[Choosing a model](../models/choosing.md) helps pick the architecture, and [Loss functions](../training/losses.md) matters when the
class you care about is rare.

```{admonition} "pygeovision ai train" does not train
:class: warning
That command prints a configuration and the Python lines to run. It writes no model. Training happens in Python so that the
decisions about data, splits and losses are visible in your code.
```

### 5. Check on a scene it has not seen

The validation score in the training log is a first sign. The real test is a scene that played no part in training:

```bash
pygeovision infer predict data/held_out_scene.tif \
    --model unet-r50 --checkpoint runs/demo/checkpoints/best.pth \
    --classes 2 --in-channels 4 --no-normalise \
    --output predictions/held_out.tif
```

Open the prediction over the imagery. A good model looks right in the places you know, not just on average. Score it against labels
you trust with [`postprocess accuracy`](../data/postprocessing.md) or the [benchmark tools](../experiments/benchmark.md).

### 6. Use it

**On more scenes**, with `infer predict`, or `infer batch` for a folder ([Inference](../inference/index.md)). Three arguments must
describe the model you trained, not the one you wish you had: `--model`, `--classes` and `--in-channels`.

**Inside a pipeline**, so that you also get the report, vectors and run record. Seven pipelines run your network, and three more
(`change_detection`, `water_bodies`, `land_cover`) can:

```bash
pygeovision channel building_footprints \
    --bbox -0.15 51.47 -0.10 51.52 --date 2024-06 \
    --checkpoint runs/demo/checkpoints/best.pth \
    --class-names "Background,Building" \
    --output out/buildings
```

The pipeline loads the checkpoint **strictly**. A different architecture, class count or band count is an error, never a silent
fallback to something else.

### 7. Keep it honest

| Question | Tool |
|---|---|
| Why did it predict that? | [Explainability](../explainability/index.md): Grad-CAM, SHAP, uncertainty |
| Is it better than the alternative? | [Experiments](../experiments/index.md): tracking, search and comparison |
| Is it still working next year? | [Monitoring](../monitoring/index.md): drift and performance |
| How do I run it somewhere else? | [Deployment](../deployment/index.md): ONNX, edge devices, cloud |

## What usually goes wrong

| Symptom | Likely cause | Fix |
|---|---|---|
| The prediction is one class everywhere | The rare class was swamped by background | A [class-balanced loss](../training/losses.md); more examples of the rare class |
| Fine on validation, wrong on new scenes | Validation chips came from the same scenes as training | Hold out whole scenes or whole areas |
| The prediction looks washed out or noisy | Scaling at prediction differs from training | Use `--no-normalise` if you trained on 0 to 1 values, and the stretch only if you trained with it ([Inference](../inference/index.md#matching-the-input-to-training)) |
| An error about shapes or channels | `--classes` or `--in-channels` differ from training | Give the values you trained with |
| Labels are shifted from the buildings | Imagery and labels are not on one grid | Re-create the labels with `--reference` |
| The map is checkered with seams | Tiles without overlap | Raise `--overlap` ([Tiles and seams](../inference/tiles.md)) |

## Related

- [Training](../training/index.md), [Inference](../inference/index.md), [Labeling](../labeling/index.md): the sections behind each step.
- [Models](../models/index.md): the 79 architectures and how to choose.
- [Reading results](../pipelines/results.md): the checks that apply to a pipeline run with your model too.
