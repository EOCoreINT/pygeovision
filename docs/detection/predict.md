# Predict with a detector

A detector returns **boxes**, not a raster, so it has its own prediction recipe. The tiled-inference engine and
`pygeovision infer predict` do not run detectors; they expect per-pixel scores.

| Task | Use |
|---|---|
| Ships or cars with the generic pretrained YOLO | `pygeovision ai detect ships --input port.tif --output ships.geojson` (needs `pip install ultralytics`). Writes **GeoJSON** boxes; the command refuses a `.tif` path and reports the detector's error if there is one |
| Your own YOLO weights | `GeoYOLO(weights="best.pt").detect_tiled("ortho.tif", tile_size=640, output_path="found.geojson")` |
| A torchvision detector you trained with `GeoTrainer(task="detection")` | The loop below |

```{admonition} "ai detect custom" is not for box detectors
:class: warning
`pygeovision ai detect custom` runs your model through the per-pixel engine, so it suits models that output a **score
map per class** (for example a segmentation model used as a detector of blobs). It does not run a torchvision detector.
```

## A trained torchvision detector over a scene

The steps: cut the scene into overlapping chips, run the model on each, keep boxes above a score threshold, move the boxes
from chip to scene pixels, remove duplicates where chips overlap, and convert pixels to map coordinates.

```python
import json
import rasterio
import torch
from rasterio.windows import Window
from torchvision.ops import nms
from pygeovision.models import get_model

model = get_model("faster-rcnn-r50", num_classes=1, in_channels=4, pretrained=False, min_size=128, max_size=128)
model.load_state_dict(torch.load("runs/det/checkpoints/best.pth", weights_only=True)["model_state_dict"])
model.eval()

CHIP, STEP, THRESHOLD = 128, 96, 0.5          # STEP < CHIP means the chips overlap by CHIP - STEP pixels
boxes, scores = [], []
with rasterio.open("scene.tif") as src, torch.no_grad():
    for row in range(0, src.height - CHIP + 1, STEP):
        for col in range(0, src.width - CHIP + 1, STEP):
            chip = torch.from_numpy(src.read(window=Window(col, row, CHIP, CHIP)).astype("float32"))
            out = model([chip])[0]                        # {'boxes': (N, 4), 'labels': (N,), 'scores': (N,)}
            keep = out["scores"] >= THRESHOLD
            offset = torch.tensor([col, row, col, row])   # chip pixels -> scene pixels
            boxes.append(out["boxes"][keep] + offset)
            scores.append(out["scores"][keep])
    boxes, scores = torch.cat(boxes), torch.cat(scores)
    keep = nms(boxes, scores, iou_threshold=0.5)          # the same object seen in two chips becomes one box
    features = []
    for (x1, y1, x2, y2), s in zip(boxes[keep].tolist(), scores[keep].tolist()):
        west, north = src.transform * (x1, y1)
        east, south = src.transform * (x2, y2)
        features.append({"type": "Feature", "properties": {"score": round(s, 3)},
                         "geometry": {"type": "Polygon", "coordinates": [[[west, north], [east, north],
                                      [east, south], [west, south], [west, north]]]}})
    crs = str(src.crs)
json.dump({"type": "FeatureCollection", "crs": {"type": "name", "properties": {"name": crs}}, "features": features},
          open("found.geojson", "w"))
```

Verified end to end on a synthetic 256 × 256 scene with the three-epoch toy detector from the [training page](train.md):
with `THRESHOLD = 0.05` (a model that barely trained keeps nothing above 0.5), the script wrote 8 boxes. With a properly
trained model use 0.5 and raise or lower it from the precision and recall you measure.

| Choice | Guidance |
|---|---|
| `CHIP` | The size you trained on. Set `min_size`/`max_size` to the same value |
| `STEP` | Smaller than `CHIP` so an object cut by a chip edge is whole in the next chip; an overlap of about the largest object is a good start |
| `THRESHOLD` | Start at 0.5. Lower finds more objects and more false alarms |
| `iou_threshold` of `nms` | 0.5 is usual. Lower merges more, higher keeps overlapping objects |

Apply the same pixel scaling to the chip as in training.

## DETR-style models

`detr-r50`, `detr-r101` and `rt-detr-l` take a batch tensor and return `logits` and `pred_boxes` (boxes as
centre-x, centre-y, width, height, **as fractions of the image**). Turn the logits into scores with a softmax over the last
axis (the last class is "no object") and convert the boxes to corners in pixels before using the recipe above.

## After detection

| You want | Do |
|---|---|
| A count | `len(features)` after the duplicate removal and a threshold you can defend |
| An accuracy figure | Match boxes to reference boxes at IoU 0.5 and report precision and recall; see [validation](../data/validation.md) |
| A map and a report | Load `found.geojson` into a GIS, or [Reporting](../reporting/index.md) |
