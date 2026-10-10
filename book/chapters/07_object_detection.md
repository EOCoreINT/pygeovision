# Chapter 7: Object Detection

## 7.1 Building Detection

```python
import pygeovision as pgv
client = pgv.PyGeoVision()

result = client.pipeline(
    "building_footprints",
    bbox       = (-0.30, 5.50, -0.05, 5.70),
    date       = "2026-06",
    output_dir = "./results/",
)
print(f"Buildings detected: {result.stats['n_buildings']:,}")
print(f"Total area: {result.stats['total_area_m2']/1e4:.0f} ha")
```

## 7.2 Ship Detection

```python
from pygeovision.models.detection import ShipDetector

detector = ShipDetector(confidence_threshold=0.7)
ships    = detector.detect("./data/sentinel2_sea.tif")

print(f"Ships detected: {len(ships)}")
for ship in ships:
    print(f"  {ship['lat']:.4f}, {ship['lon']:.4f}  conf={ship['confidence']:.2f}")
```

## 7.3 Solar Panel Detection

```python
result = client.pipeline(
    "solar_panels",
    bbox       = (8.5, 47.3, 8.6, 47.4),   # Zurich
    date       = "2026-06",
    output_dir = "./solar/",
)
capacity_kwp = result.stats.get("estimated_capacity_kwp", 0)
print(f"Solar capacity: {capacity_kwp:.0f} kWp")
```

## Exercises

1. Run building detection over a 5x5km study area.
2. Detect ships in a Sentinel-2 scene over the Strait of Gibraltar.
3. Compare building footprint accuracy vs OpenStreetMap ground truth.

## 7.4 Custom Training for Object Detection

```python
from pygeovision.models import get_model
from pygeovision.training import GeoTrainer
import torch

# YOLO-based detector for small objects (ships, vehicles, solar panels)
detector = get_model(
    "yolo_v8",
    in_channels = 3,
    num_classes = 4,    # background, building, vehicle, ship
    variant     = "medium",
)

# Training requires bounding box annotations in YOLO format:
# class_id center_x center_y width height (normalised 0-1)
trainer = GeoTrainer(
    model         = detector,
    output_dir    = "./checkpoints/detector/",
    task          = "detection",
    loss_fn       = "yolo",
)
trainer.train(train_loader=train_dl, val_loader=val_dl, epochs=100)
print(f"Best mAP@50: {trainer.best_metric:.4f}")
```

## 7.5 Post-Processing Detections

```python
import numpy as np

def nms(boxes, scores, iou_threshold=0.5):
    """Non-maximum suppression — remove duplicate detections."""
    if len(boxes) == 0:
        return []
    keep = []
    order = scores.argsort()[::-1]
    while order.size > 0:
        i = order[0]; keep.append(i)
        if order.size == 1: break
        xx1 = np.maximum(boxes[i,0], boxes[order[1:],0])
        yy1 = np.maximum(boxes[i,1], boxes[order[1:],1])
        xx2 = np.minimum(boxes[i,2], boxes[order[1:],2])
        yy2 = np.minimum(boxes[i,3], boxes[order[1:],3])
        inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
        area  = ((boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1]))
        union = area[i] + area[order[1:]] - inter
        iou   = inter / (union + 1e-8)
        order = order[1:][iou <= iou_threshold]
    return keep

# Compute mAP@50
from sklearn.metrics import average_precision_score

def compute_map(pred_boxes, true_boxes, iou_threshold=0.5):
    aps = []
    for class_id in range(num_classes):
        # Match predicted and ground truth boxes
        # Compute precision-recall curve
        # AP = area under PR curve
        pass  # Implementation depends on annotation format
    return float(np.mean(aps))
```

## Summary

Object detection localises and classifies individual objects within a scene.
PyGeoVision pipelines wrap this for common use cases (buildings, ships, solar
panels). Custom detectors require bounding box annotations and YOLO-format
training data.

## Exercises

1. Run `client.pipeline("building_footprints")` on a dense urban area.
2. Detect ships in a Sentinel-2 scene over a busy shipping lane.
3. Implement NMS post-processing on raw detection outputs.
4. Compute mAP@50 on a validation set of annotated aerial images.
