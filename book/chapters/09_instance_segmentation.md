# Chapter 9: Instance Segmentation

## 9.1 Individual Building Extraction

Instance segmentation separates individual objects (each building has its
own ID and boundary), unlike semantic segmentation which labels classes.

```python
from pygeovision.models.detection import InstanceSegmenter

seg    = InstanceSegmenter(model="mask_rcnn", in_channels=3)
result = seg.predict("./data/naip_scene.tif")

print(f"Buildings found: {len(result['instances'])}")
for inst in result['instances'][:5]:
    print(f"  ID={inst['id']}  area={inst['area_m2']:.0f}m2  "
          f"conf={inst['confidence']:.2f}")
```

## 9.2 Tree Crown Detection

```python
from pygeovision.models.detection import TreeCrownDetector

detector = TreeCrownDetector(min_area_m2=5)
crowns   = detector.detect("./data/lidar_chm.tif")

n_trees    = len(crowns)
mean_crown = sum(c["area_m2"] for c in crowns) / n_trees
print(f"Trees: {n_trees:,}  Mean crown: {mean_crown:.1f} m2")
```

## Exercises

1. Run instance segmentation on a high-resolution aerial image.
2. Compute the building footprint distribution (area histogram).
3. Detect tree crowns in a forest scene. Estimate stem density (trees/ha).

## 9.3 Evaluation Metrics for Instance Segmentation

```python
import numpy as np

def compute_instance_iou(pred_mask, true_mask):
    """Compute IoU between two binary instance masks."""
    inter = (pred_mask & true_mask).sum()
    union = (pred_mask | true_mask).sum()
    return inter / (union + 1e-8)

def match_instances(pred_instances, true_instances, iou_threshold=0.5):
    """Match predicted instances to ground truth using Hungarian matching."""
    from scipy.optimize import linear_sum_assignment
    import numpy as np

    n_pred = len(pred_instances)
    n_true = len(true_instances)
    iou_matrix = np.zeros((n_pred, n_true))

    for i, pred in enumerate(pred_instances):
        for j, true in enumerate(true_instances):
            iou_matrix[i, j] = compute_instance_iou(pred, true)

    pred_idx, true_idx = linear_sum_assignment(-iou_matrix)
    matched = [(pi, ti) for pi, ti in zip(pred_idx, true_idx)
               if iou_matrix[pi, ti] >= iou_threshold]

    tp = len(matched)
    fp = n_pred - tp
    fn = n_true - tp
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2*precision*recall / (precision + recall + 1e-8)
    return {"tp":tp, "fp":fp, "fn":fn, "precision":precision,
            "recall":recall, "f1":f1}
```

## 9.4 Applications

### Individual tree crown delineation

```python
# Tree crowns in high-resolution imagery
from pygeovision.models.detection import TreeCrownDetector

det     = TreeCrownDetector(min_area_m2=4, nms_iou=0.3)
crowns  = det.detect("./data/lidar_chm.tif")

n_trees = len(crowns)
ha      = sum(c["area_m2"] for c in crowns) / 1e4
density = n_trees / (total_area_ha)
print(f"Trees:   {n_trees:,}")
print(f"Density: {density:.0f} trees/ha")
print(f"Mean crown: {sum(c['area_m2'] for c in crowns)/n_trees:.1f} m2")
```

## Summary

Instance segmentation produces individual object masks. It requires panoptic-
style annotations and NMS post-processing. Useful for building footprint
separation, tree crown mapping, and vessel instance tracking.

## Exercises

1. Run instance segmentation on a dense urban block. Compute building count.
2. Detect tree crowns in a forest patch. Estimate stems per hectare.
3. Implement NMS for overlapping building predictions.
