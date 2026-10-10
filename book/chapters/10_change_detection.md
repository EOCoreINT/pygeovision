# Chapter 10: Change Detection

## 10.1 Bi-Temporal Change Detection

```python
from pygeovision.models.change_detection.changeformer import ChangeDetection

cd = ChangeDetection(model_variant="changeformer",
                      in_channels=6, num_classes=2)
cd.build()

result = cd.detect(
    pre_path    = "./data/before.tif",
    post_path   = "./data/after.tif",
    output_path = "./change/change_mask.tif",
)
print(f"Changed area: {result['change_pct']:.1f}%")
```

## 10.2 Deforestation Detection

```python
result = client.pipeline("forest_monitoring",
                          bbox=BBOX, date="2026-06", output_dir="./forest/")
print(f"Forest loss: {result.stats.get('loss_area_km2', 0):.1f} km2")
print(f"Loss rate:   {result.stats.get('annual_loss_pct', 0):.2f}%/year")
```

## 10.3 4-Class Damage Assessment

```python
# 4-class: 0=no damage, 1=minor, 2=moderate, 3=severe
cd4 = ChangeDetection(model_variant="changeformer",
                       in_channels=6, num_classes=4)
cd4.build()

damage = cd4.detect("./before.tif", "./after.tif", output_path="./damage.tif")

for cls, label in enumerate(["No damage","Minor","Moderate","Severe"]):
    pct = (damage.prediction == cls).mean() * 100
    print(f"  {label:<12}: {pct:.1f}%")
```

## 10.4 Urban Growth Analysis

```python
result = client.pipeline("urban_growth",
                          bbox=BBOX, date="2026-06", output_dir="./urban/")
growth_km2 = result.stats.get("new_urban_area_km2", 0)
print(f"New urban area (last decade): {growth_km2:.1f} km2")
```

## Exercises

1. Compare pre/post Sentinel-2 for a major construction project.
2. Map deforestation in a rainforest region over 5 years.
3. Run 4-class damage assessment on earthquake imagery.

## 10.5 Multi-Temporal Change Detection

```python
import numpy as np, rasterio

paths = ["./data/ndvi_2023.tif","./data/ndvi_2024.tif","./data/ndvi_2025.tif"]
stack = []
for path in paths:
    with rasterio.open(path) as src:
        stack.append(src.read(1).astype("float32"))
stack  = np.stack(stack, axis=0)   # (T, H, W)
trend  = np.polyfit(range(len(paths)), stack.reshape(len(paths),-1), 1)[0]
trend  = trend.reshape(stack.shape[1], stack.shape[2])
print(f"Declining pixels: {(trend < -0.05).mean()*100:.1f}%")
```

## 10.6 Accuracy Assessment

```python
from sklearn.metrics import confusion_matrix
import numpy as np

def change_accuracy(true, pred):
    cm    = confusion_matrix(true.ravel(), pred.ravel(), labels=[0,1])
    oa    = cm.diagonal().sum() / cm.sum()
    tp,tn,fp,fn = cm[1,1],cm[0,0],cm[0,1],cm[1,0]
    po    = oa
    pe    = ((cm[0,:].sum()*cm[:,0].sum()) + (cm[1,:].sum()*cm[:,1].sum())) / cm.sum()**2
    kappa = (po - pe) / (1 - pe + 1e-8)
    f1    = 2*tp / (2*tp+fp+fn+1e-8)
    print(f"OA={oa:.4f}  Kappa={kappa:.4f}  F1={f1:.4f}")
    return {"oa":oa,"kappa":kappa,"f1":f1}
```

## Summary

Change detection compares multi-temporal imagery. Kappa coefficient is the
standard metric. Multi-temporal stacking reveals gradual changes invisible
in bi-temporal comparisons.

## Exercises

1. Detect deforestation in the Amazon between two Sentinel-2 dates.
2. Map urban expansion in a fast-growing city over 10 years.
3. Compute Kappa coefficient for your change detection results.
