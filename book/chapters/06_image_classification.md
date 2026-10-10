# Chapter 6: Image Classification

## 6.1 Scene Classification

Classify an entire image into a single category (urban, forest, water, etc.):

```python
import pygeovision as pgv
from pygeovision.models import get_model
import torch

# ResNet-based scene classifier
classifier = get_model("resnet50_scene",
                        in_channels=6, num_classes=20)

# Inference
with rasterio.open("./data/prepared.tif") as src:
    patch = torch.tensor(src.read()).unsqueeze(0).float()

logits = classifier(patch)
probs  = logits.softmax(dim=1)
top3   = probs.topk(3)
print(f"Top class: {top3.indices[0][0].item()}")
```

## 6.2 Land Cover Classification

Multi-class per-pixel classification with Prithvi:

```python
from pygeovision.models.foundation.prithvi import PrithviTasks

tasks  = PrithviTasks("prithvi_eo_2_0")
result = tasks.land_cover(ready_array, source="sentinel2")
classes = result["class_distribution"]

print("Land cover distribution:")
for cls, pct in classes.items():
    print(f"  {cls:<20}: {pct:.1f}%")
```

## 6.3 Zero-Shot with CLIP

Classify scenes without labelled training data:

```python
from pygeovision.models.vlm.clip_zero_shot import CLIPZeroShot

clip = CLIPZeroShot()
labels = ["flooded area", "vegetation", "urban", "bare soil", "water body"]

result = clip.classify(image_path="./data/scene.tif", labels=labels)
print(f"Predicted: {result['label']}  ({result['confidence']:.1%})")
```

## Exercises

1. Classify 10 Sentinel-2 patches using scene classification.
2. Apply Prithvi land cover to your study area. Create a pie chart.
3. Test CLIP zero-shot classification with 5 domain-specific labels.

## 6.4 Multi-Label Classification

Some scenes contain multiple land-cover types. Multi-label classification predicts
a binary vector of class presence:

```python
import torch, torch.nn as nn

class MultiLabelClassifier(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.head     = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(backbone.out_channels, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))   # sigmoid applied at loss time

# Training with BCEWithLogitsLoss
criterion = nn.BCEWithLogitsLoss()
logits    = model(images)
loss      = criterion(logits, labels.float())

# Threshold predictions at 0.5
preds = (torch.sigmoid(logits) > 0.5).long()
```

## 6.5 Training Custom Classifiers

```python
from pygeovision.training import GeoTrainer
from pygeovision.training.data import GeoSegDataset, make_geo_loaders
from pygeovision.models import get_model
import torch

# 1. Prepare dataset
train_dl, val_dl = make_geo_loaders(
    image_dir  = "./data/classification/train/images/",
    mask_dir   = "./data/classification/train/labels/",
    batch_size = 32,
    tile_size  = 224,
    augment    = True,
)

# 2. Load model (ResNet50 for scene classification)
model = get_model("resnet50_scene", in_channels=6, num_classes=10)

# 3. Train with one-cycle learning rate
trainer = GeoTrainer(
    model         = model,
    output_dir    = "./checkpoints/scene_classifier/",
    loss_fn       = "cross_entropy",
    metrics       = ["accuracy", "f1_macro"],
)
trainer.train(
    train_loader  = train_dl,
    val_loader    = val_dl,
    epochs        = 30,
    lr            = 1e-3,
    weight_decay  = 0.01,
    scheduler     = "cosine",
)
print(f"Best val accuracy: {trainer.best_metric:.4f}")
```

## 6.6 Model Inference and Deployment

```python
import torch
import numpy as np

# Load trained model
model = get_model("resnet50_scene", in_channels=6, num_classes=10)
model.load_state_dict(torch.load("./checkpoints/best_model.pth"))
model.eval()

# Batch inference
class_names = ["Forest","Cropland","Urban","Water","Desert",
               "Wetland","Grassland","Cloud","Snow","Other"]

with torch.no_grad():
    for images, paths in dataloader:
        logits = model(images)
        probs  = logits.softmax(dim=1)
        preds  = probs.argmax(dim=1)

        for path, pred, prob in zip(paths, preds, probs):
            top3  = prob.topk(3)
            label = class_names[pred.item()]
            conf  = prob[pred].item()
            print(f"{path}: {label} ({conf:.1%})")
            for idx, score in zip(top3.indices, top3.values):
                print(f"    {class_names[idx]}: {score:.1%}")
```

## Summary

Scene classification assigns a single category to an entire image tile.
Prithvi foundation model handles zero-shot classification. CLIP extends this
to arbitrary text descriptions. Custom classifiers need geographically
stratified train/val splits to avoid data leakage.

## Exercises

1. Classify 50 Sentinel-2 patches using Prithvi. Which class is most common?
2. Apply CLIP zero-shot classification with 8 custom land-cover labels.
3. Train a ResNet50 scene classifier on the EuroSAT benchmark.
4. Compare per-class accuracy. Which class has the lowest recall?
