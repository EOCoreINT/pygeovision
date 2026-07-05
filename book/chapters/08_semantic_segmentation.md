# Chapter 8: Semantic Segmentation

## 8.1 Building Footprint Extraction

```python
from pygeovision.models import get_model
import torch

# SegFormer-B2 (state-of-the-art semantic segmentation)
model = get_model("segformer", variant="b2",
                   in_channels=6, num_classes=2)

# Tiled inference for large scenes
from pygeovision.inference.tiled import TiledInference

tiler = TiledInference(model=model, tile_size=512, overlap=64)
pred  = tiler.predict("./data/prepared.tif", output_path="./pred/buildings.tif")
```

## 8.2 Road Network Extraction

```python
result = client.pipeline("road_network",
                          bbox=BBOX, date="2026-06", output_dir="./roads/")
print(f"Road length: {result.stats.get('total_length_km', 0):.1f} km")
```

## 8.3 Water Body Segmentation

```python
result = client.pipeline("water_bodies",
                          bbox=BBOX, date="2026-06", output_dir="./water/")
print(f"Water coverage: {result.stats.get('water_area_km2', 0):.2f} km2")
```

## 8.4 Custom Training

```python
from pygeovision.training.data import GeoSegDataset, make_geo_loaders

train_dl, val_dl = make_geo_loaders(
    image_dir = "./data/train/images/",
    mask_dir  = "./data/train/masks/",
    batch_size = 8,
    tile_size  = 512,
    augment    = True,
)

from pygeovision.training import GeoTrainer
trainer = GeoTrainer(model=model, output_dir="./checkpoints/")
trainer.train(train_dl, val_dl, epochs=100, lr=3e-4)
```

## Exercises

1. Run semantic segmentation on your city with SegFormer-B2.
2. Extract the road network and compare length vs OSM data.
3. Segment agricultural fields. Compute mean field area.

## 8.5 Tiled Inference for Large Scenes

Sentinel-2 scenes are typically 10,980 × 10,980 pixels — too large to process
in a single GPU forward pass:

```python
from pygeovision.inference.tiled import TiledInference

# Auto-tile with overlap to avoid boundary artefacts
tiler = TiledInference(
    model     = segformer_b2,
    tile_size = 512,
    overlap   = 64,        # pixels of overlap to average at boundaries
    batch_size = 4,         # tiles per GPU batch
)
pred = tiler.predict("./data/large_scene.tif",
                      output_path="./pred/segmentation.tif")

print(f"Processed: {pred.shape[1]:,} x {pred.shape[2]:,} pixels")
print(f"Output:    {pred.output_path}")
```

## 8.6 Multi-Scale Inference (Test-Time Augmentation)

```python
def tta_predict(model, image, scales=(0.75, 1.0, 1.25), flips=True):
    """Test-time augmentation for improved accuracy."""
    import torch, torch.nn.functional as F

    preds = []
    for scale in scales:
        h, w = image.shape[-2:]
        img_scaled = F.interpolate(image, size=(int(h*scale), int(w*scale)),
                                    mode="bilinear", align_corners=False)
        with torch.no_grad():
            pred = model(img_scaled)
        pred = F.interpolate(pred, size=(h,w), mode="bilinear", align_corners=False)
        preds.append(pred)

        if flips:
            # Horizontal flip
            pred_flip = model(img_scaled.flip(-1)).flip(-1)
            pred_flip = F.interpolate(pred_flip, size=(h,w), mode="bilinear",
                                       align_corners=False)
            preds.append(pred_flip)

    return torch.stack(preds, 0).mean(0).argmax(1)
```

## 8.7 Domain Adaptation

```python
# When training and test domains differ (e.g. Ghana vs Ethiopia buildings)
# Use domain adaptation to improve generalisation:

class AdversarialDA(nn.Module):
    def __init__(self, encoder, task_head, domain_head):
        super().__init__()
        self.encoder     = encoder
        self.task_head   = task_head
        self.domain_head = domain_head  # predicts: source or target?

    def forward(self, x, alpha=1.0):
        feats = self.encoder(x)
        task_out   = self.task_head(feats)
        # Gradient reversal: confuse domain classifier
        domain_out = self.domain_head(GradientReversal.apply(feats, alpha))
        return task_out, domain_out
```

## Summary

Semantic segmentation assigns a class to every pixel. SegFormer-B2 is the
recommended baseline for most EO tasks. Use TiledInference for large scenes,
FocalDice loss for imbalanced classes, and TTA for production accuracy.

## Exercises

1. Run SegFormer-B2 over a full Sentinel-2 scene. How long does tiled inference take?
2. Compare TTA vs single-pass accuracy on a validation set.
3. Train on European cities and evaluate on an African city (domain shift).
4. Implement connected-component post-processing to remove small noise regions.
