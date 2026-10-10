# Chapter 12: Foundation Models

## 12.1 What Are Foundation Models?

Foundation models are pre-trained on massive datasets and adapted for
downstream tasks with minimal fine-tuning. In EO, they learn general
representations of the Earth's surface that transfer across scenes, seasons,
and regions.

## 12.2 Prithvi-EO-2.0

NASA's Prithvi-EO-2.0 is a 600M parameter masked autoencoder pre-trained on
HLS (Harmonised Landsat Sentinel) data covering the continental United States.

```python
from pygeovision.models.foundation.prithvi import (
    PrithviTasks, validate_prithvi_input,
    LAND_COVER_CLASSES, CROP_CLASSES,
)

# Available task methods
tasks = PrithviTasks("prithvi_eo_2_0")

# Validate input before inference
val = validate_prithvi_input(ready_array, source="sentinel2", n_prithvi_bands=6)
if not val["valid"]:
    print(f"Input error: {val}")

# Land cover (9-class ESA WorldCover)
lc = tasks.land_cover(ready_array, source="sentinel2")
print(f"Classes: {LAND_COVER_CLASSES}")
print(f"Distribution: {lc['class_distribution']}")

# Flood detection
fl = tasks.flood_detection(ready_array, source="sentinel2")
print(f"Flood pixels: {fl['flood_pixels']:,}")

# Crop mapping (10 crop types)
cr = tasks.crop_mapping(ready_array, source="sentinel2")
print(f"Crop classes: {CROP_CLASSES}")

# Burn scar mapping
bs = tasks.burn_scar(ready_array, source="sentinel2")
print(f"Burn severity: {bs['severity_distribution']}")
```

## 12.3 DINOv3

DINOv3 is trained with self-supervised learning on satellite imagery,
producing strong spatial representations for unsupervised analysis:

```python
from pygeovision.models.foundation.dinov3 import DINOv3Backbone

dino     = DINOv3Backbone(variant="vitl14", satellite_type="optical")
features = dino.extract_features(image_tensor)  # (B, 1024, H/14, W/14)

# Unsupervised habitat clustering
from sklearn.cluster import KMeans
flat     = features.reshape(-1, 1024).numpy()
km       = KMeans(n_clusters=8, random_state=42)
clusters = km.fit_predict(flat)
print(f"Found {len(set(clusters))} habitat clusters")
```

## 12.4 Fine-tuning Foundation Models

```python
from pygeovision.training import GeoTrainer
from pygeovision.training.data import GeoSegDataset

# Freeze encoder, fine-tune decoder
for param in model.encoder.parameters():
    param.requires_grad = False

dataset = GeoSegDataset("./data/labelled/", tile_size=224)
trainer = GeoTrainer(model=model, output_dir="./finetuned/")
trainer.train(
    train_loader = DataLoader(dataset, batch_size=16),
    epochs       = 30,
    lr           = 1e-4,
)
```

## Exercises

1. Run Prithvi land cover on 3 different landscapes. Compare results.
2. Fine-tune Prithvi on 50 labelled patches. What is the improvement?
3. Use DINOv3 embeddings for nearest-neighbour scene retrieval.

## 12.5 SAM for Interactive Segmentation

```python
from pygeovision.models.foundation.sam import GeoSAM

sam = GeoSAM(model_type="vit_h")
sam.set_image("./data/scene.tif")

# Segment from a single point prompt
mask = sam.predict_from_point(lat=5.62, lon=-0.22)
print(f"Segment area: {mask.sum() * 100 / 1e6:.2f} km2")

# Automatic mode
all_masks = sam.predict_all(points_per_side=32)
print(f"Total segments: {len(all_masks)}")
```

## 12.6 DINOv3 Embeddings for Retrieval

```python
from pygeovision.models.foundation.dinov3 import DINOv3Backbone
import torch, numpy as np

dino = DINOv3Backbone(variant="vitl14", satellite_type="optical")

embeddings = []
for path in scene_paths:
    with rasterio.open(path) as src:
        arr = src.read()[:6].astype("float32")
    x   = torch.tensor(arr).unsqueeze(0)
    with torch.no_grad():
        emb = dino.extract_features(x).mean((-2,-1)).squeeze().numpy()
    embeddings.append(emb)

embeddings = np.stack(embeddings)  # (N, 1024)
# Nearest-neighbour retrieval
dists = np.linalg.norm(embeddings - embeddings[0], axis=1)
similar = np.argsort(dists)[1:6]
print(f"Most similar scenes: {similar}")
```

## Summary

Prithvi, DINOv3, and SAM are the three key EO foundation models in
PyGeoVision. Each requires different input preparation and serves different
task types. LoRA fine-tuning adapts large models efficiently.

## Exercises

1. Use SAM to segment all buildings in an aerial image.
2. Build a scene retrieval system from DINOv3 embeddings.
3. Fine-tune Prithvi on 50 patches. Compare zero-shot vs fine-tuned mIoU.
