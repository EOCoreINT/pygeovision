# Chapter 13: Vision-Language Models

## 13.1 CLIP for Zero-Shot Analysis

CLIP learns joint embeddings of images and text, enabling zero-shot
classification without any labelled satellite data:

```python
from pygeovision.models.vlm.clip_zero_shot import CLIPZeroShot

clip = CLIPZeroShot(model="openai/clip-vit-large-patch14")

# Zero-shot scene classification
result = clip.classify(
    image_path = "./data/scene.tif",
    labels     = ["flooded farmland", "drought-stressed crops",
                   "healthy dense vegetation", "bare soil",
                   "urban expansion", "deforestation"],
)
print(f"Predicted: {result['label']}  ({result['confidence']:.1%})")

# Batch classification
results = clip.batch_classify(
    image_dir = "./data/patches/",
    labels    = ["forest", "savanna", "water", "urban", "agriculture"],
    top_k     = 3,
)
```

## 13.2 Moondream Visual Question Answering

```python
from pygeovision.models.vlm.moondream import MoondreamVQA

vqa = MoondreamVQA()

answer = vqa.ask(
    image_path = "./data/sentinel2_crop.tif",
    question   = "How many distinct agricultural fields can you see?",
)
print(f"Answer: {answer}")

# Automatic captioning
caption = vqa.caption("./data/disaster_scene.tif")
print(f"Caption: {caption}")
```

## 13.3 Image-Text Retrieval

```python
from pygeovision.models.vlm.clip_zero_shot import CLIPZeroShot

clip   = CLIPZeroShot()
images = ["./data/flood_2024.tif", "./data/forest_ghana.tif",
          "./data/city_accra.tif", "./data/farm_nile.tif"]

# Find images matching a text query
matches = clip.retrieve(
    query  = "a flooded urban area with submerged buildings",
    images = images,
    top_k  = 2,
)
for img_path, score in matches:
    print(f"  {img_path}: {score:.3f}")
```

## Exercises

1. Apply CLIP zero-shot classification to 20 Sentinel-2 patches.
2. Ask Moondream to count buildings in an aerial image.
3. Build a text-based image retrieval system for your data archive.

## 13.4 Automated Caption Generation

```python
from pygeovision.models.vlm.moondream import MoondreamVQA

vqa  = MoondreamVQA()

# Caption a disaster scene
cap  = vqa.caption("./data/flood_scene.tif")
print(f"Auto-caption: {cap}")

# Targeted question answering
ans  = vqa.ask("./data/urban_scene.tif",
                "Are there any visible signs of damage to the buildings?")
print(f"Answer: {ans}")

# Count objects
count = vqa.ask("./data/parking_lot.tif",
                 "How many vehicles are visible in the parking lot?")
print(f"Vehicle count: {count}")
```

## 13.5 Building a Geospatial VQA System

```python
from pygeovision.models.vlm.clip_zero_shot import CLIPZeroShot

clip   = CLIPZeroShot()
LABELS = ["flooded urban area", "drought-stressed cropland",
           "dense tropical forest", "active wildfire smoke",
           "coastal erosion", "industrial facility"]

# Process an archive of scenes
import pathlib
results = {}
for path in pathlib.Path("./data/scenes/").glob("*.tif"):
    result = clip.classify(str(path), labels=LABELS)
    results[path.name] = result

# Count by category
from collections import Counter
cats = Counter(v["label"] for v in results.values())
for cat, count in cats.most_common():
    print(f"  {cat}: {count} scenes")
```

## Summary

Vision-language models bring natural language into geospatial analysis.
CLIP enables zero-shot classification from any text description. Moondream
answers questions about satellite images without any training data.

## Exercises

1. Classify 20 Sentinel-2 scenes using 8 CLIP labels.
2. Ask Moondream to describe damage severity in disaster imagery.
3. Build a text-based image retrieval system for your scene archive.
4. Compare CLIP zero-shot vs Prithvi supervised for land cover accuracy.
