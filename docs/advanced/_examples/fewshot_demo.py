"""Few-shot classification end to end, offline.

The real backbone (DINOv2) is downloaded from Hugging Face on first use. To keep this script runnable without a
network it builds a TINY, RANDOMLY INITIALISED DINOv2-shaped model, saves it to a folder and passes that folder as
``backbone`` (which is also how you would use a copy of the real model on an air-gapped machine). A random network
has no semantic knowledge: it only separates things that differ in simple colour/texture statistics, so the numbers
below show the API and the output format, not the accuracy you will get with the real DINOv2.

Run from the repository root:  PYTHONPATH=. python docs/advanced/_examples/fewshot_demo.py
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import rasterio
import transformers
from rasterio.transform import from_origin

from pygeovision.advanced.few_shot import FewShotLearner

work = Path(tempfile.mkdtemp())
rng = np.random.default_rng(0)

# 1. a tiny local backbone -------------------------------------------------------------------------------------------
cfg = transformers.Dinov2Config(hidden_size=64, num_hidden_layers=2, num_attention_heads=2, image_size=56,
                                patch_size=14, intermediate_size=128)
torch_seed = __import__("torch").manual_seed(0)
transformers.Dinov2Model(cfg).save_pretrained(work / "tiny-dino")
transformers.BitImageProcessor(size={"shortest_edge": 56}, crop_size={"height": 56, "width": 56}).save_pretrained(
    work / "tiny-dino")

# 2. synthetic 3-band tiles: water (dark blue, smooth), forest (green, grainy), bare soil (tan, smooth) -------------
COLOURS = {"water": (30, 60, 120), "forest": (40, 110, 50), "bare": (170, 140, 100)}
GRAIN = {"water": 4, "forest": 30, "bare": 6}


def tile(path, cls, size=64):
    base = np.array(COLOURS[cls], dtype="float32")[:, None, None]
    data = np.clip(base + rng.normal(0, GRAIN[cls], (3, size, size)), 0, 255).astype("uint8")
    with rasterio.open(path, "w", driver="GTiff", height=size, width=size, count=3, dtype="uint8", crs="EPSG:32630",
                       transform=from_origin(500000, 4500000, 10, 10)) as dst:
        dst.write(data)
    return str(path)


support = {c: [tile(work / f"{c}_{i}.tif", c) for i in range(3)] for c in COLOURS}      # 3 examples per class
queries = [tile(work / f"q_{c}.tif", c) for c in COLOURS]

learner = FewShotLearner(backbone=str(work / "tiny-dino"), device="cpu")
learner.fit_support(support)
print("classes:", learner.class_names)

print("\n== predict ==")
for path, res in zip(queries, learner.predict(queries)):
    probs = {k: round(v, 2) for k, v in res["probabilities"].items()}
    print(f"{Path(path).name:12s} -> {res['class']:7s} confidence {res['confidence']:.2f}  {probs}")

print("\n== an unreadable query does not stop the batch ==")
bad = work / "broken.tif"
bad.write_bytes(b"not a tiff")
for res in learner.predict([queries[0], str(bad)]):
    print({k: (v if k != "error" else str(v)[:48] + "...") for k, v in res.items() if k != "probabilities"})

# 3. a map: a 128x128 scene, left half water, right half forest ----------------------------------------------------
scene = work / "scene.tif"
left = np.clip(np.array(COLOURS["water"], "float32")[:, None, None] + rng.normal(0, GRAIN["water"], (3, 128, 64)), 0, 255)
right = np.clip(np.array(COLOURS["forest"], "float32")[:, None, None] + rng.normal(0, GRAIN["forest"], (3, 128, 64)), 0, 255)
with rasterio.open(scene, "w", driver="GTiff", height=128, width=128, count=3, dtype="uint8", crs="EPSG:32630",
                   transform=from_origin(500000, 4500000, 10, 10)) as dst:
    dst.write(np.concatenate([left, right], axis=2).astype("uint8"))
res = learner.predict_geotiff(str(scene), str(work / "classes.tif"), chip_size=64, stride=32)
print("\n== predict_geotiff ==")
print({k: v for k, v in res.items() if k != "output_path"})
with rasterio.open(work / "classes.tif") as src:
    arr = src.read(1)
    print("dtype", src.dtypes[0], "| nodata", src.nodata, "| tags", json.loads(src.tags().get("class_map", "{}")) or src.tags())
for code, name in res["class_map"].items():
    code = int(code)
    print(f"  code {code} = {name:7s}: left half {(arr[:, :64] == code).mean():.2f}   right half {(arr[:, 64:] == code).mean():.2f}")
