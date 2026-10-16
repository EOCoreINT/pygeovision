"""What the vision-language models actually see, and what the retrieval index looks like. Works offline.

The CLIP and Moondream weights are downloaded from the Hugging Face Hub on first use, which is not possible here, so this
script covers the parts that do not need them: how a GeoTIFF becomes a picture, and the search-index file.

Run from the repository root:  PYTHONPATH=. python docs/_examples/advanced/vlm_inputs_demo.py
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from pygeovision.advanced.vlm.retrieval import GeoImageRetrieval
from pygeovision.models.vlm._image import list_images, load_rgb_pil

work = Path(tempfile.mkdtemp())
rng = np.random.default_rng(0)

# A 4-band Sentinel-2-like scene (B2, B3, B4, B8; reflectance x 10000) with a nodata border (value 0)
data = np.zeros((4, 64, 64), "uint16")
data[0], data[1], data[2], data[3] = 400, 700, 900, 3000
data += rng.integers(0, 300, data.shape).astype("uint16")
data[:, :8, :] = 0
with rasterio.open(work / "scene.tif", "w", driver="GTiff", height=64, width=64, count=4, dtype="uint16", crs="EPSG:32630",
                   transform=from_origin(500000, 4500000, 10, 10), nodata=0) as dst:
    dst.write(data)

for bands_note, kwargs in (("default: bands 1,2,3", {}), ("per-band stretch", {"stretch": "per_band"}),
                           ("bands=(3, 2, 1)", {"bands": (3, 2, 1)})):
    img = np.asarray(load_rgb_pil(str(work / "scene.tif"), **kwargs))
    print(f"{bands_note:24s} shape {img.shape} dtype {img.dtype}  mean R,G,B in valid area = "
          f"{[round(float(img[8:, :, c].mean())) for c in range(3)]}  nodata border max = {int(img[:8].max())}")
print("(the file is written B2, B3, B4, B8: with the default the model's 'red' is band 1 = B2, so the picture is blue-tinted;")
print(" bands=(3, 2, 1) shows B4 as red and gives natural colour)")

print("\nsupported image files :", [p.name for p in list_images(work)])

# The search index is a plain JSON file
class Stub:                                    # stands in for CLIP so that no download is needed
    def embed_image(self, path):
        v = np.array([1.0, 0.0, 0.0, 0.0]); return v
r = GeoImageRetrieval.__new__(GeoImageRetrieval)
r.model, r._clip, r._index, r.skipped = "openclip-b32", Stub(), {}, {}
(work / "tiles").mkdir()
for n in ("a.tif", "b.tif"):
    (work / "tiles" / n).write_bytes(b"x")
r.build_index(str(work / "tiles"), save_path=str(work / "index.json"))
print("index file keys       :", list(json.loads((work / "index.json").read_text())))
print("index model / entries :", json.loads((work / "index.json").read_text())["model"], len(json.loads((work / "index.json").read_text())["embeddings"]))
