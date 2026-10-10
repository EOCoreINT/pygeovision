## 🧩 Object-Based Image Analysis (OBIA)

Analyse imagery as objects rather than pixels: segment, describe, classify, assess.

```python
from pygeovision import obia

scene = obia.make_demo_scene()                      # synthetic scene with known truth (offline)
seg   = obia.segment(scene.raster, "multires", scale=20)
F     = obia.extract_features(seg.labels, scene.raster,
                              extra_layers=obia.spectral_layers(scene.raster))
rules = obia.RuleBasedClassifier([("water", "mean_ndwi > 0.3"),
                                  ("bare_field", "mean_ndvi <= 0.55"),
                                  ("cropland", "mean_ndvi > 0.55")])
result = rules.classify(F)
```

Includes scale-controlled region merging (Baatz & Schäpe criterion — *not* a clone of eCognition),
five other segmenters, spectral/shape/GLCM-texture/context features, rule-based, fuzzy and supervised
classifiers, and object-level accuracy and segmentation-quality metrics. `pip install "pygeovision[obia]"`;
CLI: `pygeovision obia --help`. See `docs/obia` and the tutorial scripts in `examples/obia/`.

