"""Session 5: how good is it?  Object vs pixel accuracy, weighting, segmentation quality.

Run:  python examples/obia/04_accuracy.py
"""
from _common import REFINED_RULES
import numpy as np
import pandas as pd
from pygeovision import obia

pd.set_option("display.width", 160)
scene = obia.make_demo_scene(160, seed=0)
seg = obia.segment(scene.raster, "multires", scale=20)
F = obia.extract_features(seg.labels, scene.raster, extra_layers=obia.spectral_layers(scene.raster))
result = obia.RuleBasedClassifier(REFINED_RULES).classify(F)
ref = obia.samples_from_geometries(seg.labels, scene.reference, "class", scene.raster.transform)

print("The same classification, three different accuracy numbers:")
r_obj = obia.confusion_report(ref, result.classes)
r_area = obia.confusion_report(ref, result.classes, weights=F["area_px"])
print(f"  object count, every object equal : {r_obj['overall_accuracy']:.3f}")
print(f"  area-weighted                    : {r_area['overall_accuracy']:.3f}")

# pixel-wise: paint predicted classes, compare with the true per-pixel class map
codes = result.codes()
pred_raster = result.to_raster(seg.labels)
true_codes = np.zeros_like(scene.class_raster)
for i, name in enumerate(scene.class_names, start=1):
    true_codes[scene.class_raster == i] = codes.get(name, 0)
rp = obia.pixel_report(pred_raster, true_codes, result.class_names)
print(f"  pixel-wise                       : {rp['overall_accuracy']:.3f}   (area-weighted ~ pixel-wise)")

print("\nper-class accuracy (area-weighted):")
print(r_area["per_class"].round(3).to_string())
print("\nconfusion matrix (area in pixels; rows = reference):")
print(r_area["confusion"].astype(int).to_string())

# segmentation quality, per reference object
ref_ids = scene.ref_labels
table, summ = obia.segmentation_quality(seg.labels, ref_ids)
print(f"\nsegmentation vs reference: mean over-seg {summ['mean_over_seg']:.3f}, "
      f"under-seg {summ['mean_under_seg']:.3f}, D {summ['mean_D']:.3f}, QR {summ['mean_QR']:.3f}")
worst = table.sort_values("D", ascending=False).head(3).join(scene.classes_by_ref.rename("class"))
print("three worst-matched reference objects:")
print(worst[["class", "ref_area_px", "seg_area_px", "over_seg", "under_seg", "D"]].round(3).to_string())
