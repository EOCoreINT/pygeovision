"""Session 6: two object levels, super-objects and sub-objects.

Run:  python examples/obia/05_hierarchy.py
"""
from _common import REFINED_RULES
import pandas as pd
from pygeovision import obia

pd.set_option("display.width", 160)
scene = obia.make_demo_scene(160, seed=0)
levels = obia.segment_levels(scene.raster, [10, 40])
fine, coarse = levels[10.0], levels[40.0]
print(f"fine level (scale 10): {fine.n_objects} objects;  coarse level (scale 40): {coarse.n_objects} objects")

layers = obia.spectral_layers(scene.raster)
F = obia.extract_features(fine.labels, scene.raster, extra_layers=layers)
S = obia.super_object_features(fine.labels, coarse.labels, scene.raster, extra_layers=layers)
F = F.join(S)
print("\nhow a fine object relates to its super-object (first rows):")
print(F[["area_px", "parent_id", "parent_area_px", "rel_area_of_parent", "diff_to_parent_ndvi"]].head().round(3).to_string())

# classify the fine level, then ask: what share of each coarse object is 'building'?
res = obia.RuleBasedClassifier(REFINED_RULES).classify(F)
frac = obia.sub_object_fraction(res.classes, F["area_px"], F["parent_id"], "building")
print("\ncoarse objects that contain building sub-objects (share of area):")
print(frac[frac > 0].round(3).to_string())

# class-dependent context: share of each object's border touching a given class
ctx = obia.class_context(fine.labels, res.classes)
print("\nobjects with the most border shared with roads:")
print(ctx.sort_values("rel_border_road", ascending=False).head(3)["rel_border_road"].round(3).to_string())
