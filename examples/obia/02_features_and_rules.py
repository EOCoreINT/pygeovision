"""Session 3: describe objects with features, then classify with rules you write.

Run:  python examples/obia/02_features_and_rules.py
"""
from _common import OUT, SPECTRAL_RULES, REFINED_RULES
import os
import pandas as pd
from pygeovision import obia
from pygeovision.obia import viz

pd.set_option("display.width", 160)
scene = obia.make_demo_scene(160, seed=0)
seg = obia.segment(scene.raster, "multires", scale=20)
F = obia.extract_features(seg.labels, scene.raster,
                          extra_layers=obia.spectral_layers(scene.raster), texture_layers=["nir"])
print(f"{seg.n_objects} objects x {F.shape[1]} features")

# Which features separate the classes?  Look before you write rules.
ref = obia.samples_from_geometries(seg.labels, scene.reference, "class", scene.raster.transform)
cols = ["mean_ndvi", "std_nir", "brightness", "area_px", "compactness", "elongation"]
print("\nmedian feature value per reference class:")
print(F.join(ref.rename("ref")).groupby("ref")[cols].median().round(2).to_string())

# Rules: first match wins
for name, rules in [("spectral rules only", SPECTRAL_RULES), ("+ shape and size", REFINED_RULES)]:
    result = obia.RuleBasedClassifier(rules).classify(F)
    by_count = obia.confusion_report(ref, result.classes)
    by_area = obia.confusion_report(ref, result.classes, weights=F["area_px"])
    print(f"\n{name}: object accuracy {by_count['overall_accuracy']:.3f}   "
          f"area-weighted {by_area['overall_accuracy']:.3f}   kappa {by_count['kappa']:.3f}")

# explain(): why did object X get its class?
rb = obia.RuleBasedClassifier(REFINED_RULES)
oid = F.index[F["area_px"] < 400][0]
print(f"\nwhy did object {oid} get its class?")
print(rb.explain(F, oid).to_string())
viz.show_classes(rb.classify(F), seg.labels, scene.raster, bands=(3, 2, 1)).savefig(
    os.path.join(OUT, "classes_rules.png"), dpi=80)
