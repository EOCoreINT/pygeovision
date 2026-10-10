"""Session 4: fuzzy memberships (with stability) and supervised classification.

Run:  python examples/obia/03_fuzzy_and_supervised.py
"""
from _common import OUT
import os
import pandas as pd
from pygeovision import obia
from pygeovision.obia import LargerThan, SmallerThan

pd.set_option("display.width", 160)
scene = obia.make_demo_scene(160, seed=0)
seg = obia.segment(scene.raster, "multires", scale=20)
F = obia.extract_features(seg.labels, scene.raster, extra_layers=obia.spectral_layers(scene.raster))
ref = obia.samples_from_geometries(seg.labels, scene.reference, "class", scene.raster.transform)

# ---- fuzzy: soft thresholds instead of hard ones --------------------------------
def describe(min_area_bare):
    return {
        "water":      {"mean_ndwi": LargerThan(0.1, 0.3)},
        "road":       {"elongation": LargerThan(4, 8), "brightness": SmallerThan(105, 120)},
        "building":   {"area_px": SmallerThan(300, 600), "compactness": LargerThan(0.4, 0.6),
                       "mean_ndvi": SmallerThan(0.2, 0.5)},
        "forest":     {"mean_ndvi": LargerThan(0.4, 0.55), "std_nir": LargerThan(7, 11),
                       "area_px": LargerThan(300, 600)},
        "cropland":   {"mean_ndvi": LargerThan(0.4, 0.55), "std_nir": SmallerThan(7, 11)},
        "bare_field": {"mean_ndvi": SmallerThan(0.2, 0.5), "area_px": LargerThan(*min_area_bare)},
    }

def report(label, res):
    rep = obia.confusion_report(ref, res.classes)
    n_un = int((res.classes == "unclassified").sum())
    print(f"{label}: object accuracy {rep['overall_accuracy']:.3f}  kappa {rep['kappa']:.3f}  "
          f"unclassified {n_un}/{len(res.classes)}")
    return n_un

# Attempt 1: bare fields must be at least 100-200 px.
res1 = obia.FuzzyClassifier(describe((100, 200)), combine="min", min_membership=0.1).classify(F)
report("attempt 1", res1)
print("what do the unclassified objects have in common?")
un = F.loc[res1.classes == "unclassified", ["area_px", "mean_ndvi", "compactness"]]
print(un.describe().loc[["min", "50%", "max"]].round(2).to_string())

# Attempt 2: the unclassified ones are small fragments, so relax the area membership.
res2 = obia.FuzzyClassifier(describe((10, 40)), combine="min", min_membership=0.1).classify(F)
report("attempt 2", res2)

# Stability among CLASSIFIED objects: small values flag a coin-flip between two classes.
ok = res2.classes != "unclassified"
shaky = res2.stability[ok].sort_values().head(4)
print("\nleast stable classified objects (best minus second-best membership):")
print(pd.DataFrame({"class": res2.classes[shaky.index], "stability": shaky.round(3),
                    "area_px": F.loc[shaky.index, "area_px"]}).to_string())

# ---- supervised: train on the LEFT half of the scene, test on the RIGHT half
left = F["centroid_col"] < 80
train_ref = ref[ref.index.isin(F.index[left])]
test_ref = ref[ref.index.isin(F.index[~left])]
print(f"\ntraining objects: {len(train_ref)} ({train_ref.nunique()} classes),  test objects: {len(test_ref)} "
      f"({test_ref.nunique()} classes)")
missing = set(test_ref) - set(train_ref)
if missing:
    print(f"  NOTE: classes in the test half but never seen in training: {sorted(missing)}")
for method in ("rf", "knn", "tree"):
    clf = obia.ObjectClassifier(method).fit(F[left], train_ref)
    pred = clf.predict(F[~left])
    r = obia.confusion_report(test_ref, pred.classes)
    print(f"{method:5s} test accuracy {r['overall_accuracy']:.3f} on {r['n_objects']} objects")
tree = obia.ObjectClassifier("tree").fit(F[left], train_ref)
print("\nthe decision tree is readable, it learned these rules:")
print(tree.export_rules())
