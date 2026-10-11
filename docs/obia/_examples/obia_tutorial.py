"""OBIA tutorial: every step of docs/obia/tutorial.md in one runnable file.

Run:  python obia_tutorial.py      (about a minute, no downloads)
Writes classified.gpkg and classes.tif in the current folder.
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd
from pygeovision import obia

# Step 1
A = obia.make_demo_scene(seed=0)          # build everything here
B = obia.make_demo_scene(seed=1)          # judge everything here
print(A.raster.image.shape, len(A.reference))

# Step 2
seg = obia.segment(A.raster, "multires", scale=20)
print(seg.n_objects)
quality, summary = obia.segmentation_quality(seg.labels, A.ref_labels)
print(round(summary["mean_over_seg"],3), round(summary["mean_under_seg"],3), round(summary["mean_D"],3))

# Step 3
table = obia.scale_vs_reference(A.raster, [5, 10, 15, 20, 30, 40, 60, 80], A.ref_labels)
print(table.round(3)); print(table["mean_D"].idxmin())

# Step 4
layers = obia.spectral_layers(A.raster)
feats = obia.extract_features(seg.labels, A.raster, extra_layers=layers)
truth = obia.samples_from_geometries(seg.labels, A.reference, "class", A.raster.transform)
print(feats[["area_px","compactness","elongation","mean_ndvi","std_nir","brightness"]].groupby(truth).median().round(2))

# Step 5
rules = obia.RuleBasedClassifier([
    ("water", "mean_ndwi > 0.3"),
    ("forest", "mean_ndvi > 0.4 and std_nir > 8"),
    ("cropland", "mean_ndvi > 0.55"),
    ("building", "compactness > 0.55 and area_px < 400"),
    ("bare_field", "brightness > 90"),
])
result = rules.classify(feats)
report = obia.assess_classification(result, seg.labels, A.reference, "class", A.raster.transform)
print(round(report["overall_accuracy"],3), round(report["kappa"],3), report["n_objects"])
print(report["confusion"].astype(int))
wrong = feats.join(truth.rename("truth")).assign(predicted=result.classes)
wrong = wrong[wrong.truth != wrong.predicted]
print(wrong[["truth","predicted","area_px","std_nir","elongation"]].round(1).head(12))

rules2 = obia.RuleBasedClassifier([
    ("water", "mean_ndwi > 0.3"),
    ("forest", "mean_ndvi > 0.4 and std_nir > 8 and area_px > 300"),
    ("cropland", "mean_ndvi > 0.55"),
    ("road", "elongation > 8"),
    ("building", "compactness > 0.55 and area_px < 400"),
    ("bare_field", "brightness > 90"),
])
report2 = obia.assess_classification(rules2.classify(feats), seg.labels, A.reference, "class", A.raster.transform)
print(round(report2["overall_accuracy"],3), round(report2["kappa"],3))

# Step 6
clf = obia.ObjectClassifier("rf", random_state=0).fit(feats, truth)
print(clf.feature_importances_.head(5).round(3))
trained = obia.assess_classification(clf.predict(feats), seg.labels, A.reference, "class", A.raster.transform)
print("trap", trained["overall_accuracy"])

# Step 7
segB = obia.segment(B.raster, "multires", scale=20)
featsB = obia.extract_features(segB.labels, B.raster, extra_layers=obia.spectral_layers(B.raster))
def score(res):
    x = obia.assess_classification(res, segB.labels, B.reference, "class", B.raster.transform)
    w = obia.assess_classification(res, segB.labels, B.reference, "class", B.raster.transform, weights=featsB["area_px"])
    return round(x["overall_accuracy"],3), round(x["kappa"],3), round(w["overall_accuracy"],3), x["n_objects"]
print("rules ", score(rules.classify(featsB)))
print("rules2", score(rules2.classify(featsB)))
print("rf    ", score(clf.predict(featsB)))
for m in ("tree","knn","svm"):
    print(m, score(obia.ObjectClassifier(m, **({"random_state":0} if m=="tree" else {})).fit(feats, truth).predict(featsB)))

# Step 9
import os
d = '.'
final = clf.predict(featsB)
gdf = final.to_vector(segB.labels, B.raster.transform, B.raster.crs, features=featsB[["area_m2","mean_ndvi"]])
gdf.to_file(os.path.join(d,"classified.gpkg"), driver="GPKG")
obia.save_raster(os.path.join(d,"classes.tif"), final.to_raster(segB.labels), B.raster.transform, B.raster.crs)
print(gdf.columns.tolist()); print(final.summary(featsB["area_m2"]).round(3))
