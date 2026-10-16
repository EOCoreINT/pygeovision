"""OBIA tutorial: the steps of docs/obia/tutorial.md in one file, on real data.

Needs a network and the provider credentials PyGeoVision already uses (Sentinel-2 via the pipelines' download code,
ESA WorldCover via client.labeling). Run:  python obia_tutorial.py

Optional input: reference_points.gpkg, points with a 'class' column (forest, farmland, built_up, water) that YOU digitised
on high-resolution imagery. Without it the script stops before any accuracy figure, on purpose.
Writes classified.gpkg and classes.tif in the current folder.
"""
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
import pygeovision as pgv
from pygeovision import obia

BBOX = (-1.70, 6.18, -1.655, 6.225)         # about 5 km x 5 km: the region merging is pure Python
DATE = "2024-01"                           # a dry month; cloud pixels come back as NaN
CLASS_MAP = {10: "forest", 30: "farmland", 40: "farmland", 50: "built_up", 80: "water"}   # ESA WorldCover codes

# Step 1: real data, through PyGeoVision's own search and download
client = pgv.PyGeoVision()
rd = obia.fetch_scene(client, BBOX, DATE, "./download", bands=("blue", "green", "red", "nir"))
wc, _ = obia.fetch_worldcover(client, BBOX, "./download/worldcover.tif", like=rd)
valid = obia.valid_mask(rd)
print(rd.image.shape, f"{100 * (1 - valid.mean()):.1f}% no data")

# Step 2: segment; cloud and shadow pixels get label 0 and belong to no object
seg = obia.segment(rd, "multires", scale=16)
assert ((seg.labels == 0) == ~valid).all()
print(seg.n_objects, "objects")

# Step 3: scale evidence without reference objects (candidates only: look at the objects too)
print(obia.scale_diagnostics(rd, [8, 12, 16, 20, 30]).round(2))

# Step 4: describe the objects, then see what the weak-label classes look like
feats = obia.extract_features(seg.labels, rd, extra_layers=obia.spectral_layers(rd), texture_layers=["nir"],
                              transform=rd.transform)
weak = obia.samples_from_raster(seg.labels, wc, CLASS_MAP, min_purity=0.7)["class"]
cols = ["area_px", "compactness", "elongation", "mean_ndvi", "mean_ndwi", "std_nir", "brightness"]
print(feats.loc[weak.index, cols].groupby(weak).median().round(2))

# Step 5: rules written from that table (thresholds are YOURS to read off the table, these are placeholders)
rules = obia.RuleBasedClassifier([
    ("water", "mean_ndwi > 0.2"),
    ("forest", "mean_ndvi > 0.7"),
    ("farmland", "mean_ndvi > 0.45"),
    ("built_up", "compactness > 0.5"),
], default="unclassified")
rule_result = rules.classify(feats)

# Step 6: learn from the weak labels, but keep the objects of your reference points OUT of training
ref = None
if Path("reference_points.gpkg").exists():
    import geopandas as gpd
    ref = gpd.read_file("reference_points.gpkg")
ref_objs = (obia.samples_from_geometries(seg.labels, ref, "class", rd.transform, crs=rd.crs).index
            if ref is not None else pd.Index([]))
train = weak.drop(index=ref_objs, errors="ignore")
clf = obia.ObjectClassifier("rf", random_state=0, class_weight="balanced").fit(feats, train)
result = clf.predict(feats)
print(clf.feature_importances_.head(5).round(3))

# Step 7: spatial hold-out against the weak labels (agreement with the weak labels, NOT accuracy) ...
from sklearn.model_selection import GroupKFold
H, W = seg.labels.shape
blocks = (feats["centroid_row"] // (H / 3)).astype(int) * 3 + (feats["centroid_col"] // (W / 3)).astype(int)
pooled = pd.Series(index=weak.index, dtype=object)
for tr, te in GroupKFold(3).split(weak.index, groups=blocks.loc[weak.index]):
    m = obia.ObjectClassifier("rf", random_state=0, class_weight="balanced").fit(feats, weak.iloc[tr])
    pooled.loc[weak.index[te]] = m.predict(feats).classes.loc[weak.index[te]]
held = obia.confusion_report(weak, pooled)
print("agreement with the weak labels on held-out blocks:", round(held["overall_accuracy"], 3), "on", held["n_objects"], "objects")

# ... and accuracy proper, only if you have independent reference points
if ref is None:
    print("no reference_points.gpkg: no accuracy reported")
else:
    for name, res in {"rules": rule_result, "random forest": result}.items():
        a = obia.assess_classification(res, seg.labels, ref, "class", rd.transform, crs=rd.crs)
        print(name, round(a["overall_accuracy"], 3), round(a["kappa"], 3), "on", a["n_objects"], "objects")

# Step 8-9: export for a GIS
gdf = result.to_vector(seg.labels, rd.transform, rd.crs, features=feats[["area_m2", "mean_ndvi"]])
gdf.to_file("classified.gpkg", driver="GPKG")
codes = result.codes()
obia.save_raster("classes.tif", result.to_raster(seg.labels), rd.transform, rd.crs, dtype="uint8")
print("class codes in classes.tif:", codes)
print(result.summary(feats["area_m2"]).round(3))
