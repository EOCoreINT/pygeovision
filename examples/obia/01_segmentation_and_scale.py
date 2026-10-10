"""Session 1-2: create image objects, see what the scale parameter does, choose one.

Run:  python examples/obia/01_segmentation_and_scale.py
"""
from _common import OUT
import os
from pygeovision import obia
from pygeovision.obia import viz

scene = obia.make_demo_scene(160, seed=0)
print(f"Scene: {scene.raster.image.shape}, {len(scene.reference)} reference objects")

# 1. Different algorithms, same image
for method in obia.METHODS:
    seg = obia.segment(scene.raster, method)
    print(f"{method:13s} -> {seg.n_objects:4d} objects")
    viz.show_overlay(seg, bands=(3, 2, 1)).savefig(os.path.join(OUT, f"seg_{method}.png"), dpi=80)

# 2. The scale parameter, multiresolution segmentation
print("\nmultires at increasing scale:")
for s in (5, 10, 20, 40, 80):
    seg = obia.segment(scene.raster, "multires", scale=s)
    print(f"  scale {s:3d} -> {seg.n_objects:4d} objects")

# 3. Choose a scale with evidence: reference objects (needs ground truth)
sweep = obia.scale_vs_reference(scene.raster, [5, 10, 15, 20, 30, 40, 60, 80], scene.ref_labels)
print("\nsegmentation quality vs scale (D: lower = better):")
print(sweep.round(3).to_string())
print("best scale by D:", sweep["mean_D"].idxmin())
viz.plot_scale(sweep).savefig(os.path.join(OUT, "scale_vs_reference.png"), dpi=80)

# 4. ... or without ground truth: a heuristic hint, to be checked by eye
diag = obia.scale_diagnostics(scene.raster, list(range(4, 62, 2)))
print("\nROC-LV candidate scales (heuristic):", diag.index[diag.candidate].tolist())
viz.plot_scale(diag).savefig(os.path.join(OUT, "scale_diagnostics.png"), dpi=80)
