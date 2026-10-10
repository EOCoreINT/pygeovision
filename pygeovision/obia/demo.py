"""A synthetic teaching scene with known ground truth.

Why synthetic?  Students can run every exercise offline in seconds, and because the
truth is known exactly, segmentation quality and classification accuracy can be
measured without any digitising.  The scene is deliberately built so that spectral
information alone is *not* enough - bare fields and roofs overlap spectrally, forest
is textured, roads are linear - which is the situation where shape, texture and
context features earn their keep.

Classes: ``water``, ``forest``, ``cropland``, ``bare_field``, ``building``, ``road``.
Bands: ``blue, green, red, nir``.  Pixel size 2 m, UTM 33N (near Salzburg); the
values are artificial - this is not a real sensor's reflectance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from .io import RasterData
from .vector import labels_to_gdf

CLASS_NAMES = ["water", "forest", "cropland", "bare_field", "building", "road"]
# (blue, green, red, nir) mean and within-object noise
_SIG = {
    "water": ((55, 45, 30, 12), 3.0),
    "forest": ((25, 45, 28, 130), 14.0),
    "cropland": ((35, 80, 40, 170), 5.0),
    "bare_field": ((95, 115, 135, 125), 5.0),
    "building": ((110, 125, 140, 118), 6.0),
    "road": ((100, 100, 105, 90), 4.0),
}


@dataclass
class DemoScene:
    raster: RasterData
    ref_labels: np.ndarray       # reference object ids, 1..n (every pixel belongs to one)
    class_raster: np.ndarray     # class code per pixel, 1..6 (CLASS_NAMES order)
    reference: object            # GeoDataFrame: ref_id, class, geometry
    class_names: list[str]

    @property
    def classes_by_ref(self) -> pd.Series:
        return self.reference.set_index("ref_id")["class"]


def make_demo_scene(size: int = 160, seed: int = 0) -> DemoScene:
    """Build the scene.  ``size`` is the width/height in pixels (>= 96)."""
    from affine import Affine
    if size < 96:
        raise ValueError("size must be >= 96")
    rng = np.random.default_rng(seed)
    H = W = int(size)
    yy, xx = np.mgrid[0:H, 0:W]
    ref = np.zeros((H, W), dtype=np.int32)
    cls = np.zeros((H, W), dtype=np.int32)
    code = {c: i + 1 for i, c in enumerate(CLASS_NAMES)}
    next_id = [1]
    obj_class: dict[int, str] = {}

    def paint(mask, cname):
        ref[mask] = next_id[0]
        cls[mask] = code[cname]
        obj_class[next_id[0]] = cname
        next_id[0] += 1

    # fields: Voronoi cells, alternating crop / bare
    n_f = max(10, size // 12)
    seeds = rng.uniform([0, 0], [H, W], size=(n_f, 2))
    d = (yy[..., None] - seeds[:, 0]) ** 2 + (xx[..., None] - seeds[:, 1]) ** 2
    owner = d.argmin(-1)
    kinds = rng.permutation(["cropland"] * (n_f * 6 // 10) + ["bare_field"] * (n_f - n_f * 6 // 10))
    for i in range(n_f):
        paint(owner == i, kinds[i])
    # forest blobs and a lake (ellipses)
    def ellipse(cy, cx, ry, rx, ang):
        c, s = np.cos(ang), np.sin(ang)
        u = (xx - cx) * c + (yy - cy) * s
        v = -(xx - cx) * s + (yy - cy) * c
        return (u / rx) ** 2 + (v / ry) ** 2 <= 1
    paint(ellipse(.25 * H, .72 * W, .13 * H, .17 * W, .4), "forest")
    paint(ellipse(.80 * H, .22 * W, .11 * H, .16 * W, -.3), "forest")
    paint(ellipse(.62 * H, .62 * W, .09 * H, .20 * W, .9), "water")
    # a road: straight band across the scene, not over water
    r0 = int(.42 * H)
    road = (yy >= r0) & (yy < r0 + max(3, size // 40)) & (cls != code["water"])
    paint(road, "road")
    # buildings: small rectangles on fields, never on water/forest/road
    placed, tries = 0, 0
    while placed < max(8, size // 14) and tries < 500:
        tries += 1
        h, w = rng.integers(max(4, size // 28), max(6, size // 14), size=2)
        y, x = rng.integers(2, H - h - 2), rng.integers(2, W - w - 2)
        box = (slice(y, y + h), slice(x, x + w))
        under = cls[box]
        if np.isin(under, [code["water"], code["forest"], code["road"], code["building"]]).any():
            continue
        if (cls[max(0, y - 2):y + h + 2, max(0, x - 2):x + w + 2] == code["building"]).any():
            continue
        m = np.zeros((H, W), bool); m[box] = True
        paint(m, "building"); placed += 1

    # radiometry: class signature + per-object brightness offset + spatially smooth noise
    img = np.zeros((4, H, W))
    offs = {oid: rng.normal(0, 7) for oid in obj_class}
    off_map = np.vectorize(offs.get)(ref)
    for cname in CLASS_NAMES:
        m = cls == code[cname]
        mean, sd = _SIG[cname]
        for b in range(4):
            noise = ndi.gaussian_filter(rng.normal(0, 1, (H, W)), 0.7)
            noise /= noise.std()
            img[b][m] = (mean[b] + off_map[m] * (0.3 if b == 3 else 1.0) + sd * noise[m])
    img = np.clip(img, 0, 255).astype(np.float32)

    tr = Affine(2.0, 0, 352000.0, 0, -2.0, 5297000.0)
    raster = RasterData(img, ["blue", "green", "red", "nir"], tr, "EPSG:32633")
    gdf = labels_to_gdf(ref, tr, "EPSG:32633")
    gdf = gdf.rename(columns={"obj_id": "ref_id"})
    gdf["class"] = gdf["ref_id"].map(obj_class)
    return DemoScene(raster, ref, cls, gdf, list(CLASS_NAMES))
