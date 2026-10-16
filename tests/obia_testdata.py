"""A small hand-built raster for the OBIA tests.

This is a *test fixture*, not part of the package: five flat rectangles whose class, extent and reflectance are written
down below, so every expected value in the tests follows from the construction (an object that is a rectangle has the
area of a rectangle). It makes no claim to look like imagery and nothing in ``pygeovision`` imports it.
"""
import numpy as np
from affine import Affine

from pygeovision import obia

BANDS = ["blue", "green", "red", "nir"]
CLASSES = ["water", "cropland", "bare_field", "forest", "building"]
# (blue, green, red, nir) of each rectangle
MEANS = {"water": (55, 45, 30, 12), "cropland": (35, 80, 40, 170), "bare_field": (95, 115, 135, 125),
         "forest": (25, 45, 28, 130), "building": (150, 160, 175, 90)}
PIXEL = 2.0
CRS = "EPSG:32633"
TRANSFORM = Affine(PIXEL, 0, 352000.0, 0, -PIXEL, 5297000.0)


class BlockScene:
    """``raster`` (4 bands), ``ref_labels`` (ids 1..5), ``class_raster`` (codes 1..5 in CLASSES order), ``reference``."""

    def __init__(self, n: int = 64):
        if n < 32 or n % 4:
            raise ValueError("n must be a multiple of 4 and >= 32")
        h = n // 2
        q = n // 8
        ref = np.zeros((n, n), np.int32)
        ref[:h, :h] = 1                         # water        top left
        ref[:h, h:] = 2                         # cropland     top right
        ref[h:, :h] = 3                         # bare field   bottom left (a building is cut out of it)
        ref[h:, h:] = 4                         # forest       bottom right
        ref[h + q:h + 2 * q, q:2 * q] = 5       # building     (n/8) x (n/8) inside the bare field
        img = np.zeros((4, n, n), np.float32)
        for i, cname in enumerate(CLASSES):
            m = ref == i + 1
            for b in range(4):
                img[b][m] = MEANS[cname][b]
        # tiny deterministic texture; the forest gets a strong one (a checkerboard in NIR) so std_nir separates it
        rr, cc = np.indices((n, n))
        img += ((rr + cc) % 2 * 1.0).astype(np.float32)[None]
        forest = ref == 4
        img[3][forest] += np.where(((rr + cc) % 2 == 0)[forest], 14.0, -14.0).astype(np.float32)
        self.ref_labels = ref
        self.class_raster = ref.copy()
        self.class_names = list(CLASSES)
        self.raster = obia.RasterData(img, list(BANDS), TRANSFORM, CRS)
        gdf = obia.labels_to_gdf(ref, TRANSFORM, CRS).rename(columns={"obj_id": "ref_id"})
        gdf["class"] = gdf["ref_id"].map(lambda i: CLASSES[i - 1])
        self.reference = gdf

    @property
    def n_objects(self) -> int:
        return 5


def block_scene(n: int = 64) -> BlockScene:
    return BlockScene(n)
