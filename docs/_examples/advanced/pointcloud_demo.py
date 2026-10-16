"""PointCloudProcessor on a synthetic classified LiDAR tile: gently sloping ground, 40 trees and one building.

Run from the repository root:  PYTHONPATH=. python docs/_examples/advanced/pointcloud_demo.py
Needs laspy.  The tile is 100 m x 100 m, UTM zone 30N (EPSG:32630), about 24 000 points.
"""
import json
import tempfile
from pathlib import Path

import laspy
import numpy as np
import pyproj
import rasterio

from pygeovision.advanced.pointcloud import PointCloudProcessor

work = Path(tempfile.mkdtemp())
rng = np.random.default_rng(0)
X0, Y0 = 500000.0, 4500000.0


def terrain(x, y):
    return 100.0 + 0.05 * (x - X0) + 0.02 * (y - Y0)                      # 5 % slope west to east


# ground returns (class 2), 1 point per m2 plus noise
n_g = 10000
gx, gy = rng.uniform(0, 100, n_g) + X0, rng.uniform(0, 100, n_g) + Y0
gz = terrain(gx, gy) + rng.normal(0, 0.05, n_g)
cls = np.full(n_g, 2)

# 40 trees: a crown of 60 returns, 8-20 m above ground, radius 3 m (class 5 = high vegetation)
tx, ty, tz, tc = [], [], [], []
for _ in range(40):
    cx, cy, h = rng.uniform(5, 95), rng.uniform(5, 60), rng.uniform(8, 20)
    r = 3 * np.sqrt(rng.uniform(0, 1, 60)); a = rng.uniform(0, 2 * np.pi, 60)
    px, py = cx + r * np.cos(a) + X0, cy + r * np.sin(a) + Y0
    tx += list(px); ty += list(py); tz += list(terrain(px, py) + h * (1 - 0.5 * r / 3)); tc += [5] * 60

# one 20 m x 12 m flat-roofed building, 9 m tall, 1500 roof returns (class 6)
bx, by = rng.uniform(65, 85, 1500) + X0, rng.uniform(70, 82, 1500) + Y0
bz = terrain(bx, by) + 9.0

# two stray returns far above the canopy, tagged as high noise (class 18)
nx, ny, nz = np.array([30.0, 50.0]) + X0, np.array([30.0, 40.0]) + Y0, np.array([400.0, 380.0])

x = np.concatenate([gx, tx, bx, nx]); y = np.concatenate([gy, ty, by, ny]); z = np.concatenate([gz, tz, bz, nz])
c = np.concatenate([cls, tc, np.full(1500, 6), [18, 18]])
header = laspy.LasHeader(point_format=3, version="1.2")
header.offsets, header.scales = [x.min(), y.min(), 0], [0.001] * 3
las = laspy.LasData(header)
las.x, las.y, las.z, las.classification = x, y, z, c.astype(np.uint8)
las.intensity = rng.integers(0, 60000, len(x)).astype(np.uint16)
las.header.add_crs(pyproj.CRS.from_epsg(32630))
src = work / "tile.las"
las.write(str(src))

proc = PointCloudProcessor()
pc = proc.read(str(src))
print("== read ==")
print({k: (v if not hasattr(v, "shape") else f"array{v.shape}") for k, v in pc.items()})

print("\n== canopy_height_model ==")
r = proc.canopy_height_model(str(src), str(work / "chm.tif"), resolution=1.0)
print(json.dumps({k: v for k, v in r.items() if k != "output_path"}, indent=1))
with rasterio.open(work / "chm.tif") as ds:
    a = ds.read(1)
    print("raster:", ds.width, "x", ds.height, "cell", ds.res, "crs", ds.crs, "nodata", ds.nodata, "max", round(float(a[a != ds.nodata].max()), 1))

print("\n== the same without noise exclusion ==")
r2 = proc.canopy_height_model(str(src), str(work / "chm_noisy.tif"), resolution=1.0, exclude_noise=False)
print("max_m:", r2["height_stats"]["max_m"], "(the two 400 m returns)")

print("\n== no ground classification (nothing classified, noise still tagged) ==")
las2 = laspy.read(str(src)); las2.classification = np.where(c == 18, 18, 1).astype(np.uint8); las2.write(str(work / "unclassified.las"))
try:
    proc.canopy_height_model(str(work / "unclassified.las"), str(work / "chm2.tif"))
except ValueError as e:
    print("ValueError:", str(e)[:120], "...")
r3 = proc.canopy_height_model(str(work / "unclassified.las"), str(work / "chm3.tif"), filter_ground=False)
print("filter_ground=False -> max_m", r3["height_stats"]["max_m"], "canopy_cover_2m", r3["height_stats"]["canopy_cover_2m"])

print("\n== extract_buildings ==")
rb = proc.extract_buildings(str(src), str(work / "b.geojson"), resolution=0.5, gap_fill_m=1.5)
print(rb)
gj = json.loads((work / "b.geojson").read_text())
print("area_m2:", [f["properties"]["area_m2"] for f in gj["features"]], "| true roof 20 x 12 =", 20 * 12, "| crs", gj["crs"]["properties"]["name"])
print(proc.extract_buildings(str(src), str(work / "none.geojson"), building_class=9))

print("\n== classify_points without a trained model ==")
rc = proc.classify_points(str(src), str(work / "out.las"), num_classes=3, class_names=["ground", "vegetation", "building"], model="randlanet")
print({k: v for k, v in rc.items() if k not in ("output_path", "warning")})
print("warning:", rc["warning"][:90], "...")
