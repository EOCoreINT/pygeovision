"""GeoTimeSeries on 36 monthly synthetic Sentinel-2-like scenes (3 years) with a known drought and a known failure.

* NDVI follows a seasonal cycle (low in winter, high in summer) with a slight upward trend.
* July 2023 is a drought: NDVI is 0.25 lower than normal.
* Scene 2022-03 is a corrupt file.  Scene 2022-08 is half covered by cloud (bright, NDVI ~0.05) and has a mask.

Run from the repository root:  PYTHONPATH=. python docs/_examples/advanced/timeseries_demo.py
"""
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from pygeovision.advanced.timeseries import GeoTimeSeries

work = Path(tempfile.mkdtemp())
rng = np.random.default_rng(0)
dates = [f"{y}-{m:02d}" for y in (2021, 2022, 2023) for m in range(1, 13)]
tf = from_origin(500000, 4500000, 10, 10)


def season(i):
    month = i % 12
    return 0.50 + 0.25 * np.sin(2 * np.pi * (month - 3) / 12) + 0.002 * i       # +0.002 per month trend


def write_scene(path, ndvi_mean, cloud=False):
    ndvi = np.clip(ndvi_mean + rng.normal(0, 0.03, (64, 64)), -0.9, 0.95)
    red = np.full((64, 64), 0.08)
    nir = red * (1 + ndvi) / (1 - ndvi)
    blue, green = np.full((64, 64), 0.04), np.full((64, 64), 0.07)
    bands = np.stack([blue, green, red, nir])
    if cloud:
        bands[:, :, :32] = 0.30 + rng.normal(0, 0.01, (4, 64, 32))          # bright cloud over the left half
    with rasterio.open(path, "w", driver="GTiff", height=64, width=64, count=4, dtype="uint16", crs="EPSG:32630",
                       transform=tf) as dst:
        dst.write(np.round(bands * 10000).astype("uint16"))                  # reflectance x 10000, as in L2A products


paths, masks = [], []
for i, d in enumerate(dates):
    p = work / f"s2_{d}.tif"
    mean = season(i) - (0.25 if d == "2023-07" else 0.0)
    write_scene(p, mean, cloud=(d == "2022-08"))
    paths.append(str(p))
    mk = np.zeros((1, 64, 64), "uint8")
    if d == "2022-08":
        mk[:, :, :32] = 1
    mp = work / f"mask_{d}.tif"
    with rasterio.open(mp, "w", driver="GTiff", height=64, width=64, count=1, dtype="uint8", crs="EPSG:32630", transform=tf) as dst:
        dst.write(mk)
    masks.append(str(mp))
Path(paths[2]).write_bytes(b"corrupt")                                       # 2021-03 is unreadable

ts = GeoTimeSeries(sensor="sentinel2")
raw = ts.compute_index_series(paths, "ndvi", dates)
print("== without cloud mask ==")
print("scene 2022-08 :", raw["mean"][dates.index("2022-08")], "valid", raw["valid_fraction"][dates.index("2022-08")])
print("failed scenes :", {Path(k).name: v[:60] for k, v in raw["errors"].items()})

s = ts.compute_index_series(paths, "ndvi", dates, mask_paths=masks)
i = dates.index("2022-08")
print("\n== with mask_paths ==")
print("scene 2022-08 :", s["mean"][i], "valid", s["valid_fraction"][i], "(season says %.3f)" % season(i))
print("2023-07 mean/std/p10/p90:", [s[k][dates.index("2023-07")] for k in ("mean", "std", "p10", "p90")])

print("\n== anomalies ==")
for method, kw in (("zscore", {}), ("mad", {}), ("zscore, period=12", {"period": 12})):
    m = method.split(",")[0]
    out = ts.detect_anomalies(s, method=m, **kw)
    print(f"{method:18s}:", [(a["date"], a["zscore"], a["type"]) for a in out][:6])

print("\n== trend ==")
print("plain fit         :", {k: v for k, v in ts.compute_trend(s).items() if k in ("slope", "r_squared", "p_value", "significant", "direction")})
print("period=12 (seasonal):", {k: v for k, v in ts.compute_trend(s, period=12).items() if k in ("slope", "r_squared", "p_value", "significant", "direction")})
# the drought pulls the fitted slope down: drop the flagged dates and fit again
flagged = {a["date"] for a in ts.detect_anomalies(s, method="mad", period=12)}
clean = {**s, "mean": [None if d in flagged else v for d, v in zip(s["dates"], s["mean"])]}
print("period=12, anomalies removed:", {k: v for k, v in ts.compute_trend(clean, period=12).items() if k in ("slope", "r_squared", "p_value", "significant", "direction")})
print("true slope        : 0.002 per month")
