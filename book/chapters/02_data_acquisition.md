# Chapter 2: Satellite Data Acquisition

## 2.1 Understanding the Satellite Data Landscape

Modern Earth observation produces more than 150 TB of new satellite data every
day. Accessing that data without knowing which providers offer what products,
which authentication methods they require, and how to efficiently download
only the bands you need — is the first major barrier in geospatial AI.

PyGeoVision delegates all data access to PyGeoFetch, which abstracts 22+
providers behind a uniform `search()` and `download()` interface.

### Provider categories

| Category | Providers | Notes |
|----------|-----------|-------|
| **Free, open** | Planetary Computer, Copernicus CDSE, USGS, NASA | No cost |
| **Commercial optical** | Planet, Maxar, Airbus | Subscription |
| **SAR** | Sentinel-1 (CDSE), JAXA ALOS | Free; commercial at 1-3m |
| **Multispectral stacks** | NASA HLS, USGS Landsat C2 | Pre-harmonised for ML |
| **Derived products** | Copernicus DEM, MODIS LST, Sentinel-5P | Ready indices |

## 2.2 Authentication

```python
import pygeovision as pgv
client = pgv.PyGeoVision()

# Planetary Computer (free, optional subscription key)
client.add_credentials("planetary_computer",
                        subscription_key="your-pc-key")

# Copernicus CDSE (free, EU login)
client.add_credentials("copernicus_cdse",
                        username="you@example.com",
                        password="your-password")

# Planet (commercial)
client.add_credentials("planet", api_key="PL-API-your-key")

# USGS EarthExplorer
client.add_credentials("usgs",
                        username="your-usgs-username",
                        password="your-usgs-password")

print(client.list_providers())
```

## 2.3 Searching for Imagery

### Basic search

```python
results = client.search(
    bbox            = (-0.30, 5.50, -0.05, 5.70),  # Accra, Ghana
    date_range      = ("2026-06-01", "2026-06-30"),
    cloud_cover_max = 20,
)

print(f"Found {len(results)} scenes")
for r in results[:5]:
    print(f"  {r.provider:<22} {r.datetime[:10]}  cloud={r.cloud_cover:4.1f}%")
```

### Filtering results

```python
# Specific providers and satellites
s2 = client.search(
    bbox            = (-0.30, 5.50, -0.05, 5.70),
    date_range      = ("2026-01-01", "2026-12-31"),
    providers       = ["planetary_computer"],
    satellites      = ["Sentinel-2"],
    cloud_cover_max = 10,
    max_results     = 50,
)
s2.sort(key=lambda r: r.cloud_cover)
print(f"Best: {s2[0].datetime[:10]}  cloud={s2[0].cloud_cover:.1f}%")

# SAR search (cloud cover irrelevant)
sar = client.search(
    bbox         = (-0.30, 5.50, -0.05, 5.70),
    date_range   = ("2026-06-01", "2026-06-30"),
    satellites   = ["Sentinel-1"],
    product_type = "GRD",
)
print(f"SAR acquisitions: {len(sar)}")
```

## 2.4 Downloading Data

### Single scene

```python
downloads = client.download(
    scenes       = results[:1],
    output_dir   = "./data/accra/",
    bands        = ["B02","B03","B04","B08","B8A","B11"],
    post_process = ["reproject:EPSG:32630", "cog"],
    max_workers  = 4,
    resume       = True,
)

for dl in downloads:
    if dl.success:
        print(f"  {dl.path}  ({dl.bytes_downloaded/1e6:.1f} MB)")
    else:
        print(f"  FAILED: {dl.error}")
```

### Time-series download

```python
quarters = [
    ("2026-01-01", "2026-03-31"),
    ("2026-04-01", "2026-06-30"),
    ("2026-07-01", "2026-09-30"),
    ("2026-10-01", "2026-12-31"),
]

for start, end in quarters:
    batch = client.search(bbox=BBOX, date_range=(start, end),
                          cloud_cover_max=15)
    if batch:
        client.download(batch[:1], output_dir=f"./data/{start[:7]}/",
                        bands=BANDS)
        print(f"Downloaded {start[:7]}")
```

## 2.5 Cloud Optimised GeoTIFFs (COGs)

COGs allow partial spatial reads without downloading the full file:

```python
import rasterio
from rasterio.windows import from_bounds

url = "https://storage.googleapis.com/pgv-demo/sentinel2_accra_cog.tif"

with rasterio.open(url) as src:
    # Read only a 500m x 500m window — no full download
    window = from_bounds(-0.28, 5.55, -0.27, 5.56, src.transform)
    tile   = src.read(window=window)
    print(f"Tile shape: {tile.shape}")
```

## 2.6 Best Practices

```python
# Dynamic cloud threshold
def find_best_scene(bbox, date_range, max_cloud=30):
    for threshold in [5, 10, 15, 20, 30, max_cloud]:
        r = client.search(bbox=bbox, date_range=date_range,
                          cloud_cover_max=threshold)
        if r:
            print(f"Found at cloud <= {threshold}%")
            return r[0]
    return None

# Enable cache to avoid re-downloading
import os
os.environ["PGV_CACHE_DIR"] = "/data/pgv_cache"
downloads = client.download(results[:3], output_dir="./data/",
                             bands=BANDS, skip_existing=True)
```

## Summary

- PyGeoVision provides uniform search and download across 22+ providers.
- Planetary Computer needs no key for basic searches.
- Post-processing chains (reproject, COG) run automatically on download.
- SAR data uses the same API as optical data.

## Exercises

1. Search all Sentinel-2 scenes over your city for the past year.
   Plot cloud cover by month.
2. Download two scenes from different dates and compare visually.
3. Download a Sentinel-1 SAR scene — when was the last acquisition?
4. Implement a function finding the cloud-free scene closest to a target date.
