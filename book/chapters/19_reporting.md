# Chapter 19: Reporting and Export

## 19.1 Export Formats

### GeoTIFF and COG

```python
import rasterio
from rasterio.enums import Resampling

# Save as Cloud Optimised GeoTIFF
with rasterio.open("./results/mask.tif") as src:
    data    = src.read()
    profile = src.profile.copy()

profile.update(driver="GTiff",
               compress="lzw",
               tiled=True,
               blockxsize=512,
               blockysize=512)

with rasterio.open("./results/mask_cog.tif", "w", **profile) as dst:
    dst.write(data)
```

### GeoJSON and GeoParquet

```python
import geopandas as gpd

# Vectorise raster mask
gdf = gpd.read_file("./results/buildings.geojson")

# Export formats
gdf.to_file("./results/buildings.geojson", driver="GeoJSON")
gdf.to_file("./results/buildings.shp")
gdf.to_parquet("./results/buildings.geoparquet")

print(f"Exported {len(gdf)} features")
```

## 19.2 HTML Report Generation

```python
from pygeovision.insar.interpretation import InSARInterpreter, DeformationReport

report = interp.interpret("./insar/disp.tif", study_area="Accra, Ghana")
report.export("./outputs/insar_report.json", format="json")
report.export("./outputs/insar_report.md",   format="md")
```

## 19.3 Statistics Summary

```python
import json

def generate_analysis_report(study_area, results_dir):
    report = {
        "study_area":    study_area,
        "date_generated": "2026-07-04",
        "results": {}
    }

    # Building footprints
    gdf = gpd.read_file(f"{results_dir}/buildings.geojson")
    report["results"]["buildings"] = {
        "count":        len(gdf),
        "total_area_m2": float(gdf.area.sum()),
        "mean_area_m2":  float(gdf.area.mean()),
    }

    with open(f"{results_dir}/analysis_report.json", "w") as f:
        json.dump(report, f, indent=2)

    return report
```

## Exercises

1. Export your analysis results as COG + GeoJSON + GeoParquet.
2. Generate an HTML report with embedded maps using the InSAR interpreter.
3. Build a summary report that aggregates statistics from multiple pipelines.

## 19.4 Automated Report Generation

```python
import json, datetime

def generate_report(study_area, results):
    report = {
        "study_area":  study_area,
        "generated":   datetime.datetime.utcnow().isoformat(),
        "results":     results,
        "data_source": "Sentinel-2 L2A / PyGeoVision v2.1.7",
    }
    with open("./outputs/report.json","w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Report saved")
    return report

report = generate_report("Accra, Ghana", {
    "flood_area_km2": 12.4,
    "buildings_affected": 847,
    "affected_population": 4200,
})
```

## Summary

Export to GeoTIFF/COG, GeoJSON, GeoParquet, and structured JSON/Markdown
reports. Always export with metadata (date, model version, data source).

## Exercises

1. Export your analysis results as COG + GeoJSON.
2. Generate a structured JSON report with flood statistics.
3. Build an HTML report with embedded static maps.
