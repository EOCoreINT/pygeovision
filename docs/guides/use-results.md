# Use the results in a GIS, a report or a script

A finished run leaves a folder of ordinary files. This guide shows how to turn them into what you actually need: a map in your GIS,
numbers for a table, polygons for another tool, a report for someone who does not use GIS. Nothing here needs PyGeoVision to be
running; the files open anywhere.

## What the folder holds

```text
out/london/
├── water_mask.tif          class raster: 0 = not water, 1 = water, 255 = no data
├── ndwi.tif                continuous raster: the index itself, NaN = no data
├── run_summary.json        every number, setting and input of the run
└── deliverables/
    ├── water_bodies_report.pdf  .docx  .xlsx
    ├── maps/               PNG maps and charts
    └── vectors/            water_bodies.gpkg and geojson/*.geojson
```

The file names and classes are listed in the **What you get** table on each [pipeline's page](../pipelines/catalog/index.md).

Two conventions apply to every raster and are the cause of most surprises:

| In a… | The no-data value is | It means |
|---|---|---|
| Class raster (`uint8`) | `255` | Cloud, shadow, or outside the imagery. **Not** a class and not "none". |
| Continuous raster (`float32`) | `NaN` | The same |

Both are written into the file's no-data tag, so a GIS hides them without being told. Anything you compute yourself must leave them
out as well.

## Open them in a GIS

**In QGIS or ArcGIS**, drag the files in. For the class raster, give each value a colour and a name (a *paletted* or *unique values*
style); the pipeline page lists the classes. The no-data pixels are transparent already.

**The vectors are easier still.** Open `deliverables/vectors/<pipeline>.gpkg`, which holds every layer in one file. Each polygon
carries its class (`class_name`), its area in hectares (`area_ha`), its share of your box (`pct_aoi`) and the colour used on the
report's map (`color_hex`), so you can style by class and match the report. [Vector layers](../reporting/vectors.md) lists every
attribute. Areas are measured in an equal-area projection, so they are comparable between places.

**Check the map over a basemap** before trusting it, particularly at hills, cities and the edges of water, where false alarms
cluster. The map PNG in `deliverables/maps/` is for slides, not for checking.

## Get numbers

The headline numbers are in `result.stats` and in `run_summary.json`. For your own tables, count from the raster, **over the valid
pixels only**:

```python
from pygeovision import PyGeoVision

client = PyGeoVision()
stats = client.postprocess.class_statistics("out/london/water_mask.tif")
for value, row in stats.items():
    print(value, f"{row['area_ha']:,.1f} ha", f"{row['pct']:.1f} % of the valid pixels")
```

`pct` is a share of the **valid** pixels, not of your whole box. When a lot of the box is cloud the two differ a lot, and a report
that says "12 % water" should say 12 % of what. The vector layer's `pct_aoi` is the share of the box.

### Numbers for your own districts

To report per district, parish or field, give the polygons to `zonal_statistics`:

```python
client.postprocess.zonal_statistics(
    "out/london/ndwi.tif", "districts.geojson",
    output_path="districts_with_ndwi.geojson", stats=["mean", "count"],
)
```

This writes your polygons back with one attribute per statistic, named `stat_mean` and `stat_count` here. The raster's no-data pixels are ignored.

### The true outline, not the box

Pipelines work on the **box** of your area. To restrict a raster to an irregular boundary, clip it to the polygon:

```python
client.preprocess.clip_to_polygon("out/london/water_mask.tif", "district.geojson", output_path="out/london/water_mask_district.tif")
```

Clipped-away pixels become no data, so `class_statistics` on the clipped file counts the district only.

## Compare two runs

Two class rasters can be compared pixel by pixel **only if they are on the same grid**: the same coordinate system, pixel size
and origin. Runs over the same box and the same pipeline normally are; runs over different boxes or sensors are not.

```python
import numpy as np
import rasterio

with rasterio.open("out/june/water_mask.tif") as a, rasterio.open("out/august/water_mask.tif") as b:
    if (a.crs, a.transform, a.shape) != (b.crs, b.transform, b.shape):
        raise SystemExit("the two runs are on different grids; resample one before comparing")
    water_a, water_b = a.read(1), b.read(1)

both_valid = (water_a != 255) & (water_b != 255)          # compare only where both runs saw the ground
lost = both_valid & (water_a == 1) & (water_b == 0)
gained = both_valid & (water_a == 0) & (water_b == 1)
print(f"water lost: {lost.sum():,} pixels, gained: {gained.sum():,} pixels, compared: {both_valid.sum():,}")
```

Without the `both_valid` line, a cloud in one run looks like change. [Comparing runs](../pipelines/results.md#comparing-runs) shows
the same idea for the numbers in `run_summary.json`.

## Share results with people who do not use GIS

| Audience | Give them | How |
|---|---|---|
| Decision-makers | The **PDF report**: question, method, map, tables, limits | Written for you after every run |
| Colleagues who edit text | The **Word report** | Same content, editable |
| Analysts | The **Excel workbook** | Tables as sheets, ready to filter and chart |
| A web map or a developer | **GeoJSON** or a **Cloud-Optimised GeoTIFF** | `--vector-formats geojson`; `pygeovision postprocess cog water_mask.tif` |
| A colleague with an old GIS | A **Shapefile** | `--vector-formats shp`; field names are cut to 10 characters |

You can change the title, add a format, or name the classes of your own network afterwards, **without downloading anything again**:

```bash
pygeovision report out/london --formats pdf,docx --title "Surface water, central London, June 2024"
```

[Rebuilding deliverables](../reporting/rebuilding.md) lists every option.

## Before you share a number

Reports read confidently, so check what stands behind them:

1. **How much of the box was seen?** `aoi_coverage_fraction` and `valid_fraction` in `run_summary.json`.
2. **Is a share a share of the valid area or of the box?** The report says which.
3. **Does the map look right over a basemap?**
4. **Do the limitations on the pipeline's page apply to your site?**

[Reading results](../pipelines/results.md) has the full checklist, and the [case studies](../case-studies/index.md) show what it
catches in real reports.

## Related

- [Reports and deliverables](../reporting/index.md): what each file contains.
- [Postprocessing toolbox](../data/postprocessing.md): vectorise, smooth, sieve, accuracy and more.
- [Python guide](python.md): reading the files from a script.
