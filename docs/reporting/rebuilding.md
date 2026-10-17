# Rebuilding deliverables

A run is expensive (it downloads imagery); its deliverables are cheap. `pygeovision report` rebuilds the map, vectors and reports of an earlier run from the saved rasters and
`run_summary.json`. **Nothing is downloaded and nothing is re-analysed**, so the numbers stay identical.

## When to use it

- You trained a network and want your own **class names** on the maps, tables and polygons (`--class-names`).
- You need **another format**: Word for editing, Excel for tables, a Shapefile for a colleague with an old GIS, a KML for Google Earth.
- You want a **different title**, for a client report.
- You **upgraded PyGeoVision** and want the newest layout of an old run.
- The report step **failed** (for example the optional libraries were missing) and you have now installed them.

```{note}
A run made with `--no-report` (or `PYGEOVISION_REPORT=0`) writes **no** `run_summary.json`, and `pygeovision report` rebuilds from that file,
so there is nothing to rebuild from: run the analysis again without the flag. To make a run quick but keep the ability to add reports
later, use `--report-formats none --vector-formats none` instead. That skips the PDF, Word and Excel files and the polygons, and keeps
the maps and `run_summary.json`.
```

## Command line

```bash
pygeovision report ./camp_fire                                               # rebuild everything
pygeovision report ./camp_fire --formats pdf,docx --title "Camp Fire severity, 2018"
pygeovision report ./landslides --class-names "No change,Landslide" --vector-formats gpkg,shp,kml
pygeovision report ./camp_fire --no-vectors --no-maps --formats xlsx         # tables only
```

| Option | Meaning |
|---|---|
| `OUTPUT_DIR` | The `-o` folder of a `pygeovision channel` run (the one that holds `run_summary.json`). |
| `--formats pdf,xlsx,docx` | Which reports to write (`none` for none). |
| `--vector-formats gpkg,geojson,shp,kml` | Which vector files (`none` for none). |
| `--class-names "A,B,C"` | Names for the classes `0,1,2…` of a network you trained. |
| `--title TEXT` | Replace the report title. |
| `--no-vectors`, `--no-maps` | Skip those parts. |

**Expected output**

```text
✓ deliverables rebuilt

  Deliverables: ./landslides/deliverables
    PDF report    landslide_detection_report.pdf
    Map           maps/landslide_detection_map.png
    Vectors geojson landslide_detection_aoi.geojson, landslide_detection_landslides.geojson
    Vectors kml   landslide_detection_aoi.kml, landslide_detection_landslides.kml
```

Files with the same names are replaced; files from an earlier build in other formats are left in place.

## Python

```python
from pygeovision.reporting import regenerate

d = regenerate("./landslides", report_formats=["pdf"], class_names=["No change", "Landslide"],
               vector_formats=["gpkg", "shp"], title="Nepal landslides, 2015")
print(d.reports, d.vectors, d.warnings)
```

`regenerate(output_dir, **kwargs)` accepts the same options as `generate_deliverables` (`report_formats`, `vector_formats`, `make_maps`, `make_vectors`, `class_names`, `title`, `max_polygons`).
The `parameters` shown in the report come from the run summary unless you pass `parameters=`.

## What it needs

- `run_summary.json` in the folder, and the rasters it refers to **in the same folder**. The folder can be moved or copied to another machine: layers are found by file name relative to the folder, not by the absolute paths in the summary.
- The optional reporting libraries (`pip install "pygeovision[report]"`).

A run from a version that did not write `run_summary.json` cannot be rebuilt: the command says so (`… not found: … is not the output directory of a pipeline run (or the run predates deliverables)`).

## Editing the numbers

The report is a faithful rendering of the rasters. If you change a raster (for example you sieve it or correct a class in a GIS), the rebuilt report measures the **edited raster** for its class tables
and maps, while the headline sentences and "Key statistics" still quote the pipeline's original statistics. If you edit results, say so in your report.

## What to do when…

| Symptom | Cause | Fix |
|---|---|---|
| `run_summary.json not found` | Wrong folder, or a run before deliverables existed | Point at the folder given with `-o`; re-run the pipeline |
| Rebuilt report still shows class numbers | `--class-names` was not given, or fewer names than classes | Give one name per class value, in order: `0,1,2…` |
| `pdf report skipped … Install the reporting extras` | Optional libraries missing | `pip install "pygeovision[report]"` |
| "none of the pipeline's output rasters could be read" | The `.tif` files were moved out of the folder | Put them back next to `run_summary.json` |
| Numbers differ from the first report | You edited a raster, or upgraded PyGeoVision (a layout or measurement fix) | Expected; the rebuilt report measures what is on disk |
| Old format files remain | Rebuild never deletes | Delete them yourself |
