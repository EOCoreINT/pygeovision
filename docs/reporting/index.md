# Reports and deliverables

An analysis is only useful once someone else can open it. After every successful pipeline run, PyGeoVision turns the rasters into the things people
actually use: **a map for a slide, polygons for a GIS, and a report for a decision-maker**, all measured from the delivered rasters so the numbers in the report
match the files.

```{list-table}
:widths: 28 72
:header-rows: 1
* - You want…
  - Go to
* - To know what each file is and who it is for
  - This page
* - To read the report: what each section says and how it is built
  - [What the reports contain](contents.md)
* - To use the polygons: attributes, formats, size limits
  - [Vector layers](vectors.md)
* - To change class names, add a format or refresh a report without re-running
  - [Rebuilding deliverables](rebuilding.md)
* - To produce deliverables for your own analysis
  - [From Python](#from-python)
```

## Why this exists

A raster of 0, 1, 2 and 255 is not a deliverable. The people who need the answer (a project officer, a ministry, a client) want to know
*how much*, *where*, *how sure we are* and *what the limits are*, in a form they can open without a GIS. The GIS user wants individual polygons with areas,
not one raster. The scientist wants the method and the numbers. PyGeoVision produces all three from one run, from the same measurements, so they cannot disagree.

## What you get

```text
output/
├── <raster>.tif                          # the analysis (see the pipeline page)
├── run_summary.json                      # machine-readable record of the run
└── deliverables/
    ├── <pipeline>_report.pdf             # for reading and sharing
    ├── <pipeline>_report.docx            # for editing and merging into your own document
    ├── <pipeline>_report.xlsx            # for tables, charts and further calculation
    ├── maps/
    │   ├── <pipeline>_map.png            # the main map, for slides and documents
    │   ├── <pipeline>_map_<layer>.png    # one map per further layer
    │   ├── <pipeline>_chart.png          # area per class (or the chart that fits the pipeline)
    │   └── <pipeline>_histogram.png      # distribution of values, for continuous results
    └── vectors/
        ├── <pipeline>.gpkg               # all layers in one GeoPackage
        └── geojson/<pipeline>_<layer>.geojson
```

| File | Open it with | Use it for |
|---|---|---|
| PDF report | any PDF viewer | Sharing a finished, fixed document. |
| Word report | Word, LibreOffice, Google Docs | Editing the text, adding your own findings, merging into a larger report. |
| Excel workbook | Excel, LibreOffice Calc | Sorting and filtering tables, re-computing with the live formulas, making your own charts. |
| Map PNG | anything | Slides, web pages, emails. |
| GeoPackage | QGIS, ArcGIS, `geopandas` | The right default for GIS: several layers in one file, no field-name limit. |
| GeoJSON | web maps, GitHub, `geopandas` | Web use and quick sharing. |
| Shapefile / KML | legacy GIS / Google Earth | Only when asked for (`--vector-formats shp,kml`). |
| `run_summary.json` | any program | Provenance, automation, [comparing runs](../pipelines/results.md#comparing-runs). |

## Switching it on and off, and choosing formats

Deliverables are on by default for every pipeline run.

```bash
pygeovision channel flood_mapping … --no-report                        # skip all of it, including run_summary.json
pygeovision channel flood_mapping … --report-formats pdf,docx          # only these reports (also xlsx; 'none' for none)
pygeovision channel flood_mapping … --vector-formats gpkg,shp,kml      # which vector files (gpkg, geojson, shp, kml; 'none')
pygeovision channel flood_mapping … --class-names "Dry,Flooded,Permanent water"
pygeovision channel flood_mapping … --report-title "Volta flood, June 2024"
export PYGEOVISION_REPORT=0                                            # off everywhere
```

```python
result = client.pipeline("flood_mapping", bbox=bbox, date_before="2024-02", date_after="2024-06",
                         report_formats=["pdf"], vector_formats=["gpkg"],
                         class_names=["Dry", "Flooded", "Permanent water"], report_title="Volta flood, June 2024")
```

Format names are forgiving: `word`, `excel`, `geopackage`, `shapefile` and `json` are accepted as aliases. An unknown format is reported in `deliverables.warnings` rather than stopping the run.

## It never costs you the analysis

The deliverables are built **after** the analysis has been written. If a library is missing, a file cannot be read or a picture cannot be drawn, then:

- the rasters and `run_summary.json` are still there;
- the part that failed is skipped, the other parts are still delivered;
- the reason is in `result.deliverables.warnings` (Python) and in `result.metadata["deliverables_error"]` when nothing could be built, and is logged.

The libraries are optional: `pip install "pygeovision[report]"` (matplotlib, reportlab, python-docx, openpyxl, geopandas and friends). Without them the analysis still runs.

## Honesty built in

The report states only what the pipeline computed. Each pipeline has a **report specification** with its question, method, interpretation hints and
limitations, so a flood report says that radar misses flooded buildings and a land-cover report says that WorldCover 2021 cannot show change. Reports of
pipelines that run **your network** carry a standing warning that accuracy is not assessed, and show generic class names until you pass `--class-names`.
Warnings appear at the top when the imagery covered less than 95 % of your box or less than half the area was cloud-free.

## From Python

You can produce the same deliverables for any raster result, including your own analyses:

```python
from pygeovision.pipelines import PipelineResult
from pygeovision.reporting import generate_deliverables

result = PipelineResult(
    pipeline="my_analysis",                       # becomes the file names and the default title
    output_path="out/my_map.tif",                 # a class raster (uint8, 255 = no data) or a continuous one
    stats={"valid_fraction": 0.97},               # anything you want in the "Other statistics" table
    success=True,
)
d = generate_deliverables(
    result, bbox=(-0.15, 51.47, -0.10, 51.52), output_dir="out",
    parameters={"threshold": 0.3},                # shown under "Method and parameters"
    report_formats=["pdf", "docx"], vector_formats=["gpkg"],
    class_names=["Background", "Target"], title="My analysis",
)
print(d.reports["pdf"], d.maps, d.vectors["gpkg"], d.warnings)
```

For pipelines that are not in the catalogue the report uses a **generic specification**: a title taken from the name, the output raster as the one layer, and
your statistics in a table. To give your own analysis a specific question, method and limitations, add a `ReportSpec`
([What the reports contain](contents.md#report-specifications)).

`generate_deliverables` raises `ValueError` if `result.success` is false (there is nothing to report) and
`ReportingUnavailable` internally when a library is missing (reported as a warning, not raised).

## What to do when…

| Symptom | Cause | Fix |
|---|---|---|
| No `deliverables/` folder | `--no-report`, `PYGEOVISION_REPORT=0`, or the build failed | After `--no-report`, run again without it (`pygeovision report` needs the `run_summary.json` that flag skips). After a failure, check `result.metadata["deliverables_error"]`, and run `pygeovision report <output>` if `run_summary.json` exists |
| "pdf report skipped … Install the reporting extras" | Optional libraries missing | `pip install "pygeovision[report]"` |
| Report has class numbers, not names | A network you trained: PyGeoVision cannot know your labels | `--class-names` now, or [rebuild](rebuilding.md) |
| Map has no basemap | By design: it must work offline and draw the analysis layer clearly | Open the GeoPackage in QGIS over any basemap |
| Report says "reduced-resolution read" | The raster was very large; statistics were computed on a down-sampled read | Use `pygeovision postprocess class-stats` for exact counts |
| Far fewer polygons than expected | Speckle on a large area was merged by a sieve to stay under the size limit | See [Vector layers](vectors.md#size-limits) |
| Excel feature sheet is cut off | Excel is limited to the first 5,000 features per layer | Use the GeoPackage for the full table |
| Shapefile field names are cut at 10 characters | A Shapefile limit | Use the GeoPackage |

```{toctree}
:hidden:
:maxdepth: 1

contents
vectors
rebuilding
```
