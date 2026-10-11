# What the reports contain

The PDF, Word and Excel reports are three renderings of **one** format-neutral report, so the numbers and the wording are identical. This page
lists every section, where its numbers come from, and how to extend the system.

## Sections

1. **Header and key figures.** The title, period and area, and up to four headline figures (for example *Flooded area*, *Flooded share of land*). The first map is the cover image.
2. **Executive summary.** Plain-language bullet points written for this pipeline from its own statistics (for example *"62.9 ha (3.5 % of the land) is flooded between 2024-01-05 and 2024-06-10: 62.9 ha confirmed by radar and optical…"*). Underneath, **callouts**:
   - a *warning* if the imagery covered less than 95 % of your box, or less than half the area had valid pixels;
   - a *warning* on pipelines that run your network, saying accuracy is not assessed;
   - an *information* note when class names are generic, when statistics came from a reduced-resolution read, or when the pipeline has a note about its data (`product_note`, `source_note`, `note`).
3. **Study area and inputs.** The bounding box, centre, area in km², period, output grid (size and CRS), the share of your box that the imagery covered, the valid share, and a table
   of **imagery used** (each input, its coverage, scenes used and scenes that failed).
4. **Results.** The map, a summary chart, **class-area tables** (value, name, pixels, area in hectares, share of valid area, share of your box), **key statistics** (the pipeline's own figures, formatted), **other statistics**, and for continuous
   layers a **value-statistics table** (min, max, mean, median, percentiles) with a histogram. Time-series pipelines show a table of scene means by date and a panel of maps.
5. **Additional layers.** One map per further layer (for example the radar change in dB, the height above drainage).
6. **Method and parameters.** The question, the method step by step, and **every parameter the run used**, including defaults, so the analysis can be repeated.
7. **Interpretation and limitations.** How to read the result and, separately, what it cannot tell you.
8. **Deliverables.** A table of every file written, with type and purpose, and a table of vector layers (features, total area).
9. **References.** The publications and data products the method rests on.

## Where the numbers come from

Areas and class shares in the tables are **measured from the delivered rasters**, not copied from the pipeline's statistics, so you can compare the two (they should agree
within rounding). Areas use the true area of each pixel: constant for a metric grid, and computed on the WGS 84 ellipsoid row by row for a latitude/longitude grid. Percentages in a
class table are over **valid pixels**; the "share of your box" column is over everything. Statistics of very large rasters (over 40 million pixels) come from a reduced-resolution read, and the report says so.

## The three formats

````{tab-set}
```{tab-item} PDF
Fixed layout with a cover, running header and page numbers; the font embeds so it prints the same everywhere. Good for sharing and archiving.
```
```{tab-item} Word
The same sections as editable headings, tables and pictures (Arial, 17 cm text width). Good for adding your own conclusions and merging into a larger document.
```
```{tab-item} Excel
A workbook with one sheet per kind of content, formulas that **recalculate when you open it**, and native charts:

| Sheet | Contents |
|---|---|
| Summary | Title, key figures, headline sentences, warnings |
| Metrics | The key and other statistics |
| Classes | Class tables with counts, areas, shares |
| Statistics | Value statistics of continuous layers |
| Timeseries | Per-date figures (time-series pipelines) |
| Inputs | Imagery used and coverage |
| Vectors | Vector layers, feature counts, total area |
| Files | Every file written |
| Parameters | Every parameter used |
| Maps | The map images |
| Features … | The first 5,000 features of each vector layer with `patch_id`, class, area, share and centroid |

Sheets that would be empty are left out.
```
````

## The map

Drawn with matplotlib on a plain background: the analysis layer, a graticule, a scale bar, a north arrow, a legend with class areas, up to four key figures, a footer with the CRS and
a warning banner if the data quality is low. **No web basemap is used**, so the map is reproducible offline and the analysis stays readable. Class colours are fixed per product
(the ESA WorldCover palette is ESA's own; the rest are chosen to stay distinguishable for common colour-vision deficiencies, avoiding red/green as the only difference), so a map made today looks like one made next year.

## Report specifications

Everything that depends on *which* analysis was run lives in a `ReportSpec`, one per pipeline, in `pygeovision.reporting.specs`:

| Field | Meaning |
|---|---|
| `pipeline`, `title` | Name and report title |
| `question` | The question the analysis answers (opens "Method and parameters") |
| `method` | Tuple of method sentences |
| `layers` | The rasters to report: key, file name, title, kind (`classes` or `continuous`), class definitions or a binning, colour map, which classes become polygons |
| `metrics` | Statistics to show as key figures: key, label, format code (`frac` for shares, `ha`, `km2`, `int`, `num1`–`num4`, `c` for °C, `m`, `deg`, `text`, `bool`) |
| `headline` | A function returning the executive-summary sentences from the statistics |
| `reading`, `limitations`, `references` | Interpretation hints, limitations and citations |
| `tables`, `charts` | Extra tables and charts built from list or dict statistics |
| `multi_date`, `chart` | Time-series behaviour |

```python
from pygeovision.reporting import spec_for, SPECS

spec = spec_for("flood_mapping")
spec.title; spec.method; spec.limitations
sorted(SPECS)                       # every pipeline that has a specification
```

A pipeline without a specification gets a generic one (`generic_spec`). The text is meant to say only what the code computes, and a test fails when a pipeline is added without a specification.

## What to do when…

| Symptom | Fix |
|---|---|
| A figure in the report differs slightly from `result.stats` | The report measures the raster; rounding and a reduced-resolution read explain small differences. Large differences mean the raster was edited after the run |
| Wording says things I do not want | Edit the Word report, or write your own specification |
| I need the report in another language | Reports are English; the Excel and JSON outputs carry the numbers for your own template |
