# Streams and height above drainage in the Tyrol: reading HAND on steep ground

`drainage_analysis` needs no imagery and no date. It reads a 30 m elevation model and answers three questions: where does water
collect, which cells are streams, and how high does the ground stand above the nearest stream? This run covers an 84 km² box in the
Inn valley of the Tyrol, in the Austrian Alps, and the report is a good place to learn what each number means because the terrain is steep enough
that the answers are not obvious: **only 19.5 % of the ground lies within 15 m of a stream**, and the mean height above the
nearest stream is 220 m. The page also shows the three places where the report's arithmetic is easy to misread.

```{admonition} What this page is based on
:class: note
- **Numbers:** copied from the PDF report PyGeoVision wrote for this run (`drainage_analysis_report.pdf`, PyGeoVision 2.1.8,
  generated 2026-10-11 08:31 UTC). Pixel counts, areas, the grid and the density are re-derived in `tests/test_case_studies.py`; the
  grid is rebuilt with the package's own `make_grid` and matches the report exactly (264 × 379).
- **Not re-run:** the elevation tiles were not downloaded again.
- **A terrain screen, not a flood map:** HAND knows nothing about rainfall, discharge or defences, and the report says so.
```

## The result at a glance

| Figure | Value | What it measures |
|---|---:|---|
| Stream length | **64.8 km** | Streams draining at least 1.00 km² |
| Drainage density | **0.72 km per km²** | Stream length over the area of the analysed grid |
| Highest Strahler order | 3 | A confluence of two order-2 streams makes order 3 |
| Largest contributing area | 58.80 km² | The biggest catchment found, at an outlet |
| Mean HAND | 220.1 m | Mean height above the nearest stream, over cells that have one |
| Median HAND | 123.3 m | |
| Area within 5 m / 10 m / 15 m of a stream | 12.5 % / 16.7 % / 19.5 % | |
| Cells raised by depression filling | 5,801 (5.8 %) | |
| Deepest fill | 30.3 m | |
| Cells with a HAND value | 93.2 % | The rest drain out of the box before reaching a stream |

## Study area and inputs

| | |
|---|---|
| Area of interest (west, south, east, north) | 11.3000, 47.2000, 11.4000, 47.3000 (the Inn valley around Innsbruck, Tyrol, by its coordinates) |
| Size requested | 84.16 km² |
| Output grid | 264 × 379 = **100,056 cells**, EPSG:32632, 30 m, covering **90.05 km²** |
| Elevation | Copernicus GLO-30 digital surface model, `cop30`, from the Planetary Computer |
| Dates | none (an elevation model) |
| Stream threshold | 1 km² |

The grid is 7 % larger than the box because the box, drawn in degrees, is slightly skewed in the UTM projection, so the grid is the
rectangle that contains it. It matters below, because the density is taken over the grid.

## Run it

```bash
pygeovision channel drainage_analysis \
    --bbox 11.3 47.2 11.4 47.3 \
    --param stream_threshold_km2=1 \
    --output out/drain
```

```python
from pygeovision import PyGeoVision

client = PyGeoVision()
result = client.pipeline(
    "drainage_analysis",
    bbox=(11.3, 47.2, 11.4, 47.3),
    stream_threshold_km2=1,
    output_dir="out/drain",
)
if not result.success:
    raise SystemExit(result.error)
print(result.stats["stream_length_km"], result.stats["drainage_density_km_per_km2"])
```

`stream_threshold_km2=1` is the default, which is why the report lists it as its only pipeline argument. It is the one setting
that shapes the whole result: the amount of catchment you decide makes a channel.

## What the pipeline did

```{figure} ../_static/img/flow-case-drainage.svg
:alt: Drainage analysis: download a Copernicus 30 m elevation model, fill depressions with Priority-Flood, route water D8, accumulate flow, mark streams above a catchment threshold and number them by Strahler order, compute height above the nearest drainage
:width: 100%
:class: pgv-figure

From an elevation model to streams and HAND. The dashed card is why streams near the edge of a box read short.
```

1. **Elevation.** The Copernicus GLO-30 *surface* model: heights include trees and buildings, and its vertical accuracy is a few metres.
2. **Fill.** Priority-Flood raises every sink and resolves flat areas so that water always has a way out. Here 5,801 cells (5.8 %) were
   raised, by up to 30.3 m. A 30 m fill in a 30 m model is large; in an alpine valley it is usually a surface-model artefact (a
   bridge, a dense forest edge, a dam), not a real lake.
3. **Route.** Every cell drains to its steepest downhill neighbour among eight (D8).
4. **Accumulate.** The area draining through each cell, in km². A cell that drains 1 km² or more is a stream.
5. **Streams and order.** Stream cells are numbered by Strahler order: first-order streams have no tributary, and two streams of
   the same order meeting make the next order up.
6. **HAND.** Each cell's height above the stream cell it drains into.

## The result

```{figure} ../_static/img/case-studies/drainage-hand.webp
:alt: Map of height above the nearest drainage in metres. A wide dark-blue band runs east to west across the middle of the box and branches into dark-blue tributaries; everything else fades to white as height increases
:width: 100%
:class: pgv-figure

Height above the nearest drainage. The darkest cells are on or beside a stream; white is 24 m or more above one (the colour scale stops there).
```

The wide dark band is the valley floor, low and flat enough that it counts as within a few metres of the stream. HAND looks like
a drainage map because it *is* one, turned into heights.

| Class | Value | Pixels | Area (ha) | Share of valid area |
|---|---:|---:|---:|---:|
| No stream | 0 | 98,202 | 8,838.2 | 98.1 % |
| Order 1 | 1 | 917 | 82.5 | 0.9 % |
| Order 2 | 2 | 654 | 58.9 | 0.7 % |
| Order 3 | 3 | 283 | 25.5 | 0.3 % |

Each 30 m cell is 0.09 ha, so 917 cells are 82.5 ha. The three stream rows add to 1,854 cells, which is the "stream cells" figure.

```{figure} ../_static/img/case-studies/drainage-stream-order.webp
:alt: Map of the stream network coloured by Strahler order, thin light-blue lines in a branching pattern across the box
:width: 100%
:class: pgv-figure
```

```{figure} ../_static/img/case-studies/drainage-contributing-area.webp
:alt: Map of contributing area in square kilometres, a dense dendritic pattern of thin dark lines over a pale background, with the strongest lines in the middle
:width: 100%
:class: pgv-figure

Contributing area (km²). Streams are the cells above 1 km²; the pattern below that is the hillside drainage that feeds them.
```

## Three numbers that are easy to misread

### The density is over the grid, not the box

The report says 0.72 km per km². The box you asked for is 84.16 km², and 64.8 / 84.16 is 0.77. The pipeline divides by the area of
the grid it analysed (90.05 km²), which is what 0.72 is. Neither is wrong, but they are different quantities, and if you compare
with a published density you should know which area was used. The same function works on any report:

```{literalinclude} ../_examples/case_studies/checks.py
:language: python
:pyobject: drainage_density
```

```python
# check: the report's density divides by the analysed grid, not the requested box
print(round(drainage_density(64.8, 90.05), 2), round(drainage_density(64.8, 84.16), 2))
print(round(1854 * 30 / 1000, 1), "km if every step were straight;", round(64.8 * 1000 / 1854, 1), "m average step")
```

```text
0.72 0.77
55.6 km if every step were straight; 35.0 m average step
```

The second line is a check on the length. 1,854 stream cells at 30 m would be 55.6 km, but the pipeline measures each cell's step to
the cell it drains to, and a diagonal step is 42.4 m. An average step of 35.0 m means roughly 40 % of the steps are diagonal (if
the only steps are straight and diagonal), which is what a D8 stream looks like.

### Mean HAND is a long-tailed number

The mean is 220 m and the median is 123 m. The distribution is a spike near zero (the cells next to a stream) and a long tail to
1,438 m (a ridge). The mean is pulled up by the tail, so "typical height above a stream" is better read from the median, or from
the three percentages:

```{figure} ../_static/img/case-studies/drainage-hand-hist.webp
:alt: Histogram of height above nearest drainage with a tall spike in the first bin near zero and a long right-hand tail to about 1,200 metres
:width: 100%
:class: pgv-figure

HAND histogram: a spike at the streams and a long tail up the slopes.
```

**12.5 %** of the ground is within 5 m of a stream, **16.7 %** within 10 m, **19.5 %** within 15 m. Each percentage is a share of
*all valid cells*, including the 6.8 % that have no HAND value, so they cannot reach 100 %.

### 93.2 % have a HAND value

A cell has HAND only if the water leaving it reaches a stream cell **inside the box**. The other 6.8 % drain out through the edge
first. Contributing areas are cut at the edge of the box in the same way, so streams near the border read short and the largest
catchment (58.80 km², about two thirds of the grid) is only what lies inside. Use a box that contains the whole catchment where it
matters.

## What this tells a flood map

This run connects to the two [flood studies](flood-accra.md). `multisensor_flood_mapping` computes HAND the same way and, by
default, removes any flood candidate more than 15 m above the nearest stream. In *this* box that rule would apply to about 74 %
of the ground (the 93.2 % of cells that have a HAND value, minus the 19.5 % within 15 m), which is the point of the filter: in steep
terrain most of the ground is too high to flood. It also shows what the filter does **not** do. It only removes a pixel whose HAND is a number, so the 6.8 % of cells without a
HAND value are never removed. And it is a necessary condition, not a flood map: low ground *may* flood, and a pixel under 15 m is
kept whether or not water ever reaches it.

## Things to be careful about

- **A surface model, not the ground.** Canopy and buildings can block or divert modelled flow. Heights are good to a few metres.
- **The threshold is a choice.** 1 km² makes a stream here. Lower it and you get more, shorter first-order streams; raise it and the
  network thins. Density and the number of stream cells follow directly, so state the threshold with the density.
- **Channels finer than 30 m are absent.** Culverts, ditches and embankments are not represented.
- **Edges.** See above. Streams and HAND are least reliable within a few kilometres of the box boundary.

## Change something

| To | Use | Note |
|---|---|---|
| A denser or sparser network | `--param stream_threshold_km2=0.25` (or `5`) | Changes stream length, density, order and HAND together. |
| Other height bands | `--param hand_thresholds_m=2,5,10` | Reports the share of area within each (the run used 5, 10, 15). |
| Another elevation source | `--param dem_source=copernicus_aws` | The public AWS copy of the same product. |
| A larger box | `--param max_cells=...` | Raises the 6,000,000-cell safety limit; memory grows with the cells. |
| A coarser grid | `--param resolution_m=60` | Faster and smoother. Finer than 30 m only interpolates. |

## The files it wrote

| File | What it holds |
|---|---|
| `hand_m.tif` | Height above the nearest drainage, metres |
| `stream_order.tif` | Strahler order, 0 for no stream |
| `flow_accumulation_km2.tif` | Contributing area, km² |
| `stream_network.gpkg` | The streams as lines, one feature per order (written by the code; the report does not list it) |
| `deliverables/drainage_analysis_report.pdf`, `.xlsx`, `.docx` | The report |
| `deliverables/vectors/drainage_analysis.gpkg` | Layer `streams` (3 features, one per Strahler order) and the `aoi` outline |
| `run_summary.json` | The machine-readable record |

## Related

- [`drainage_analysis`](../pipelines/catalog/drainage_analysis.md): every argument, the statistics and the failure messages.
- [Accra flood run](flood-accra.md): the terrain filter at work (9.0 ha removed).
- [Nepal flood run](flood-nepal.md): the run where the terrain stage was skipped for size.
- [`terrain_analysis`](../pipelines/catalog/terrain_analysis.md): slope, aspect and relief from the same DEM.
