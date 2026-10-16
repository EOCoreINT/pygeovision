# Buildings and roads in central London: two reference datasets, and what the rasters leave out

`building_footprints` and `road_network` do not look at imagery at all. They read published vector datasets (Microsoft's
building footprints, OpenStreetMap's roads), clip them to your box, measure them, and summarise them on a grid. That makes them the
cheapest pipelines in PyGeoVision, and it makes the *interesting* part of reading their reports not the totals but the **difference
between the total and the map**: in both runs, the density raster describes less than the headline figure says. The page shows
where the missing amount goes, with arithmetic you can repeat on your own results.

```{admonition} What this page is based on
:class: note
- **Numbers:** copied from two PDF reports PyGeoVision wrote for the same 19.32 km² box (`building_footprints_report.pdf`,
  generated 2026-10-11 07:55 UTC, and `road_network_report.pdf`, generated 2026-10-11 09:55 UTC, both PyGeoVision 2.1.8).
  The totals, the per-class lengths and the group sums are re-derived in `tests/test_case_studies.py`.
- **Not re-run:** neither dataset was downloaded again.
- **Reproduced:** the two ways a density raster loses part of the total, on the real pipeline, with three invented rectangles.
- **Not checked:** whether the datasets are complete or correct for this part of London. They are "what the source has", and the
  reports say so.
```

## The result at a glance

| Buildings (Microsoft ML footprints) | Value | Roads (OpenStreetMap) | Value |
|---|---:|---|---:|
| Buildings | **5,898** | Road length | **1,030.1 km** |
| Footprint area | 648.0 ha | Road density | 53.32 km per km² |
| Share of the box | 33.5 % | Mapped ways | 28,454 |
| Buildings per km² | 305.3 | Share of length with no name | 49.9 % |
| Mean / median footprint | 1,098.7 / 359.8 m² | Largest class | footway, 493.39 km |
| With a height | 5,515 (93.5 %) | Largest group | tracks and paths, 598.77 km |
| Mean height | 10.53 m | Cells in the density raster | 75 of 104 |

## Study area and inputs

| | |
|---|---|
| Area of interest (west, south, east, north) | -0.1500, 51.4700, -0.1000, 51.5200 (central London, by its coordinates) |
| Size | 19.32 km² |
| Buildings | Microsoft Global ML Building Footprints, `source=microsoft`; no date (a published dataset) |
| Roads | OpenStreetMap `highway` ways through the Overpass API; no date |
| Grids | UTM zone 30 (EPSG:32630): buildings in **100 m** cells (38 × 58), roads in **500 m** cells (8 × 13) |

The same box is used by the [land cover and habitat pages](london-landcover-habitat.md) and the
[vegetation page](london-vegetation.md).

## Run it

```bash
pygeovision channel building_footprints \
    --bbox -0.15 51.47 -0.10 51.52 \
    --param source=microsoft \
    --output out/bld_ms

pygeovision channel road_network \
    --bbox -0.15 51.47 -0.10 51.52 \
    --output out/roads
```

```python
from pygeovision import PyGeoVision

client = PyGeoVision()
box = (-0.15, 51.47, -0.10, 51.52)

buildings = client.pipeline("building_footprints", bbox=box, source="microsoft", output_dir="out/bld_ms")
roads = client.pipeline("road_network", bbox=box, output_dir="out/roads")
for r in (buildings, roads):
    if not r.success:
        raise SystemExit(r.error)
print(buildings.stats["building_count"], roads.stats["total_length_km"])
```

`source=microsoft` is the default and is shown because the report lists it. The roads run uses every default (`cell_m=500`, and
ways that are proposed, under construction, razed, abandoned, planned or platforms are left out).

## What the pipelines did

```{figure} ../_static/img/tree-case-london.svg
:alt: One central-London box feeding five pipelines side by side: land cover, biodiversity hotspot, building footprints, road network and crop health
:width: 100%
:class: pgv-figure

The same box, five pipelines. This page is the third and fourth.
```

**Buildings.** The footprint polygons are read, clipped to the box (a building that crosses the edge is cut, and only the part inside is
measured), and their area is taken in the local UTM zone. Each building is then assigned to the 100 m cell that holds its
**centroid**, and two rasters are written: the share of each cell covered by buildings, and the number of buildings per cell.

**Roads.** Every OSM way tagged `highway` is clipped to the box, measured in UTM, and given a group. Each way is then cut into pieces
of 100 m (a fifth of the cell) and each piece goes to the 500 m cell holding its midpoint. Density is the kilometres of road in a
cell divided by its area.

## Buildings

```{figure} ../_static/img/case-studies/buildings-density.webp
:alt: Map of the share of each 100 m cell covered by buildings, in yellow to dark red. The top of the box is mostly orange and red with dark-red cells, the middle has large pale areas, and the bottom is a patchwork of pale yellow and orange
:width: 100%
:class: pgv-figure

Share of each 100 m cell covered by buildings. The pale areas are cells with few or no building centroids; compare them with the river and the parks on the [land-cover map](london-landcover-habitat.md).
```

| Figure | Value | Check |
|---|---:|---|
| Buildings per km² | 305.3 | 5,898 / 19.32 |
| Share of the box covered | 33.5 % | 648.0 ha / 1,931.7 ha |
| Mean footprint | 1,098.7 m² | 648.0 ha / 5,898 |
| Median footprint | 359.8 m² | (not derivable) |

```python
# check: the building figures agree with each other
buildings, footprint_ha, area_km2 = 5_898, 648.0, 19.32
print(f"{buildings / area_km2:.1f} per km2")
print(f"{100 * footprint_ha / (area_km2 * 100):.1f} % of the box covered")
print(f"{footprint_ha * 1e4 / buildings:.1f} m2 mean footprint, against a median of 359.8")
```

```text
305.3 per km2
33.5 % of the box covered
1098.7 m2 mean footprint, against a median of 359.8
```

The mean is three times the median. A few very large buildings (a station, a hospital, a block of flats drawn as one polygon) pull the
mean up, so "the typical building" is better read from the median, 360 m².

The Microsoft dataset carries heights for 5,515 of the 5,898 buildings (93.5 %), with a mean of 10.53 m. The report does not say how those heights were obtained, so treat them as the dataset's estimates, not a survey.

```{figure} ../_static/img/case-studies/buildings-density-hist.webp
:alt: Histogram of the share of each cell covered by buildings, with a tall spike in the first bin at zero, a broad low hump between 0.1 and 0.5, and a smaller spike in the last bin at 1.0
:width: 100%
:class: pgv-figure

The cell-coverage histogram. The spike at 0 is cells with no building centroid; the spike at 1 is a cap.
```

### The raster adds up to less than the headline

The report says 648.0 ha of buildings. The density raster says the mean cell is 29.9 % covered, over 1,929 cells of 1 ha:

```python
# check: what the density and count rasters add up to
cells, cell_ha = 1_929, 1.0
raster_ha = 0.299 * cells * cell_ha
raster_count = 2.983 * cells
print(f"density raster: {raster_ha:,.0f} ha against 648 ha reported ({raster_ha - 648.0:+.0f} ha, {100 * (raster_ha - 648.0) / 648.0:+.0f} %)")
print(f"count raster:   {raster_count:,.0f} buildings against 5,898 reported ({raster_count - 5_898:+,.0f})")
```

```text
density raster: 577 ha against 648 ha reported (-71 ha, -11 %)
count raster:   5,754 buildings against 5,898 reported (-144)
```

The raster holds about 11 % less area and about 144 fewer buildings than the headline. Both come from how the grid is built, and both
are visible in the code:

- **Each cell's share is capped at 100 %.** A building's whole area goes to the cell holding its centroid, and a cell cannot
  show more than 1.0. A large building in a small cell gives a number above 1, which is then cut to 1: the 100 % spike at the right of
  the histogram (the 95th percentile is exactly 1.0).
- **Only cells whose centre is inside the box are filled.** The 100 m grid is the rectangle that contains the box in UTM, and cells
  with their centre outside are empty. A building whose centroid is in one of them is counted in the headline, but not in either
  raster.

The report does not split the 71 ha between the two. You can, with the files it wrote: sum `building_density.tif` times the cell
area, and compare with the footprints in `building_footprints.gpkg`. The test suite reproduces both effects on the real pipeline: a
150 m building in a 100 m cell gives a density of exactly 1.0 and the raster then holds less area than the building, and a building
centred outside the filled cells appears in the stats and in neither raster.

**What to do with it.** Use the headline (33.5 %, 305 per km²) as the figure for the box, and use the rasters to see *where*. Do not
add up the raster and quote it as the total.

## Roads

```{figure} ../_static/img/case-studies/roads-density.webp
:alt: Map of road density in 500 m cells from pale yellow to dark brown. The densest, darkest cells are in the top (north) of the box, with a few dark cells further down the middle, and the rest is orange
:width: 100%
:class: pgv-figure

Road density in 500 m cells (km of road per km²). The darkest cells are 95 km per km².
```

| Figure | Value | Check |
|---|---:|---|
| Road density | 53.32 km per km² | 1,030.1 km / 19.32 km² |
| Mean length of a way | 36.2 m | 1,030.1 km / 28,454 ways |
| Raster minimum / median / maximum | 21.9 / 45.1 / 95.3 | over 75 cells |

### What is a "road" here?

The report's summary says "the longest group is tracks and paths (598.8 km)". In central London that does not mean tracks:

```{figure} ../_static/img/case-studies/roads-length-by-class.webp
:alt: Horizontal bar chart of road length by OpenStreetMap highway class. Footway is by far the longest bar at 493.4 km, followed by residential 151.3, service 103.3, primary 58.6, unclassified 56.5 and pedestrian 56.0 km; the remaining classes are short
:width: 100%
:class: pgv-figure

Road length by highway class. Footways are nearly half of the whole network.
```

```python
# check: the groups add up, and what the density is without the paths
groups = {"local roads": 333.48, "major roads": 92.11, "other": 5.70, "tracks and paths": 598.77}
total = sum(groups.values())
for name, km in groups.items():
    print(f"{name:<18} {km:>7.2f} km {100 * km / total:>5.1f} %")
vehicle = groups["local roads"] + groups["major roads"]
print(f"total {total:,.1f} km; major + local roads {vehicle:.1f} km = {vehicle / 19.32:.1f} km per km2, against 53.3 with paths")
```

```text
local roads         333.48 km  32.4 %
major roads          92.11 km   8.9 %
other                 5.70 km   0.6 %
tracks and paths    598.77 km  58.1 %
total 1,030.1 km; major + local roads 425.6 km = 22.0 km per km2, against 53.3 with paths
```

The "tracks and paths" group is `track`, `path`, `footway`, `cycleway`, `bridleway`, `steps` and `pedestrian`. In this box it is almost
entirely footways (493.39 km, a mapped pavement or crossing, one way each), pedestrian streets (56.01 km) and cycleways (31.73 km); the
`track` class has 0.77 km. So **58 % of the "road" length here is for people on foot**, and the density of ground a vehicle can use is
22 km per km², not 53. The headline is correct for what OpenStreetMap holds; it is not a measure of the road network a driver sees.

That also explains the 28,454 ways averaging 36 m: pavements and crossings are drawn as many short ways. The report's own limitation says
the same from the other side, "Each OSM way counts once: a divided road mapped as two ways is counted twice", so length is *mapped
length*, not centreline length.

The 49.9 % of length "without a name" is the same story: a pavement is rarely named. (That is an expectation from how OSM is
mapped; the report only gives the share.)

### The density raster covers 75 of 104 cells

```python
# check: how much road is in the cells the density raster fills
mean_density, filled_cells, cell_km2 = 50.466, 75, 0.25
in_raster = mean_density * filled_cells * cell_km2
print(f"{in_raster:.1f} km in the raster, {1030.1 - in_raster:.1f} km ({100 * (1030.1 - in_raster) / 1030.1:.1f} %) in cells with their centre outside the box")
print(f"the raster covers {filled_cells * cell_km2:.2f} km2 of the {19.32} km2 box")
```

```text
946.2 km in the raster, 83.9 km (8.1 %) in cells with their centre outside the box
the raster covers 18.75 km2 of the 19.32 km2 box
```

The raster mean (50.5) is lower than the headline (53.3) for the second of the two reasons seen with the buildings: the 8 × 13 grid is a rectangle round
the box, and a cell is filled only if its centre is inside it. The 29 outer cells hold about 84 km of road that the density raster
does not show. The headline length is not affected, because it comes from the vector layer.

A single 500 m cell is a coarse unit: the highest value (95.3 km per km², 23.8 km of road in one 0.25 km² cell) is a very small
part of the city, and the report notes that "very high values in a single cell can be a junction or dual carriageway rather than a dense
grid".

## Does the footprint layer agree with the land-cover map?

The land-cover run has 73.8 % built-up and this run has 33.5 % covered by building footprints. The ratio is 0.45:

```python
# check: building footprints against WorldCover built-up
print(round(33.5 / 73.8, 2), "of the built-up area is building footprint;", round(73.8 - 33.5, 1), "% of the box is built-up but not a building")
```

```text
0.45 of the built-up area is building footprint; 40.3 % of the box is built-up but not a building
```

That is not a disagreement, because the two measure different things. WorldCover's built-up class is everything artificial at 10 m,
which includes roads, pavements, squares, car parks and yards. This run's footprints are the buildings only. The 40.3 percentage points
between them is where the streets, paved ground and surfaces other than buildings would be, but this run cannot split it (the road
lines have no width). Had the footprints come out *larger* than the built-up area, that would have pointed to a mismatch
between the two datasets (a different date, say) worth looking into.

## Things to be careful about

- **Datasets, not detections.** What is missing from the source is missing here, and the footprints differ in date: a new building may
  be absent and a demolished one still present.
- **Footprints are clipped to the box.** A building that crosses the edge is counted once, with only its inside area.
- **Heights are the dataset's estimates.** Do not turn them into volumes without checking them against something you trust.
- **Road length is mapped length.** Divided roads count twice and pavements count as roads.
- **Densities depend on mapping effort.** Compare places only where OSM has been mapped as thoroughly.
- **Rasters are for where, headlines are for how much.** See the two sections above.

## Change something

| To | Use | Note |
|---|---|---|
| Another building dataset | `--param source=google` (or `osm`) | Run it three times to compare the sources over the same box. `google` also takes `min_confidence`. |
| Finer or coarser grid | `--param cell_m=50` for buildings or roads | Changes the detail and the size of the two gaps above; 500 m cells are coarse for a single district. |
| Keep ways that are not built yet | `--param exclude=proposed,construction` on `road_network` | The ways named are dropped, the others kept. The default drops proposed, construction, razed, abandoned, planned and platform. |
| Roads without paths | Filter `road_network.gpkg` by `group` | Keep `major roads` and `local roads`; the density is then 22 km per km². |
| Detect your own buildings | `--checkpoint weights.pth` on `building_footprints` | Runs a network you trained instead; at 10 m, houses are not resolvable. |

## The files they wrote

| File | What it holds |
|---|---|
| `out/bld_ms/building_density.tif`, `building_count.tif` | Share of each cell covered, and buildings per cell (NaN outside the box) |
| `out/bld_ms/building_footprints.gpkg` | Every footprint, clipped, with `area_m2` and `source` (layer `buildings`) |
| `out/bld_ms/deliverables/…` | The report (PDF, Excel, Word), three maps and the vector files |
| `out/roads/road_density.tif` | Kilometres of road per km² in 500 m cells |
| `out/roads/road_network.gpkg` | Every way (28,454), with `highway`, name, `group` and `length_m` (layer `roads`) |
| `out/roads/deliverables/…` | The report, the map, the class chart and the histogram |
| `run_summary.json` | The machine-readable record in each folder |

## Related

- [`building_footprints`](../pipelines/catalog/building_footprints.md) and [`road_network`](../pipelines/catalog/road_network.md): every
  argument, statistic and failure message.
- [Land cover and habitat in the same box](london-landcover-habitat.md): the 73.8 % built-up this page compares with.
- [`road_extraction`](../pipelines/catalog/road_extraction.md): train a model on the lines this pipeline writes.
- [Reading results](../pipelines/results.md).
