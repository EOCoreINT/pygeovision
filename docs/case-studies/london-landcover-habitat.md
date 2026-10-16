# Central London from space: land cover, then habitat diversity

Two pipelines, one box, and the second reads the first's answer. `land_cover` clips the ESA WorldCover map to a 19.3 km² box in
central London and counts what is on the ground. `biodiversity_hotspot` then asks where that ground is **most mixed**, using the same
map, and flags the top tenth. The pair is a small, complete example of going from a reference map to a derived indicator, and
of the thing to watch while doing it: **a number that is true by construction is not a finding**. Here that is the "10 % hotspot
share".

```{admonition} What this page is based on
:class: note
- **Numbers:** copied from two PDF reports PyGeoVision wrote for this box (`land_cover_report.pdf`, PyGeoVision 2.1.8,
  generated 2026-10-10 16:23 UTC; `biodiversity_hotspot_report.pdf`, PyGeoVision 2.1.9, generated 2026-10-11 17:22 UTC). The Shannon
  index, the evenness and the shares are re-derived from the class pixel counts in `tests/test_case_studies.py`.
- **Not re-run:** nothing was downloaded again. Both reports use the same published map.
- **A reference map, not a classification.** The land-cover numbers are WorldCover 2021, whose own validation reports about 75 %
  overall accuracy. There is no ground truth here.
```

## The result at a glance

| Figure | Value |
|---|---:|
| Built-up | **1,426.4 ha (73.8 %)** |
| Tree cover | 348.0 ha (18.0 %) |
| Permanent water | 124.7 ha (6.5 %) |
| Grassland | 30.7 ha (1.6 %) |
| Habitat classes present | 6 (one of them is a single pixel) |
| Regional Shannon index H' | 0.782 |
| Regional evenness | 0.437 |
| Highest local H' | 1.363 |
| Hotspot threshold | H' above 0.835 |
| Hotspots | 193.2 ha, **10.0 %** of the box, 53 polygons |

## Study area and inputs

| | |
|---|---|
| Area of interest (west, south, east, north) | -0.1500, 51.4700, -0.1000, 51.5200 (central London, by its coordinates) |
| Size | 19.32 km² (1,931.7 ha) |
| Output grid | 347 × 553 = **191,891 pixels**, EPSG:4326, about 10 m |
| Input | ESA WorldCover **2021**, 10 m, from the Planetary Computer; no satellite scene |
| Date passed | `2023-06` in the land-cover run, ignored; the period shown is "WorldCover 2021" |

Both reports use the same grid and the same 191,891 pixels, so every land-cover share below is the same in both.

## Run it

```bash
pygeovision channel land_cover \
    --bbox -0.15 51.47 -0.10 51.52 \
    --date 2023-06 \
    --output out/lc

pygeovision channel biodiversity_hotspot \
    --bbox -0.15 51.47 -0.10 51.52 \
    --output out/bio
```

```python
from pygeovision import PyGeoVision

client = PyGeoVision()
box = (-0.15, 51.47, -0.10, 51.52)

lc = client.pipeline("land_cover", bbox=box, date="2023-06", output_dir="out/lc")
bio = client.pipeline("biodiversity_hotspot", bbox=box, output_dir="out/bio")
for r in (lc, bio):
    if not r.success:
        raise SystemExit(r.error)
print(bio.stats["region_shannon_index"], bio.stats["hotspot_fraction"])
```

`--date 2023-06` does nothing here and is on the page because the run had it: with the default source the pipeline returns the
2021 map whatever date is passed, and the report says so ("it describes 2021 whatever date is passed"). Pass `source=dynamic_world`
for a map of a specific month (that needs an Earth Engine account). The habitat run has no date at all.

## What the pipelines did

```{figure} ../_static/img/tree-case-london.svg
:alt: One central-London box feeding five pipelines side by side: land cover, biodiversity hotspot, building footprints, road network and crop health
:width: 100%
:class: pgv-figure

One box, five pipelines. This page covers the first two; [buildings and roads](london-buildings-roads.md) and
[vegetation](london-vegetation.md) cover the rest.
```

**Land cover.** The WorldCover map is clipped to the box and the pixels of each class are counted. Nothing is classified; PyGeoVision
does the clipping, the areas, the vector layer and the report.

**Habitat diversity.** For every pixel, take the 21 × 21 pixels around it (210 m by 210 m), compute the Shannon index of the class
proportions in that window, H' = -sum(p ln p), and mark the pixels in the top decile of H' as hotspots. H' is 0 where one class fills
the window, and it grows with the number of classes and with how evenly they share the window.

## Land cover

```{figure} ../_static/img/case-studies/land-cover-map.webp
:alt: WorldCover map of central London. Built-up red covers most of the box; a wide blue river sweeps from the left edge through a bend to the top right; dark-green tree patches mark parks and the river corridor, with a few yellow grassland patches
:width: 100%
:class: pgv-figure

WorldCover 2021 over the box. The river is the wide blue band; the big green patches are parks.
```

| Class | Value | Pixels | Area (ha) | Share |
|---|---:|---:|---:|---:|
| Tree cover | 10 | 34,566 | 348.0 | 18.0 % |
| Grassland | 30 | 3,052 | 30.7 | 1.6 % |
| Cropland | 40 | **1** | 0.0 | 0.0 % |
| Built-up | 50 | 141,703 | 1,426.4 | 73.8 % |
| Bare / sparse vegetation | 60 | 184 | 1.9 | 0.1 % |
| Permanent water | 80 | 12,385 | 124.7 | 6.5 % |

The shares are re-derived from the pixel counts. One pixel is 1,931.7 ha / 191,891 = 0.01007 ha (a degree grid, so a little more than
100 m²):

```python
# check: re-derive the class areas and shares from the pixel counts
pixels = {"tree": 34_566, "grass": 3_052, "crop": 1, "built": 141_703, "bare": 184, "water": 12_385}
pixel_ha = 1931.7 / sum(pixels.values())
for name, (ha, pct) in class_areas(pixels, pixel_ha).items():
    print(f"{name:<6} {ha:>8.1f} ha {pct:>5.1f} %")
```

```text
tree      348.0 ha  18.0 %
grass      30.7 ha   1.6 %
crop        0.0 ha   0.0 %
built    1426.5 ha  73.8 %
bare        1.9 ha   0.1 %
water     124.7 ha   6.5 %
```

The built-up area comes out 0.1 ha above the report's 1,426.4 because the pixel area used here is itself rounded; the shares match exactly.

The **single cropland pixel** matters in one place: it counts as a "habitat class present", so the report says 6 classes and divides
the diversity by ln 6 (below). The vector layer has 1,776 polygons covering the 1,931.7 ha.

## Habitat diversity

```{figure} ../_static/img/case-studies/biodiversity-shannon.webp
:alt: Map of local Shannon diversity in central London. Dark purple covers most of the box, with bright yellow-green bands along both banks of the river bend and bright rings and blobs around parks and tree patches; faint square blocks show in the dark areas
:width: 100%
:class: pgv-figure

Local H' (natural-log units). Bright is a window that mixes several classes; dark is a window of one class.
```

The bright band follows both banks of the river, because several classes meet there (water, trees, buildings); the bright rings and blobs are the
edges of parks, where trees or grass meet buildings. The middle of a big park or of a solid built-up block is dark, because a window
inside it sees one class. **Diversity here is a property of edges**, not of what is inside a patch.

| Statistic of local H' | Value |
|---|---:|
| Minimum / 5th percentile | 0.000 / 0.000 |
| 25th percentile | 0.181 |
| Median | 0.451 |
| Mean | 0.446 |
| 75th percentile | 0.676 |
| 95th percentile | 0.954 |
| Maximum | 1.363 |
| Standard deviation | 0.299 |

```{figure} ../_static/img/case-studies/biodiversity-hotspots.webp
:alt: Map of diversity hotspots in orange on a pale grey background: a long curved band along the river bend, rounded patches around parks, and a few blobs near the bottom of the box
:width: 100%
:class: pgv-figure

The hotspots: the top decile of local H'. They trace the river and the park edges.
```

### Two things that are true by construction

**The hotspot share is 10 %, and it could not have been anything else.** The pipeline marks the pixels above the 90th percentile of
local H', so about a tenth of the box is a hotspot in every box on earth. The report's 193.2 ha is 10.0 % of 1,931.7 ha, and the
threshold H' of 0.835 is where the 90th percentile happened to fall *in this box*. Compare hotspots **across places** only by their
threshold and their H' values, never by their share. The report gives the share because it is a convenient way to say how much to
look at.

```python
# check: the hotspot share is the percentile, not a finding
print(round(100 * 193.2 / 1931.7, 1), "% of the area is a hotspot;", round(100 * 19_190 / 191_891, 1), "% of the pixels")
```

```text
10.0 % of the area is a hotspot; 10.0 % of the pixels
```

**Evenness depends on how many classes you count.** The regional index H' = 0.782 over the whole box is computed from the class
proportions above, and evenness is H' divided by its maximum, ln of the number of classes. With the single cropland pixel counted
that is 6 classes, and with it ignored 5:

```python
# check: Shannon index and evenness from the class pixel counts
pixels = {"tree": 34_566, "grass": 3_052, "crop": 1, "built": 141_703, "bare": 184, "water": 12_385}
h, evenness = shannon(list(pixels.values()))
print(f"H' = {h:.3f}, evenness over 6 classes = {evenness:.3f}")
h5, evenness5 = shannon([n for n in pixels.values() if n > 1])
print(f"H' = {h5:.3f}, evenness over 5 classes = {evenness5:.3f}")
```

```text
H' = 0.782, evenness over 6 classes = 0.437
H' = 0.782, evenness over 5 classes = 0.486
```

The index does not change (one pixel adds almost nothing), but the evenness does, from 0.437 to 0.486. The report is right to count what is
present; just do not compare evenness between two boxes whose class lists differ by a single pixel.

### The window leaves marks

The Shannon map shows faint square blocks in the dark areas. They are the window: a 21-pixel window around a one-pixel feature
produces a 210 m square of non-zero H'. It also means the orange hotspot band along the river is **wider than the river**: any pixel
within about 105 m of a boundary between classes sees both. For a finer picture use a smaller window (`--param window_px=11`); for a
district-scale picture, a larger one.

## What the two maps say together

| Observation | From |
|---|---|
| Three quarters of the box is built-up, so regional diversity is low (H' 0.78 against a maximum of 1.79 for six equal classes). | Land cover |
| The most mixed ground is a **band along the river and the margins of the parks**, not the parks' interiors. | Hotspot map |

What it does **not** say is that those places have more species. The report says it plainly: "Heterogeneity is not a species count".
It is a proxy for the variety of habitat types within a short distance, from a map of one year, at a 10 m resolution that does not see
a garden, a street tree or a green roof as distinct from the buildings around it.

## Things to be careful about

- **Land cover is a 2021 product.** Passing a date changes nothing. For a month, use `source=dynamic_world`.
- **Accuracy is about 75 % overall.** In a dense city, where trees overhang buildings, expect confusion between tree cover and built-up,
  and a river's edge can be off by a pixel or two. This is a limit of the map and applies to every area shown here.
- **Do not compare WorldCover 2020 with 2021 for change.** They are different algorithm versions. Use
  [`landcover_change`](../pipelines/catalog/landcover_change.md) for change.
- **A hotspot is relative.** It is relative to the rest of *this box*.
- **The box is small.** 19 km² gives the window plenty of room, but a box of one land cover has no hotspots worth the name.

## Change something

| To | Use | Note |
|---|---|---|
| Another year, month or source | `--param source=dynamic_world --date 2024-06` | Needs an Earth Engine account. |
| Your own classes | `--param source=<model> --checkpoint weights.pt` | Runs your trained network instead of WorldCover. |
| A finer or coarser diversity pattern | `--param window_px=11` (or `41`) | 21 is 210 m. Larger windows give smoother, coarser patterns. |
| A stricter hotspot | `--param hotspot_percentile=95` | The top 5 %; 80 gives the top fifth. |
| Names for classes | `--class-names "Tree,Shrub,Grass,Crop,Built,Bare,Snow,Water,Wetland,Mangrove,Moss"` | As in the [`land_cover` page](../pipelines/catalog/land_cover.md). |

## The files they wrote

| File | What it holds |
|---|---|
| `out/lc/land_cover.tif` | The class raster (255 is no data) |
| `out/lc/deliverables/vectors/land_cover.gpkg` | Layer `land_cover`: 1,776 polygons, 1,931.7 ha, with `class_id`, `class_name`, `area_ha` and more |
| `out/bio/shannon_diversity.tif` | Local H' (a continuous raster; NaN is no data) |
| `out/bio/hotspot_mask.tif` | 1 for a hotspot, 0 for the rest |
| `out/bio/land_cover/land_cover.tif` | The WorldCover raster the diversity was computed from |
| `out/bio/deliverables/vectors/biodiversity_hotspot.gpkg` | Layers `diversity_hotspots` (53 polygons, 193.2 ha) and `land_cover` |
| `deliverables/*_report.pdf`, `.xlsx`, `.docx`, `maps/*.png` | The reports and the maps in both folders |
| `run_summary.json` | Every figure and parameter |

## Related

- [`land_cover`](../pipelines/catalog/land_cover.md) and [`biodiversity_hotspot`](../pipelines/catalog/biodiversity_hotspot.md): every
  argument and failure message.
- [Buildings and roads in the same box](london-buildings-roads.md): the built-up 73.8 % in more detail.
- [Vegetation change in the same box](london-vegetation.md): what the 18 % tree cover did between two seasons.
- [Reading results](../pipelines/results.md).
