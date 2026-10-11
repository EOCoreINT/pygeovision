# Urban

Urban work mixes three kinds of evidence, and the common mistake is to treat them as one: **published vector data** (buildings,
roads), **annual 10 m land-cover maps** (growth), and **thermal imagery** (heat). Each has a different resolution, date and failure
mode. This page shows the workflow for a city-planning question and says where each result stops being trustworthy.

```{admonition} Resolution decides what you can claim
:class: warning
Sentinel-2 pixels are 10 m. A detached house is smaller than two pixels, so **no pipeline here detects individual houses from Sentinel-2**.
Building counts come from published footprint datasets (Microsoft, Google, OpenStreetMap), not from your imagery. Land-cover
"built area" is a neighbourhood-scale class. Landsat thermal is 30 m resampled from 100 m.
```

## 1. What is built, and how much?

```bash
pygeovision channel building_footprints \
    --bbox -0.25 5.58 -0.15 5.66 \
    --param source=microsoft --param cell_m=100 \
    --output ./accra_buildings
```

**Expect.** `building_footprints.gpkg` (every polygon with `area_m2`), `building_density.tif` (share of each 100 m cell covered)
and `building_count.tif`; statistics `building_count`, `building_area_ha`, `building_coverage`, `buildings_per_km2`,
`max_cell_coverage`, plus a `source_note` that states the dataset's vintage and known error.

**Compare sources before you quote a count.** Run the same box with `source=google` and `source=osm`:

| Source | Strength | Weakness |
|---|---|---|
| `microsoft` | Machine-learning detections, wide coverage | Dates of imagery vary; about 1 % false positives reported by the project |
| `google` | Open Buildings v3, has a confidence you can filter (`min_confidence`) | Coverage limited to its release regions |
| `osm` | Includes attributes where mapped | Only as complete as volunteers made it; a 0 may mean "unmapped" |

If counts differ by more than the datasets' own stated error, trust none of them as a census and use the spread as your range.

## 2. How much road, of which kind?

```bash
pygeovision channel road_network \
    --bbox -1.70 6.60 -1.55 6.75 \
    --param cell_m=500 \
    --output ./kumasi_roads_osm
```

**Expect.** `road_network.gpkg` (every road with its class), `road_density.tif` (km per km²), and `km_by_group` / `km_by_highway`
tables. **Completeness is the limit:** minor roads and tracks vary by place, so densities are comparable between mapped areas, not
against unmapped ones. Detecting roads from imagery needs a trained network (`road_extraction`\*, refuses to run without `--checkpoint`).

## 3. How fast is the city growing?

```bash
pygeovision channel urban_expansion \
    --bbox -0.35 5.55 -0.10 5.75 \
    --date-before 2018 --date-after 2023 \
    --output ./accra_growth
```

**Expect.** `built_area_change.tif` (0 not built, 1 built both years, 2 new built area, 3 built area lost) and
`gain_ha`, `loss_ha`, `net_change_ha`, `built_share_before/after`, `annual_growth_rate`.

**Check it before you publish.**

- **Loss should be small.** A city rarely un-builds. A large `loss_ha` signals classifier disagreement between years, not demolition.
- **Compare gain with the footprint layer.** New built area should sit where building density is high; a gain far from any footprint
  is either a quarry/bare ground classed as built, or a gap in the footprint data.
- **Years 2017-2023 only.** For other years, or classes of your own, `urban_growth`\* uses a Landsat network you train.

Cut the growth to districts with `pygeovision postprocess class-stats built_area_change.tif` or `zonal-stats` on a district layer.

## 4. Where is it hottest?

```bash
pygeovision channel urban_heat_island \
    --bbox 16.25 48.15 16.50 48.30 \
    --date 2024-07 \
    --param threshold_c=3 \
    --output ./vienna_uhi
```

**Expect.** `lst_celsius.tif` (land surface temperature, °C), `heat_island_mask.tif` (1 = more than `threshold_c` above the **scene
mean**), and `mean/min/max_temp_celsius`, `pct_heat_island`.

```{admonition} This is surface temperature on one clear morning
:class: caution
It is **not** air temperature, not a heat-stress index, and not "urban minus rural": the hot mask is relative to the whole scene.
Landsat passes in the late morning, so night-time heat, which matters most for health, is not seen. For an urban–rural difference, draw a rural polygon
and compare the two means with `postprocess zonal-stats`.
```

`land_surface_temperature` is the same product without the city framing.

## 5. Parcels and rooftops with your own imagery

When you have high-resolution imagery, the right tool is [object-based analysis](../obia/index.md): segment the image into
objects, compute features (spectra, shape, texture), then classify with rules or a small training set. The OBIA pages include a
[tutorial](../obia/tutorial.md) and object-level [accuracy](../obia/accuracy.md) measures. After classification,
[`postprocess regularise`](../reference/cli/postprocess.md) squares building outlines and `vectorise` writes polygons.

## Putting it together

| Question | Order of work |
|---|---|
| "What has the city added since 2018?" | `urban_expansion` → compare with `building_footprints` → `class-stats` per district → report |
| "Where should we plant trees?" | `urban_heat_island` (hot) + `building_footprints` (density) + `road_network` (density) → overlay in your GIS; name the dates |
| "Is the data good enough?" | Compare footprint sources; check `loss_ha`; check `valid_fraction` of the thermal run |

\* needs a network **you** trained.
