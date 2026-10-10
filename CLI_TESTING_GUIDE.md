# PyGeoVision CLI testing guide

A runnable example for **every** command-line component — all 91 CLI commands, including the 25 end-to-end
pipelines that run through `pygeovision channel` — with what each should print or write, so you can test the
whole tool from the shell.

## How to read this guide

Every code block starts with a tag that says what the commands need:

| Tag | Meaning |
|---|---|
| `# [offline]` | Runs on your machine with **no network, no credentials, no GPU, no downloaded weights**, using the synthetic data from section 1. These blocks were executed in order, from a clean directory, to produce this guide (see [How this was verified](#how-this-was-verified)). |
| `# [network]` | Needs internet access (satellite providers, OpenStreetMap, Hugging Face, ...). Not executed for this guide. |
| `# [credentials]` | Needs an account/API key (for example OpenTopography, Copernicus). |
| `# [weights]` | Needs weights you train yourself, or pretrained weights downloaded from the internet. |
| `# [cloud]` | Needs a cloud account and its SDK. |
| `# [manifest]` | Needs `MANIFEST.sha256` from the delivery zip (only for checking that your files match it). |

Two rules that matter when you read results:

* **Exit codes are meaningful.** Every command exits `0` on success and non-zero on failure (usage errors
  exit `2`). The examples use that: a command that is *supposed* to refuse is wrapped in `expect_fail`.
* **Wiring checkpoints prove plumbing, not accuracy.** PyGeoVision ships network *architectures*, not trained
  weights. Everything that runs a network needs `--checkpoint your_model.pth`. Where this guide has no trained
  model, it uses a *wiring checkpoint* (random weights of the right shape) — enough to prove the command loads,
  checks, tiles and writes correctly. **Its predictions are random and mean nothing.**

---

## 1. Setup

```bash
# [offline]
# From the repository root. Python 3.10+.
pip install -e ".[ai,obia,geo,edge,dev]"      # everything the offline examples use
pip install ultralytics                         # only for `ai detect generic|ships|cars` (section 12)
pygeovision --version
```

Create the synthetic dataset (a 256 × 256 px, 10 m, 6-band scene with five land-cover classes at known
positions, a second date with a known 60 × 60 px change, a time series, labels, chips):

```bash
# [offline]
python tools/make_test_data.py pgv_test
```

Define the shortcuts and helpers used by the rest of the guide (re-run this block in each new shell):

```bash
# [offline]
export D=pgv_test O=out
mkdir -p $O

# Use a scratch home so the credential, cache and schedule examples never touch your real ~/.pygeofetch,
# ~/.pygeovision or model caches (delete the folder afterwards).
export HOME=$PWD/pgv_home; mkdir -p $HOME

# expect_fail CMD...  -> passes only if CMD fails (exit != 0) and shows why; used for commands that must refuse.
expect_fail() { local out rc=0; out=$("$@" 2>&1) || rc=$?; if [ $rc -eq 0 ]; then echo "UNEXPECTED SUCCESS: $*"; return 1; fi; echo "refused as expected (exit $rc): $(printf '%s' "$out" | grep -v ' INFO ' | tail -n 1 | cut -c1-120)"; }

# A wiring checkpoint for the 6-band scene (5 classes). Random weights: plumbing tests only.
python tools/make_wiring_checkpoint.py unet_resnet50 --classes 5 --bands 6 -o $O/wiring.pth
```

| File in `pgv_test/` | What it is |
|---|---|
| `scene_t1.tif`, `scene_t2.tif` | 6 bands (blue, green, red, NIR, SWIR1, SWIR2), reflectance 0–1, UTM 33N. `t2` has a forest→bare change at rows 40–99, cols 120–179 |
| `scene_rgb.tif`, `scene_rgb.png` | 3-band 8-bit view of `t1` |
| `scene_bands/B02…B12.tif`, `SCL.tif` | the six bands as separate Sentinel-2-style files, plus a scene classification layer with a cloud patch |
| `labels.tif` | ground-truth class map (0 water, 1 forest, 2 crop, 3 urban, 4 bare; nodata 255) |
| `reference.tif` | truth with ~8 % of pixels flipped, so accuracy comes out below 100 % |
| `classified.tif` | a "prediction" identical to the truth |
| `series/s2_<date>.tif` | five dates with a seasonal NDVI cycle |
| `ref_chips/`, `cur_chips/` | 128 px chips; the current ones are brightened so drift detection fires |

---

## 2. System checks

```bash
# [offline]
pygeovision --help                 # lists every group below
pygeovision status                 # versions, device, provider counts
pygeovision status --json          # same, machine-readable
pygeovision doctor                 # one ✓/✗ per component; exits 1 if any check fails
```

Expect: `doctor` prints `7/7 components healthy` and exits `0`; `status` shows `22 providers | 10 open-access`.

If `doctor` shows `✗ credentials` / `✗ cache` with no reason, says `5/5 components healthy`, or `status` shows
`0 open-access`, some files in your tree are older than the rest (a partly copied update behaves like neither
version). Check with the manifest that ships with the delivery:

```bash
# [manifest]   (needs MANIFEST.sha256 from the delivery zip in the repo root)
python tools/verify_install.py        # lists every file that is missing or differs; exit 1 if any
```

---

## 3. The 25 pipelines (`pygeovision channel`)

`channel <name>` downloads imagery with pygeofetch and then computes an index, a terrain product, a reference
map, or runs *your* network. `channel list` shows each one and whether it runs out of the box (`ready`) or
needs your trained weights (`needs_weights`).

```bash
# [offline]
pygeovision channel list                                       # 25 pipelines (19 ready, 6 need your weights)
expect_fail pygeovision channel not_a_pipeline --bbox 0 0 1 1  # unknown name: exit 2 with the reason
expect_fail pygeovision channel building_footprints --bbox -0.15 51.47 -0.10 51.52 --date 2024-06
                                                               # needs_weights without --checkpoint: refuses
```

### 3.1 Test all 25 at once, offline

`tools/offline_pipeline_smoke.py` runs every pipeline through the **real CLI** with only the three network
calls replaced (scene download, OpenTopography DEM, WorldCover tile) by synthetic data. Parsing, validation,
band maths, nodata handling, the strict checkpoint load, file writing and statistics are the real code.

```bash
# [offline]
python tools/offline_pipeline_smoke.py                       # all 25; exit 0 only if all succeed
python tools/offline_pipeline_smoke.py water_bodies flood_mapping   # just some
python tools/offline_pipeline_smoke.py --commands            # print the real-data command for each
```

Expect: 25 lines starting `OK` (each also states how many reports, maps and vector files it produced), then
`25/25 pipelines succeeded`. The smoke run fails a pipeline whose run did not leave a map, a GeoPackage and the PDF,
Word and Excel reports.

### 3.1.1 What every run produces: map, vectors and a report

Unless you pass `--no-report`, each `channel` run (and `client.pipeline(...)` in Python) writes, next to the
GeoTIFFs:

```
<out>/run_summary.json                    machine-readable stats + the list of deliverables
<out>/deliverables/<pipeline>_report.pdf  presentation-ready report (cover, key figures, map, tables, method, limits, references)
<out>/deliverables/<pipeline>_report.docx the same content as an editable Word file
<out>/deliverables/<pipeline>_report.xlsx workbook: Summary, Metrics, Classes (live formulas + chart), Statistics, Inputs, Vectors, Files, Parameters
<out>/deliverables/maps/*.png             map with scale bar, north arrow, legend with areas, key figures; charts
<out>/deliverables/vectors/<pipeline>.gpkg        GeoPackage: aoi, <layer>, <layer>_classes
<out>/deliverables/vectors/geojson/*.geojson      one file per layer (EPSG:4326)
<out>/deliverables/vectors/shapefile|kml/         only if requested
```

Options: `--no-report` · `--report-formats pdf,xlsx,docx` · `--vector-formats gpkg,geojson[,shapefile,kml]` ·
`--class-names "1=Water,2=Crops"` (for the classes of your own network) · `--report-title TEXT`. The environment
variable `PYGEOVISION_REPORT=0` switches it off globally. Needs `pip install "pygeovision[report]"`; without it
the analysis still runs and says what to install. A failing report writer never fails the analysis: it is
logged and listed under `warnings` in `run_summary.json`.

`pygeovision report OUTPUT_DIR` rebuilds the deliverables of an earlier run (other formats, other title, class
names, no maps or no vectors) from the saved rasters, without re-downloading anything:

```bash
# [offline]
python tools/offline_pipeline_smoke.py flood_mapping land_cover --keep out/smoke
pygeovision report out/smoke/out/flood_mapping --formats pdf,docx --vector-formats gpkg,shapefile --title "Flood review"
ls out/smoke/out/flood_mapping/deliverables out/smoke/out/flood_mapping/deliverables/vectors/shapefile
pygeovision report out/smoke/out/land_cover --class-names "10=Trees,30=Grass" --no-vectors
expect_fail pygeovision report out/smoke                       # not a pipeline output folder: exit non-zero with the reason
```

Check: the PDF opens with a cover, key figures and an executive summary; areas in the report equal the
`*_area_ha` numbers printed by the run; the GeoPackage opens in QGIS with the classes coloured by `color_hex`; the
Excel `Classes` sheet shows shares that add to 100 % after Excel calculates the workbook (formulas are computed on
open). The maps use no web basemap; vectors are polygonised from the output raster (very large rasters are
generalised or skipped, and the report says so).

### 3.2 The real-data commands

These need the network (and for three of them a key). Replace the bounding box
(`MIN_LON MIN_LAT MAX_LON MAX_LAT`) and dates with your own; `--date` is `YYYY-MM`. After a run check that
the output files exist, the printed `valid_fraction` is high (clouds/nodata are excluded, not counted), and
open the GeoTIFF in QGIS next to the imagery.

What a real run looks like (checked against logs of live Sentinel-2 runs):

* Only the bands the pipeline needs, plus the cloud-mask layer (`SCL`), are downloaded: tens to a few hundred MB
  per scene. If you see `Downloading 15 asset(s)` for a Sentinel-2 scene, the tree is out of date.
* A box that crosses a tile boundary downloads one scene per tile and mosaics them. The result reports
  `aoi_coverage_fraction`: `1.0` means the downloaded scenes cover the whole box. Lower means the output is
  smaller than you asked for (the log also warns `covers only NN% of the requested bbox`); all other statistics
  then describe only that part. Start with a box under 0.1° on a side; a 1° x 1° box is 4+ tiles and 1-3 GB.
* `--date 2024-06` searches the whole month (`2024-06-01` to `2024-06-30`).
* Searches skip providers that cannot answer (no stored credentials, or `copernicus` for STAC collection names)
  instead of failing on every request; name one with `--provider` to force it.
* `buildings.pth`, `forest.pth` and similar names in the `needs_weights` rows are placeholders for **your own**
  trained file. A name that does not exist stops at once with `checkpoint not found` before anything is downloaded.
* Sentinel-2 scenes from 2022-01-25 on are stored with a +1000 DN offset; it is removed before scaling (the file
  tag `boa_add_offset_dn` records it). Results from versions before this one were biased for all indices.

Common options: `-o DIR` output directory · `--date` / `--date-before` / `--date-after` / `--dates a,b,c` ·
`--checkpoint FILE` · `--param KEY=VALUE` (repeatable, JSON-parsed) · `--provider NAME` (repeatable) ·
`--source` · `--model`. See `pygeovision channel --help` and `docs/pipelines/`.

#### Ready — no weights needed

| Pipeline | Command | Writes | Check |
|---|---|---|---|
| `water_bodies` | `pygeovision channel water_bodies --bbox -0.15 51.47 -0.10 51.52 --date 2024-06 -o out/wb` | `water_mask.tif`, `ndwi.tif` | `water_coverage` plausible for the area (the Thames/reservoirs for this box) |
| `flood_mapping` | `pygeovision channel flood_mapping --bbox 67.5 25.5 68.5 26.5 --date-before 2022-05 --date-after 2022-09 -o out/flood` | `flood_extent.tif` (0 dry, 1 flooded, 2 permanent, 255 nodata) | `flooded_area_ha` > 0 and `permanent_water_area_ha` ≈ the river; `date-before` is the *dry* reference |
| `wetland_mapping` | `pygeovision channel wetland_mapping --bbox -81.0 25.5 -80.5 26.0 --date 2024-06 -o out/wet` | `wetland_classification.tif` | `pct_open_water`, `pct_wet_vegetation` |
| `coastal_monitoring` | `pygeovision channel coastal_monitoring --bbox 4.0 52.0 4.5 52.4 --date-before 2020-01 --date-after 2024-01 -o out/coast` | `shoreline_change.tif` | `land_to_water_ha`, `water_to_land_ha`, `net_water_change_ha` |
| `water_quality` | `pygeovision channel water_quality --bbox -76.6 38.8 -76.2 39.2 --date 2024-06 -o out/wq` | `water_quality.tif` | relative NDCI / red-reflectance inside water only (`note` says "calibrate against water samples") |
| `snow_cover` | `pygeovision channel snow_cover --bbox 7.8 46.4 8.2 46.7 --date 2024-01 -o out/snow` | `snow_extent.tif` | `snow_cover_fraction` (NDSI > 0.4) |
| `glacier_monitoring` | `pygeovision channel glacier_monitoring --bbox 7.9 46.4 8.2 46.6 --date-before 2020-08 --date-after 2024-08 -o out/glacier` | `glacier_extent_after.tif` | `area_change_pct` negative for a retreating glacier |
| `vegetation_indices` | `pygeovision channel vegetation_indices --bbox -0.15 51.47 -0.10 51.52 --dates 2024-04,2024-06,2024-08 -o out/veg` | one folder per date with NDVI/EVI/NDWI | `n_dates_used` = number of dates you gave |
| `crop_health` | `pygeovision channel crop_health --bbox -0.15 51.47 -0.10 51.52 --date-before 2024-05 --date-after 2024-08 --param anomaly_threshold=0.2 -o out/crop` | `ndvi_change.tif`, `crop_health_anomaly.tif` | `pct_flagged`; compare the same season |
| `wildfire_severity` | `pygeovision channel wildfire_severity --bbox -122.6 39.7 -121.9 40.2 --date-before 2018-10 --date-after 2018-12 -o out/fire` | `dnbr.tif`, `severity_class.tif` | `burned_area_ha`, `pct_high_severity` (the 2018 Camp Fire area) |
| `change_detection` | `pygeovision channel change_detection --bbox -0.15 51.47 -0.10 51.52 --date-before 2022-06 --date-after 2024-06 -o out/chg` | `change_mask.tif`, `change_magnitude.tif` | `otsu_threshold`, `changed_area_ha`; it flags *any* spectral change (also seasonal) |
| `land_cover` | `pygeovision channel land_cover --bbox -0.15 51.47 -0.10 51.52 --date 2023-06 -o out/lc` | `land_cover.tif` | ESA WorldCover classes (10 tree, 30 grass, 40 crop, 50 built, 80 water, …) |
| `biodiversity_hotspot` | `pygeovision channel biodiversity_hotspot --bbox -0.15 51.47 -0.10 51.52 -o out/bio` | `shannon_diversity.tif`, `hotspot_mask.tif` | habitat-heterogeneity *proxy*, not a species count (the `note` says so) |
| `land_surface_temperature` ¹ | `pygeovision channel land_surface_temperature --bbox -0.15 51.47 -0.10 51.52 --date 2024-07 -o out/lst` | `lst_celsius.tif`, `above_threshold_mask.tif` | `mean_temp_celsius` in a believable range for the date |
| `urban_heat_island` ¹ | `pygeovision channel urban_heat_island --bbox -0.15 51.47 -0.10 51.52 --date 2024-07 -o out/uhi` | `lst_celsius.tif`, `heat_island_mask.tif` | `pct_heat_island` (hotter than scene mean + threshold) |
| `volcano_monitoring` ¹ | `pygeovision channel volcano_monitoring --bbox 15.0 37.6 15.2 37.8 --date 2024-07 -o out/volc` | `lst_celsius.tif`, `thermal_anomaly_mask.tif` | thermal-anomaly *candidates*, not confirmed activity |
| `archaeological_site` ² | `pygeovision channel archaeological_site --bbox -1.9 51.1 -1.7 51.2 -o out/arch` | `local_relief_model.tif`, `multidirectional_hillshade.tif` | visualisation layers for expert interpretation |
| `solar_potential` ² | `pygeovision channel solar_potential --bbox 11.3 47.2 11.5 47.3 --date 2024-06-21 --param hour=12 -o out/solar` | `solar_potential.tif` | `mean_relative_irradiance` 0–1; clear-sky, no shadows from terrain |
| `wind_farm_siting` ² | `pygeovision channel wind_farm_siting --bbox 8.0 54.0 8.6 54.4 -o out/wind` | `wind_suitability.tif`, `landcover_for_exclusion.tif` | screening only: no wind speed or yield |

¹ Landsat thermal: needs Landsat Collection 2 Level-2 scenes (Sentinel-2 has no thermal band).
² DEM pipelines need a free OpenTopography key: `pygeovision data auth add opentopography --api-key KEY`.

#### Need your trained weights (`--checkpoint`)

Each of these refuses to run without `--checkpoint` (it never returns an untrained network's output). The
checkpoint must match the pipeline's architecture, class count and band count exactly — a mismatch fails with
a message naming the mismatching tensor. Train it with `pygeovision.ai.training.trainer.GeoTrainer`.

| Pipeline | Command | Model it expects (default) |
|---|---|---|
| `building_footprints` | `pygeovision channel building_footprints --bbox -0.15 51.47 -0.10 51.52 --date 2024-06 --checkpoint buildings.pth -o out/bld` | `unet_resnet50`, 2 classes, RGB |
| `solar_detection` | `pygeovision channel solar_detection --bbox -0.15 51.47 -0.10 51.52 --date 2024-06 --checkpoint solar.pth -o out/solar_det` | `unet_efficientnet_b4`, 2 classes, RGB |
| `crop_monitoring` | `pygeovision channel crop_monitoring --bbox -0.15 51.47 -0.10 51.52 --date 2023-06 --param num_classes=3 --checkpoint crops.pth -o out/crops` | `segformer_b2`, your class count |
| `deforestation` | `pygeovision channel deforestation --bbox -62.5 -9.0 -62.0 -8.5 --date-before 2023-06 --date-after 2024-06 --checkpoint forest.pth -o out/defor` | `changeformer`, 3 classes |
| `urban_growth` | `pygeovision channel urban_growth --bbox -0.15 51.47 -0.10 51.52 --date-before 2015-06 --date-after 2024-06 --checkpoint urban.pth -o out/urban` | `siamese_unet`, 2 classes |
| `disaster_assessment` | `pygeovision channel disaster_assessment --bbox -95.5 29.5 -95.0 30.0 --date-before 2017-08 --date-after 2017-09 --param disaster_type=flood --checkpoint damage.pth -o out/disaster` | `siamese_unet`, 4 classes |

To prove the wiring of one of them before you have a trained model (output is meaningless), see
`tools/offline_pipeline_smoke.py` — it does exactly this for all six — or:

```bash
# [offline]
python tools/make_wiring_checkpoint.py unet_resnet50 --classes 2 --bands 3 -o $O/wiring_rgb.pth
python tools/make_wiring_checkpoint.py unet_resnet50 --classes 4 --bands 3 -o $O/wiring_bad.pth
```

---

## 4. Data acquisition (`pygeovision data`)

```bash
# [offline]
pygeovision data providers                       # all 22 providers
pygeovision data providers --open-only           # 10 that need no credentials
pygeovision data providers --sar                 # SAR-capable providers
pygeovision data providers --sub-meter           # sub-metre providers
pygeovision data auth list                       # stored credentials ("No credentials stored." on a fresh machine)
pygeovision data cache stats                     # entries / size / location of the search cache
pygeovision data cache clear --dry-run           # says what it would clear, clears nothing
```

Credentials (stored in `~/.pygeofetch`; use throw-away values to test the commands):

```bash
# [offline]
pygeovision data auth add usgs --username demo --password demo     # stores locally; nothing is sent
pygeovision data auth list
pygeovision data auth remove usgs --yes
expect_fail pygeovision data auth test usgs      # no stored credentials -> exit 1
```

Live checks:

```bash
# [network]
pygeovision data auth test planetary_computer    # open provider: HTTP GET on its endpoint; exit 0 iff it answers
pygeovision data search --bbox -0.15 51.47 -0.10 51.52 --start-date 2024-06-01 --end-date 2024-06-30 \
    --providers planetary_computer --satellite sentinel-2 --cloud-max 20 --max-results 5
pygeovision data search --bbox -0.15 51.47 -0.10 51.52 --date 2024-06 --providers planetary_computer \
    --format json --output results.json
pygeovision data search --bbox -0.15 51.47 -0.10 51.52 --date 2024-06 --providers planetary_computer --format ids
pygeovision data download --bbox -0.15 51.47 -0.10 51.52 --date 2024-06 --providers planetary_computer \
    --max-items 1 --output ./data/ --post-process cog
pygeovision data download --from-search results.json --output ./data/ --parallel 4 --verify-checksum --resume
```

Expect: `data search` prints a table of scenes (id, date, cloud %, provider). **If the provider cannot be
reached it prints `No results found.` and exits `0`** — the real cause is in the warning lines above it, so
read them before concluding the area has no imagery.

### YAML data pipelines (pygeofetch dialect: search / filter / download / process / export)

```bash
# [offline]
cat > $O/data_pipeline.yaml <<'EOF'
name: demo-search
description: search then filter
schedule: "0 6 * * 1"
steps:
  - search:
      providers: [planetary_computer]
      bbox: "-0.15,51.47,-0.10,51.52"
      date_range: "2024-06-01/2024-06-30"
  - filter:
      expression: "data.cloud_cover < 20"
EOF
printf 'name: broken\nsteps:\n  - search\n' > $O/data_pipeline_bad.yaml

pygeovision data pipeline validate $O/data_pipeline.yaml         # ✓ Pipeline valid
expect_fail pygeovision data pipeline validate $O/data_pipeline_bad.yaml   # a step must be a one-key mapping
pygeovision data pipeline schedule $O/data_pipeline.yaml --name nightly --cron "0 2 * * *"
pygeovision data pipeline list                                    # nightly: 0 2 * * *  [file]
expect_fail pygeovision data pipeline run $O/data_pipeline.yaml --step no_such_step
```

```bash
# [network]
pygeovision data pipeline run $O/data_pipeline.yaml               # runs search -> filter
pygeovision data pipeline run $O/data_pipeline.yaml --step search # only the search step
pygeovision data pipeline history --limit 5                       # ✓/✗, start time, duration (a local run log)
```

Notes: `schedule` records *what should run when* in pygeofetch's registry; something still has to run the
scheduler loop (cron, `pygeofetch`) for it to fire. A step that finds nothing (including a provider that
could not be reached) is **not** a pipeline failure in pygeofetch's dialect, so `run` can report success with
0 scenes — check the step output.

---

## 5. Preprocessing (`pygeovision preprocess`)

Every command writes a new file and accepts `--validate` to check the result afterwards.

```bash
# [offline]
# stack per-band files into one GeoTIFF (band order = the order given)
pygeovision preprocess stack $D/scene_bands --bands B02,B03,B04,B08,B11,B12 --output $O/stack.tif --validate

# clip to a WGS84 box: ONE comma-separated string minlon,minlat,maxlon,maxlat (inside the scene!)
pygeovision preprocess clip $D/scene_t1.tif --bbox 15.005,45.135,15.025,45.148 --output $O/clip.tif

# normalise: minmax | zscore | percentile | scale_factor
pygeovision preprocess normalise $D/scene_t1.tif --method minmax --output $O/norm_minmax.tif
pygeovision preprocess normalise $D/scene_t1.tif --method zscore --output $O/norm_z.tif
pygeovision preprocess normalise $D/scene_t1.tif --method percentile --percentile 2 --output $O/norm_pct.tif
pygeovision preprocess normalise $D/scene_t1.tif --method scale_factor --scale-factor 10000 --output $O/norm_sf.tif

# resample to 20 m (nearest | bilinear | cubic | lanczos | average)
pygeovision preprocess resample $D/scene_t1.tif --resolution 20 --method average --output $O/res20.tif

# the whole chain in one go: stack -> cloud-mask with SCL -> clip -> normalise -> resample
pygeovision preprocess pipeline $D/scene_bands --output $O/ready.tif \
    --bands B02,B03,B04,B08,B11,B12 --scl $D/scene_bands/SCL.tif \
    --bbox 15.0,45.13,15.05,45.155 --normalise scale_factor --resample-m 20 --validate
```

Expect: `res20.tif` is 128 × 128 (256 px at 10 m → 20 m); `clip.tif` is smaller than the scene. Check with
`rio info $O/res20.tif` (or `gdalinfo`). **`preprocess clip --bbox` takes one comma string, whereas
`channel` / `data search --bbox` take four space-separated numbers.**

---

## 6. Validation (`pygeovision validate`)

```bash
# [offline]
pygeovision validate run $D/scene_t1.tif                                          # nulls, dtype, CRS, bounds, outliers
pygeovision validate run $D/scene_t1.tif --required-bands 6 --value-range 0,1 --target-crs EPSG:32633
pygeovision validate run $D/scene_t1.tif --report $O/validation.html              # HTML report
pygeovision validate run $D/scene_t1.tif --report $O/validation.json --report-fmt json --mode warn
pygeovision validate for-inference $D/scene_t1.tif --model-type segmentation
pygeovision validate for-inference $D/scene_t1.tif --model-type change_detection
```

---

## 7. Spectral indices (`pygeovision indices`)

Band numbers are 1-based and default to blue 1, green 2, red 3, NIR 4, SWIR1 5, SWIR2 6 — which is how the
synthetic scene is laid out.

```bash
# [offline]
pygeovision indices list                                         # the 21 indices and transforms
pygeovision indices compute $D/scene_t1.tif ndvi --output $O/ndvi.tif
pygeovision indices compute $D/scene_t1.tif mndwi --output $O/mndwi.tif --green 2 --swir1 5
pygeovision indices compute $D/scene_t1.tif nbr --output $O/nbr.tif --nir 4 --swir2 6 --validate
pygeovision indices all $D/scene_t1.tif --output-dir $O/indices --indices ndvi,evi,nbr,ndwi,ndbi
```

Check: NDVI over the forest block (rows 20–119, cols 100–199) is high (≈ 0.8) and over the water strip
(cols 0–59) is below 0; MNDWI over water is clearly positive.

```bash
# [offline]
python - <<'EOF'
import rasterio
v = rasterio.open("out/ndvi.tif").read(1)
print("NDVI forest  :", round(float(v[30:110, 110:190].mean()), 2))
print("NDVI water   :", round(float(v[:, :50].mean()), 2))
assert v[30:110, 110:190].mean() > 0.6 and v[:, :50].mean() < 0
EOF
```

---

## 8. Post-processing (`pygeovision postprocess`)

```bash
# [offline]
pygeovision postprocess class-stats $D/classified.tif                      # per-class pixel count and area
pygeovision postprocess accuracy $D/classified.tif $D/reference.tif \
    --class-names Water,Forest,Crop,Urban,Bare                             # overall accuracy 0.938, kappa 0.912
pygeovision postprocess sieve $D/classified.tif --min-pixels 25 --output $O/sieved.tif
pygeovision postprocess cog $D/classified.tif --output $O/classified_cog.tif
pygeovision postprocess vectorise $D/classified.tif --output $O/forest.geojson --target-class 1 --min-area 100 --simplify 5
pygeovision postprocess vectorise $D/classified.tif --output $O/classes.geojson --validate
pygeovision postprocess smooth $O/forest.geojson --output $O/forest_smooth.geojson --tolerance 5
pygeovision postprocess regularise $O/forest.geojson --output $O/forest_rect.geojson --min-area 100
pygeovision postprocess export $O/forest.geojson --output $O/forest.gpkg --format gpkg
pygeovision postprocess export $O/forest.geojson --output $O/forest.kml --format kml
pygeovision postprocess zonal-stats $O/ndvi.tif $O/classes.geojson --output $O/zonal.geojson --stats mean,std,min,max,count
```

Expect: `accuracy` reports overall accuracy 0.938 and Cohen's κ 0.912 (below 1.0 because `reference.tif` has
~8 % flipped pixels; against `labels.tif` it is exactly 1.0) with a per-class precision/recall/F1/IoU table; `vectorise --target-class 1` yields one large forest polygon; the exported files
open in QGIS.

---

## 9. Object-based image analysis (`pygeovision obia`)

```bash
# [offline]
pygeovision obia demo $O/obia_demo --size 160 --seed 0      # writes scene.tif, reference.gpkg, reference_ids.tif

# segment (multires | slic | felzenszwalb | quickshift | watershed | chessboard)
pygeovision obia segment $O/obia_demo/scene.tif --method multires --scale 20 \
    --output-labels $O/seg.tif --output-vector $O/seg.gpkg
pygeovision obia segment $O/obia_demo/scene.tif --method slic --param n_segments=150 --output-labels $O/slic.tif

# scale selection diagnostics, and segmentation quality against reference objects
pygeovision obia scale $O/obia_demo/scene.tif --scales 10,20,40 --reference $O/obia_demo/reference.gpkg --output $O/scale.csv
pygeovision obia assess-segmentation $O/seg.tif $O/obia_demo/reference.gpkg --output $O/segassess.csv

# object features (spectral, shape, texture, context)
pygeovision obia features $O/obia_demo/scene.tif $O/seg.tif --output $O/features.csv --texture nir
pygeovision obia features $O/obia_demo/scene.tif $O/seg.tif --output $O/features.gpkg   # keeps geometry

# classify: supervised from sample polygons ...
pygeovision obia classify $O/obia_demo/scene.tif $O/seg.tif --samples $O/obia_demo/reference.gpkg \
    --class-column class --method rf --output $O/classified.gpkg --output-raster $O/classified_obia.tif
# ... or rule-based from YAML
cat > $O/rules.yaml <<'EOF'
rules:
  - {class: water,      when: "mean_ndwi > 0.3"}
  - {class: vegetation, when: "mean_ndvi > 0.4"}
  - {class: other,      when: "mean_ndvi <= 0.4"}
EOF
pygeovision obia classify $O/obia_demo/scene.tif $O/seg.tif --rules $O/rules.yaml --output $O/rule_classified.gpkg

# accuracy of a classification against reference objects (needs the classified VECTOR; CSV output)
pygeovision obia assess-classification $O/seg.tif $O/classified.gpkg $O/obia_demo/reference.gpkg \
    --class-column class --weight-by-area --output $O/confusion.csv
```

Expect: `assess-classification` prints overall accuracy / kappa. **The demo trains and evaluates on the same
objects, so a near-perfect score here proves the pipeline runs, not that the classifier generalises** — for
a real accuracy figure hold out reference objects the classifier never saw.

---

## 10. Time series (`pygeovision timeseries`)

```bash
# [offline]
pygeovision timeseries analyze $D/series/*.tif --index ndvi --sensor sentinel2 --output $O/ndvi_series.json
pygeovision timeseries anomaly $D/series/*.tif --threshold 1.2
```

Expect: `analyze` prints five mean-NDVI values (0.24, 0.34, 0.40, 0.38, 0.12 — a seasonal curve peaking in
June), `Trend: decreasing`, and writes the JSON; `anomaly --threshold 1.2` reports exactly one anomaly,
`[4] value=0.1218 z=-1.68 (low)` — the 2024-10-15 scene. (With only five dates a z-score above the default
2.5 is almost unreachable, hence the lower threshold here.)

---

## 11. Labelling (`pygeovision label`, `ai chips`)

```bash
# [offline]
pygeovision label quality $D/labels.tif                 # grade, class balance, coverage, slivers, connectivity
pygeovision label quality $D/labels.tif --html          # also writes pgv_test/labels_quality.html (next to the input)
pygeovision ai chips --image $D/scene_t1.tif --label $D/labels.tif -o $O/chips --chip-size 64 --overlap 8
ls $O/chips | head
```

Expect: `Quality Grade: A (97%)` for the clean synthetic labels; `ai chips` prints `✓ 16 chips exported` and
writes aligned pairs into `out/chips/images/` and `out/chips/labels/`.

```bash
# [network]
pygeovision label osm 15.0 45.13 15.03 45.15 --categories buildings,roads,water --output osm_labels.geojson
```

Expect: a GeoJSON of OpenStreetMap features via the Overpass API (three public mirrors are tried; if all fail
it says `Overpass request failed` and exits `1`).

---

## 12. AI commands (`pygeovision ai …`)

### 12.1 Index/threshold-based (no weights)

```bash
# [offline]
pygeovision ai segment water --input $D/scene_t1.tif --output $O/seg_water.tif    # NDWI threshold (not a model)
pygeovision ai cloud-mask --input $D/scene_t1.tif --output $O/cloud.tif --brightness-threshold 0.7
pygeovision ai change --before $D/scene_t1.tif --after $D/scene_t2.tif -o $O/change.tif
pygeovision ai change --before $D/scene_t1.tif --after $D/scene_t1.tif -o $O/nochange.tif --min-magnitude 0.1
```

Expect for `ai change` on the synthetic pair: `change_fraction` ≈ 0.055 (the 60 × 60 px block is 3 600 of
65 536 px); the identical pair with a magnitude floor reports `change_fraction: 0.0`. The method is change
vector analysis with an Otsu threshold; nodata in either raster is written as 255 and excluded.

### 12.2 Your own network (`--checkpoint` required)

```bash
# [offline]
# change detection with a siamese network (changeformer | bit | dsamnet) needs weights
expect_fail pygeovision ai change --before $D/scene_t1.tif --after $D/scene_t2.tif -o $O/c.tif --method changeformer
python - <<'EOF'
import torch
from pygeovision.models.change_detection.changeformer import ChangeDetection
for v in ("changeformer", "bit", "dsamnet"):
    torch.save(ChangeDetection(model_variant=v, num_classes=2, in_channels=6).build().state_dict(), f"out/{v}.pth")
EOF
for m in changeformer bit dsamnet; do
  pygeovision ai change --before $D/scene_t1.tif --after $D/scene_t2.tif -o $O/chg_$m.tif --method $m --checkpoint $O/$m.pth
done

# tiled inference, batch inference, custom segmentation -- all strict about the checkpoint
expect_fail pygeovision ai infer -i $D/scene_t1.tif -m unet_resnet50 -o $O/x.tif --num-classes 5   # --checkpoint missing
pygeovision ai infer -i $D/scene_t1.tif -m unet_resnet50 --checkpoint $O/wiring.pth -o $O/pred_ai.tif \
    --num-classes 5 --tile-size 128 --overlap 16
pygeovision ai segment custom -i $D/scene_t1.tif -o $O/seg_custom.tif --model unet_resnet50 \
    --checkpoint $O/wiring.pth --num-classes 5
expect_fail pygeovision ai segment custom -i $D/scene_t1.tif -o $O/x.tif --model unet_resnet50 --num-classes 5
expect_fail pygeovision ai infer -i $D/scene_t1.tif -m unet_resnet50 --checkpoint $O/wiring.pth -o $O/x.tif --num-classes 3
```

The last line shows the strict load: a checkpoint trained for 5 classes cannot be used as a 3-class model; the
message names the mismatching tensor (`segmentation_head.0.weight: shape [5,16,3,3] vs [3,16,3,3]`).
Output rasters are `uint8` class maps; every class id is `< num_classes`.

### 12.3 Pretrained / foundation models

```bash
# [weights]
pygeovision ai segment buildings --input $D/scene_rgb.tif --output $O/buildings.tif     # SAM weights from Hugging Face
pygeovision ai segment sam --input $D/scene_rgb.tif --output $O/sam.tif
pygeovision ai classify scene --input $D/scene_rgb.tif --classes "forest,urban,water,agriculture"   # CLIP zero-shot
pygeovision ai classify land-cover --input $D/scene_t1.tif --output $O/lc.tif            # also needs the network (WorldCover)
```

Expect: these need to download model weights (or WorldCover tiles) on first use, so they fail offline with a
connection error (`httpx.ProxyError`, `Can't load the configuration of ...`). That is the expected offline
behaviour — they never fall back to something else.

```bash
# [network]
pip install ultralytics
pygeovision ai detect generic --input $D/scene_rgb.tif --output $O/generic.geojson
pygeovision ai detect ships   --input port.tif --output $O/ships.geojson
pygeovision ai detect cars    --input parking.tif --output $O/cars.geojson
```

`ai detect generic|ships|cars` download the COCO-trained `yolov8m.pt` (about 50 MB, saved in the **current
directory**) on first use. It is a general photo detector, not trained for overhead 10 m imagery, so on the
synthetic scene it correctly finds nothing (an empty `FeatureCollection`); treat it as a wiring test, and use
`ai detect custom` with your own detector for real work.

```bash
# [offline]
expect_fail pygeovision ai detect custom --input $D/scene_rgb.tif --output $O/x.geojson --model unet_resnet50
```

### 12.4 Training

`ai train` does **not** train; it prints `✓ Config ready. This command does NOT train anything itself`, the
configuration, and the Python to run (see `docs/training/`). Exit `0` here means "config built", nothing more.

```bash
# [offline]
pygeovision ai train segmentation --data $O/chips --output $O/model.pth --num-classes 5 --epochs 2 --backbone resnet50
```

---

## 13. Inference (`pygeovision infer`)

```bash
# [offline]
expect_fail pygeovision infer predict $D/scene_t1.tif -o $O/x.tif                     # --checkpoint is required
pygeovision infer predict $D/scene_t1.tif --model unet_resnet50 --checkpoint $O/wiring.pth \
    --output $O/pred.tif --classes 5 --chip-size 128 --overlap 16 --blend gaussian
pygeovision infer predict $D/scene_t1.tif --model unet_resnet50 --checkpoint $O/wiring.pth \
    --output $O/pred_lin.tif --classes 5 --chip-size 128 --overlap 16 --blend linear --device cpu

mkdir -p $O/batch_in && cp $D/ref_chips/*.tif $O/batch_in/
pygeovision infer batch $O/batch_in $O/batch_out --model unet_resnet50 --checkpoint $O/wiring.pth \
    --classes 5 --workers 1 --pattern "*.tif"                                           # 4/4 succeeded

expect_fail pygeovision infer predict $D/scene_t1.tif --checkpoint $O/wiring.pth -o $O/x.tif --classes 3   # wrong class count
expect_fail pygeovision infer predict $D/scene_t1.tif --model no_such_model --checkpoint $O/wiring.pth -o $O/x.tif --classes 5
```

Check the output: `rio info $O/pred.tif` → `uint8`, same size and CRS as the input, 1 band;
`postprocess class-stats $O/pred.tif` shows class ids `0–4` only. With the wiring checkpoint the *classes*
are random — that is expected. With a trained checkpoint they should resemble `labels.tif`.

`--device cuda` needs a GPU (`# [gpu]`, not run here): `pygeovision infer predict … --device cuda`.

---

## 14. Explainability, edge export, monitoring

```bash
# [offline]
# Grad-CAM saliency (registry names use hyphens here: see `pygeovision models list`)
pygeovision explain gradcam $D/scene_t1.tif --model unet-r50 --checkpoint $O/wiring.pth --classes 5 \
    --class-idx 1 --output $O/gradcam.tif

# ONNX export, then a latency benchmark
pygeovision edge export-onnx unet-r50 --output $O/unet.onnx --classes 5 --in-channels 6 --input-size 64 \
    --checkpoint $O/wiring.pth
pygeovision edge benchmark-onnx $O/unet.onnx --device cpu --runs 5       # Mean / P95 ms and FPS
expect_fail pygeovision edge benchmark-onnx $O/missing.onnx

# data drift between a reference set and current chips (PSI / KL per band)
pygeovision monitor drift $D/ref_chips $D/cur_chips --output $O/drift.json         # level: major (cur_chips are brightened)
pygeovision monitor drift $D/ref_chips $D/ref_chips --output $O/drift_none.json    # level: none (identical), PSI 0.0
```

Without `--checkpoint`, `export-onnx` / `gradcam` use the registry's pretrained *initialisation* (this
downloads ImageNet weights, so it needs the network): fine for a latency test, not a trained model.
`monitor drift` also writes a `monitoring/` folder in the current directory.

---

## 15. Vision-language models (`pygeovision vlm`)

```bash
# [weights]
pygeovision vlm caption  $D/scene_rgb.png
pygeovision vlm query    $D/scene_rgb.png "What land-cover types are visible?"
pygeovision vlm search   "river and forest" $D/ref_chips --top-k 3
```

Expect: these download Moondream2 / CLIP from Hugging Face on first use. Offline they raise
`ModelLoadError: Could not load vikhyatk/moondream2 … Needs network access to the Hugging Face Hub (or a
populated cache)` and exit `1` — they never return a made-up caption.

---

## 16. Cloud deployment (`pygeovision cloud`)

```bash
# [offline]
# without the SDKs/credentials these must fail cleanly and exit 1
expect_fail pygeovision cloud deploy-aws $O/wiring.pth my-endpoint --region eu-west-1
expect_fail pygeovision cloud deploy-gcp $O/wiring.pth my-endpoint --project my-project
```

```bash
# [cloud]
pip install boto3 sagemaker                  # AWS SageMaker
pygeovision cloud deploy-aws model.pth my-endpoint --region eu-west-1 --instance ml.m5.large
pip install google-cloud-aiplatform google-cloud-storage     # GCP Vertex AI
pygeovision cloud deploy-gcp model.pth my-endpoint --project my-project --region us-central1
```

Not exercised here: a real deployment creates billable resources.

---

## 17. Registries and catalogues

PyGeoVision has three model listings. They answer different questions:

| Command group | Registry | Use it for |
|---|---|---|
| `ai-models` | the native registry behind `ModelHub` (14 models, underscore names such as `unet_resnet50`) | the names accepted by `infer`, `ai infer`, `ai segment custom` and `--model` of `channel` |
| `models` | the general registry `pygeovision.models.registry` (79 entries, hyphen names such as `unet-r50`) | names for `explain gradcam`, `edge export-onnx`; browse architectures and their weight sources |
| `zoo` | a metadata catalogue (compare sizes and tasks) | choosing a model; it cannot build one |

```bash
# [offline]
pygeovision ai-models list
pygeovision ai-models list --task change_detection --pretrained-only
pygeovision ai-models info unet_resnet50
pygeovision ai-models cache                        # cached weights ("No models cached." initially)
pygeovision ai-models cache --clear --model unet_resnet50   # local file operation

pygeovision models list
pygeovision models list --task segmentation --max-params 30
pygeovision models list --family segformer
pygeovision models info segformer-b0
pygeovision models summary
pygeovision models download unet-r50               # "No downloadable weights" -- ImageNet weights load on first use

pygeovision zoo stats
pygeovision zoo list --task segmentation --max-params 30
pygeovision zoo list --hf-only
pygeovision zoo search segformer
pygeovision zoo top segmentation --n 3
pygeovision zoo info segformer_b0

pygeovision benchmark tasks
pygeovision benchmark datasets segmentation --n 3
pygeovision benchmark models segmentation --n 3
```

```bash
# [network]
pygeovision models download segformer-b0           # fetches nvidia/segformer-b0-... from Hugging Face
```

### Dataset catalogue (`pygeovision datasets`, 503 entries; metadata only)

Dataset names may contain spaces — quote them.

```bash
# [offline]
pygeovision datasets stats
pygeovision datasets domains
pygeovision datasets list --domain wildfire --limit 5
pygeovision datasets list --modality multispectral --task segmentation --min-year 2020 --max-res-m 10 --limit 5
pygeovision datasets search sentinel --limit 5
pygeovision datasets info BigEarthNet
pygeovision datasets similar BigEarthNet --n 3
pygeovision datasets top segmentation --n 3
expect_fail pygeovision datasets info NoSuchDataset          # clean error, no traceback
expect_fail pygeovision datasets download BigEarthNet        # prints the page to visit; never pretends to download
```

---

## 18. YAML pipelines (`pygeovision pipeline`)

A second, simpler pipeline system: a YAML file of steps (`search`, `download`, `infer`, `export`) with
dependencies. Unlike the 25 `channel` pipelines it is a scaffold you fill in. Every step does its work or
fails — an empty search, a failed download, a missing checkpoint or an unsupported action makes the run exit
`1` rather than report success.

```bash
# [offline]
pygeovision pipeline list-templates          # agriculture, climate, disaster, forestry, urban, water
cat pygeovision/pipelines/templates/urban.yaml | head -20

# structure check without running anything
pygeovision pipeline run pygeovision/pipelines/templates/urban.yaml --dry-run

# a template cannot run until you give it an area, dates and YOUR checkpoint -> it refuses
expect_fail pygeovision pipeline run pygeovision/pipelines/templates/urban.yaml

# unsupported actions are rejected at load time (they used to run as silent no-ops)
printf 'name: x\nsteps:\n  - {name: a, action: notify}\n' > $O/bad_action.yaml
expect_fail pygeovision pipeline run $O/bad_action.yaml
```

```bash
# [network]
pygeovision pipeline run pygeovision/pipelines/templates/urban.yaml \
    --set search.bbox='[-0.15,51.47,-0.10,51.52]' \
    --set 'search.date_range=["2024-06-01","2024-06-30"]' \
    --set infer.checkpoint=buildings.pth --set infer.num_classes=2
```

`--set STEP.PARAM=VALUE` overrides a step parameter (JSON-parsed, else a string). Expect, on success, one
prediction raster per downloaded scene (`infer.output_dir`) and its vectorised GeoJSON/GPKG/SHP/KML
(`export.format`, `export.output_dir`). Without network access the `search` step fails (`search returned no
scenes`), `download`/`infer`/`export` are skipped, and the exit code is `1`.

---

## 19. Pass/fail checklist

Tick each line when the command ran as described. `●` = executed offline for this guide on synthetic data;
`○` = needs network, credentials, downloaded weights or a cloud account (the failure mode offline is
described in its section).

<!-- CHECKLIST:BEGIN -->
`●` ran successfully offline for this guide · `◐` only its refusal/failure path ran offline (the real run needs the tag shown) · `○` not run offline.

| ☐ | Command | Offline | Needs | Notes |
|---|---|---|---|---|
| ☐ | `pygeovision ai change` | ● | — | spectral_diff and the three network methods (with wiring checkpoints) offline |
| ☐ | `pygeovision ai chips` | ● | — |  |
| ☐ | `pygeovision ai classify` | ○ | weights | scene needs CLIP weights; land-cover needs network |
| ☐ | `pygeovision ai cloud-mask` | ● | — |  |
| ☐ | `pygeovision ai detect` | ◐ | network | generic/ships/cars need `ultralytics` + a YOLO download; custom refusal offline |
| ☐ | `pygeovision ai infer` | ● | — |  |
| ☐ | `pygeovision ai segment` | ● | weights | water, custom offline; buildings, sam need network + weights |
| ☐ | `pygeovision ai train` | ● | — |  |
| ☐ | `pygeovision ai-models cache` | ● | — |  |
| ☐ | `pygeovision ai-models info` | ● | — |  |
| ☐ | `pygeovision ai-models list` | ● | — |  |
| ☐ | `pygeovision benchmark datasets` | ● | — |  |
| ☐ | `pygeovision benchmark models` | ● | — |  |
| ☐ | `pygeovision benchmark tasks` | ● | — |  |
| ☐ | `pygeovision channel` | ● | — | 19 ready + 6 needs-weights pipelines: all 25 run offline via tools/offline_pipeline_smoke.py; real data needs network |
| ☐ | `pygeovision cloud deploy-aws` | ◐ | cloud | cloud account (offline: clean failure) |
| ☐ | `pygeovision cloud deploy-gcp` | ◐ | cloud | cloud account (offline: clean failure) |
| ☐ | `pygeovision data auth add` | ● | — |  |
| ☐ | `pygeovision data auth list` | ● | — |  |
| ☐ | `pygeovision data auth remove` | ● | — |  |
| ☐ | `pygeovision data auth test` | ◐ | network | open providers: network; others: stored credentials |
| ☐ | `pygeovision data cache clear` | ● | — |  |
| ☐ | `pygeovision data cache stats` | ● | — |  |
| ☐ | `pygeovision data download` | ○ | network | network |
| ☐ | `pygeovision data pipeline history` | ○ | network | after a run; needs network for a run that finds scenes |
| ☐ | `pygeovision data pipeline list` | ● | — |  |
| ☐ | `pygeovision data pipeline run` | ◐ | network | network (search step); structure/refusals offline |
| ☐ | `pygeovision data pipeline schedule` | ● | — |  |
| ☐ | `pygeovision data pipeline validate` | ● | — |  |
| ☐ | `pygeovision data providers` | ● | — |  |
| ☐ | `pygeovision data search` | ○ | network | network |
| ☐ | `pygeovision datasets domains` | ● | — |  |
| ☐ | `pygeovision datasets download` | ◐ | — |  |
| ☐ | `pygeovision datasets info` | ● | — |  |
| ☐ | `pygeovision datasets list` | ● | — |  |
| ☐ | `pygeovision datasets search` | ● | — |  |
| ☐ | `pygeovision datasets similar` | ● | — |  |
| ☐ | `pygeovision datasets stats` | ● | — |  |
| ☐ | `pygeovision datasets top` | ● | — |  |
| ☐ | `pygeovision doctor` | ● | — |  |
| ☐ | `pygeovision edge benchmark-onnx` | ● | — |  |
| ☐ | `pygeovision edge export-onnx` | ● | — |  |
| ☐ | `pygeovision explain gradcam` | ● | — |  |
| ☐ | `pygeovision indices all` | ● | — |  |
| ☐ | `pygeovision indices compute` | ● | — |  |
| ☐ | `pygeovision indices list` | ● | — |  |
| ☐ | `pygeovision infer batch` | ● | — |  |
| ☐ | `pygeovision infer predict` | ● | — |  |
| ☐ | `pygeovision label osm` | ○ | network | network (Overpass API) |
| ☐ | `pygeovision label quality` | ● | — |  |
| ☐ | `pygeovision models download` | ● | network | no-weights entry offline; Hugging Face entries need network |
| ☐ | `pygeovision models info` | ● | — |  |
| ☐ | `pygeovision models list` | ● | — |  |
| ☐ | `pygeovision models summary` | ● | — |  |
| ☐ | `pygeovision monitor drift` | ● | — |  |
| ☐ | `pygeovision obia assess-classification` | ● | — |  |
| ☐ | `pygeovision obia assess-segmentation` | ● | — |  |
| ☐ | `pygeovision obia classify` | ● | — |  |
| ☐ | `pygeovision obia demo` | ● | — |  |
| ☐ | `pygeovision obia features` | ● | — |  |
| ☐ | `pygeovision obia scale` | ● | — |  |
| ☐ | `pygeovision obia segment` | ● | — |  |
| ☐ | `pygeovision pipeline list-templates` | ● | — |  |
| ☐ | `pygeovision pipeline run` | ● | network | dry-run and refusals offline; a real run needs network + your checkpoint |
| ☐ | `pygeovision postprocess accuracy` | ● | — |  |
| ☐ | `pygeovision postprocess class-stats` | ● | — |  |
| ☐ | `pygeovision postprocess cog` | ● | — |  |
| ☐ | `pygeovision postprocess export` | ● | — |  |
| ☐ | `pygeovision postprocess regularise` | ● | — |  |
| ☐ | `pygeovision postprocess sieve` | ● | — |  |
| ☐ | `pygeovision postprocess smooth` | ● | — |  |
| ☐ | `pygeovision postprocess vectorise` | ● | — |  |
| ☐ | `pygeovision postprocess zonal-stats` | ● | — |  |
| ☐ | `pygeovision preprocess clip` | ● | — |  |
| ☐ | `pygeovision preprocess normalise` | ● | — |  |
| ☐ | `pygeovision preprocess pipeline` | ● | — |  |
| ☐ | `pygeovision preprocess resample` | ● | — |  |
| ☐ | `pygeovision preprocess stack` | ● | — |  |
| ☐ | `pygeovision report` | ● | — | rebuilds PDF/Word/Excel, maps and vectors from a saved run; runs offline on the smoke-test outputs |
| ☐ | `pygeovision status` | ● | — |  |
| ☐ | `pygeovision timeseries analyze` | ● | — |  |
| ☐ | `pygeovision timeseries anomaly` | ● | — |  |
| ☐ | `pygeovision validate for-inference` | ● | — |  |
| ☐ | `pygeovision validate run` | ● | — |  |
| ☐ | `pygeovision vlm caption` | ○ | weights | Hugging Face download |
| ☐ | `pygeovision vlm query` | ○ | weights | Hugging Face download |
| ☐ | `pygeovision vlm search` | ○ | weights | Hugging Face download |
| ☐ | `pygeovision zoo info` | ● | — |  |
| ☐ | `pygeovision zoo list` | ● | — |  |
| ☐ | `pygeovision zoo search` | ● | — |  |
| ☐ | `pygeovision zoo stats` | ● | — |  |
| ☐ | `pygeovision zoo top` | ● | — |  |
<!-- CHECKLIST:END -->

---

## 20. Things to know

* **Two bounding-box syntaxes.** `channel` and `data search|download` take four space-separated numbers
  (`--bbox MIN_LON MIN_LAT MAX_LON MAX_LAT`); `preprocess clip|pipeline --bbox` take one comma-separated string
  (`minlon,minlat,maxlon,maxlat`). `label osm` takes the four numbers as positional arguments.
* **Two model-name spellings.** `ai-models`/`ModelHub` names use underscores (`unet_resnet50`);
  `models`/`explain`/`edge` use hyphens (`unet-r50`). `ModelHub.load` finds a hyphen name in the general
  registry, so `infer predict --model unet-r50` also works; the reverse is not true.
* **Nothing is substituted.** A model that cannot be built, or a checkpoint that does not match, raises. Random
  weights are only ever created by `tools/make_wiring_checkpoint.py` (and by you), never silently.
* **Reports are generated by default.** Disable per run with `--no-report`, globally with `PYGEOVISION_REPORT=0`.
  Accuracy is not assessed (no reference data): reports state areas and the method, and say so. For pipelines that
  run your own network the class names default to `Class N`; pass `--class-names`.
* **Rasters are written without GDAL block-size warnings** and class maps are `uint8` with a nodata value that
  fits the type.
* **`ultralytics` downloads `yolov8m.pt` into the current directory** the first time a detection command runs.
* **`preprocess normalise --scale-factor 10000` is a plain divisor.** It does not know the scene, so it does not
  subtract the +1000 DN offset of 2022+ Sentinel-2 scenes; the `channel` pipelines and `stack_and_prepare_bands`
  do. Subtract it yourself (or use those) before computing indices on such scenes.
* **A search is never served from the cache by a pipeline.** Cached results lose their download handle and used
  to take a different, degraded download route (no engine, no cloud mask). Plain `data search` still caches
  (the key now includes the result cap and sort order).
* **Accuracy of 1.0 in the OBIA demo** comes from evaluating on the training objects; it is a plumbing check.
* **Ranking in `datasets top` / `zoo top`** is a catalogue heuristic (it can put point-cloud sets under
  "segmentation"); read the modality column.
* **Not covered by any command-line example here:** the Python-only APIs (`client.labeling.*`, foundation-model
  labelers, `GeoTrainer`), a GPU, and real satellite scenes.

## How this was verified

The `# [offline]` blocks were extracted from this file and executed in order, in an empty directory, against
the data from `tools/make_test_data.py` with network access blocked: every command that should succeed exited
`0`, every `expect_fail` command exited non-zero. The 25 pipelines were additionally run with
`tools/offline_pipeline_smoke.py` (25/25, including the map, GeoPackage and PDF/Word/Excel check). The repository test suite also passes offline. **No `# [network]`,
`# [credentials]`, `# [weights]` or `# [cloud]` block was executed successfully** (there is no network, no
trained or pretrained weights, no GPU and no cloud account in the environment where this was written); for the
ones that were attempted offline the section describes how they fail. Treat the first run of each on your own
machine and data as a validation run.

The first live runs (`change_detection`, `water_bodies`, `flood_mapping` on real Sentinel-2 scenes) exposed the
defects fixed in `tests/test_real_run_regressions.py`: a multi-tile mosaic that kept only the last tile, a search
that saw too few candidates, 15-asset downloads, the missing Sentinel-2 offset, a month window that stopped on the
28th and duplicated log lines. Those are covered offline now with synthetic tiles; the live behaviour itself is
still yours to confirm.
