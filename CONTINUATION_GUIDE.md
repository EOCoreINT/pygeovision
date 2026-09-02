# PyGeoVision — Continuation Guide

**Read this before touching the codebase.** This documents an in-progress
production-readiness audit: what's been verified and fixed, what's
confirmed broken and not yet fixed, and — most importantly — the
*recurring architectural traps* that will bite you if you don't know
about them going in.

If you're picking this up fresh: this is not a "mostly done, polish the
edges" situation. Real, load-bearing bugs have been found in code that
looked completely fine (98/98 models "worked" until you actually ran a
forward pass; a whole pipeline module silently returned pre-fix, buggy
classes because nothing forced an import error). Assume nothing is
correct until you've actually run it against real or synthetic data
with hand-checkable expected output. That's the standard this whole
audit was held to, and it found real bugs every time it was applied.

---

## 1. The single most important thing to understand: duplicate parallel systems

This codebase has been through at least one incomplete refactor. The
symptom, found repeatedly and independently, in unrelated parts of the
code: **the same capability implemented twice, under different module
paths — one side real/audited/fixed, the other stale, unaudited, or
outright broken — with no import error or warning connecting them.**

Confirmed instances (there may be more not yet found — see §6):

| Capability | Real, audited path | Separate / stale path | Status |
|---|---|---|---|
| 10 core AI pipelines | `pygeovision.ai.pipelines.*Pipeline` | `pygeovision.pipelines.*Pipeline` | **FIXED** — was a complete stale duplicate with pre-fix bugs still present (confirmed: the old NDVI-via-post_process bug, still there). Now re-exports the real classes. See §3. |
| `Pipeline` YAML orchestrator | `pygeovision.pipelines.orchestrator.Pipeline` | `from pygeovision.pipelines import Pipeline` | **FIXED** — was unconditionally broken (`ImportError`), used by `data/fetch.py`'s real `run_pipeline()` fallback. |
| Tiled inference | `pygeovision.ai.inference.tiled_inference.TiledInference` (real windowed reads, memory-aware batch sizing) | `pygeovision.inference.tiled.TiledInference` (used by `client.inference.tiled()`, `client.detection.custom`) | **NOT fixed** — confirmed genuinely separate code (different defaults: `overlap=128` vs `64`), not a re-export. Memory-efficiency fixes do not apply to the second one. |
| ESA WorldCover labeling | `pygeovision.ai.labeling.esa_worldcover.ESAWorldCoverLabeler` (3 real bugs fixed: nodata, class-name mapping, tile-origin int/float) | `pygeovision.labeling.landcover.ESAWorldCoverLabeler` (used by `client.labeling.esa_worldcover()`) | **NOT fixed** — confirmed genuinely separate class body. Fixes do not apply. |
| "Pipeline" (as a concept) | means 3 different things | see below | Documented, not "fixed" (it's a naming problem, not a bug per se) |
| Model registries | `pygeovision.ai.models.registry.registry` (14 models, real factories) | `pygeovision.models.registry.model_registry` (121 names) AND `pygeovision.ai.models.zoo.model_zoo` (98 `ModelSpec` entries) | Three overlapping catalogs, different counts, different naming conventions (`vit-b16` vs `vit_b16`). `hub.load()` tries the 14-model native registry first, falls back to the 121-model one. Not consolidated. |

**The "pipeline" naming collision specifically** — worth internalizing since it will confuse you if you don't know it going in:
1. **`pygeovision.ai.pipelines.*Pipeline` classes** (10 real + 23 in `domains.py`, run via `pygeovision channel <name>`) — the actual AI task pipelines.
2. **`client.pipeline(name)`** — NOT an AI pipeline runner. It's PyGeoFetch's chainable *data processing* builder (`cloud_mask().clip().reproject().ndvi().run(...)`). A docstring in the old stale module literally showed `client.pipeline("building_footprints", bbox=..., date=...)` as if it worked — it doesn't; that's not what this method does.
3. **`pygeovision.pipelines.orchestrator.Pipeline`** — a YAML-based multi-step workflow orchestrator (search → download → label → train chains), unrelated to either of the above.

**Practical rule going forward:** before assuming a fix "just works" everywhere, grep for the class/function name across the whole tree. If it appears in more than one file with the same name, check whether they're the same object or two separate definitions. This one check would have caught the `pipelines/__init__.py` bug immediately.

---

## 2. What's genuinely verified and fixed this cycle

"Verified" here means: read the real code, understood the real bug,
fixed it, then confirmed with either (a) a real forward pass on real or
carefully-constructed synthetic data with a **hand-calculated expected
value**, or (b) a revert-check (temporarily undo the fix, confirm the
test now fails for the right reason, then restore it). Every fix below
was held to this standard — not just "the code runs without an
exception."

### Data layer (`pygeovision/data/`)
- Multi-asset band stacking (Landsat/Sentinel-2), mission-aware band
  aliases (`S2_BAND_ALIASES`, `LANDSAT_ETM_TM_ALIASES`,`LANDSAT_OLI_ALIASES` in `radiometric.py` — Landsat 7/5 and Landsat
  8/9 have genuinely different band-number-to-wavelength mappings)
- Real radiometric scaling (`LANDSAT_SR_SCALE=0.0000275, offset=-0.2`;
  Sentinel-2 `/10000`) — confirmed against real USGS/ESA documentation
- Real cloud masking (QA_PIXEL / SCL), real bbox cropping
  (`crop_to_bbox`), real unzip handling, cache-hit asset recovery
- **New this cycle**: real SWIR1/SWIR2 band aliases (needed for
  dNBR/NDSI/MNDWI — didn't exist before), real Landsat thermal band
  aliases (`ST_B10`/`ST_B6`) with the real USGS Collection 2 scale/offset
  (`LANDSAT_ST_SCALE=0.00341802, LANDSAT_ST_OFFSET=149.0`) — deliberately
  kept isolated from the SR reflectance scaling path since it's a
  different formula for a different physical quantity
- **New this cycle**: `real_pixel_area_ha()` — a real, geodesically-accurate
  (`pyproj.Geod`) area calculation. Fixes a confirmed, severe bug: the
  old pattern `abs(src.res[0]*src.res[1])/10000` assumes raster resolution
  is in meters, but the real production pipeline always outputs
  `EPSG:4326` (degrees) — this made area/density values wrong by roughly
  **10 billion times** (confirmed directly: a real test scene produced
  "4.7 billion vehicles per km²" before the fix). This bug was found in
  4 separate pipelines and fixed in all of them — see §3.

### The 10 core AI pipelines (`pygeovision/ai/pipelines/__init__.py`)
`building_footprints`, `carbon_estimation`, `change_detection`,
`crop_monitoring`, `deforestation`, `disaster_assessment`, `land_cover`,
`solar_detection`, `urban_growth`, `water_bodies` — real bug fixes
including bi-temporal common-grid alignment, real NDVI computation
(previously requested via a `post_process` step that silently never ran),
and (this cycle) the area-calculation bug fix in `carbon_estimation`.

### Tiled inference (`pygeovision.ai.inference.tiled_inference.TiledInference`)
Real windowed reads (not whole-image loads), memory-aware automatic
batch sizing. **Only this specific class** — see the duplicate-systems
table above.

### Object detection (`pygeovision/models/detection/yolo.py`)
Fixed a severe bug: `client.detection.ships()`/`.cars()` relabeled
*every* detection with the requested class name regardless of what the
model actually detected (a detected person and a detected car were both
labeled "ship"). Now filters to real, relevant COCO classes.

### Agent (`pygeovision/agent/`)
Planner and 8 real tools verified; SAR/InSAR routing removed (see §4).

### Model loading — `.to()` device placement bug (`pygeovision/ai/models/hub.py`, `pygeovision/models/registry.py`)
Two **separate, independent** `model.to(device)` call sites both
unconditionally assumed every model supports `.to()`. Real wrapper
classes (`CLIPGeo`, `TesseraGeo`, `AlphaEarthGeo`, `MoondreamGeo`) either
didn't implement it, or genuinely have no local device-bound model at
all (`TesseraGeo`/`AlphaEarthGeo` are Google Earth Engine query
services). Fixed both call sites defensively (`hasattr` check), and
added a real `.to()` to `CLIPGeo`/`MoondreamGeo` since those do wrap a
real HuggingFace model and should support real device placement, not
just silently skip it. **Finding this took two rounds** — the first fix
(in `hub.py`) didn't fully resolve `tessera`/`alphaearth`; a second,
independent call site in `registry.py`'s `get_model()` fallback had to
be found and fixed separately. Don't assume one fix location is the
only one.

### A misleading error message (`pygeovision/models/registry.py::_build_pytorch_fallback`)
Every model that failed to build via `timm`/HuggingFace showed *"This
architecture has no timm_id/hf_id"* — **false** for many of them.
Confirmed directly: `vit-b16` has a real `hf_id` defined; the actual
failure was `HTTPError: couldn't connect to huggingface.co` (network),
silently caught and replaced with a false claim that no metadata
existed. Fixed to surface the real underlying reason instead.

### 7 of 25 `_make_simple` domain pipelines converted to real implementations
See §3 for full detail. `wildfire_severity` (real dNBR), `glacier_monitoring`
+ `snow_cover` (real NDSI), `wetland_mapping` (real MNDWI+EVI),
`urban_heat_island` (real Landsat thermal LST), `parking_occupancy` +
`port_monitoring` (real vehicle/ship counts via the fixed detector).

### Documentation
Full ReadTheDocs rewrite distinguishing verified claims from unverified
ones, page by page. See `/mnt/user-data/outputs/pygeovision-docs.zip`
from earlier in this session (or wherever it was ultimately placed in
the repo) for the full picture — `architecture.md` in particular has a
detailed "what does verified even mean here" section worth reading in
full.

---

## 3. The pipeline catalog — exact, current status

`pygeovision.ai.pipelines.domains.list_pipelines()` (and the top-level
`pygeovision.ai.pipelines.domains._PIPELINE_REGISTRY`) currently returns
**51 names**. Breakdown, as of this cycle:

### Tier 1 — 10 original, thoroughly audited pipelines
`building_footprints`, `carbon_estimation`, `change_detection`,
`crop_monitoring`, `deforestation`, `disaster_assessment`, `land_cover`,
`solar_detection`, `urban_growth`, `water_bodies`
— in `pygeovision/ai/pipelines/__init__.py`, real, fixed, tested.

### Tier 2 — 7 newly converted this cycle (real, hand-verified)
All in `pygeovision/ai/pipelines/domains.py`:

| Pipeline | Real technique | Honest limitation noted |
|---|---|---|
| `wildfire_severity` | Real dNBR (Key & Benson 2006), USFS/MTBS 4-class thresholds | — |
| `glacier_monitoring` | Real NDSI (Hall et al. 1995), bi-temporal area trend | — |
| `snow_cover` | Real NDSI extent | Does **not** estimate SWE (original catalog claimed it; SWE needs microwave/ground data, not optical) |
| `wetland_mapping` | Real MNDWI (Xu 2006) + EVI (Huete et al. 2002) | Distinguishes wetland from open water — MNDWI alone can't |
| `urban_heat_island` | Real Landsat thermal band → Kelvin → Celsius (real USGS Collection 2 formula) | Requires Landsat explicitly; Sentinel-2 has no thermal sensor |
| `parking_occupancy` | Real vehicle count/density via the fixed `client.detection.cars()` | Does **not** compute occupancy % (needs real parking capacity data) |
| `port_monitoring` | Real ship count via the fixed `client.detection.ships()` | Does **not** compute throughput (needs a real multi-date time series) |

Each has a dedicated test file with hand-calculated expected values —
see `tests/test_wildfire_severity_real.py`,
`tests/test_cryosphere_pipelines_real.py`,
`tests/test_wetland_mapping_real.py`,
`tests/test_urban_heat_island_real.py`,
`tests/test_area_fix_and_detection_pipelines.py`. **Use these as the
template** for converting the remaining ones — the pattern (real formula
→ synthetic data with a hand-computable expected result → assert exact
match) is consistent and works well.

### Tier 3 — 16 named classes, real code, **NOT independently verified**, confirmed widespread bug pattern
`CropTypeMappingPipeline`, `CropHealthPipeline`, `IrrigationDetectionPipeline`,
`CanopyHeightPipeline`, `TreeSpeciesPipeline`, `ForestFirePipeline`,
`RoadExtractionPipeline`, `InfrastructureMonitoringPipeline`,
`FloodMappingPipeline`, `WaterQualityPipeline`, `CoastalMonitoringPipeline`,
`LandslideDetectionPipeline`, `VolcanoMonitoringPipeline`,
`LandSurfaceTemperaturePipeline`, `VegetationIndicesPipeline`,
`OceanShipDetectionPipeline`

**Confirmed bug pattern, not yet fixed**: `domains.py` has **9 instances**
of bare `except Exception:` (no `as e`, no logging), all within these 16
classes (lines 101–488). Two are confirmed, specific silent-fallback
bugs:
- `CropTypeMappingPipeline`: if its default model name (`"crop_type_model"`)
  isn't registered — and it almost certainly isn't — it silently falls
  back to returning the **raw, unprocessed input image** as the "crop
  type map," with `success=True` and no visible error.
- `OceanShipDetectionPipeline`: identical pattern —
  `except Exception: ships_out = out` — returns `success=True` with the
  output *directory* standing in for a real result file.

**This is the next highest-priority work after finishing Tier 4.** All 9
`except Exception:` sites need individual inspection: does the fallback
silently produce a plausible-looking-but-wrong result (fix it to fail
loudly instead), or is it a reasonable degradation (leave it, but add
`as e` and log it)? Don't assume — check each one.

### Tier 4 — 18 remaining `_make_simple` pipelines (generic stub, does nothing task-specific)
`oil_spill_detection`, `air_quality_index`, `solar_potential`,
`mangrove_mapping`, `permafrost_thaw`, `mine_detection`,
`powerline_extraction`, `dam_safety`, `crop_yield_forecast`,
`aquaculture_mapping`, `landcover_change`, `biodiversity_hotspot`,
`construction_progress`, `reef_bleaching`, `dust_storm_tracking`,
`archaeological_site`, `pipeline_leak_detection`, `wind_farm_siting`

Each of these is built from `_make_simple(name, description, sensors,
tags)` — a factory producing a pipeline that only searches, downloads
one scene, and validates it. Despite specific-sounding descriptions
("SAR oil slick detection via adaptive backscatter threshold," "dNBR
burn severity mapping" — wait, that one's now real, see above), **none
of the remaining 18 run any task-specific model or algorithm.**

**Realistic triage for these 18**, based on what was already assessed:

- **Quick wins, real implementation feasible** (similar shape to what's
  already been done — a real, published formula or a straightforward
  orchestration of already-real pieces):
  - `landcover_change` — orchestrate two already-real `LandCoverPipeline`
    runs at different dates, compute a real class-transition matrix.
    More involved than a single-formula pipeline (it's two full pipeline
    runs plus a real comparison step), but everything it needs already
    exists and is real.
  - `oil_spill_detection` — real SAR dark-pixel thresholding is a
    well-known, standard technique. Requires calling
    `pygeofetch.processing.sar.SARProcessor` directly (SAR/InSAR was
    removed from pygeovision's own layer this session — see §4) rather
    than through `_search_and_download`.
  - `mangrove_mapping` — likely combinable from NDVI-like vegetation
    index + SAR via pygeofetch, similar shape to `wetland_mapping`.
  - `pipeline_leak_detection` — "vegetation stress anomaly" is
    implementable as a real NDVI-anomaly-vs-baseline comparison.
  - `dam_safety`, `permafrost_thaw` — both described as needing InSAR
    deformation; delegate to `pygeofetch.insar.*` directly, same
    reasoning as `oil_spill_detection`.

- **Genuinely infeasible without real trained models or non-satellite
  data this session doesn't have** — the honest move for these is a
  clear `raise NotImplementedError(...)` with a real explanation (matching
  the pattern already used elsewhere in this codebase for genuinely
  unimplemented things), **not** a fake pass and **not** silent removal
  from the catalog (removal would hide that someone asked for this
  capability and it doesn't exist — a clear error is more honest and
  more useful to a future contributor who might implement it for real):
  - `wind_farm_siting` — needs wind resource data (meteorological, not
    satellite imagery)
  - `archaeological_site` — needs specialized sub-surface anomaly
    detection, not a standard remote-sensing technique
  - `solar_potential` — needs a real DSM/LiDAR elevation product for
    roof-level irradiance, not standard optical bands
  - `mine_detection`, `construction_progress`, `aquaculture_mapping` —
    all need specialized, trained object/change detectors that don't
    exist in this codebase
  - `air_quality_index`, `dust_storm_tracking` — need specialized
    atmospheric data products (Sentinel-5P NO2/AOD), not standard
    optical bands; possible but would need real verification that
    pygeofetch can actually fetch these products before claiming it works
  - `crop_yield_forecast` — "forecast" implies a real predictive model;
    an NDVI-time-series proxy could be built honestly (matching the
    `carbon_estimation` "uncalibrated proxy" pattern) but needs the same
    honesty treatment, not a silent claim of real yield prediction
  - `biodiversity_hotspot`, `reef_bleaching` — could potentially use
    real DINOv3 embeddings for the former (clustering as a diversity
    proxy), but neither has a standard, published remote-sensing formula
    the way NDSI/NBR/MNDWI do — would need real domain research before
    implementing, not just picking a plausible-sounding index

---

## 4. SAR/InSAR — removed, not stubbed

PyGeoVision's own SAR/InSAR processing layer (`client.sar`, plus a
`pygeovision.insar` module that existed in an earlier snapshot of this
project but was missing from the codebase this cycle started from) was
**removed entirely** this session — not stubbed, not deprecated, deleted.
Reasoning: [PyGeoFetch](https://pypi.org/project/pygeofetch/) — a real,
independent, installed dependency (confirmed: v2.6.2.1, inspected
directly) — already has a `SARProcessor` with exactly the four methods
(`calibrate`, `coherence`, `despeckle`, `flood_map`) pygeovision's own
wrapper delegated to, plus a comprehensive `pygeofetch.insar.*` suite
pygeovision never had an equivalent of.

**If you're implementing any of the remaining SAR/InSAR-flavored
pipelines** (`oil_spill_detection`, `dam_safety`, `permafrost_thaw`,
`mangrove_mapping`), call `pygeofetch.processing.sar.SARProcessor` /
`pygeofetch.insar.*` **directly** — do not re-introduce a pygeovision-side
SAR wrapper. That's exactly the duplication this removal was meant to
eliminate.

---

## 5. The model registry — 121 names, verified breakdown

A systematic harness (`hub.load()` + real forward pass with
task-appropriate synthetic input, for all 121 registered model names)
found:

- **6 fully working** (build + real forward pass both succeed, verified
  this cycle): `unet-r50`, `unet-r101`, `unet-efficientb4` (segmentation),
  `bit-r50`, `dsamnet` (change detection), `srcnn` (super-resolution)
- **9 fixed this cycle** — the `.to()` bug (§2). All 9 now build
  successfully: `clip-vit-b32`, `remoteclip-b32`, `remoteclip-l14`,
  `georsclip`, `openclip-b32`, `openclip-l14`, `moondream2`, `tessera`,
  `alphaearth`
- **36 have real `timm_id`/`hf_id` metadata but couldn't be verified**
  from this sandbox (no network access to HuggingFace Hub /
  torchvision's weight servers). Very likely real and working in a
  normal, network-connected environment — this needs re-verification in
  an environment with real network access, not more sandbox work.
- **26 build successfully but fail on forward pass** — mostly 3D
  point-cloud architectures (`pointnet2-ssg`, `randlanet`, `kpconv`,
  `ptv3`, etc.) where the test harness's synthetic input shape likely
  doesn't match what each architecture actually expects. **Needs
  individual verification per architecture** — this is plausibly mostly
  a test-harness problem, not 26 real bugs, but that needs confirming
  one at a time, not assumed.
- **43 genuinely raise `NotImplementedError`** — no factory exists at
  all. Matches the same honest-stub pattern found elsewhere (SAM,
  DINOv3 heads, ChangeFormer, Prithvi, most detection/timeseries/
  super-resolution architectures). The error message is now honest
  (§2) about *why* — check whether it's a missing factory or a network
  failure before assuming it's a "real" gap in the codebase.

The verification script itself is worth keeping —
`verify_all_models.py` (repo root at the time this was run; may need
moving into `tests/` or `scripts/`) — it's a real, working harness, not
a one-off. Re-run it with real network access to get a much more
complete picture of the 36+26 buckets above.

---

## 6. Places not yet checked for the duplicate-parallel-systems pattern

A comprehensive sweep this cycle checked every top-level module that
mirrors an `ai/` submodule name (`data`, `inference`, `labeling`,
`models`, `monitoring`, `pipelines`, `training`) and every function-level
import across all 223 modules. Result: no other instance at the severity
of the `pipelines/__init__.py` case was found — the others checked
(`models/architectures/*`, `monitoring/`, `training/trainer.py` + 3
other same-named files, `labeling/osm.py`) are genuinely different,
coexisting implementations serving different purposes, not abandoned
stale copies.

**That does not mean they're bug-free** — most of them are simply
unaudited, which is a different (lower-severity, but still real) risk.
If you're working in any of these areas, don't assume "genuinely
different from its counterpart" means "correct."

**Not yet checked at all**: `advanced/`, `api/`, `benchmark/`, `cloud/`,
`core/`, `datasets/`, `edge/`, `enterprise/`, `explainability/`,
`losses/`, `preprocess/`, `serving/`, `utils/`, `viz/` (partially — the
API signatures were spot-checked for docs, not the implementation
logic), and most of the ~80 CLI commands outside `channel`.

---

## 7. Recommended priority order from here

1. **Finish the Tier 3 sweep** (§3) — 9 `except Exception:` sites across
   16 classes, at least 2 confirmed producing silently-wrong results.
   This is higher priority than converting more Tier 4 stubs, because a
   pipeline that *looks* like it works but silently returns garbage is
   worse than one that's honestly a generic stub.
2. **Convert the Tier 4 "quick win" pipelines** (§3) — `landcover_change`,
   `oil_spill_detection`, `mangrove_mapping`, `pipeline_leak_detection`,
   `dam_safety`, `permafrost_thaw`.
3. **Convert the genuinely-infeasible Tier 4 pipelines to honest
   `NotImplementedError`** rather than leaving them as silently-fake
   `_make_simple` stubs — this is a small amount of work per pipeline
   and directly serves the "no stub is needed here" production goal,
   even where a full real implementation isn't feasible right now.
4. **Re-run `verify_all_models.py` with real network access** to get a
   real picture of the 36 network-blocked and 26 forward-fail model
   buckets, rather than leaving them as an educated guess.
5. Only after 1–4: start on the genuinely-unaudited modules in §6.

---

## 8. Testing methodology — replicate this, don't skip steps

Every real fix in this audit followed the same pattern; deviating from
it is how bugs get missed:

1. **Read the real code path**, not just the function you're editing —
   trace what actually calls it, with what real arguments.
2. **Construct synthetic data with a hand-computable expected result.**
   Not just "does it run without crashing" — pick input values where you
   can work out the correct answer with a calculator, then assert the
   code produces that exact answer (within floating-point tolerance).
   Every pipeline fix in §3 did this — e.g. `wildfire_severity`'s test
   uses NIR/SWIR2 values chosen so the expected dNBR is exactly 0.9.
3. **Revert-check structural fixes.** For fixes that aren't a pure
   formula (defensive checks, inheritance changes, import fixes),
   temporarily undo the fix and confirm the test fails for the *right*
   reason, then restore it. This caught real gaps twice this cycle (the
   `.to()` fix needed a second call site; a test's own assertion
   direction was backwards on first write, caught by checking actual
   output against real math rather than trusting the assertion).
4. **Run the full test suite after every fix**, not just the new tests.
   Structural changes (the `BasePipeline` inheritance fix affecting 41
   pipelines at once) need this to catch unintended breakage.
5. **Grep for the same bug pattern elsewhere** before considering a fix
   "done." The area-calculation bug was found once, then found in 3
   more places by grepping for the exact buggy expression across the
   whole codebase — and that same sweep is what led to finding the
   stale `pipelines/__init__.py` module in the first place.

Current test suite size: 1049 passed, 10 skipped, 1 xfailed (skips are
optional-dependency related; check `pytest.ini`/`conftest.py` for the
skip conditions before assuming they're all fine to ignore).
