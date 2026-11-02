# Issues to file on EOCoreINT/pygeovision (22)

## 1. Search cache hit makes download() silently use a different code path (direct HTTP, bypasses the pygeofetch engine)

**Labels:** bug, data-layer, help wanted

**What happens**
`client.search(..., use_cache=True)` is the default (1-hour TTL, `~/.pygeovision/search_cache`). Items read back from the cache
have `satellite_data=None`, so `download()` treats them as "pystac-fallback" items and downloads them with
`_download_stac_item_direct`. The pygeofetch engine (retries, `parallel`, its progress and auth handling) is never used. The
`pyproject.toml` comment says the pystac fallback is "not a normal operating path", yet a second identical search takes it.

**Evidence** (benchmark of 2 Nov 2026, pygeovision 2.2.0, three Sentinel-2 L2A scenes from Planetary Computer)
- Run 1, empty cache: the log shows `Searching 1 provider(s) [planetary_computer]` and `Downloading 3 scene(s) via PyGeoFetch`.
- Runs 2 and 3 within the hour: no search line, then `Downloading 3 pystac-fallback item(s) directly (bypassing PyGeoFetch engine).`

**To reproduce**
```python
import pygeovision as pgv
c = pgv.PyGeoVision()
kw = dict(bbox=(-0.35, 5.45, 0.05, 5.75), date_range=("2026-06-27", "2026-07-06"),
          collections=["sentinel-2-l2a"], providers=["planetary_computer"], cloud_cover_max=100, max_results=3)
for i in range(2):                      # the second call is served from the cache
    res = c.search(**kw)
    c.download(res, output_dir=f"out{i}", bands=["B03", "B11"])   # compare the two logs
```
Workaround: `use_cache=False`.

**Suggested approach** Persist what is needed to rebuild `satellite_data` (or re-resolve the item by id on a cache hit), so
cached and fresh results take the same path. At minimum log a WARNING when the path changes.

**Done when** a test shows cached and uncached results go through the same download path, and the benchmark above logs the
engine in all three runs.

**Where to look** `pygeovision/data/fetch.py`: search (~L727), `_load_cache` (~L2915), `download` (~L1160-1220),
`_download_stac_item_direct` (~L1513).

---
**Want to take this?** Comment below so nobody duplicates the work, then open a pull request that mentions this issue.
Please read [CONTRIBUTING.md](https://github.com/EOCoreINT/pygeovision/blob/main/CONTRIBUTING.md) first. Questions are welcome here too; if you hit a problem
while reproducing, include the output of `pygeovision doctor`.


## 2. download() returns a scene as successful when a requested band was not downloaded

**Labels:** bug, data-layer, help wanted

**What happens**
With `bands=["B03", "B11", "SCL"]` on the direct path, the log shows `✓ <scene> 64 MB (2 assets)` followed by
`✗ <scene>: pip install rasterio`. Only B03 and B11 were on disk, and the call returned without raising. A caller cannot tell
that a requested band is missing unless it inspects the folder.

Two causes are visible:
1. `rasterio` is only in the `geo` and `obia` extras, not in the core dependencies, but `data/fetch.py` imports it for
   post-processing (~L196, ~L1864, ~L1903). A plain install fails on this step.
2. The result object does not record per-asset success.

A related behaviour was seen with the pygeofetch engine and `on_failure="skip"`: one failed band still printed
`3/3 complete` and left a partial 267 MB file behind. That part belongs upstream in pygeofetch, but pygeovision should not
rely on it.

**Suggested approach**
- Make `DownloadResult` list each requested asset with its status, and set `success=False` (or `partial=True`) when any is missing.
- Either add `rasterio` to the core dependencies or avoid importing it when no post-processing was requested.
- After each download, check that every requested band exists with a non-zero size.

**Done when** a test with one unreachable asset returns a non-success result that names the asset, and `pip install pygeovision`
followed by a plain band download works without `rasterio`.

---
**Want to take this?** Comment below so nobody duplicates the work, then open a pull request that mentions this issue.
Please read [CONTRIBUTING.md](https://github.com/EOCoreINT/pygeovision/blob/main/CONTRIBUTING.md) first. Questions are welcome here too; if you hit a problem
while reproducing, include the output of `pygeovision doctor`.


## 3. sar_flood_mapping accepts identical reference and flood windows and reports 0.0 ha without any warning

**Labels:** bug, flood, good first issue

**What happens**
A radar-only flood run for Greater Accra was given the same dates for the reference and the flood window
(4 to 16 June 2026). The report shows 0.0 ha flooded, a backscatter change of 0.000 dB at every percentile, and no warning.
Its header also prints the two windows run together (`2026-06-04 to 2026-06-16 to 2026-06-04 to 2026-06-16`), which hides the
problem. A change analysis whose two windows coincide cannot find change; it should refuse or warn.

**Suggested approach**
- In `sar_flood_mapping` and `multisensor_flood_mapping`, raise `ReferenceDataError` when the windows are identical or overlap,
  and warn when the gap is shorter than a few days.
- Put the check in one shared helper so every before/after pipeline can use it.
- Print the two windows with a clear separator in the report header (for example `reference A to B | flood C to D`).

**Done when** tests cover identical, overlapping and reversed windows, and the report header shows both windows unambiguously.

**Where to look** `pygeovision/ai/pipelines/multisensor_flood.py` (~L188-200, `date_window`), `pygeovision/reporting/specs.py` (~L642).

---
**Want to take this?** Comment below so nobody duplicates the work, then open a pull request that mentions this issue.
Please read [CONTRIBUTING.md](https://github.com/EOCoreINT/pygeovision/blob/main/CONTRIBUTING.md) first. Questions are welcome here too; if you hit a problem
while reproducing, include the output of `pygeovision doctor`.


## 4. Flood pipeline skips the terrain (HAND) stage for grids above 6 million cells

**Labels:** enhancement, flood, help wanted

**What happens**
`DrainageAnalysisPipeline` and `hand_from_dem` refuse grids larger than `max_cells=6_000_000` (`hy.check_size`). In
`multisensor_flood_mapping` the resulting `ValueError` is caught and only recorded as the note `terrain (HAND) not applied`.
In the Nepal run (5,232 km², 13.5 million cells) the terrain filter therefore did not run, and flooded pixels well above the
drainage stayed in the map. The report does say so, but it is a quiet loss of one of the pipeline's three evidence sources.

**Suggested approach** Compute HAND in overlapping tiles (with a halo for flow routing), or on a coarsened DEM and resample, or
expose `max_cells` as a flood-pipeline argument. Show "terrain not applied" in the report's summary block, not only in the notes.

**Done when** an AOI above 6 million cells gets a HAND filter, and a test shows the tiled result matches the single-grid result
within a stated tolerance on a small AOI.

**Where to look** `pygeovision/ai/pipelines/terrain.py` (~L104-107, ~L215-223), `multisensor_flood.py` (~L281-290).

---
**Want to take this?** Comment below so nobody duplicates the work, then open a pull request that mentions this issue.
Please read [CONTRIBUTING.md](https://github.com/EOCoreINT/pygeovision/blob/main/CONTRIBUTING.md) first. Questions are welcome here too; if you hit a problem
while reproducing, include the output of `pygeovision doctor`.


## 5. Flood reports: show how much the result depends on the radar water threshold

**Labels:** enhancement, flood, help wanted

**What happens**
`water_db="auto"` calibrates the radar water threshold against optical water. When no clear optical view exists, the
conventional -15 dB is used and recorded as a default. In the Nepal run the default (-15.0 dB) was used while the scene's own
Otsu threshold was -7.1 dB, 8 dB apart. The report states the choice but not how much the flooded area changes with it.

**Suggested approach** Add a small sensitivity table to the flood reports: flooded area at the used threshold, at plus and minus
3 dB, and at the Otsu value. Flag clearly when the used threshold is a fallback.

**Done when** both flood reports include the table, with a test on a synthetic scene whose answer changes with the threshold.

**Where to look** `pygeovision/ai/pipelines/multisensor_flood.py` (~L119, ~L202, ~L352), `pygeovision/reporting/specs.py` (~L642-700).

---
**Want to take this?** Comment below so nobody duplicates the work, then open a pull request that mentions this issue.
Please read [CONTRIBUTING.md](https://github.com/EOCoreINT/pygeovision/blob/main/CONTRIBUTING.md) first. Questions are welcome here too; if you hit a problem
while reproducing, include the output of `pygeovision doctor`.


## 6. Validate multisensor_flood_mapping against labelled flood data (optical-only evidence is 72% and 90% of the mapped flood)

**Labels:** validation, flood, help wanted

**Why**
In two real runs the evidence classes show that most of the mapped flood rests on optical imagery alone: 58.2 of 81.4 ha (72%)
in Accra, and 407.1 of 451.7 ha (90%) in Nepal. Whether that optical-only class is flood or false water (for example turbid or
debris-laden water, shadow) has not been measured. No accuracy figure exists for this pipeline on real ground.

**Suggested approach**
- Run the pipeline on the hand-labelled Sen1Floods11 chips (Bonafilia et al. 2020) and report IoU and F1 per evidence class.
- For the Rasuwa flash flood of 26 August 2026, compare with the high-resolution imagery published for 27 to 28 August.
- Publish a notebook and a results table in `docs/`.

**Done when** a documented evaluation exists with per-class numbers and the data/code to reproduce it.

---
**Want to take this?** Comment below so nobody duplicates the work, then open a pull request that mentions this issue.
Please read [CONTRIBUTING.md](https://github.com/EOCoreINT/pygeovision/blob/main/CONTRIBUTING.md) first. Questions are welcome here too; if you hit a problem
while reproducing, include the output of `pygeovision doctor`.


## 7. Tracking: model registry entries that are registered but cannot be built (get_model raises NotImplementedError)

**Labels:** enhancement, models, help wanted

These entries are discoverable in `pygeovision models list` but have no verified build path, so `get_model()` raises
`NotImplementedError` on purpose (better than silently building a different network). Contributions that add a verified loader
are welcome. Please pick one, say so below, and include a test and a "verified how" note.

- [ ] `dinov3_segmentor`: segmentation head on a DINOv3 backbone (`pygeovision.models.foundation.dinov3` gives features)
- [ ] `changestar-r18`: no implementation available (use changeformer, bit-r50 or dsamnet meanwhile)
- [ ] `satlas-pretrain`: no verified checkpoint or construction code
- [ ] `chatearthnet`: no verified checkpoint or construction code
- [ ] `lisat-7b`: needs the authors' `lisat_code` repository, flash-attn and a separate RemoteCLIP checkpoint
- [ ] `rf-detr-b`: distributed by Roboflow's `rfdetr` package, not transformers
- [ ] `prithvi_burn_scar`: TerraTorch/MMSegmentation artefact

**Also needed:** the documentation disagrees about how many there are. The README says five of 79 entries are placeholders,
`docs/project/roadmap.md` calls all 79 "buildable", and the descriptions in `models/registry.py` mark seven as raising
`NotImplementedError`. Please add a test that calls `get_model` on every entry and generates the counts, so the docs cannot drift.

**Where to look** `pygeovision/models/registry.py` (~L74-148, ~L323-330, ~L399).

---
**Want to take this?** Comment below so nobody duplicates the work, then open a pull request that mentions this issue.
Please read [CONTRIBUTING.md](https://github.com/EOCoreINT/pygeovision/blob/main/CONTRIBUTING.md) first. Questions are welcome here too; if you hit a problem
while reproducing, include the output of `pygeovision doctor`.


## 8. Tracking: labeling features that are not implemented (clay, satmae, Dynamic World probability composite, LabelingOrchestrator)

**Labels:** enhancement, help wanted

Four pieces of the labeling layer raise `NotImplementedError` today. Each was disabled because the earlier version returned
something misleading. A real implementation of any of them is welcome.

- [ ] `clay` and `satmae` in `ai/labeling/foundation_labeler.py`: no verified loader in the package
- [ ] `dynamic_world`: `composite_method="probability"` and `probability_bands=True` (it used to silently return the mode)
- [ ] `LabelingOrchestrator.label()` in `ai/labeling/__init__.py`: was a stub that returned `status="completed"` without labelling
      anything. Either implement it on top of `AutoLabelPipeline` or remove the class.

**Done when** each item either works with a test, or the option is removed from the API and the docs.

---
**Want to take this?** Comment below so nobody duplicates the work, then open a pull request that mentions this issue.
Please read [CONTRIBUTING.md](https://github.com/EOCoreINT/pygeovision/blob/main/CONTRIBUTING.md) first. Questions are welcome here too; if you hit a problem
while reproducing, include the output of `pygeovision doctor`.


## 9. Seven pipelines need a checkpoint you train yourself: provide training recipes or reference weights

**Labels:** enhancement, models, help wanted

`crop_monitoring`, `deforestation`, `disaster_assessment`, `landslide_detection`, `road_extraction`, `solar_detection` and
`urban_growth` refuse to run without `--checkpoint`. That is the right behaviour (an untrained network gives confident-looking
random output), but it means 7 of the 41 pipelines are unusable out of the box.

**Ways to help**
- Write an end-to-end recipe: open training data, the `TrainingConfig`, the command, and the metrics on a held-out split.
- Train one on open data and publish the weights with a model card (data, licence, metrics, limits) on the Hugging Face Hub.
- Add a loader that downloads such a published checkpoint explicitly, never silently.

**Done when** at least one of the seven runs from a bounding box with a documented, published checkpoint.

---
**Want to take this?** Comment below so nobody duplicates the work, then open a pull request that mentions this issue.
Please read [CONTRIBUTING.md](https://github.com/EOCoreINT/pygeovision/blob/main/CONTRIBUTING.md) first. Questions are welcome here too; if you hit a problem
while reproducing, include the output of `pygeovision doctor`.


## 10. DatasetLoader.download never downloads: add an opt-in download for open-licence datasets

**Labels:** enhancement, help wanted

The 460-entry dataset catalogue is metadata only; `DatasetLoader.download` explains where to get a dataset but never fetches it.
That avoids licence problems, but users must do every download by hand.

**Suggested approach** For datasets with an open licence, add an explicit `download(accept_licence=True)` that shows the licence
text and source, verifies a checksum, and resumes partial files. Datasets with restrictive terms keep the current behaviour.

**Done when** two or three open datasets can be downloaded and loaded in one call, with tests that use a local mirror.

**Where to look** `pygeovision/datasets/`, `docs/datasets/index.md`.

---
**Want to take this?** Comment below so nobody duplicates the work, then open a pull request that mentions this issue.
Please read [CONTRIBUTING.md](https://github.com/EOCoreINT/pygeovision/blob/main/CONTRIBUTING.md) first. Questions are welcome here too; if you hit a problem
while reproducing, include the output of `pygeovision doctor`.


## 11. Configuration sections gpu.* and training.* are stored but never read; about 27 of 36 exception classes are never raised

**Labels:** bug, good first issue

`docs/reference/known-issues.md` and `docs/project/roadmap.md` both state it: the `gpu` and `training` sections of the
configuration file are accepted but training and inference take device and batch size from their own arguments, and about nine
of the 36 classes in `pygeovision.core.exceptions` are used while the rest are reserved names. Settings that do nothing mislead
users.

**Choose one for each**
- Wire the setting in (and test that it changes behaviour), or
- Remove it with a deprecation warning and update the docs.

For the exceptions: delete the unused classes, or use them where a built-in is raised today.

**Done when** no documented setting is silently ignored and no unused exception class remains (or each has a test that raises it).

**Where to look** `pygeovision/core/exceptions.py`, `docs/reference/config.md`.

---
**Want to take this?** Comment below so nobody duplicates the work, then open a pull request that mentions this issue.
Please read [CONTRIBUTING.md](https://github.com/EOCoreINT/pygeovision/blob/main/CONTRIBUTING.md) first. Questions are welcome here too; if you hit a problem
while reproducing, include the output of `pygeovision doctor`.


## 12. Tracking: reintroduce removed pipelines and add missing ones with validated methods

**Labels:** enhancement, help wanted

Twenty-one of the original 49 pipelines were removed because they could not produce a correct result (untrained networks,
placeholder indices, invented constants); the reasons are in `docs/pipelines/removed.md`. SAR/InSAR processing, the inference
server, the enterprise/RBAC layer and the planner agent are also not part of the package any more. No air-quality or dust
pipeline exists because it would need aerosol data (Sentinel-5P or MODIS AOD) and an uncalibrated proxy could be read as health
guidance.

**How to contribute** Pick one removed or missing pipeline and propose it in a comment first. A new pipeline should have a
published method with a citation, a synthetic test with a known answer, at least one run on real data, a report that states its
limits, and no invented constants.

---
**Want to take this?** Comment below so nobody duplicates the work, then open a pull request that mentions this issue.
Please read [CONTRIBUTING.md](https://github.com/EOCoreINT/pygeovision/blob/main/CONTRIBUTING.md) first. Questions are welcome here too; if you hit a problem
while reproducing, include the output of `pygeovision doctor`.


## 13. Turn the real-data runs into repeatable regression tests (the pytest suite itself is synthetic or mocked)

**Labels:** testing, help wanted

**What exists**
- The pytest suite (159 files, 2,271 `def test_` functions by static count) uses synthetic rasters or mocked clients. 75 of the 159
  files mention synthetic data, and several docstrings say so (`tests/test_real_run_regressions.py`: "synthetic tiles and a fake
  download client (no network)"; `tests/test_obia_realdata.py`: "behaviour on real-data conditions" built from `block_scene`).
- The only test that calls live providers, `tests/test_live.py`, is a stand-alone script excluded from pytest (`conftest.py`,
  `addopts = --ignore=tests/test_live.py`). Its docstring still says "208 tests".
- Real-data runs do exist, but they are results, not tests: the 14 report runs, the Accra and Nepal flood runs, the Piura and
  San Donà notebooks, and the live Sentinel-2 OBIA notebook for Obuasi. `docs/project/roadmap.md` ("No test ran against live
  imagery") is out of date on this point and should say what was run and where the outputs are.

**Suggested approach**
- Mark network tests (`@pytest.mark.integration`) and run them in a scheduled CI job, never on every push.
- Use credential-free sources (Planetary Computer, AWS Earth Search) and small, fixed boxes and dates; start with the runs above.
- Assert on invariants (valid fraction, class totals sum to the area, output files exist), not on exact numbers.
- Update the roadmap and README wording to match, and refresh the `test_live.py` docstring.

**Done when** a nightly workflow runs at least five pipelines on real scenes and fails loudly when a provider changes.

**Where to look** `.github/workflows/ci.yml`, `tests/test_live.py`, `tests/test_real_run_regressions.py`, `conftest.py`.

---
**Want to take this?** Comment below so nobody duplicates the work, then open a pull request that mentions this issue.
Please read [CONTRIBUTING.md](https://github.com/EOCoreINT/pygeovision/blob/main/CONTRIBUTING.md) first. Questions are welcome here too; if you hit a problem
while reproducing, include the output of `pygeovision doctor`.


## 14. Verification matrix: pretrained weights, GPU and cloud paths have never been run (help wanted from people with a GPU or cloud account)

**Labels:** testing, help wanted

The documentation environment had no network, GPU or cloud account, so these were only tested with tiny random local models.
If you have the hardware, please run one and report back with a short note ("Verified here?" line in its docs page).

- [ ] Real DINOv2 weights
- [ ] CLIP / RemoteCLIP
- [ ] Moondream
- [ ] DOFA
- [ ] Prithvi
- [ ] SAM
- [ ] LiDAR classification accuracy (only the plumbing is tested; there is no pretrained point-cloud network)
- [ ] Ray Tune
- [ ] MLflow, Weights & Biases, TensorBoard logging
- [ ] Mixed precision, DDP, TensorRT
- [ ] AWS, Azure and GCP paths

**Done when** each ticked item has the output of a real run attached and its docs page updated.

---
**Want to take this?** Comment below so nobody duplicates the work, then open a pull request that mentions this issue.
Please read [CONTRIBUTING.md](https://github.com/EOCoreINT/pygeovision/blob/main/CONTRIBUTING.md) first. Questions are welcome here too; if you hit a problem
while reproducing, include the output of `pygeovision doctor`.


## 15. No pipeline has a published accuracy on real reference data

**Labels:** validation, documentation, help wanted

`docs/reference/known-issues.md`: methods are documented and checked on scenes with known answers; no pipeline has an accuracy
figure on real ground, and none is claimed. That is honest, but it leaves users unable to judge the tools.

**Suggested approach** For each pipeline that has an open reference (for example building footprints against a hand-digitised
sample, water against a labelled water dataset, land cover against reference points), write a short evaluation: data, code,
metrics with confidence intervals, and the limits. Note that ESA WorldCover is itself a model output, so agreement with it is not
accuracy. See `docs/data/validation.md`.

**Done when** at least three pipelines have an evaluation page with reproducible code.

---
**Want to take this?** Comment below so nobody duplicates the work, then open a pull request that mentions this issue.
Please read [CONTRIBUTING.md](https://github.com/EOCoreINT/pygeovision/blob/main/CONTRIBUTING.md) first. Questions are welcome here too; if you hit a problem
while reproducing, include the output of `pygeovision doctor`.


## 16. Reproducible acquisition and flood benchmark against EODAG and Google Earth Engine

**Labels:** benchmark, help wanted

**First result** (2 Nov 2026, one consumer connection, three Sentinel-2 L2A scenes of Accra, bands B03 and B11 from Planetary
Computer, tools alternated, one process per run): pygeovision and EODAG each moved 488.6 MB; times were 477 to 528 s for
pygeovision against 489 to 502 s for EODAG, i.e. no measurable difference. Fetching whole products instead of bands moved
2,305.5 MB (4.7 times more) in a single EODAG run. The three pygeovision runs did not all use the same code path (see the
search-cache issue), so a clean repeat is needed.

**To do**
- Repeat with `use_cache=False`, five runs per tool, on several connections and at different times.
- Add the flood arm: `multisensor_flood_mapping` against the UN-SPIDER recommended practice in Earth Engine (needs
  `earthengine authenticate`), same box and dates.
- Commit the scripts under `benchmarks/` with the raw results and a README describing the protocol.

**Done when** the scripts run end to end from the README and the results table is reproducible.

---
**Want to take this?** Comment below so nobody duplicates the work, then open a pull request that mentions this issue.
Please read [CONTRIBUTING.md](https://github.com/EOCoreINT/pygeovision/blob/main/CONTRIBUTING.md) first. Questions are welcome here too; if you hit a problem
while reproducing, include the output of `pygeovision doctor`.


## 17. OBIA runs entirely in memory: add tiled or out-of-core segmentation for large scenes

**Labels:** enhancement, performance, help wanted

`docs/reference/known-issues.md`: everything in OBIA runs in memory on one machine, and scenes of a few thousand pixels on a side
are comfortable. Larger scenes need tiling with consistent object boundaries across tile edges.

**Suggested approach** Segment in overlapping tiles and merge objects that cross tile borders, or process with Dask. Keep the
current single-tile path as the reference for tests.

**Done when** a scene of at least 20,000 x 20,000 pixels segments within a stated memory limit, and the merged result matches the
single-tile result on a small scene.

**Where to look** `pygeovision/obia/`, `docs/obia/segmentation.md`.

---
**Want to take this?** Comment below so nobody duplicates the work, then open a pull request that mentions this issue.
Please read [CONTRIBUTING.md](https://github.com/EOCoreINT/pygeovision/blob/main/CONTRIBUTING.md) first. Questions are welcome here too; if you hit a problem
while reproducing, include the output of `pygeovision doctor`.


## 18. Version, DOI, URL and count inconsistencies across pyproject, CITATION.cff, README and docs

**Labels:** documentation, packaging, good first issue

A user or a citation manager gets different answers depending on which file they read. Found on 2 Nov 2026:

- `pyproject.toml` `[project.urls]` point to `github.com/pygeovision/pygeovision` and `docs.pygeovision.org`; the repository is
  `EOCoreINT/pygeovision`. `CITATION.cff` uses `pygeovision.readthedocs.io`.
- `CITATION.cff`: `doi: 10.5281/zenodo.XXXXXXX` (a placeholder), `date-released: 2026-07-10`.
- The package version is 2.2.0; the README citation block names release 2.1.9 with its own DOI; `docs/project/roadmap.md` says
  "this release, 2.1.8".
- Architecture counts: the README badge and text say 77 registered architectures, the registry has 79 entries, the README says
  five of 79 are placeholders, and the roadmap says "79 buildable architectures" (see the registry tracking issue).
- Test counts: the README says "about 1,300 tests"; a static count finds 2,271 `def test_` functions in 159 files
  (parametrisation and skips may explain part of it). State how the figure is counted.

**Suggested approach** Fix each, then add a small script in `tools/` that generates the counts from the code and a CI check that
fails when the docs disagree.

**Done when** `pyproject.toml`, `CITATION.cff`, README and roadmap agree and the CI check passes.

---
**Want to take this?** Comment below so nobody duplicates the work, then open a pull request that mentions this issue.
Please read [CONTRIBUTING.md](https://github.com/EOCoreINT/pygeovision/blob/main/CONTRIBUTING.md) first. Questions are welcome here too; if you hit a problem
while reproducing, include the output of `pygeovision doctor`.


## 19. Package metadata describes features that were removed (FastAPI serving, SAR/InSAR, GeoAgent)

**Labels:** documentation, packaging, good first issue

The README says the inference server, the enterprise/RBAC layer, SAR/InSAR processing and the planner agent are not part of the
package. Other files still advertise them:

- `pyproject.toml` description: "foundation models, FastAPI serving, and end-to-end pipelines" and the keyword `SAR`.
- `CITATION.cff` abstract: "SAR and optical preprocessing", "SAR interferometry (InSAR)" and "A natural-language query interface
  (GeoAgent)", and the keyword `InSAR`.

This text is what PyPI and citation tools show. Rewrite it to describe what ships: the 41 pipelines, the model registry, OBIA,
the reporting layer, and acquisition through pygeofetch.

**Done when** no metadata file mentions a removed feature, and a grep for `GeoAgent`, `InSAR` and `FastAPI serving` finds only the
roadmap/history pages.

---
**Want to take this?** Comment below so nobody duplicates the work, then open a pull request that mentions this issue.
Please read [CONTRIBUTING.md](https://github.com/EOCoreINT/pygeovision/blob/main/CONTRIBUTING.md) first. Questions are welcome here too; if you hit a problem
while reproducing, include the output of `pygeovision doctor`.


## 20. Tidy the repository root (example outputs, overlapping guides, layout)

**Labels:** documentation, packaging, good first issue

The root of the repository holds example output folders (`accra_buildings` 2.1 MB, `alps_winter_sun` 13 MB, `rondonia_forest`
2.4 MB), an empty `_max/`, `patches/`, `book/` and `paper/`, and four overlapping start guides (`START_HERE.md`, `QUICKSTART.md`,
`CONTINUATION_GUIDE.md`, `CLI_TESTING_GUIDE.md`), and `tests/` contains a stray copy named `test_data_layer (1).py`. The package sits at the top level rather than under `src/`. This makes the
project harder to approach for a new contributor and bloats clones.

**Suggested approach**
- Move example outputs to `examples/` or to release assets; delete the empty folder.
- Merge the start guides into `docs/` and leave one pointer in the README.
- Consider a `src/` layout (check that packaging and the docs build still work) and note the licence file name your venue expects
  (`LICENSE` today; some journals ask for `LICENSE.txt`).

**Done when** the root contains only code, docs, configuration and one README, and CI still passes.

---
**Want to take this?** Comment below so nobody duplicates the work, then open a pull request that mentions this issue.
Please read [CONTRIBUTING.md](https://github.com/EOCoreINT/pygeovision/blob/main/CONTRIBUTING.md) first. Questions are welcome here too; if you hit a problem
while reproducing, include the output of `pygeovision doctor`.


## 21. pygeovision/models/weights is missing from the GitHub repository: .gitignore excludes it (`weights/`)

**Labels:** bug, packaging, good first issue

**What happens**
`.gitignore` line 108 contains `weights/`, which also matches the source package `pygeovision/models/weights/`
(`__init__.py`, `downloader.py`, `manager.py`). `git ls-files pygeovision/models/weights` lists nothing, and
`git check-ignore -v pygeovision/models/weights/downloader.py` reports the rule. The PyPI wheel for 2.2.0 does contain the folder.

`pygeovision models download <name>` imports `pygeovision.models.weights.downloader` (`cli/main.py` ~L1392), so a checkout from
GitHub (`git clone`, `pip install git+https://...`) fails with `ModuleNotFoundError` on that command, while a PyPI install works.
The source and the published package differ without anyone noticing.

**Suggested approach** Anchor the ignore rule to the repository root (`/weights/`) or add `!pygeovision/models/weights/`, commit the
three files, and add a CI step that builds the wheel and a clean checkout and compares their file lists.

**Done when** the folder is tracked, `pygeovision models download` works from a fresh clone, and the CI comparison passes.

---
**Want to take this?** Comment below so nobody duplicates the work, then open a pull request that mentions this issue.
Please read [CONTRIBUTING.md](https://github.com/EOCoreINT/pygeovision/blob/main/CONTRIBUTING.md) first. Questions are welcome here too; if you hit a problem
while reproducing, include the output of `pygeovision doctor`.


## 22. OBIA run on a live Sentinel-2 scene reports 0 ha for every class (scene is in EPSG:4326, areas are in square degrees)

**Labels:** bug, good first issue

**What happens** In `notebooks/obuasi_mining_landcover_obia_pygeovision.ipynb` (live Sentinel-2 scene, 499 x 499 px, no-data
44.3%, pygeovision 2.1.9) `obia.fetch_scene` returns a raster in `EPSG:4326`. The notebook then computes hectares from
`F["area_m2"] / 1e4`, but `area_m2` is in map units squared (see `docs/reference/known-issues.md`), here square degrees. The
saved `notebooks/obuasi_obia_outputs/area_by_class.csv` shows 0.0 ha for all five classes, the no-data line prints "= 0 ha", and
`mine_site.geojson` has 0 parts. The same run reports that the OSM mine outlines labelled 0 objects as `mine_disturbed`, so the
class has only 15 objects from the classifier; please check whether the outline file was found and reprojected.

**Suggested approach**
- Reproject the scene to a local UTM grid in `fetch_scene` (or an explicit helper) so areas are in metres, or have the feature
  extraction raise or warn when `area_m2` is requested in a geographic CRS.
- Fix the notebook and re-run it, and add a test that a geographic-CRS scene gives areas within 1% of the geodesic area.
- Investigate why the OSM outlines labelled no objects.

**Done when** the notebook's `area_by_class.csv` has plausible hectares for the 5 km x 5 km window (about 2,500 ha in total) and
the test passes.

---
**Want to take this?** Comment below so nobody duplicates the work, then open a pull request that mentions this issue.
Please read [CONTRIBUTING.md](https://github.com/EOCoreINT/pygeovision/blob/main/CONTRIBUTING.md) first. Questions are welcome here too; if you hit a problem
while reproducing, include the output of `pygeovision doctor`.


