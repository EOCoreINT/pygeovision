# Contributing

PyGeoVision's value is that its outputs can be trusted, so the rules below are about **evidence**: a feature is finished when its
behaviour has been run on data with a known answer, documented, and covered by a test that fails when it breaks.

## Set up

```bash
git clone https://github.com/<org>/pygeovision.git
cd pygeovision
python -m venv .venv && source .venv/bin/activate
pip install -e ".[all,dev]"            # or a smaller extra, e.g. ".[geo,dev]"
PYTHONPATH=. pytest -q                 # the test-suite uses synthetic data; no network, GPU or credentials
```

Optional packages (Optuna, laspy, transformers …) are skipped by tests that need them, so a minimal install runs a subset.

## The standard

Every fix in this project's history followed the same loop:

1. **Read the code, not the docstring.** Then run it on data whose answer you know (a synthetic raster with one square of water, a
   point cloud with a 12 m tree).
2. **Write the test first**, and check that it **fails** when the fix is removed (mutation check). A test that passes either way proves nothing.
3. **Fail loudly.** A function that cannot do its job returns an error naming the cause, or raises. It never returns a plausible empty
   result, and never substitutes a different model, band or date without saying so.
4. **Document the behaviour that surprised you**, including what the tool does *not* do.

## Add a pipeline

A pipeline is a class in `pygeovision/ai/pipelines/` registered by name. Use an existing one as the template, for example
`SnowCoverPipeline` in `domains.py` (single date) or `GlacierMonitoringPipeline` (two dates).

```python
class SnowCoverPipeline(BasePipeline):
    """Snow/ice extent from NDSI = (green - SWIR1) / (green + SWIR1) > 0.4. ... (method, limits, outputs)"""
    name = "snow_cover"
    description = "Snow/ice extent from NDSI > 0.4 on one date"
    domain = "cryosphere"
    tags = ["snow", "ndsi"]

    def run(self, bbox, output_dir="./output", date="2024-01", **kwargs) -> PipelineResult:
        t0 = time.time()
        ...
        img = self._search_and_download(bbox, date, out, providers=kwargs.get("providers"), bands=("green", "swir1"))
        if img is None:
            return self._fail("No usable imagery found.", t0)
        ...
        return self._ok(path, stats, {"date": date}, t0)
```

Checklist:

- [ ] **Return, do not raise**: `self._fail("specific reason", t0)` for everything that can go wrong with data; `self._ok(path, stats, metadata, t0)` otherwise.
- [ ] **Nodata is nodata.** Cloud, shadow and outside-imagery pixels are 255 in class rasters and NaN in continuous ones, excluded from every statistic, and
  reported as `valid_fraction`. Use the helpers (`write_classes`, `write_float`, `valid_mask`, `area_ha`, `fraction`).
- [ ] **Neural network?** Set `kind = "needs_weights"` and obtain the network only through `self._load_trained_model(model_name, checkpoint, num_classes, in_channels)`,
  which raises `ModelWeightsRequired` without a checkpoint and loads strictly. Call it **before** any download so a missing `checkpoint` fails at once.
  Never return a network with random weights as an answer.
- [ ] **A docstring that states the method, the limits and the outputs.** The pipeline page is generated from it.
- [ ] **Register** it in the module's dictionary (`DOMAIN_PIPELINES`, `REFERENCE_PIPELINES`, …).
- [ ] **A report spec** in `pygeovision/reporting/specs.py`: layers with class colours, metrics, "how to read", limitations, references. Without one, the run still works
  but writes no map or report.
- [ ] **A test with a known answer** in `tests/` (see `tests/test_pipeline_algorithms.py`): synthetic rasters via `tests/pipeline_synth.py`, an expected number computed by hand.
- [ ] **Docs.** Add notes to `docs/_data/pipeline_notes.yaml` (question, when to use, when not, how to read, what to do when), an entry in `tools/offline_pipeline_smoke.py`
  so an example run can be captured, then regenerate:

  ```bash
  PYTHONPATH=. python tools/capture_pipeline_examples.py     # offline example run -> docs/_data/pipeline_examples.json
  PYTHONPATH=. python tools/generate_pipeline_pages.py       # docs/pipelines/catalog/<name>.md and the index
  PYTHONPATH=. python tools/generate_pipeline_docs.py        # docs/pipelines/removed.md
  ```

`tests/test_pipeline_docs.py` and `tests/test_reference_docs.py` fail if a pipeline has no page or a generated page is stale.

## Add a model to the registry

- A real `hf_id` / `timm_id` that **you** have confirmed is the model you claim, not a same-family model or another generation.
- If no verified build path exists, register the entry with accurate metadata and a `NotImplementedError` that says exactly what is missing (see `lisat-7b`).
- Failed pretrained loads must raise `ModelLoadError`; no silent fallback. Random weights only with `allow_random_init=True`.
- Regenerate the catalogue: `PYTHONPATH=. python tools/generate_reference_docs.py`.

## Add a CLI command

- Give the command a docstring with a runnable example; it appears in the generated reference.
- Errors are `click.ClickException` with a message naming the cause, and a non-zero exit code.
- Add a test with `click.testing.CliRunner`, including the failure path.
- Add or extend the group's notes in `docs/_data/cli_notes.yaml` (`intro`, `when`, `expect`, `watch`, `examples`), then
  `PYTHONPATH=. python tools/generate_reference_docs.py`. **Never edit `docs/reference/cli/*.md` by hand**; it is overwritten.

## Write documentation

Each feature page answers four questions: **why** use it, **how**, **what output to expect**, **what to do when** it misbehaves.

- **Only real output.** Paste what the code printed. If you could not run something (no GPU, no network, a paid account), say so on the page.
- **Say what it is not.** Every tool page has a "not for" line.
- **Examples run.** Put scripts under the section's `_examples/` folder and include the captured output.
- Use `:::` fences, not backtick fences, when a directive title contains backticks (a backtick fence silently swallows the page).

Build exactly as Read the Docs does (warnings are errors):

```bash
pip install -r docs/requirements.txt
make -C docs html SPHINXOPTS="-W --keep-going"
```

Markdown links between pages are relative file links (`../data/index.md`), so they are checked by the build and by
`tests/test_reference_docs.py::test_relative_links_point_at_existing_files`.

## Before you open a pull request

```bash
PYTHONPATH=. pytest -q tests/test_<your_area>*.py      # then the whole suite
ruff check <the files you changed>
PYTHONPATH=. python tools/generate_reference_docs.py --check
PYTHONPATH=. python tools/generate_pipeline_pages.py --check
make -C docs html SPHINXOPTS="-W --keep-going"
```

Describe in the pull request what you ran and what you could **not** run.

## Good first issues

- `pygeovision/training/callbacks.py` is imported by nothing; wire it into the trainer or delete it ([roadmap](roadmap.md)).
- Merge another [duplicate pair](../architecture.md#systems-that-exist-twice) by re-export, as was done for the trainer.
- Bring back a [removed pipeline](../pipelines/removed.md): each entry says what it needs, for example a trained road-segmentation model or aerosol data.
- Move the `PyGeoVision` client out of `pygeovision/__init__.py`.

## Licence

See `LICENSE` in the repository root.
