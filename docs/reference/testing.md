# Testing

PyGeoVision's tests check behaviour against **known answers**, not against the code's own output: a raster with one square of water,
a point cloud with a 12 m tree, a scene with a fixed number of burned pixels. They need no network, GPU or credentials.

## Run the tests

```bash
pip install -e ".[all,dev]"            # or a smaller extra, e.g. ".[geo,dev]"
PYTHONPATH=. pytest -q                 # about 2,100 tests; the full run takes several minutes
PYTHONPATH=. pytest tests/test_obia.py -q          # one file
PYTHONPATH=. pytest -m "not slow" -q               # skip the slow ones
```

Tests that need an optional package (Optuna, laspy, transformers, scikit-image …) skip cleanly when it is missing, so a minimal
install runs a subset.

| Marker | Meaning |
|---|---|
| `slow` | Takes noticeably longer; deselect with `-m "not slow"` |
| `gpu` | Needs a GPU |
| `integration` | Needs a network |
| `live` | Needs live pygeofetch credentials |

`tests/test_live.py` is a separate, real end-to-end check against live providers. It is excluded from the normal run; start it by hand
(`python tests/test_live.py --bbox … --providers planetary_computer`) when you want to confirm that the search and download path works on
your connection.

## What the tests cover

| Area | Examples |
|---|---|
| Pipelines | Every pipeline on a synthetic scene with a known answer; refusal without a checkpoint; unknown arguments are rejected |
| Data layer | Search, caching, download, cloud masking, radiometry, band derivation |
| OBIA | Segmentation, features, classifiers, accuracy measures, the CLI, and the real-data fetch path with the network replaced |
| Training | Losses, trainer fixes, resume, nodata labels, metrics, tracking |
| Inference | Tiled prediction, blending, nodata, ensembles, streaming |
| Models | Every builder in the registry, foundation models, vision-language models |
| Reporting | PDF, Word, Excel, maps and vectors |
| Documentation | The generated pages must match the installed package (below) |

## Documentation checks

Three pages are generated from the code, so they cannot drift without a test failing.

| Generated page | Generator | Test |
|---|---|---|
| `docs/reference/cli/*.md` (one per command group) | `tools/generate_reference_docs.py` | `tests/test_reference_docs.py` |
| `docs/pipelines/catalog/*.md` (one per pipeline) | `tools/generate_pipeline_pages.py` | `tests/test_pipeline_docs.py` |
| `docs/pipelines/removed.md` | `tools/generate_pipeline_docs.py` | `tests/test_pipeline_docs.py` |

After you add a pipeline or a command, regenerate and commit:

```bash
PYTHONPATH=. python tools/generate_reference_docs.py
PYTHONPATH=. python tools/generate_pipeline_pages.py
PYTHONPATH=. python tools/generate_pipeline_docs.py
```

## Build the documentation

```bash
pip install -r docs/requirements.txt
make -C docs html                       # English plus every translation, strict: a warning fails the build
DOCS_SKIP_OTHER_LANGS=1 make -C docs html    # English only, quicker while you write
python docs/tools/check_translations.py      # translation files are well formed and cover the navigation bar
```

The build uses `-W`, so a broken link, a bad cross-reference or a malformed table fails it. Read the Docs runs the same build.

## The standard for a new test

1. Write the test first and check that it **fails** when the fix is removed. A test that passes either way proves nothing.
2. Use data whose answer you know, not the function's own output.
3. Make failures loud: a function that cannot do its job raises or returns an error naming the cause.

See [Contributing](../project/contributing.md) for the full loop.
