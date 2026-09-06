# Dataset Catalog

`pygeovision.datasets.registry.dataset_registry` is a real, 503-entry
discovery catalog of published remote-sensing benchmark datasets
(UC-Merced, EuroSAT, BigEarthNet, DOTA, and hundreds more) — for
finding out what real datasets exist for a task, not for automated
downloading.

```bash
pygeovision datasets top segmentation --n 5
pygeovision datasets similar EuroSAT
```

```python
from pygeovision.datasets.registry import dataset_registry

top5 = dataset_registry.top_for_task("segmentation", n=5)
similar = dataset_registry.similar_to("EuroSAT", n=10)
```

## A real, confirmed bug found and fixed while writing this page

Only 5 of the 503 entries have a real `download_url` populated at
all. That in itself isn't a bug — `dataset_registry` has no
`download()` method; it's honestly a pure discovery catalog. But a
separate class, `DatasetLoader`, has a real, CLI-reachable `download()`
method (`pygeovision datasets download <name>`) whose own docstring
claims to "download and extract a dataset by name."

Checking what it actually did: for the 498 entries with no
`download_url`, it correctly raised a clear error. But for the 5
entries that *do* have one, it created an **empty output directory**,
printed the URL and expected size, and returned the directory path —
with no real file-fetching code anywhere. It looked like a successful
download (a real path returned, no error raised) while downloading
nothing.

Checking the real URLs explained why no real download was ever
implemented: all 5 (`EuroSAT`, `BigEarthNet`, `DOTA`, `LEVIR-CD`,
`CropHarvest`) point to human-facing project pages —
`https://github.com/phelber/EuroSAT`, `https://bigearth.net` — not
direct file links. A real HTTP request against these would fetch HTML,
not the dataset, which would have been a worse bug than the one it
replaced: a real-looking file that's actually the wrong content
entirely.

`DatasetLoader.download()` now raises clearly for every real dataset
name, pointing to the real URL to visit, rather than creating a
misleading empty directory:

```python
from pygeovision.datasets.loader import DatasetLoader

DatasetLoader().download("EuroSAT")
# ValueError: 'EuroSAT' has no real, automated download path -- its real
# download_url is a human-facing project page, not a direct file link.
# Visit 'https://github.com/phelber/EuroSAT' directly to obtain it.
```

```{note}
This is a genuine capability gap, not just a bug fix — none of these
503 real, published datasets can currently be fetched automatically
through `pygeovision`. See [Roadmap](../reference/roadmap.md).
```

## Benchmark configuration

`pygeovision.datasets.benchmark` builds and saves real benchmark
configurations (which datasets to use for which task) as JSON files —
this part does write real files, verified by checking that it calls a
real `cfg.save(path)` for each one rather than just creating an empty
directory.

```python
from pygeovision.datasets.benchmark import BenchmarkBuilder

BenchmarkBuilder().save_all(output_dir="./benchmarks", n=5)
```
