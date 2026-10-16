# Dataset catalogue

**Package:** `pygeovision.datasets` · **Classes:** `DatasetRegistry` (singleton `dataset_registry`), `DatasetInfo`, `DatasetAnalyzer`,
`BenchmarkBuilder`, `BenchmarkConfig`, `DatasetLoader` · **CLI:** `pygeovision datasets list | search | info | top | similar | domains | stats | download`

## Why

Before you train a model you ask: *which public datasets exist for this task, at this resolution, from this sensor, and
how big are they?* Answering that means reading dozens of papers. The catalogue puts one line of **metadata** for about 460
remote-sensing datasets behind `search`, `filter` and a few ranking helpers, so you can shortlist candidates in a minute,
then go to the dataset's own page for the real details.

```{admonition} It is a catalogue of metadata, not a data store
:class: warning
- **Nothing is downloaded.** `datasets download` only tells you where to look (below).
- **The numbers are approximate and hand-entered** (year, sample count, class count, resolution, size). Use them to
  shortlist, then confirm against the dataset's paper or page before you rely on a figure.
- **Licences are not recorded.** Every entry's `license` is `"unknown"`. Check the dataset's own terms before using it,
  especially commercially.
- **Only a handful of entries have links**: 5 of 460 have a `download_url` and 7 a `paper_url`. For the rest, search the web for the name and year.
- **The catalogue is not exhaustive and not an endorsement.** A dataset being absent says nothing about it.
```

## How to use

### Search and filter

```python
from pygeovision.datasets import dataset_registry as reg

len(reg)                                     # 460
reg.search("flood")                          # name, description, domain, modality, tasks, tags; case-insensitive
reg.search("sentinel", field="description")  # restrict where to look: name | description | domain | modality | tasks
reg.filter(task="change_detection", modality="sar")
reg.filter(domain="urban", max_resolution_m=0.5, min_year=2020, max_volume_gb=50)
reg["LoveDA"]                                # a DatasetInfo; KeyError with a "did you mean" hint if the name is off
```

`filter` takes any combination of `domain`, `modality`, `task`, `min_year`, `max_year`, `min_samples`, `max_volume_gb`,
`min_resolution_m`, `max_resolution_m`, `n_classes_min`; **all** given conditions must hold. The resolution bounds are on the ground
sampling distance, so `max_resolution_m=0.5` means "pixels of 0.5 m or finer". `0` is a real bound (`max_volume_gb=0` returns nothing),
not "no filter".

```console
$ pygeovision datasets search flood --limit 5

Found 5 dataset(s) matching 'flood':

Name                           Domain          Modality         Year    Samples   Res(m)  Vol(GB)
────────────────────────────────────────────────────────────────────────────────────────────────────
CEMS Flood                     disaster        multimodal       2023      1,000     10.0     20.0
FLOODNET-VQA                   disaster        rgb              2022      2,343      0.1      2.1
FloodNet                       disaster        rgb              2021      2,343      0.1      1.9
FloodSat                       disaster        sar              2022     12,000     10.0      4.1
GloFAS-Flood                   disaster        multispectral    2023     30,000    250.0     45.0

5 shown; the catalogue holds 460 datasets
```

```console
$ pygeovision datasets list --task change_detection --modality sar --limit 4
$ pygeovision datasets info EuroSAT
$ pygeovision datasets domains
$ pygeovision datasets stats
```

`datasets info` prints every recorded field:

```text
  name              : EuroSAT
  domain            : land_cover
  year              : 2019
  n_samples         : 27000
  sample_size       : 64×64
  n_classes         : 10
  modality          : multispectral
  resolution_m      : 10.0
  volume_gb         : 2.8
  tasks             : ['classification']
  description       : Sentinel-2 13-band land cover
  download_url      : https://github.com/phelber/EuroSAT
  paper_url         : https://arxiv.org/abs/1709.00029
  license           : unknown
```

(Empty fields are not printed.)

### What the fields mean

| Field | Meaning | Caveat |
|---|---|---|
| `domain` | Research area (28 values in use, e.g. `urban`, `land_cover`, `disaster`, `climate`) | Free-form: values such as `sar`, `vlm`, `3d` and `foundation` appear as domains too, and the `DOMAINS` constant in the source lists only some. Use `reg.domains()` for what is actually there |
| `year` | Release year | Approximate |
| `n_samples` | Number of images, patches or scenes | For **point-cloud** sets it counts points (SensatUrban: 3 billion), for others it may count annotations. It is not comparable across kinds |
| `sample_size` | Typical size, as text | `"varies"` is common |
| `n_classes` | Number of classes | `0` for 28 entries (pretraining corpora and similar without a label set) |
| `modality` | `rgb`, `multispectral`, `sar`, `sar_optical`, `multimodal`, `lidar`, `point_cloud`, ... (15 values; `reg.modalities()`) | |
| `resolution_m` | Ground sampling distance in metres | A single number for datasets that mix resolutions |
| `volume_gb` | Download size | Approximate |
| `tasks` | `classification`, `segmentation`, `detection`, `change_detection`, `regression`, ... (23 values; `reg.tasks()`) | A dataset can have several |
| `download_url`, `paper_url` | Links | Mostly empty |
| `license` | | Always `"unknown"` for now |

### How the ranking works

`top_for_task(task, n, modality=None)` and the commands `datasets top`, `benchmark datasets` order the datasets of a task by

```
0.30 · recency + 0.35 · scale + 0.20 · class diversity + 0.15 · pixel fineness
   recency   = (year − 2000) / 24
   scale     = min(1, n_samples / 100 000)
   diversity = min(1, n_classes / 50)
   fineness  = 1 / (1 + resolution_m)
```

```console
$ pygeovision datasets top segmentation --n 3

Name                           Domain          Modality         Year    Samples   Res(m)  Vol(GB)
────────────────────────────────────────────────────────────────────────────────────────────────────
TUM-Urban                      urban           point_cloud      2023  1,000,000      0.1      8.0
3D-RomaSet                     3d              lidar            2023 120,000,000      0.1      8.4
SensatUrban                    urban           point_cloud      2021 3,000,000,000      0.0     23.0
```

It is a **metadata heuristic, not a quality ranking**. It has no notion of how widely a dataset is used or how reliable its
labels are. In the real output above, the top "segmentation" datasets are 3-D point clouds, and well-known benchmarks sit
lower: out of 218 segmentation datasets LoveDA ranks 57th and ISPRS Potsdam 166th; in change detection LEVIR-CD is 18th.
Use `--modality rgb` (or `multispectral`, `sar`, ...) to compare like with like, and treat the result as a reading list.

### Similar datasets

```console
$ pygeovision datasets similar EuroSAT --n 4
Name                           Domain          Modality         Year    Samples   Res(m)  Vol(GB)
COPER-BENCH                    land_cover      multispectral    2023     25,000     10.0     18.0
Copernicus-Bench               land_cover      multispectral    2024     50,000     10.0     12.0
GeoMultiSens                   land_cover      multimodal       2022     30,000     10.0     50.0
PlanetSCOPE-LC                 land_cover      multispectral    2022     80,000      3.0     25.0
```

`similar_to(name, n)` scores every other dataset by `0.35·same domain + 0.2·same modality (0.5 if not) + 0.2·resolution
closeness + 0.15·task overlap + 0.1·size ratio`. It is useful for "what else is like this?", for example to find a second
dataset to test transfer on.

## Analysis helpers

`DatasetAnalyzer` summarises the catalogue itself:

```python
from pygeovision.datasets import DatasetAnalyzer, dataset_registry
a = DatasetAnalyzer(dataset_registry)

a.resolution_distribution()      # counts per GSD bin; returns {label: count}
a.modality_distribution()
a.domain_distribution()
a.volume_trend()                 # cumulative TB by release year
a.correlation_matrix(["EuroSAT", "BigEarthNet", "LoveDA"])
a.radar_chart(["EuroSAT", "LoveDA", "DOTA"], save_path="radar.png")
a.cluster("tsne", save_path="map.png")        # needs scikit-learn (or umap-learn for "umap")
a.full_report()                  # dict with all of the above plus the top-5 for four tasks
```

```text
Spatial resolution distribution:
       <0.1m: █████████████ (40)
    0.1-0.5m: █████████████████████████████ (92)
      0.5-1m: █████████████████████████████ (91)
        1-5m: ██████████████████ (56)
       5-10m: █ (4)
      10-30m: ████████████████████████████████████████ (126)
     30-100m: █████████ (28)
       >100m: ███████ (23)

EarthNets Similarity Matrix (top-3 datasets):
                      EuroSAT  BigEarthNet       LoveDA
         EuroSAT         1.00         0.75         0.50
     BigEarthNet         0.75         1.00         0.47
          LoveDA         0.50         0.47         1.00

Radar comparison: ['EuroSAT', 'LoveDA', 'DOTA']
  Dimension            EuroSAT        LoveDA          DOTA
  Resolution              0.70          0.97          0.99
  Scale                   0.47          0.40          0.36
  Diversity               0.20          0.14          0.30
  Recency                 0.79          0.92          0.75
  Volume                  0.12          0.13          0.17
  Access                  1.00          0.00          1.00
```

Read these with the metadata caveats in mind:

- Bars are scaled so the largest count is 40 characters; the numbers in brackets are the counts.
- `correlation_matrix` without `names` covers only the **first 30 catalogue entries**, which is rarely what you want.
- The `Access` axis of the radar chart is 1 only when a `download_url` is recorded (5 datasets), so it says almost nothing.
- `cluster` encodes `domain` and `modality` as their alphabetical index before the embedding, so the picture is a loose
  overview, not a clustering result. `radar_chart` returns the matplotlib figure; close it with `plt.close(fig)` in loops.
- `volume_trend` is dominated by a few huge pretraining corpora (the catalogue totals 139 TB, most of it 2023-2024 foundation datasets).

## Benchmark configurations

`BenchmarkBuilder` turns the ranking into a small JSON file you can commit next to your experiments:

```python
from pygeovision.datasets import BenchmarkBuilder

b = BenchmarkBuilder()
cfg = b.build("change_detection", n=5)     # BenchmarkConfig
cfg.dataset_names                          # ['SYSU-CD', 'ChangeStar-v2', 'CDD', 'SARS-COVID', 'SkySat-TS']
b.save_all("./benchmarks", n=5)            # one benchmark_<task>.json per task that has datasets
b.print_all(n=3)                           # the top-3 table for nine tasks
b.recommended_for_paper("change_detection")   # names, years, sizes and links as a dict
```

```json
{
  "task": "change_detection",
  "dataset_names": ["SYSU-CD", "ChangeStar-v2", "CDD", "SARS-COVID", "SkySat-TS"],
  "split_ratios": {"train": 0.6, "val": 0.2, "test": 0.2},
  "seed": 42,
  "primary_metric": "iou_change",
  "notes": ""
}
```

**What it is and is not:**

- It records *names, split ratios, a seed and a metric name*. It does **not** download data, make the splits or run a model.
  You apply the 60/20/20 split yourself (with the `seed`) and score with [`ModelEvaluator`](../experiments/benchmark.md).
  `BenchmarkConfig` raises `ValueError` if the ratios do not have `train`/`val`/`test` keys that sum to 1.
- `primary_metric` is a *label per task*: `mean_iou` (segmentation), `mAP50` (detection), `accuracy` (classification, VQA),
  `mean_f1` (multi-label), `iou_change` (change detection), `rmse` (regression), `mae` (prediction),
  `linear_probe_accuracy` (self-supervised). `ModelEvaluator` produces the first four kinds; for change detection read
  `per_class_iou["iou_class_1"]`; regression, prediction and self-supervised scoring are not provided.
- The picks come from the [ranking above](#how-the-ranking-works): in the real output the change-detection top five does not
  include LEVIR-CD or OSCD, so check that the selection matches what your field uses.
- A dataset appears in several tasks' lists if it has several tasks: `cross_task_matrix()["multi_task"]` currently gives only `MLRSNet`
  (classification and multi-label).

## Getting the data

```console
$ pygeovision datasets download EuroSAT
Error: 'EuroSAT' has no automated download path -- the catalogue holds metadata only. Its recorded download_url is a project page, not a direct file link: https://github.com/phelber/EuroSAT.

$ pygeovision datasets download LoveDA
Error: 'LoveDA' has no automated download path -- the catalogue holds metadata only. No download_url or paper_url is recorded for it; search the web for 'LoveDA 2022 dataset' to find its official page.
```

This always exits with status 1 by design: the recorded links are project pages, so a blind download would save an HTML
page that looks like data. Get the data from the dataset's official page, accept its terms, and point the
[training data loaders](../training/data.md) at the folder.

To search the *satellite imagery* archives (Sentinel, Landsat, ...) rather than public training datasets, see
[Search and download](../data/search-and-download.md).

## What to do when…

| You see | Cause | Do this |
|---|---|---|
| `KeyError: Dataset 'LoveDa' not found. Did you mean: LoveDA?` | Names are case-sensitive and exact | Use the suggestion, or `reg.search("loveda")` |
| `No datasets found for task 'xyz'` | The task name is not one of the 23 in the catalogue | `pygeovision benchmark tasks` lists them |
| `top` returns point clouds for segmentation | The ranking mixes modalities | `--modality rgb` (or the one you need) |
| `No datasets found for task 'x' with modality 'y'` | No dataset has both | Drop `--modality` or choose another |
| `datasets download` always fails | By design; no automated download exists | Visit the page named in the message |
| A number looks wrong (size, count, year) | Metadata is hand-entered | Check the paper; do not quote the catalogue as a source |
| `license: unknown` | Not recorded | Read the dataset's own terms |
| Two entries seem to be the same dataset | The source listed some datasets twice; they are merged (first entry's numbers kept, links and tasks combined) | Report conflicts you notice |
| `correlation_matrix` shows unrelated names | No `names` given: it shows the first 30 entries | Pass `names=[...]` |
| `cluster` returns `None` and logs a warning | `scikit-learn` / `umap-learn` / `matplotlib` missing | `pip install scikit-learn` (umap: `umap-learn`) |
| `ValueError: split_ratios must have train/val/test keys ...` | Ratios do not add up to 1 | Fix the ratios |

```{seealso}
[Benchmarking](../experiments/benchmark.md) · [Model registry](../models/registry.md) · [Training data](../training/data.md) ·
[Search and download imagery](../data/search-and-download.md)
```
