# The leaderboard and the benchmark CLI

## The leaderboard

```python
from pygeovision.benchmark import Leaderboard

board = Leaderboard("leaderboard.json")      # loads the file when it exists
board.add(result)                             # replaces an earlier entry for the same model + dataset + split
board.print(task="segmentation", dataset="east-of-scene")
rows = board.get(task="segmentation")        # list of dicts, best primary_metric first
board.export_csv("leaderboard.csv")
```

```text
════════════════════════════════════════════════════════════════════════════════
  Leaderboard | segmentation
════════════════════════════════════════════════════════════════════════════════
  #   Model                          Dataset                 Primary     mIoU
  ───────────────────────────────────────────────────────────────────────────
  1   net-16                         east-of-scene              ...      ...
  2   quarter-res                    east-of-scene              ...      ...
  3   net-4                          east-of-scene              ...      ...
  4   untrained                      east-of-scene              ...      ...
════════════════════════════════════════════════════════════════════════════════
```

- **Entries are keyed by model + dataset + split.** Adding a model again replaces its row; to keep both versions, give
  them different `model_name`s (`unet-v1`, `unet-v2`).
- **Results from different datasets are not comparable**, yet one file can hold them; always filter by `dataset=`.
- **`print` shows at most 20 rows** and always has an `mIoU` column, even for detection or classification boards (it is 0
  there). `Primary` is the column to read.
- **In the CSV, `per_class_iou` is a Python-dict string**, and the header comes from the first row, so keep one task per file.
- The file is plain JSON. Two processes writing to the same file at once can overwrite each other: write from one place.

## The benchmark CLI

The three commands help you **choose what to benchmark**; they do not run an evaluation (that is the Python API above).
They read the dataset [catalogue](../datasets/index.md) and the model zoo ([registry](../models/registry.md)).

```console
$ pygeovision benchmark tasks
Available benchmark tasks (26):

  change_detection            31 datasets     10 models
  classification              79 datasets     19 models
  detection                   77 datasets     17 models
  segmentation               218 datasets     17 models
  ...
```

```console
$ pygeovision benchmark datasets segmentation --n 3 --modality rgb

══════════════════════════════════════════════════════════════════════
  Candidate Benchmark Datasets for 'segmentation' (EarthNets Top-3)
══════════════════════════════════════════════════════════════════════

Name                           Domain          Modality         Year    Samples   Res(m)  Vol(GB)
────────────────────────────────────────────────────────────────────────────────────────────────────
FLAIR1                         land_cover      rgb              2022     77,762      0.2     45.0
OpenCities                     urban           rgb              2020    790,000      0.3     25.0
CrowdAI Mapping                urban           rgb              2018    280,741      0.3     21.0

3 shown; the catalogue holds 460 datasets
```

The ranking is a metadata heuristic (newer, larger, more classes, finer pixels), **not** a statement of which datasets are the
accepted benchmarks. Without `--modality` the top segmentation entries are point-cloud datasets. See
[Choosing datasets](../datasets/index.md#how-the-ranking-works).

```console
$ pygeovision benchmark models change_detection
No models with pretrained weights for task 'change_detection' (10 registered without them; see 'pygeovision zoo list --task change_detection').
```

`benchmark models` ranks only models that have pretrained weights; `benchmark tasks` counts all registered models, which is why a task can
show models in one command and none in the other.
