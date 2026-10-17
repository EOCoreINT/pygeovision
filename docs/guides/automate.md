# Run many areas and dates, or run on a schedule

One run answers one question for one area and one time. Real work is usually many: ten districts, twelve months, a monitoring site
checked every week. This guide shows how to run a pipeline repeatedly, collect the results into one table, skip what is already
done, and put the whole thing on a schedule.

## Pick the right tool

| You want | Use | Why |
|---|---|---|
| The same pipeline over **many areas** | A loop over `client.pipeline(...)` | One output folder per area, one summary table |
| One area over **several dates** | A pipeline that takes `dates` (such as [`vegetation_indices`](../pipelines/catalog/vegetation_indices.md) or [`water_extent_timeseries`](../pipelines/catalog/water_extent_timeseries.md)) | One run, one grid, one chart across time |
| One area over many dates with a pipeline that takes one date | A loop over the dates | Same loop, with the date in the loop |
| Your **own trained network** over many scenes, then polygons | A [YAML workflow](../pipelines/yaml-workflows.md) | Search, download, predict and export as one file |
| A **folder of imagery kept up to date** | A [data pipeline](../data/yaml-pipelines.md) | Moves and prepares files; no analysis |
| Any of these **every week** | The scheduler of your system (cron, below) | PyGeoVision commands exit non-zero on failure, so the scheduler can alert you |

## Many areas

Put the areas in a dictionary and loop. Keep each run in its own folder and record the outcome of **every** run, including the failed
ones, so one cloudy district does not stop the other nine.

```python
from pathlib import Path

import pandas as pd
from pygeovision import PyGeoVision

client = PyGeoVision()

areas = {
    "central": (-0.15, 51.47, -0.10, 51.52),
    "east":    (-0.10, 51.47, -0.05, 51.52),
    "west":    (-0.20, 51.47, -0.15, 51.52),
}

rows = []
for name, bbox in areas.items():
    result = client.pipeline(
        "water_bodies", bbox=bbox, date="2024-06",
        output_dir=f"out/water/{name}",
        report_formats="none", vector_formats="none",    # no PDF, Word, Excel or polygons; the maps and run_summary.json stay
    )
    rows.append({
        "area": name,
        "ok": result.success,
        "water_share": result.stats.get("water_coverage"),
        "valid_share": result.stats.get("valid_fraction"),
        "problem": result.error or "",
    })

table = pd.DataFrame(rows)
table.to_csv("out/water/summary.csv", index=False)
print(table)
```

Turning the reports and polygons off makes each run quicker. Leave those two arguments out when you want the reports. Do **not**
use `report=False` (or `--no-report`) here: it also skips `run_summary.json`, which the later steps of this guide read. The keys of
`result.stats` are specific to each pipeline and are listed on its page, so change `water_coverage` when you change the pipeline.

**Compare shares of the valid area, not raw areas**, when `valid_share` differs between rows. A district with half its pixels under
cloud has half the water area for that reason alone.

### The same loop on the command line

```bash
for area in "central -0.15 51.47 -0.10 51.52" "east -0.10 51.47 -0.05 51.52"; do
    set -- $area                     # $1 is the name, $2..$5 the box
    pygeovision channel water_bodies --bbox $2 $3 $4 $5 --date 2024-06 \
        --output out/water/$1 --report-formats none --vector-formats none || echo "failed: $1" >> out/water/failures.txt
done
```

## Many dates

Where a pipeline takes one date, loop over the months. Where it takes a list, give it the list and let it do the work once:

```bash
# one run, one grid, a chart across time
pygeovision channel vegetation_indices --bbox -0.15 51.47 -0.10 51.52 \
    --dates 2024-04,2024-05,2024-06,2024-07 --output out/season
```

```python
# one run per month, when the pipeline takes a single date
for month in ["2024-04", "2024-05", "2024-06", "2024-07"]:
    result = client.pipeline("water_bodies", bbox=(-0.15, 51.47, -0.10, 51.52), date=month,
                             output_dir=f"out/months/{month}", report_formats="none", vector_formats="none")
```

## Skip what is already done

A long loop that stops halfway should not start from zero. A successful run writes a `run_summary.json` (unless you switched
reports off with `--no-report`), and a failed run does not, so its presence is the test:

```python
for name, bbox in areas.items():
    out = Path(f"out/water/{name}")
    if (out / "run_summary.json").exists():
        continue                             # done in an earlier pass
    client.pipeline("water_bodies", bbox=bbox, date="2024-06", output_dir=str(out), report_formats="none", vector_formats="none")
```

## Collect the results afterwards

Because every run records its numbers and settings, you can build the table from the folders alone, without re-running anything:

```python
import json

records = []
for summary_file in sorted(Path("out/water").glob("*/run_summary.json")):
    s = json.loads(summary_file.read_text())
    records.append({"area": summary_file.parent.name, **s["stats"], "period": s.get("period")})

print(pd.DataFrame(records))
```

`run_summary.json` also holds every parameter that was used, defaults included, so a table made this way is reproducible
([Reading results](../pipelines/results.md)).

## Be kind to the services

- **Searches are cached for an hour**, so repeating a run over the same area and dates does not hit the provider again.
- **Place-name lookups** (`client.boundary`) go to OpenStreetMap's Nominatim service, which allows about one request per second. Do
  not look up hundreds of names in a loop; resolve them once and save the boxes.
- **Large downloads add up.** Test a loop on two areas first, and use a coarser `resolution_m` on pipelines that offer it while you
  are developing.

## On a schedule

Any repeating analysis is an ordinary command, so the system scheduler runs it. On Linux or macOS, `crontab -e`:

```text
# every Monday at 06:00: this month's vegetation anomaly, in a dated folder
0 6 * * 1  cd /home/me/monitoring && /home/me/venv/bin/pygeovision channel vegetation_anomaly --bbox -0.15 51.47 -0.10 51.52 --date $(date +\%Y-\%m) --output runs/$(date +\%F) >> logs/monitor.log 2>&1
```

Four details make a scheduled job reliable:

1. **Full paths**: cron does not load your shell, so give the path of the `pygeovision` program inside your virtual environment.
2. **Escape `%` as `\%`** in a crontab line, as above.
3. **One folder per run**, named by date, so runs never overwrite each other.
4. **Log to a file** and look at it. The exit status is non-zero when a run fails, which most schedulers can turn into an alert.

On Windows, Task Scheduler runs the same command. For a workflow that includes your own network, the
[YAML workflow page](../pipelines/yaml-workflows.md#scheduling) has a ready-made schedule.

## Related

- [Python guide](python.md): the client and the result object.
- [Choose your area and dates](your-area.md): what to give each run.
- [Reading results](../pipelines/results.md): comparing runs and building a history.
- [Monitoring](../monitoring/index.md): watching a deployed model, which is a different job from watching the ground.
