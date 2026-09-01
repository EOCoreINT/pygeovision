# CLI Reference

**Verification status**: only `pygeovision channel` has been independently
tested this cycle — real bugs found and fixed across its full stack
(data layer, radiometric processing, tiled inference, WorldCover
labeling). The ~25 other command groups below are listed because they
exist in the code, not because their behavior has been verified this
cycle.

**Correction**: there is no `pgv` alias. Only `pygeovision` is registered
as a real entry point (confirmed in `pyproject.toml`'s `[project.scripts]`).
If you want a shorter alias, create one yourself (`alias pgv=pygeovision`).

```bash
pip install pygeovision
pygeovision --help
```

---

## `pygeovision channel` — verified this cycle

Runs one of the 10 real, audited pipelines. See [Pipelines](pipelines.md)
and [Architecture](../architecture.md) for which of the 51 registered
pipeline names this actually covers well.

```bash
pygeovision channel land_cover --bbox -74.1 40.6 -73.7 40.9 --date 2024-01

pygeovision channel urban_growth --bbox -0.45 5.50 0.05 5.75 \
  --date-before 2018-01 --date-after 2024-01

# Restrict to a specific search provider (a real, tested flag)
pygeovision channel land_cover --bbox -74.1 40.6 -73.7 40.9 \
  --date 2024-01 --provider planetary_computer
```

Real options: `--bbox`, `--output`, `--date`, `--date-before`, `--date-after`,
`--model`, `--source`, `--provider` (repeatable).

Run `pygeovision channel --help` for the full, current list of pipeline names.

---

## Other command groups — not independently verified this cycle

The CLI has roughly 25 additional top-level command groups beyond
`channel`, covering data search/download, model management, inference,
labeling, explainability, monitoring, edge/cloud deployment,
vision-language models, time series, dataset/model registries,
benchmarking, validation, and raster preprocessing/indices/postprocessing.

Run `pygeovision --help` for the authoritative, current list — this page
intentionally doesn't enumerate all ~80 individual commands, since doing
so without testing each one would just be repackaging unverified claims
with more detail. Where you rely on one of these for real work, verify
it against your own data first, the same way this audit verified
`channel`.

Two real, confirmed structural issues found in the CLI this cycle, worth
knowing about if you're exploring it:

- `pipeline` appears as **two different, separate command groups**
  (one nested under `data`, one standalone at the top level) — a real
  naming collision, not a documentation error.
- Command and flag names for the unaudited groups have not been
  cross-checked against their actual `click` option definitions the way
  `channel`'s were — treat any specific flag name you see in `--help`
  output as more reliable than any flag name in older documentation.
