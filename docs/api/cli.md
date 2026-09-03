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

Runs any of the 51 registered pipelines -- every one has been
individually resolved this cycle (not just the original 10). See
[Task Pipelines](task_pipelines.md) for the complete, per-pipeline
breakdown of which are real implementations, which honestly raise
`NotImplementedError`, and why.

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

The CLI has 20 additional top-level command groups beyond `channel`
(`data`, `ai`, `ai-models`, `models`, `infer`, `label`, `explain`,
`monitor`, `pipeline`, `edge`, `cloud`, `vlm`, `timeseries`, `datasets`,
`zoo`, `benchmark`, `validate`, `preprocess`, `indices`, `postprocess`
-- confirmed directly by counting real `@cli.group` decorators), plus 2
more direct top-level commands (`status`, `doctor`).

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
