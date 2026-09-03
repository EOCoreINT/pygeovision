# Utilities

Shared helpers: geo utilities, GPU detection, raster I/O, reporting,
and a data-pipeline helper.

**Checked this cycle**: `pygeovision.utils.acquire` and
`pygeovision.data.acquire` are genuinely different files (405 vs. 1038
lines, different stated purposes), not the same duplicate-bug pattern
found in `pygeovision.pipelines` (see [Architecture](../architecture.md)).
`utils.acquire`'s 4 functions (`best_scene`, `download_ok`,
`download_s1`, `preprocess_s2`) are publicly re-exported from
`pygeovision.utils` but confirmed unused by any other internal code —
worth knowing if you're relying on them, though not necessarily a bug.
`pygeovision.utils.data_pipeline` isn't re-exported or used anywhere
internally at all.

**Verification status**: import verified. Not independently audited this cycle.

::: pygeovision.utils
    options:
      members: true
