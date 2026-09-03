# Data Layer

The real, thoroughly-audited satellite data access layer this session's
verification work focused on most heavily. `pygeovision.data` wraps
[PyGeoFetch](https://pypi.org/project/pygeofetch/) for search/download
across ~24 providers, and owns the real radiometric scaling, band
alignment, and geodesic area calculations that sit between a
downloaded scene and a model-ready tensor.

**Verification status**: thoroughly audited and fixed this cycle — see
[Architecture](../architecture.md) for the specific bugs found (mission-aware
band aliases, the area-calculation fix, thermal/SWIR/red-edge band aliases
added this cycle).

---

## Client-facing exports

::: pygeovision.data
    options:
      members: true

---

## Radiometric processing (`pygeovision.data.radiometric`)

The module most of this cycle's real fixes live in — sensor
identification, real scaling formulas, band aliases (including SWIR1/2,
red-edge, and thermal, all added this cycle), bbox cropping, and the
real geodesic area helper.

::: pygeovision.data.radiometric

---

## PyGeoFetch bridge (`pygeovision.data.pgf_bridge`)

The real integration layer between PyGeoVision and the installed
PyGeoFetch package — confirmed this cycle to correctly detect and use
PyGeoFetch v2's native processing engine when available.

::: pygeovision.data.pgf_bridge
    options:
      show_source: false

---

## Spectral indices (`pygeovision.data.indices`)

::: pygeovision.data.indices
    options:
      show_source: false

---

## Post-processing (`pygeovision.data.postprocess`)

::: pygeovision.data.postprocess
    options:
      show_source: false
