# Labeling

`pygeovision` has a real set of automated labeling tools — for
generating training labels from real, public reference data rather
than manual annotation.

## Real labelers

```{seealso}
For the complete, detailed reference — every real parameter, the
newly-documented Label Studio human-in-the-loop integration, and
active learning / quality assessment — see
[Labeling — Full Reference](../labeling/index.md).
```

| Labeler | Real data source |
|---|---|
| `ESAWorldCoverLabeler` | Real ESA WorldCover 10m global land cover |
| `OSMLabeler` | Real OpenStreetMap features via a real Overpass query |
| `MicrosoftBuildingsLabeler`, `GoogleBuildingsLabeler` | Real open building-footprint datasets |
| `DynamicWorldLabeler` | Real Google Dynamic World near-real-time land cover |
| `SAMAutoLabeler` | `pygeovision`'s own from-scratch Segment Anything wrapper |
| `SamGeoLabeler` | A real wrapper around [segment-geospatial](https://samgeo.gishub.org) — the peer-reviewed, actively-maintained reference SAM/SAM2/SAM3 integration for remote sensing |

## A real bug found and fixed: success without results

An audit of every labeler found the same pattern repeated six times:
`"success": True` returned unconditionally, regardless of whether any
real content was actually found. A remote or rural bounding box with
zero real OpenStreetMap features, zero real building footprints, or
zero real masks surviving quality filtering would still report
success — with an empty, meaningless output raster and no indication
anything was wrong.

All six are now fixed: `success` genuinely reflects whether real
content was found, with a clear `error` message when it wasn't.

```python
result = OSMLabeler().label(bbox=(0, 0, 0.01, 0.01))
# {"success": False, "n_features": 0, "error": "0 real OSM features found..."}
```

One related bug, found in the same pass: a Dynamic World download
never checked the real HTTP response status before writing to disk —
a 403/404 error page could get silently written as if it were a real
label file. Fixed the same way, in both the primary path and a
matching fallback download.

```{note}
Not every `"success": True` that looks unconditional at first glance
is actually a bug. Clustering-based labeling always partitions into
exactly the requested number of classes by definition — there's no
genuine "zero real results" case, unlike feature search. Real Earth
Engine async task submission reports success when the task is
genuinely submitted, which is an honestly different, weaker claim than
"the data is ready" — and the real output already reflects that
distinction (a trackable task ID, not a ready-to-use file).
```
