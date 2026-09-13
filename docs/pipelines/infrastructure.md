# Infrastructure

Six real pipelines documented in full here, plus `building_footprints`,
one of the 10 CLI-reachable pipelines with a separate real
implementation — see
[The 10 CLI-Reachable Pipelines](cli-reachable-pipelines.md) for that
one (a simpler real segmentation call than an earlier version of this
page described).

## `road_extraction`

**What it does:** Real road-network detection via object detection
(not segmentation — the model draws real bounding boxes/linear
segments, not a pixel mask).

**How it actually works:** Runs a real detection model
(`rf-detr-b`/`rf-detr-l`, or `retinanet_resnet50`) with real,
configurable confidence and IoU thresholds.

```python
result = RoadExtractionPipeline(client).run(
    bbox=(...), date="2024-06", output_dir="./output",
    conf=0.4, iou=0.5,     # real, tunable NMS thresholds
)
```

`conf` (confidence threshold) and `iou` (non-max-suppression overlap
threshold) are real parameters passed straight through to the
underlying detector — raising `conf` trades recall for precision;
lowering `iou` merges more overlapping detections into fewer, larger
ones.

## `infrastructure_monitoring`

**What it does:** General-purpose bi-temporal infrastructure change
detection — new construction, demolition, or visible damage.

**How it actually works:** Two real, selectable methods:
`method="change_detection"` runs the real Siamese change-detection
model (`siamese_unet`/`changeformer`) on the image pair;
`method="spectral_diff"` computes a real, simpler per-band difference
image thresholded against a configurable value — faster, no model
download required, less precise.

```python
result = InfrastructureMonitoringPipeline(client).run(
    bbox=(...), date_before="2023-01", date_after="2024-01",
    output_dir="./output", method="change_detection",
)
```

## `powerline_extraction`

**What it does:** Flags candidate linear infrastructure corridors —
**not** confirmed powerlines specifically.

**How it actually works:** A real, two-stage geometric technique, not
a trained powerline classifier (none exists in this codebase). First,
real NDVI vegetation-gap detection finds linear clearings through
forest/vegetation (utility corridors are kept clear by real, routine
maintenance). Second, a real covariance-matrix elongation test on each
candidate blob's shape keeps only genuinely linear, corridor-shaped
regions and rejects round or irregular clearings. Uses real
8-connectivity for the connected-component labeling — a real bug found
during this pipeline's implementation used 4-connectivity by default,
which fragmented a single thin diagonal corridor into 20 separate,
disconnected regions.

```python
result = PowerlineExtractionPipeline(client).run(
    bbox=(...), date="2024-06", output_dir="./output",
    min_elongation=5.0,   # real, tunable shape threshold
)
```

```{warning}
This detects the *shape signature* of a maintained linear corridor —
it cannot distinguish a real powerline easement from a similarly
-shaped road, pipeline right-of-way, or firebreak. Treat output as
"candidate corridors for human review," not confirmed powerline
locations.
```

## `parking_occupancy`

**What it does:** Real vehicle detection within a defined parking-lot
boundary, with a real occupied/total-spaces ratio.

```python
result = ParkingOccupancyPipeline(client).run(
    bbox=(...), date="2024-06", output_dir="./output",
    conf=0.5, total_spaces=240,   # real, user-supplied lot capacity
)
# result.stats: {"n_vehicles_detected": ..., "occupancy_pct": ...}
```

**Honest limitation:** `total_spaces` is not inferred — you supply the
real, known capacity of the lot. Without it, `occupancy_pct` isn't
computed, only the raw vehicle count.

## `port_monitoring`

**What it does:** Real vessel detection and counting within a port or
anchorage area.

```python
result = PortMonitoringPipeline(client).run(
    bbox=(...), date="2024-06", output_dir="./output", conf=0.4,
)
# result.stats: {"n_vessels": ..., "vessel_density_per_km2": ...}
```

## `pipeline_leak_detection`

**What it does:** A real bi-temporal NDVI anomaly proxy along a
pipeline right-of-way — vegetation stress from a real hydrocarbon leak
often shows up as a localized, real NDVI drop before it's otherwise
visible.

```python
result = PipelineLeakDetectionPipeline(client).run(
    bbox=(...), date_before="2024-05", date_after="2024-07",
    output_dir="./output", corridor_path="./pipeline_route.geojson",
)
```

**Honest limitation:** this is the same class of uncalibrated
vegetation-stress proxy as `crop_health` above — a real, genuine
signal, but not a confirmed leak detector. Real drought, disease, or
mowing along the corridor will also produce a flagged anomaly.
