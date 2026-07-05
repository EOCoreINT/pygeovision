# Chapter 27: Disaster

## 27.1 Rapid Damage Assessment

```python
from pygeovision.agent import GeoAgent

agent = GeoAgent(client, output_dir="./disaster/")
agent.set_context(
    bbox = (36.0, 36.5, 38.0, 38.0),   # Kahramanmaras
    date = "2023-02",
)

# Agent decides: SAR (earthquake + no clear sky guarantee) -> damage assessment
trace = agent.run("Detect building damage after the Kahramanmaras earthquake")
print(f"Damage map: {trace.final_output}")
print(f"Execution: {trace.total_duration:.1f}s")
```

## 27.2 4-Class Damage Mapping

```python
from pygeovision.models.change_detection.changeformer import ChangeDetection

cd = ChangeDetection(model_variant="changeformer",
                      in_channels=6, num_classes=4)
cd.build()
result = cd.detect("./sar/pre.tif", "./sar/post.tif",
                    output_path="./disaster/damage.tif")

classes = ["No damage","Minor","Moderate","Severe"]
for i, label in enumerate(classes):
    pct = (result.prediction == i).mean() * 100
    print(f"  {label:<12}: {pct:.1f}%")
```

## 27.3 Flood Early Warning

```python
# Real-time flood monitoring pipeline
from pygeovision.insar import InSARProcessor

proc = InSARProcessor(output_dir="./disaster/flood_monitor/")

for new_scene in get_latest_sar_acquisitions():
    result = proc.full_pipeline(
        pre_path   = baseline_scene,
        post_path  = new_scene,
        study_area = "Odaw Basin, Accra",
    )

    if result.report.max_subsidence_m < -0.05:
        send_alert(f"WARNING: {result.report.subsidence_extent_km2:.0f} km2 flooded")
```

## 27.4 NB09: Accra Flood Intelligence Platform

The Accra Flood Intelligence Platform (Notebook SAR-09) implements a
complete flood monitoring system for the Odaw River Basin, Ghana.

Key components:
- Historical flood frequency composite (2019-2024)
- Active inundation mapping from current Sentinel-1 acquisition
- Prithvi-SAR flood segmentation (Sen1Floods11 adapted)
- 4-tier risk zone classification by ward
- FloodWatch Ghana alert payload for 6 communities

```python
# FloodWatch Ghana alert payload example
payload = {
    "alert_type":   "FLOOD_WARNING",
    "severity":     "HIGH",
    "communities":  ["Alajo", "Agbogbloshie", "Kaneshie"],
    "flood_area_km2": 12.4,
    "timestamp":    "2026-06-15T08:30:00Z",
    "message":      "Active inundation detected. Evacuate low-lying areas.",
    "source":       "PyGeoVision Sentinel-1 analysis",
}
import json, requests
# requests.post("https://floodwatch.ghana.gov.gh/api/alerts", json=payload)
print(json.dumps(payload, indent=2))
```

## Exercises

1. Run damage assessment on a pre/post earthquake image pair.
2. Map flood extent within 24 hours of a recent flood event.
3. Build a simple alert system that monitors daily SAR acquisitions.

## 27.7 Summary

Disaster response is the highest-impact EO AI application. The complete
workflow from SAR download to damage assessment report runs in 15-30 minutes,
enabling rapid response in the critical first 24h after an event.

## Exercises

1. Run 4-class damage assessment on earthquake imagery.
2. Detect landslides using SAR + slope DEM.
3. Build a flood early-warning pipeline.
4. Generate a rapid damage assessment report.
