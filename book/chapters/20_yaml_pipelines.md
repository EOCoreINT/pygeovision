# Chapter 20: YAML Pipelines

## 20.1 Introduction to Pipelines

PyGeoVision pipelines define complete geospatial workflows in YAML:

```yaml
# pipeline.yml
pipeline:
  name: flood_monitoring_accra
  schedule: "0 6 * * *"   # daily at 06:00 UTC

  steps:
    - name: search
      type: search
      providers: [planetary_computer]
      bbox: [-0.30, 5.50, -0.05, 5.70]
      date_range: [today-7d, today]
      cloud_cover_max: 20

    - name: download
      type: download
      bands: [B02, B03, B04, B08, B8A, B11]
      post_process: [reproject:EPSG:32630, cog]

    - name: prepare
      type: prepare_for_ai
      model_type: foundation
      normalise: scale_factor

    - name: classify
      type: ai_inference
      model: prithvi_eo_2_0
      task: flood_detection

    - name: export
      type: postprocess
      operations: [sieve, vectorise, cog]
      output_dir: ./outputs/
```

## 20.2 Running Pipelines

```python
import pygeovision as pgv

client = pgv.PyGeoVision()

# Run a YAML pipeline
result = client.run_pipeline("./pipeline.yml")
print(f"Status: {result.status}")
print(f"Output: {result.output_dir}")
```

## 20.3 Built-in Pipelines

```python
# 10 named end-to-end pipelines
PIPELINES = [
    "building_footprints",    "change_detection",
    "land_cover",             "water_bodies",
    "solar_panels",           "crop_monitoring",
    "disaster_assessment",    "deforestation",
    "urban_growth",           "carbon_estimation",
]

for pipe in PIPELINES:
    result = client.pipeline(pipe, bbox=BBOX, date="2026-06")
    print(f"  {pipe}: {result.status}")
```

## 20.4 Scheduling and Automation

```python
from pygeovision.pipelines.scheduler import PipelineScheduler

scheduler = PipelineScheduler()

# Schedule daily flood monitoring
scheduler.add(
    pipeline    = "./pipelines/flood_monitoring.yml",
    cron        = "0 6 * * *",   # every day at 06:00
    notify_email = "alerts@org.com",
)

# Run immediately
scheduler.run("./pipelines/flood_monitoring.yml")

# List scheduled pipelines
for job in scheduler.list():
    print(f"  {job['name']}: {job['next_run']}")
```

## Exercises

1. Write a YAML pipeline for NDVI monitoring of a crop region.
2. Schedule the pipeline to run every Monday at 08:00.
3. Modify the pipeline to send an email alert when NDVI drops below 0.3.

## 20.5 Scheduling and Monitoring

```python
from pygeovision.pipelines.scheduler import PipelineScheduler

scheduler = PipelineScheduler()
scheduler.add(pipeline="./pipelines/flood.yml",
               cron="0 6 * * *",
               notify_email="alerts@org.com")

for job in scheduler.list():
    print(f"  {job['name']}: next={job['next_run']}")
```

## Summary

YAML pipelines define complete geospatial workflows declaratively.
Schedule them as daily monitors or trigger on new satellite acquisitions.
Ten built-in named pipelines cover the most common use cases.

## Exercises

1. Write a YAML pipeline for NDVI monitoring.
2. Schedule it to run every Monday morning.
3. Add email alerting when NDVI drops below threshold.
