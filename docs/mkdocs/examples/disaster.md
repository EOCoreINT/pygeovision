# Disaster Response Examples

## Turkey Earthquake 2023 — InSAR Deformation (NB08)

```python
from pygeovision.insar import InSARProcessor

proc   = InSARProcessor(output_dir="./turkey_insar/")
result = proc.full_pipeline(
    "sar_pre_ready.tif", "sar_post_ready.tif",
    study_area="Kahramanmaraş, Turkey"
)
print(result.report.summary())
# Max subsidence: -45.2 cm
# Affected area: 2,150 km² (far larger than building damage footprint)
```

## Accra Flood Intelligence (NB09)

```python
import pygeovision as pgv
from pygeovision.agent import GeoAgent

client = pgv.PyGeoVision()
agent  = GeoAgent(client, output_dir="./accra/")
agent.set_context(bbox=(-0.30, 5.50, -0.05, 5.70), date="2026-06")
trace = agent.run("Map current flood inundation in the Odaw River Basin")
print(trace.final_output)   # flood_mask.geojson
```

## Building Damage Assessment

```python
from pygeovision.models.change_detection.changeformer import ChangeDetection

cd = ChangeDetection(model_variant="changeformer", in_channels=6, num_classes=4)
cd.build()
result = cd.detect("before.tif", "after.tif", output_path="damage_4class.tif")
# Classes: 0=no_damage, 1=minor, 2=moderate, 3=severe
```
