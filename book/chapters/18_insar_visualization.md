# Chapter 18: InSAR Visualization

## 18.1 InSAR Product Visualization

```python
from pygeovision.insar.visualization import InSARViz

viz = InSARViz(figsize=(12, 10))

# Individual products
viz.interferogram("./insar/ifg.tif").show()     # HSV rainbow colour cycle
viz.coherence("./insar/coh.tif").show()          # Red-Yellow-Green colour ramp
viz.displacement("./insar/disp.tif").show()      # RdBu + histogram
viz.deformation_rate("./insar/rate.tif").show()  # Rate map + statistics
```

## 18.2 Full Dashboard

```python
viz.dashboard(
    interferogram_path = "./insar/ifg.tif",
    coherence_path     = "./insar/coh.tif",
    displacement_path  = "./insar/disp.tif",
    rate_path          = "./insar/rate.tif",
    title              = "InSAR Analysis - Istanbul Subsidence",
).export("./insar/dashboard.png", dpi=300)
```

## 18.3 Displacement Time-Series

```python
viz.time_series(
    displacement_paths = disp_paths,
    dates              = disp_dates,
    pixel_coords       = [(256, 256), (300, 180)],  # sample pixels
    title              = "Displacement at Key Locations",
).export("./insar/timeseries.png")
```

## 18.4 Interpretation Summary

```python
from pygeovision.insar.interpretation import InSARInterpreter

interp = InSARInterpreter()
report = interp.interpret("./insar/disp.tif", coherence_path="./insar/coh.tif")

# Print structured report
print(report.summary())

# Export
report.export("./insar/report.json")
report.export("./insar/report.md")
```

## Exercises

1. Visualise an interferogram — how many phase cycles (fringes) do you count?
2. Create the full 4-panel dashboard for a SAR pair.
3. Plot the displacement time-series for the most-deforming pixel.

## 18.5 Interpretation Summary

```python
from pygeovision.insar.interpretation import InSARInterpreter

interp = InSARInterpreter()
report = interp.interpret("./insar/disp.tif", study_area="Demo")
print(report.summary())
report.export("./insar/report.json")
report.export("./insar/report.md")
```

## Summary

InSARViz provides specialised visualisation for all InSAR products. The HSV
colour wheel for wrapped phase, and RdBu for displacement maps, follow InSAR
community conventions.

## Exercises

1. Visualise a wrapped interferogram. How many colour cycles (fringes)?
2. Create the 4-panel dashboard.
3. Plot displacement time-series for the most deforming pixel.
