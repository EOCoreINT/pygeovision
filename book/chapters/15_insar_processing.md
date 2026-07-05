# Chapter 15: InSAR Processing

## 15.1 What is InSAR?

InSAR measures phase differences between two SAR acquisitions to detect
centimetre-scale ground deformation — earthquakes, subsidence, volcanoes.

| Data | Phase | Accuracy | Mode |
|------|-------|----------|------|
| SLC | Yes | mm scale | `slc_phase` |
| GRD | No | cm-dm | `amplitude_proxy` |

## 15.2 Quick Start

```python
from pygeovision.insar import InSARProcessor

proc   = InSARProcessor(output_dir="./insar/", wavelength=0.05546576)
result = proc.full_pipeline("./sar/S1_pre.tif", "./sar/S1_post.tif",
                              study_area="Istanbul, Turkey")
print(result.report.summary())
```

## 15.3 Interferogram Generation

```python
from pygeovision.insar.interferogram import generate_interferogram

ifg = generate_interferogram("./sar/pre.tif", "./sar/post.tif",
                              output_dir="./insar/", filter_type="goldstein")
print(f"Mean coherence: {ifg['stats']['mean_coherence']:.3f}")
print(f"Changed area:   {ifg['stats']['changed_pct']:.1f}%")
```

## 15.4 Coherence and Masking

```python
from pygeovision.insar.coherence import estimate_coherence, coherence_mask

coh  = estimate_coherence("./sar/pre.tif", "./sar/post.tif",
                            output_path="./insar/coh.tif", window_size=7)
mask = coherence_mask(coh, threshold=0.4)
print(f"Reliable pixels: {mask.mean()*100:.1f}%")
```

## 15.5 Displacement Measurement

```python
from pygeovision.insar.displacement import phase_to_displacement

# GRD amplitude proxy
disp = phase_to_displacement("./insar/ifg.tif", "./insar/disp.tif",
                               mode="amplitude_proxy")
print(f"Max subsidence: {disp.min()*100:.1f} cm")
print(f"Max uplift:     {disp.max()*100:.1f} cm")
```

## 15.6 Deformation Rate

```python
from pygeovision.insar.displacement import displacement_rate

rate = displacement_rate(
    displacement_paths = ["./disp_2023.tif","./disp_2024.tif","./disp_2025.tif"],
    dates              = ["2023-01-01","2024-01-01","2025-01-01"],
    output_path        = "./insar/rate_mm_yr.tif",
)
print(f"Mean rate: {rate.mean():.1f} mm/year")
```

## 15.7 Interpretation

```python
from pygeovision.insar.interpretation import InSARInterpreter

interp = InSARInterpreter(subsidence_threshold_m=-0.02, min_zone_km2=0.5)
report = interp.interpret("./insar/disp.tif", coherence_path="./insar/coh.tif",
                           study_area="Jakarta, Indonesia")
print(report.summary())
report.export("./insar/report.json")
```

## 15.8 Visualization

```python
from pygeovision.insar.visualization import InSARViz

viz = InSARViz()
viz.dashboard(interferogram_path="./insar/ifg.tif",
               coherence_path="./insar/coh.tif",
               displacement_path="./insar/disp.tif",
               rate_path="./insar/rate.tif").export("./insar/dashboard.png")
```

## 15.9 Case Study: Turkey Earthquake 2023

The Feb 6 Mw 7.8 earthquake produced >45 cm co-seismic displacement visible
in Sentinel-1:

```python
proc   = InSARProcessor(output_dir="./turkey/")
result = proc.full_pipeline("./sar/S1_pre_20230130.tif",
                              "./sar/S1_post_20230211.tif",
                              study_area="Kahramanmaras, Turkey")

print(result.report.summary())
# Max subsidence: -45.2 cm (proxy estimate)
# Affected area:  2,150 km2
```

## Summary

- GRD amplitude proxy gives cm-dm accuracy without SNAP/ISCE2.
- `full_pipeline()` runs interferogram -> coherence -> displacement -> report.
- For mm-precision: use pyroSAR + SNAP to process SLC, then import here.

## Exercises

1. Run full_pipeline on pre/post SAR scenes over a known earthquake.
2. Detect subsidence in a city with known groundwater depletion.
3. Generate the 4-panel InSAR dashboard.
