# InSAR API Reference

Complete InSAR processing chain: interferogram → coherence → displacement → interpretation.

## Quick Start

```python
from pygeovision.insar import InSARProcessor

proc   = InSARProcessor(output_dir="./insar/")
result = proc.full_pipeline("sar_pre.tif", "sar_post.tif", study_area="Istanbul")
print(result.report.summary())
```

## `InSARProcessor`

High-level façade chaining all InSAR modules:

- `proc.interferogram(pre, post)` — generate wrapped phase + coherence
- `proc.displacement(result)` — convert to displacement map
- `proc.deformation_rate(paths, dates)` — pixel-wise mm/year rate
- `proc.full_pipeline(pre, post)` — all steps + interpretation report

## `InSARViz`

```python
from pygeovision.insar.visualization import InSARViz

viz = InSARViz()
viz.interferogram("ifg.tif")
viz.coherence("coh.tif")
viz.displacement("disp.tif")
viz.dashboard(ifg.tif, coh.tif, disp.tif, rate.tif)
viz.export("insar_dashboard.png")
```

## `InSARInterpreter`

```python
from pygeovision.insar.interpretation import InSARInterpreter

interp = InSARInterpreter(subsidence_threshold_m=-0.02, uplift_threshold_m=0.01)
report = interp.interpret("displacement.tif", coherence_path="coh.tif")
print(report.summary())
report.export("report.json")
```

## Mode note

Default: **GRD amplitude proxy** — accessible without SNAP/ISCE2.
Accuracy: cm–dm scale. For mm-precision, process Sentinel-1 SLC with pyroSAR → import here.
