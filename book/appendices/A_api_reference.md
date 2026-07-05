# Appendix A: API Reference

## PyGeoVision Client

```python
import pygeovision as pgv
client = pgv.PyGeoVision()

# Data methods
client.search(bbox, date_range, providers, satellites, cloud_cover_max, max_results)
client.download(scenes, output_dir, bands, post_process, max_workers, resume)
client.prepare_for_ai(path, stack_bands, bbox, bbox_crs, scl_path, normalise, model_type, output_path)
client.pipeline(name, bbox, date, output_dir)
client.add_credentials(provider, **credentials)
client.list_providers()

# AI methods
client.indices.ndvi(path, output_path)
client.indices.ndwi(path, output_path)
client.indices.ndbi(path, output_path)
client.postprocess.sieve_filter(path, min_pixels, output_path)
client.postprocess.vectorise(path, output_path, target_class, min_area_m2)
client.postprocess.to_cog(path, output_path)
```

## GeoAgent

```python
from pygeovision.agent import GeoAgent

agent = GeoAgent(client, output_dir, api_key, model, stop_on_failure, verbose)
agent.set_context(**kwargs)       # bbox, date, output_dir, bands...
agent.bind(name, value)           # name a file path for later reference
agent.run(query, context_override)   # returns ExecutionTrace
agent.stream(query)               # yields event dicts
agent.tools()                     # list tool names
agent.tool_schema(name)           # get one tool's schema
agent.all_schemas()               # all tool schemas
agent.history()                   # conversation history
agent.save_session(path)          # persist to JSON
agent.load_session(path)          # restore from JSON
agent.reset()                     # clear history, keep context
```

## Visualization

```python
from pygeovision.viz import Map, RasterViewer, VectorViewer, ChangeViewer, TimeSeriesViewer

# Map
m = Map(center, zoom, basemap, height)
m.add_raster(path, band, colormap, vmin, vmax, opacity, name)
m.add_vector(path_or_geojson, color, fill_color, fill_opacity, weight, name)
m.add_basemap(basemap)
m.remove_layer(name)
m.export(path, format, dpi)       # "html" or "png"
m.show()
Map.split_view(left, right, center, zoom)

# RasterViewer
rv = RasterViewer(path, figsize)
rv.rgb(red, green, blue)
rv.false_color(nir, red, green)
rv.ndvi(nir_band, red_band)
rv.ndwi(green_band, nir_band)
rv.ndbi(swir_band, nir_band)
rv.single_band(band, colormap)
rv.histogram(band, bins)
rv.profile(start_px, end_px, bands)
rv.export(path, dpi)
rv.show()

# ChangeViewer
cv = ChangeViewer(before, after, band)
cv.split(before_label, after_label)
cv.difference()
cv.overlay(mask_path, change_color, alpha)
cv.statistics(mask_path, pixel_area_m2)
cv.export(path, dpi)

# TimeSeriesViewer
tsv = TimeSeriesViewer(paths, dates, band, colormap, vmin, vmax)
tsv.animate(output, fps)
tsv.trend(roi, title)
tsv.seasonal(period, title)
tsv.anomaly(n_sigma, title)
tsv.mosaic(max_cols, title)
tsv.export(path, dpi)
```

## InSAR

```python
from pygeovision.insar import InSARProcessor, InSARViz, InSARInterpreter

# InSARProcessor
proc = InSARProcessor(output_dir, wavelength, incidence_angle_deg, coherence_threshold)
proc.interferogram(pre, post, multilook, filter_type)  # returns InSARResult
proc.displacement(ifg_result, mode)
proc.deformation_rate(paths, dates, coherence_path)
proc.full_pipeline(pre, post, study_area)

# InSARViz
viz = InSARViz(figsize)
viz.interferogram(path)
viz.coherence(path)
viz.displacement(path, vmax_cm)
viz.deformation_rate(path)
viz.time_series(paths, dates, pixel_coords)
viz.dashboard(interferogram_path, coherence_path, displacement_path, rate_path)
viz.export(path, dpi)

# InSARInterpreter
interp = InSARInterpreter(subsidence_threshold_m, uplift_threshold_m,
                           coherence_threshold, min_zone_km2)
report = interp.interpret(displacement_path, rate_path, coherence_path, study_area)
report.summary()
report.export(path, format)
report.to_dict()
```

## Enterprise

```python
from pygeovision.enterprise import RBACManager, APIKeyManager, AuditLogger, ComplianceChecker

# RBAC
rbac = RBACManager()
rbac.create_user(user_id, email, roles)
rbac.check_permission(user_id, permission)
rbac.assign_role(user_id, role)

# API Keys
keys = APIKeyManager()
keys.generate(user_id, description, ttl_days)
keys.validate(raw_key)   # returns user_id or None
keys.revoke(raw_key)

# Audit
audit = AuditLogger(log_path)
audit.log_action(user_id, action, resource, outcome)
audit.query(user_id, action, since)
audit.export(path, format)

# Compliance
checker = ComplianceChecker()
checker.check_all(config)
checker.report(config)
```
