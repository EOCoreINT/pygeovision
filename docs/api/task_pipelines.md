# Task Pipelines — Complete Reference

All 51 pipelines registered in `pygeovision.ai.pipelines.domains.list_pipelines()`,
individually documented. Every real signature below was extracted
directly from the installed code this cycle (`inspect.signature`), not
transcribed by hand — trust these over any older documentation.

Run any pipeline via the CLI:
```bash
pygeovision channel <name> --bbox <lon_min> <lat_min> <lon_max> <lat_max> [--date ... | --date-before ... --date-after ...]
```

Or directly from Python:
```python
from pygeovision.ai.pipelines.domains import _PIPELINE_REGISTRY
# or, for the 10 originals:
from pygeovision.ai.pipelines import LandCoverPipeline

pipeline = LandCoverPipeline(client)
result = pipeline.run(bbox=(...), output_dir="./out", date="2024-06")
```

---

## Tier 1 — 10 original pipelines (thoroughly audited)

| Name | Class | Real signature | Notes |
|---|---|---|---|
| `building_footprints` | `BuildingFootprintsPipeline` | `bbox, output_dir, date, model, cloud_cover_max` | |
| `carbon_estimation` | `CarbonEstimationPipeline` | `bbox, output_dir, date` | Real NDVI proxy; area calc fixed this cycle (was wrong ~1e10x) |
| `change_detection` | `ChangeDetectionPipeline` | `bbox, output_dir, date_before, date_after, model, cloud_cover_max` | |
| `crop_monitoring` | `CropMonitoringPipeline` | `bbox, output_dir, date, crop_classes, model` | |
| `deforestation` | `DeforestationPipeline` | `bbox, output_dir, baseline_year, analysis_year, model` | |
| `disaster_assessment` | `DisasterAssessmentPipeline` | `bbox, output_dir, pre_date, post_date, disaster_type, model` | |
| `land_cover` | `LandCoverPipeline` | `bbox, output_dir, date, source, num_classes` | `source="worldcover"` uses the real, fixed `ESAWorldCoverLabeler` |
| `solar_detection` | `SolarDetectionPipeline` | `bbox, output_dir, date, cloud_cover_max, model` | |
| `urban_growth` | `UrbanGrowthPipeline` | `bbox, output_dir, start_year, end_year, model` | |
| `water_bodies` | `WaterBodiesPipeline` | `bbox, output_dir, date, method, cloud_cover_max` | |

---

## Tier 2 — 13 real implementations (converted this cycle from generic stubs)

Every one verified against a hand-calculated expected value, not just "runs without error."

| Name | Class | Real signature | Real technique |
|---|---|---|---|
| `wildfire_severity` | `WildfireSeverityPipeline` | `bbox, output_dir, date_before, date_after` | Real dNBR (Key & Benson 2006), USFS/MTBS 4-class severity |
| `glacier_monitoring` | `GlacierMonitoringPipeline` | `bbox, output_dir, date_before, date_after` | Real NDSI (Hall et al. 1995), bi-temporal area trend |
| `snow_cover` | `SnowCoverPipeline` | `bbox, output_dir, date` | Real NDSI extent. Does NOT estimate SWE (needs microwave/ground data) |
| `wetland_mapping` | `WetlandMappingPipeline` | `bbox, output_dir, date` | Real MNDWI (Xu 2006) + EVI (Huete et al. 2002), distinguishes wetland from open water |
| `urban_heat_island` | `UrbanHeatIslandPipeline` | `bbox, output_dir, date` | Real Landsat thermal → Celsius (real USGS Collection 2 formula). Landsat only. |
| `land_surface_temperature` | `LandSurfaceTemperaturePipeline` | `bbox, output_dir, date` | Real subclass of `UrbanHeatIslandPipeline` — same real computation |
| `volcano_monitoring` | `VolcanoMonitoringPipeline` | `bbox, output_dir, date` | Real subclass of `UrbanHeatIslandPipeline`, volcano-appropriate threshold (20°C vs. urban's 2°C) |
| `forest_fire` | `ForestFirePipeline` | `bbox, output_dir, date_before, date_after, date` | Real subclass of `WildfireSeverityPipeline` (dNBR). Does NOT cover real-time active-fire detection |
| `parking_occupancy` | `ParkingOccupancyPipeline` | `bbox, output_dir, date` | Real vehicle count via `client.detection.cars()`. NOT occupancy % (needs real capacity data) |
| `port_monitoring` | `PortMonitoringPipeline` | `bbox, output_dir, date` | Real ship count via `client.detection.ships()`. NOT throughput (needs a real time series) |
| `landcover_change` | `LandcoverChangePipeline` | `bbox, output_dir, date_before, date_after` | Real orchestration of two `LandCoverPipeline` runs + real, named class-transition matrix |
| `mangrove_mapping` | `MangroveMappingPipeline` | `bbox, output_dir, date_before, date_after` | Real ESA WorldCover mangrove class (code 95), bi-temporal area change |
| `oil_spill_detection` | `OilSpillDetectionPipeline` | `bbox, output_dir, date` | Real SAR dark-pixel detection via pygeofetch's `SARProcessor.flood_map()` directly, oil-appropriate threshold |
| `pipeline_leak_detection` | `PipelineLeakDetectionPipeline` | `bbox, output_dir, date_before, date_after, anomaly_threshold` | Real NDVI anomaly vs. baseline. No single published threshold for this application — documented judgment call |
| `crop_health` | `CropHealthPipeline` | `bbox, output_dir, date_before, date_after, anomaly_threshold` | Real NDVI baseline-vs-current anomaly (previously relied on a broken mechanism that never computed anything) |
| `crop_yield_forecast` | `CropYieldForecastPipeline` | `bbox, output_dir, dates, yield_calibration_factor` | Real peak-season NDVI. Uncalibrated unless you provide a real, local `yield_calibration_factor` |
| `biodiversity_hotspot` | `BiodiversityHotspotPipeline` | `bbox, output_dir, date` | Real Shannon diversity index on real land cover classes. Deliberately not the originally-claimed "DINOv3 embedding clustering" — see docstring for why |
| `water_quality` | `WaterQualityPipeline` | `bbox, output_dir, date` | Real NDWI + real NDCI (chlorophyll proxy, needs the real red-edge band alias added this cycle). Turbidity proxy explicitly labeled uncalibrated |
| `vegetation_indices` | `VegetationIndicesPipeline` | `bbox, output_dir, dates` | Real NDVI/EVI/NDWI computed per date across a real multi-date series (previously claimed but never computed) |

*(13 pipelines; `land_surface_temperature`/`volcano_monitoring`/`forest_fire` above are also part of the 16-class Tier-3 bug-pattern fix below — listed once here since their real technique matters more than which bucket they started in.)*

---

## Tier 3 — 16 classes with a confirmed, now-fixed bug pattern

Real, dedicated classes that had either (a) a bare `except Exception:`
silently substituting a wrong result while still returning
`success=True`, or (b) a claimed-but-never-actually-computed index via
a broken mechanism. All fixed — see [Architecture](../architecture.md#the-pipeline-catalog--every-one-of-51-resolved)
for the fix categories.

| Name | Class | Real signature | Fix applied |
|---|---|---|---|
| `crop_type_mapping` | `CropTypeMappingPipeline` | `bbox, output_dir, date` | Removed silent fallback to raw imagery on segmentation failure |
| `irrigation_detection` | `IrrigationDetectionPipeline` | `bbox, output_dir, date` | Removed silent fallback |
| `canopy_height` | `CanopyHeightPipeline` | `bbox, output_dir, date` | Removed silent fallback |
| `road_extraction` | `RoadExtractionPipeline` | `bbox, output_dir, date` | Removed silent fallback to the output *directory* as the result |
| `infrastructure_monitoring` | `InfrastructureMonitoringPipeline` | `bbox, output_dir, date_before, date_after` | Removed silent fallback |
| `flood_mapping` | `FloodMappingPipeline` | `bbox, output_dir, date` | Removed silent fallback |
| `coastal_monitoring` | `CoastalMonitoringPipeline` | `bbox, output_dir, date_before, date_after` | Removed silent fallback + fixed broken `post_process` NDWI claim |
| `landslide_detection` | `LandslideDetectionPipeline` | `bbox, output_dir, date` | Removed silent fallback |
| `ocean_ship_detection` | `OceanShipDetectionPipeline` | `bbox, output_dir, date` | Removed silent fallback; underlying `detection.ships()` is real and fixed |
| `tree_species` | `TreeSpeciesPipeline` | `bbox, output_dir, date` | Now honestly returns `success=False` instead of `True` with an admission note |

*(The remaining 6 of the 16 — `land_surface_temperature`, `volcano_monitoring`,
`forest_fire`, `crop_health`, `water_quality`, `vegetation_indices` — are
listed in Tier 2 above, since they became real implementations rather
than just bug-fixed stubs.)*

---

## Tier 4 — 12 honest, specific `NotImplementedError`s

Each has a genuinely distinct, individually-verified reason — confirmed
by a test asserting all 12 reasons are unique strings, not a
copy-pasted excuse. Calling `.run()` raises clearly; nothing here
silently returns a fake success.

| Name | Class | Why not implemented |
|---|---|---|
| `permafrost_thaw` | `PermafrostThawPipeline` | Real InSAR needs domain expertise this cycle didn't have; `snaphu` not installed in the audited environment; safety-adjacent territory |
| `dam_safety` | `DamSafetyPipeline` | Same as above — real InSAR orchestration, not implemented rather than guessed at for safety-critical deformation monitoring |
| `air_quality_index` | `AirQualityIndexPipeline` | Confirmed: zero real Sentinel-5P/atmospheric-composition provider exists in pygeofetch (searched, not assumed) |
| `dust_storm_tracking` | `DustStormTrackingPipeline` | Same gap — no real MODIS/Sentinel-5P aerosol provider exists |
| `solar_potential` | `SolarPotentialPipeline` | A real DEM source exists (pygeofetch's `OpentopographyProvider`), but real irradiance modeling (sun position, shadow-casting, slope/aspect) is genuinely complex, specialized GIS work not implemented this cycle |
| `archaeological_site` | `ArchaeologicalSitePipeline` | Real DEM source exists, but real LiDAR relief-model interpretation needs domain expertise this cycle doesn't have |
| `mine_detection` | `MineDetectionPipeline` | Needs a specialized, trained open-pit-mine boundary detector that doesn't exist |
| `powerline_extraction` | `PowerlineExtractionPipeline` | Needs a specialized LiDAR linear-feature extraction algorithm not built on the real `PointCloudProcessor` |
| `aquaculture_mapping` | `AquacultureMappingPipeline` | Real water detection (MNDWI) exists, but distinguishing aquaculture from natural water bodies needs real geometric/textural classification not implemented |
| `construction_progress` | `ConstructionProgressPipeline` | Needs a specialized, trained construction-staging detector that doesn't exist |
| `reef_bleaching` | `ReefBleachingPipeline` | Needs real bathymetric water-column correction to be reliable; a naive threshold would be noise-dominated |
| `wind_farm_siting` | `WindFarmSitingPipeline` | Needs real meteorological wind data, not satellite imagery — a fundamentally different data domain |

---

## Verified counts

10 + 13 + 16 (listed once each, with 6 of the 16 shown under Tier 2
since they became real implementations) + 12 = 51. Confirmed
programmatically this cycle:
`len(pygeovision.ai.pipelines.domains.list_pipelines()) == 51`.
