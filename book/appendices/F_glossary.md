# Appendix F: Glossary

**AGB** — Above-Ground Biomass. The total mass of living plant material above
the soil surface, typically measured in tonnes per hectare.

**AOI** — Area of Interest. The geographic region being analysed.

**bbox** — Bounding box. A rectangle defined as `[lon_min, lat_min, lon_max, lat_max]`
in WGS84 decimal degrees.

**COG** — Cloud Optimised GeoTIFF. A GeoTIFF file structured for efficient partial
reads over HTTP — no full download required for spatial subsets.

**Coherence** — In InSAR, a measure of phase stability between two SAR acquisitions
(0=decorrelated, 1=fully coherent). Vegetation typically shows low coherence.

**CRS** — Coordinate Reference System. The mathematical framework that relates
pixels to geographic coordinates (e.g., EPSG:4326 = WGS84, EPSG:32630 = UTM 30N).

**dB** — Decibel. A logarithmic unit: dB = 10 × log10(linear_power). SAR data is
often displayed in dB but must be despeckled in linear power first.

**dNBR** — Delta Normalised Burn Ratio. The difference between pre- and post-fire
NBR, used to classify wildfire burn severity.

**DINOv3** — A self-supervised vision foundation model trained on satellite imagery.
Produces general-purpose spatial feature embeddings.

**EVI** — Enhanced Vegetation Index. An improved NDVI that reduces atmospheric
effects and soil influence for dense canopy conditions.

**GRD** — Ground Range Detected. SAR product with only amplitude information
(phase discarded). Lower processing requirements than SLC.

**HLS** — Harmonised Landsat Sentinel. A NASA product that co-registers and
calibrates Landsat 8/9 and Sentinel-2 to a common grid and spectral resolution.

**InSAR** — Interferometric SAR. Measures the phase difference between two SAR
acquisitions to detect millimetre-scale ground deformation.

**IW** — Interferometric Wide-swath. The standard Sentinel-1 SAR acquisition mode
over land (250 km swath at 5×20 m resolution).

**LOS** — Line-of-Sight. The direction along the satellite-to-ground radar beam.
InSAR displacement is measured in LOS — not purely vertical.

**mIoU** — Mean Intersection over Union. The standard segmentation metric:
average of (true_positive / (true_positive + false_positive + false_negative))
across all classes.

**NDBI** — Normalised Difference Built-up Index. (SWIR - NIR)/(SWIR + NIR).
Positive values indicate built-up/impervious surfaces.

**NDSI** — Normalised Difference Snow Index. (Green - SWIR)/(Green + SWIR).
Values > 0.4 typically indicate snow.

**NDVI** — Normalised Difference Vegetation Index. (NIR - Red)/(NIR + Red).
Range −1 to +1; values > 0.3 indicate healthy vegetation.

**NDWI** — Normalised Difference Water Index. (Green - NIR)/(Green + NIR).
Positive values indicate open water.

**Prithvi-EO-2.0** — NASA's 600M parameter geospatial foundation model, pre-trained
on HLS satellite imagery. Supports land cover, flood, crop, and burn scar tasks.

**SAR** — Synthetic Aperture Radar. An active microwave imaging system that
creates high-resolution images independent of solar illumination and clouds.

**SCL** — Scene Classification Layer. Sentinel-2 per-pixel classification:
vegetation, bare soil, water, cloud shadow, cloud, snow, etc.

**SLC** — Single Look Complex. SAR data product preserving both amplitude and phase,
required for true InSAR processing.

**STAC** — SpatioTemporal Asset Catalog. An open standard API for querying and
accessing geospatial data. Planetary Computer and Copernicus CDSE support STAC.

**UTM** — Universal Transverse Mercator. A projected coordinate system that divides
the Earth into 60 zones (EPSG:326XX for northern hemisphere). Units are metres.

**WGS84** — World Geodetic System 1984. The standard geographic coordinate system
(EPSG:4326). Units are degrees of latitude and longitude.
