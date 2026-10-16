# Glossary

Terms as PyGeoVision uses them. Where a word has a broader meaning elsewhere, the entry says what it means *here*.

AOI
: Area of interest. Given as a bounding box `--bbox min_lon min_lat max_lon max_lat` in WGS 84 degrees, or as a polygon ([boundaries](../data/boundaries.md)).

Backscatter
: The share of a radar pulse that returns to a SAR satellite, in dB. Calm open water is dark (low backscatter). Used by the radar flood pipelines.

Baseline
: The earlier observations a new one is compared with: the same month in earlier years (`vegetation_anomaly`), a dry reference window (flood pipelines), a reference sample for drift ([monitoring](../monitoring/drift.md)).

bbox
: See AOI. Always `min_lon min_lat max_lon max_lat`; longitude first.

Checkpoint
: A `.pth` file with trained network weights. PyGeoVision ships none for task models. The seven "needs weights" pipelines, `infer predict` and `ai segment|detect|classify|change` require one you trained ([training](../training/index.md)).

Chip / tile
: A small window cut from a large raster to feed a network (for example 256 × 256 pixels), or the pieces of a large scene processed one at a time in [tiled inference](../inference/index.md).

Class code
: The integer stored in a class raster. **0 is background, 255 is "ignore / no data"**; classes are 1 to *n* ([concepts](../getting-started/concepts.md)).

Cloud mask
: A per-pixel flag for cloud, shadow and cirrus. Sentinel-2 uses the Scene Classification Layer (SCL). Masked pixels become nodata, never "dry" or "no change".

COG
: Cloud-Optimised GeoTIFF; a GeoTIFF laid out for partial reads from a web server (`postprocess cog`).

CRS
: Coordinate reference system. Areas and distances are computed in a metric projection (the local UTM zone), not in degrees ([units and CRS](../data/postprocessing.md#units-read-this-before-choosing-a-threshold)).

CVA
: Change vector analysis. The length of the per-band difference between two dates; the method of `change_detection`.

DEM
: Digital elevation model. The Copernicus 30 m DEM feeds `terrain_analysis`, `drainage_analysis` and the multi-sensor flood terrain filter.

dNBR
: Difference of the Normalised Burn Ratio before and after a fire; classed into unburned, low, moderate and high severity by `wildfire_severity`.

DN
: Digital number; the raw integer stored in a satellite file before conversion to reflectance or temperature.

Dynamic World
: Google's near-real-time 10 m land-cover probability product, available as a labeling source.

Epoch
: One time window of imagery in a study (before, event, control year).

EVI
: Enhanced Vegetation Index; less saturated than NDVI in dense canopy; uses the blue band.

Foundation model
: A large network pre-trained on huge unlabelled imagery (Prithvi, DOFA, DINOv2/v3, SAM, CLIP/RemoteCLIP) and adapted to a task ([foundation models](../foundation-models/index.md)).

GSD
: Ground sampling distance: the size of one pixel on the ground (10 m for Sentinel-2 bands used here).

HAND
: Height above nearest drainage. Elevation of a cell above the stream cell it drains to; low values mean ground close to a channel. Used to reject "floods" on hillsides.

Hectare (ha)
: 10 000 m². All areas in `result.stats` are in hectares unless the key says otherwise.

IoU
: Intersection over union of predicted and true pixels of a class. Reported per class and as a mean. For a rare class (flood) the class IoU is the honest number; the mean over classes hides it.

JRC GSW
: The European Commission Joint Research Centre Global Surface Water dataset (Landsat, 1984-2020); the source of `surface_water_dynamics`.

Label
: A class raster (or vector) used to train or score a model. Labels from automatic sources (WorldCover, OSM, SAM) are *noisy*; see [label quality](../labeling/quality.md).

Leakage
: Information from the test area reaching training, usually because neighbouring chips land on both sides of the split. The Volta study splits by geographic block to prevent it.

MNDWI / NDWI
: Water indices. NDWI contrasts green and near-infrared; MNDWI contrasts green and short-wave infrared and copes better with built-up land.

NBR
: Normalised Burn Ratio, from near- and short-wave infrared.

NDVI
: Normalised difference vegetation index, (NIR − red)/(NIR + red). A greenness measure. It is **not** biomass or yield and saturates in dense vegetation.

Nodata
: A pixel with no valid observation: cloud, outside the imagery, masked. **255 in class rasters, NaN in continuous ones.** Excluded from every statistic and reported as `valid_fraction`.

OBIA
: Object-based image analysis: segment the image into objects, compute features per object, then classify the objects ([OBIA](../obia/index.md)).

Otsu threshold
: An automatic threshold that best separates two groups in a histogram. Used for change magnitude and radar water.

Pipeline
: A named, documented analysis you run with `pygeovision channel NAME` or `client.pipeline(NAME, …)`: one of 40 ([catalogue](../pipelines/catalog/index.md)). Not to be confused with a YAML workflow ([YAML workflows](../pipelines/yaml-workflows.md)).

PSI
: Population Stability Index; a measure of how far a feature's distribution has drifted from a reference ([drift](../monitoring/drift.md)).

Provider
: A data source reached through pygeofetch (Planetary Computer, Copernicus, USGS, AWS Earth …); 22 in all ([providers](../data/providers.md)).

Prithvi
: IBM/NASA's geospatial foundation model, pre-trained on 30 m Harmonized Landsat-Sentinel imagery; used as an encoder in the Volta case study.

Reflectance
: The fraction of incoming light a surface reflects (0-1). Sentinel-2 L2A stores it as an integer ×10 000; PyGeoVision converts it.

SAM
: Segment Anything Model; used to propose segments for labeling ([SAM](../labeling/sam.md)).

SAR
: Synthetic-aperture radar (Sentinel-1). Sees through cloud and at night.

SCL
: Scene Classification Layer of Sentinel-2 L2A; per-pixel class (vegetation, water, cloud shadow, cloud, snow …).

Sieve
: Remove connected patches smaller than a minimum number of pixels from a class raster (`postprocess sieve`).

Speckle
: Grainy noise inherent in radar images; reduced by averaging (`speckle_filter` in the radar pipelines).

Strahler order
: A stream ranking: headwater streams are order 1; two streams of order *n* meeting make order *n* + 1.

UTM
: Universal Transverse Mercator; a metric projection in 6° zones. PyGeoVision computes areas in the zone of the AOI centre.

VCI
: Vegetation Condition Index; where this month's NDVI sits between the lowest (0) and highest (100) on record for that month.

Valid fraction
: The share of the AOI that could be assessed (not cloud, not masked, inside the imagery). Percentages and areas are computed over this part only.

Z-score
: How many standard deviations a value lies from the mean of its baseline.
