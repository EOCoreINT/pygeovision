# Appendix B: CLI Reference

## Installation

```bash
pip install pygeovision
pygeovision --version   # 2.1.7
```

## Data Commands

```bash
# Search satellite imagery
pygeovision data search \
  --bbox "-0.30 5.50 -0.05 5.70" \
  --date-range "2026-06-01" "2026-06-30" \
  --providers planetary_computer \
  --cloud-max 20

# Download
pygeovision data download \
  --scene-id S2_T30PXR_20260615 \
  --output-dir ./data/ \
  --bands B02 B03 B04 B08 B8A B11 \
  --cog

# Add credentials
pygeovision data credentials add --provider copernicus_cdse \
  --username you@example.com --password your-password
```

## AI Commands

```bash
# Run a named pipeline
pygeovision ai pipeline building_footprints \
  --bbox "-0.30 5.50 -0.05 5.70" \
  --date 2026-06 \
  --output-dir ./results/

# Run all 10 pipelines
pygeovision ai pipeline --list

# Model inference
pygeovision ai infer \
  --model prithvi_eo_2_0 \
  --task land_cover \
  --input ./data/prepared.tif \
  --output ./results/land_cover.tif
```

## SAR Commands

```bash
# Complete SAR preprocessing pipeline
pygeovision sar preprocess \
  --input ./sar/S1_GRD.tif \
  --output-dir ./sar/processed/ \
  --bbox "-0.30 5.50 -0.05 5.70" \
  --filter enhanced_lee

# Flood detection
pygeovision sar flood-detect \
  --input ./sar/processed/S1_clipped.tif \
  --output ./sar/flood_mask.tif

# Change detection
pygeovision sar change-detect \
  --pre ./sar/S1_pre.tif \
  --post ./sar/S1_post.tif \
  --output ./sar/damage.tif \
  --num-classes 4
```

## InSAR Commands

```bash
# Full InSAR pipeline
pygeovision insar pipeline \
  --pre ./sar/S1_pre.tif \
  --post ./sar/S1_post.tif \
  --output-dir ./insar/ \
  --study-area "Istanbul, Turkey"

# Individual steps
pygeovision insar interferogram --pre pre.tif --post post.tif
pygeovision insar coherence --pre pre.tif --post post.tif
pygeovision insar displacement --phase ifg.tif --mode amplitude_proxy
pygeovision insar rate --displacements d1.tif d2.tif --dates 2023-01-01 2024-01-01
```

## Agent Commands

```bash
# Start interactive chat session
pygeovision agent chat

# One-shot query
pygeovision agent run "Map flood extent in Accra using SAR" \
  --bbox "-0.30 5.50 -0.05 5.70" \
  --date 2026-06 \
  --output-dir ./results/

# Save and load sessions
pygeovision agent save ./sessions/accra.json
pygeovision agent load ./sessions/accra.json

# Show available tools
pygeovision agent tools
```

## Visualization Commands

```bash
# View raster
pygeovision viz raster --input scene.tif --composite rgb --bands 2 1 0

# Export interactive map
pygeovision viz map --raster scene.tif --vector buildings.geojson --output map.html

# Before/after comparison
pygeovision viz split --before pre.tif --after post.tif --output comparison.html

# Animate time-series
pygeovision viz animate --inputs "ndvi_*.tif" --dates dates.json --output ndvi.gif
```
