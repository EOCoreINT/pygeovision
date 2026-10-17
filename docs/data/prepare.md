# Prepare and check data

Imagery straight from a provider is not ready for analysis. Values are stored as integer counts rather than reflectance, clouds are
in the way, radar needs despeckling, and a neural network is stricter still about shape, range and missing values. This section
covers what happens between **downloaded** and **ready to use**, and what to do with the result afterwards.

```{list-table}
:header-rows: 1
:widths: 28 72
* - Page
  - What it answers
* - [Radiometry and cloud masks](radiometry.md)
  - What scale factors are applied? How are cloud and shadow masked? Why is the 2022 Sentinel-2 offset handled?
* - [Radar (SAR) preprocessing](sar.md)
  - How is Sentinel-1 prepared: georeference recovery, despeckling, dB conversion, normalisation?
* - [Spectral indices](indices.md)
  - What do NDVI, NDWI, NBR… mean, how are they computed here, and when do they mislead?
* - [Texture features](texture.md)
  - GLCM (Haralick) texture from a band.
* - [Preprocessing toolbox](preprocessing.md)
  - Stack, clip, mask, normalise, resample, mosaic, composite, tile.
* - [AI-ready data](ai-ready.md)
  - What does `prepare_for_ai()` do, and what comes back?
* - [Validation](validation.md)
  - How do I check a raster before a model sees it, and what is fixed automatically?
* - [Postprocessing toolbox](postprocessing.md)
  - Vectorise, sieve, smooth, regularise, zonal statistics, accuracy assessment, COG, export.
```

## Which page do I need?

| You have… | You want… | Read |
|---|---|---|
| Scenes you downloaded | Surface reflectance with cloud removed | [Radiometry and cloud masks](radiometry.md) |
| Sentinel-1 radar | Backscatter in dB without speckle | [Radar preprocessing](sar.md) |
| Reflectance bands | NDVI, water, burn or snow indices | [Spectral indices](indices.md) |
| A folder of GeoTIFFs of mixed sizes | One stack, one grid, one scale | [Preprocessing toolbox](preprocessing.md) |
| A raster you plan to give to a network | Proof that it will not silently break | [AI-ready data](ai-ready.md), [Validation](validation.md) |
| A prediction raster | Polygons, areas in hectares, an accuracy figure | [Postprocessing toolbox](postprocessing.md) |

## The order that works

1. **Correct each scene on its own**: scale to reflectance and mask cloud and shadow. Do this before mosaicking, so there is no
   brightness jump along a seam.
2. **Combine**: mosaic, clip to your box, put everything on one grid.
3. **Derive**: compute the indices your method uses.
4. **Check**: validate before a model sees the raster, and look at the share of valid pixels.
5. **Deliver**: vectorise, measure and export.

Steps 1 and 2 are what [scene selection](scene-selection.md) and the pipelines do for you. Steps 3 to 5 are the pages above.

```{toctree}
:hidden:
:maxdepth: 1

radiometry
sar
indices
texture
preprocessing
ai-ready
validation
postprocessing
```
