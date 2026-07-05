# Frequently Asked Questions

## Data access

**Q: Do I need credentials to use PyGeoVision?**  
A: No. The default provider is Microsoft Planetary Computer (open access). You need
credentials only for Planet, Maxar, or commercial providers.

**Q: Which Sentinel data is free?**  
Sentinel-1, -2, -3, -5P — all free via Planetary Computer or Copernicus CDSE.

## SAR

**Q: Why does `clip_sar_to_bbox()` fail with "shapes do not overlap raster"?**  
A: Your bbox is in WGS84 degrees but the raster is in UTM metres. PyGeoVision's
`clip_sar_to_bbox()` handles this automatically — pass your WGS84 bbox and it
auto-reprojects.

**Q: What's the correct SAR preprocessing order?**  
```
verify → validate_georeference → despeckle(LINEAR) → to_dB → normalise → clip
```
Despckling MUST happen before dB conversion — speckle statistics are only valid
in linear power scale.

## GeoAgent

**Q: Does GeoAgent work without an Anthropic API key?**  
A: Yes. Without a key it uses the heuristic planner which covers ~80% of common
geospatial workflows with zero external API calls.

**Q: How does the agent decide whether to use SAR or optical?**  
A: It reads your query for explicit signals ("SAR", "Sentinel-1", "cloud cover",
"night" → SAR; "Sentinel-2", "optical", "clear sky" → optical). For disaster
tasks where neither is mentioned, it defaults to SAR.

## InSAR

**Q: Is this real InSAR or an approximation?**  
A: PyGeoVision's InSAR module uses GRD amplitude change as a proxy
(centimetre-scale accuracy). For millimetre-precision true InSAR, process
Sentinel-1 SLC data with pyroSAR + SNAP, then import the unwrapped phase here.

## Training

**Q: Tests skip with "torch not installed" — is that a bug?**  
A: No. PyTorch is optional (`pip install "pygeovision[train]"`). Tests skip
cleanly when torch is absent and run when it's present.
