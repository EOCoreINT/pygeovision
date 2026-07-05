# Architecture

## Two-layer design

PyGeoVision sits between **PyGeoFetch** (satellite data access) and your analysis code:

```
User Query
    │
    ▼
┌─────────────────────────────────────────────┐
│            PyGeoVision v2.1.2               │
│                                             │
│  GeoAgent ──── natural language interface   │
│    │                                        │
│    ▼                                        │
│  Pipeline layer                             │
│    ├── search → download → prepare_for_ai   │
│    ├── AI models (Prithvi, ChangeFormer…)   │
│    ├── SAR pipeline (S0-S9 + bug fixes)     │
│    ├── InSAR chain                          │
│    └── postprocess (sieve/vector/COG)       │
│                                             │
│  Viz layer (Map, RasterViewer, ChangeViewer)│
│  Enterprise (RBAC, Audit, Compliance)       │
└─────────────────────────────────────────────┘
    │                        │
    ▼                        ▼
PyGeoFetch              AI backends
(22+ providers)    (Prithvi/ISCE2/TerraTorch)
```

## Key design decisions

| Decision | Rationale |
|---|---|
| SAR despeckle before dB | Speckle is Gamma-distributed in linear power, not dB |
| Georeference validation after reproject | Catches identity-transform CRS corruption (BUG 1) |
| Download completeness check first | Catches partial downloads before despeckle crashes (BUG 2) |
| WGS84→raster CRS before clip | Prevents no-overlap error from unit mismatch (BUG 3) |
| Heuristic planner as fallback | GeoAgent works without API key for 80% of workflows |
| GRD amplitude proxy for InSAR | Accessible without SNAP/ISCE2; upgrade path to SLC documented |
