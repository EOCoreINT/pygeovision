# Production Deployment Guide

## Docker Deployment

```bash
docker build -t pygeovision:2.1.6 .
docker run -p 8080:8080 -v ./data:/app/data pygeovision:2.1.6
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Claude API key for GeoAgent LLM planning | (none, uses heuristic) |
| `PLANETARY_COMPUTER_TOKEN` | Planetary Computer API token | (none, uses open access) |
| `PGV_CACHE_DIR` | Download cache directory | `~/.pgv_cache` |
| `PGV_LOG_LEVEL` | Logging level | `INFO` |
| `PGV_MAX_WORKERS` | Parallel download workers | `4` |

## Resource Requirements

| Workload | CPU | RAM | GPU |
|----------|-----|-----|-----|
| Data search & download | 2 vCPU | 4 GB | — |
| Prithvi inference | 4 vCPU | 8 GB | Optional (4 GB VRAM) |
| ChangeFormer training | 8 vCPU | 16 GB | Required (16 GB VRAM) |
| Full InSAR pipeline | 4 vCPU | 16 GB | — |
| GeoAgent (heuristic) | 2 vCPU | 4 GB | — |

## Scaling

- Parallel downloads: `max_workers=8` in PyGeoVision config
- Tiled inference: `TiledInference` for scenes > 10,000 × 10,000 pixels
- Batch mode: `client.batch.search()` for multi-AOI / multi-date workflows
