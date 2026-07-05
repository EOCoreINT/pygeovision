# Appendix C: Configuration

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PGV_LOG_LEVEL` | `INFO` | Logging level (DEBUG/INFO/WARNING/ERROR) |
| `PGV_CACHE_DIR` | `~/.pgv_cache` | Download cache directory |
| `PGV_MAX_WORKERS` | `4` | Parallel download workers |
| `ANTHROPIC_API_KEY` | (none) | Claude API key for GeoAgent LLM planning |
| `PLANETARY_COMPUTER_TOKEN` | (none) | Microsoft PC subscription key |
| `COPERNICUS_CDSE_USER` | (none) | Copernicus CDSE username |
| `COPERNICUS_CDSE_PASS` | (none) | Copernicus CDSE password |

## Configuration File

Create `~/.pygeovision/config.yml`:

```yaml
version: "2.1.2"

data:
  cache_dir: /data/pgv_cache
  max_workers: 8
  default_providers:
    - planetary_computer
  default_bands:
    - B02
    - B03
    - B04
    - B08
    - B8A
    - B11

sar:
  default_filter: enhanced_lee
  window_size: 7
  despeckle_before_db: true   # MUST be true — do not change

insar:
  wavelength: 0.05546576      # Sentinel-1 C-band
  incidence_angle: 39.0
  coherence_threshold: 0.4

agent:
  default_planner: heuristic  # heuristic | llm
  llm_model: claude-sonnet-4-6
  output_dir: ./agent_output/

logging:
  level: INFO
  format: "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
  file: /var/log/pygeovision.log
```

## Docker Environment

```bash
# .env file for Docker
ANTHROPIC_API_KEY=sk-ant-...
PLANETARY_COMPUTER_TOKEN=your-token
PGV_CACHE_DIR=/data/cache
PGV_LOG_LEVEL=INFO
PGV_MAX_WORKERS=8
```

## Kubernetes ConfigMap

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: pygeovision-config
  namespace: pygeovision
data:
  PGV_LOG_LEVEL: "INFO"
  PGV_CACHE_DIR: "/data/cache"
  PGV_MAX_WORKERS: "8"
```
