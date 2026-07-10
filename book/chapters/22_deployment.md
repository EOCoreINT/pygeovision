# Chapter 22: Deployment

## 22.1 Docker Deployment

```bash
# Build production image
docker build -t pygeovision:2.1.5 \
  -f deployment/docker/Dockerfile.prod .

# Run with volume mount for data persistence
docker run -d \
  -p 8080:8080 \
  -v ./data:/data \
  -v ./results:/app/results \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  -e PGV_LOG_LEVEL=INFO \
  pygeovision:2.1.5

# Verify
curl http://localhost:8080/health
```

## 22.2 Kubernetes Deployment

```bash
# Apply manifests
kubectl apply -f deployment/kubernetes/manifests/namespace.yaml
kubectl apply -f deployment/kubernetes/manifests/deployment.yaml
kubectl apply -f deployment/kubernetes/manifests/service.yaml

# Check status
kubectl -n pygeovision get pods
kubectl -n pygeovision get svc

# Scale
kubectl -n pygeovision scale deployment pygeovision-api --replicas=5
```

## 22.3 Terraform (AWS)

```bash
cd deployment/terraform/aws

terraform init
terraform plan   -var="environment=production"
terraform apply  -var="environment=production" \
                 -var="anthropic_api_key=sk-ant-..."

# Get cluster endpoint
terraform output eks_cluster_endpoint
terraform output kubeconfig_command
```

## 22.4 FastAPI Serving

```python
from pygeovision.serving.server import create_app
import uvicorn

app = create_app(
    providers    = ["planetary_computer"],
    model_cache  = "/data/model_cache/",
    output_dir   = "/data/results/",
    max_workers  = 4,
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, workers=4)
```

## 22.5 Production Best Practices

```python
# Environment configuration
import os

PGV_CONFIG = {
    "log_level":    os.getenv("PGV_LOG_LEVEL", "INFO"),
    "cache_dir":    os.getenv("PGV_CACHE_DIR", "/data/cache"),
    "max_workers":  int(os.getenv("PGV_MAX_WORKERS", "4")),
    "api_key":      os.getenv("ANTHROPIC_API_KEY", ""),
}

# Health check endpoint
from fastapi import FastAPI
app = FastAPI()

@app.get("/health")
def health():
    import pygeovision as pgv
    return {"status": "ok", "version": pgv.__version__}
```

## Exercises

1. Deploy PyGeoVision in Docker locally. Verify the health endpoint.
2. Deploy to Kubernetes using the provided manifests.
3. Run the Terraform AWS plan. What resources would be created?
4. Add a `/analyze` endpoint that accepts a bbox and returns NDVI statistics.

## 22.8 Security Checklist

```python
# Production security configuration
import os

# Never hard-code credentials
API_KEY     = os.getenv("ANTHROPIC_API_KEY", "")
PC_TOKEN    = os.getenv("PLANETARY_COMPUTER_TOKEN", "")

# Verify TLS in production
import ssl
ctx = ssl.create_default_context()
ctx.minimum_version = ssl.TLSVersion.TLSv1_2

# Use enterprise RBAC
from pygeovision.enterprise import RBACManager
rbac = RBACManager()
rbac.create_user("api_user", "api@org.com", roles=["analyst"])
```

## Summary

Complete deployment from Docker single-container to Kubernetes multi-node
to Terraform-managed cloud. All infrastructure code is in `deployment/`.

## Exercises

1. Deploy PyGeoVision locally in Docker. Hit the health endpoint.
2. Apply the Kubernetes manifests to a local kind cluster.
3. Add Prometheus metrics to monitor inference latency.
