# PyGeoVision v2.0 — Makefile
# Usage: make help

.PHONY: help install install-dev install-all test test-fast test-unit \
        lint format format-check type-check clean build docs status \
        notebooks serve benchmark validate-env

## ── Default: show help ──────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  PyGeoVision v2.0 — Development Commands"
	@echo "  ─────────────────────────────────────────"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	    awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'
	@echo ""

## ── Installation ────────────────────────────────────────────────────────────
install: ## Install core + geo + train extras
	pip install -e ".[geo,train]"

install-dev: ## Install all dev dependencies + pre-commit hooks
	pip install -e ".[dev,geo,train,serve,labeling,foundation,vlm]"
	pre-commit install
	@echo "✓ Dev environment ready"

install-all: ## Install everything (no cloud/edge)
	pip install -e ".[all,dev]"

## ── Testing ─────────────────────────────────────────────────────────────────
test: ## Run full test suite (580 tests)
	pytest tests/ -q --tb=short

test-cov: ## Run full test suite with HTML coverage report
	pytest tests/ -q --cov=pygeovision --cov-report=term-missing --cov-report=html
	@echo "Coverage report → htmlcov/index.html"

test-fast: ## Run tests, skip slow/integration/live
	pytest tests/ -q -m "not slow and not integration and not live" --tb=short

test-unit: ## Run only core unit tests (no network, no GPU)
	pytest tests/test_core.py tests/test_training.py tests/test_labeling.py \
	       tests/test_inference.py tests/test_benchmark.py -q

test-validator: ## Run DataValidator tests
	pytest tests/ -q -k "validator" --tb=short

test-bridge: ## Run PyGeoFetch bridge tests
	pytest tests/test_data_layer.py -q --tb=short

test-pipelines: ## Run pipeline tests
	pytest tests/ -q -k "pipeline" --tb=short

## ── Code Quality ────────────────────────────────────────────────────────────
lint: ## Run ruff linter
	ruff check pygeovision/ tests/

lint-fix: ## Run ruff linter with auto-fix
	ruff check pygeovision/ tests/ --fix

format: ## Auto-format with black
	black pygeovision/ tests/ --line-length 100

format-check: ## Check formatting (no modifications)
	black --check pygeovision/ tests/ --line-length 100

type-check: ## Run mypy type checking
	mypy pygeovision/ --ignore-missing-imports

pre-commit: ## Run all pre-commit hooks
	pre-commit run --all-files

## ── CLI Shortcuts ───────────────────────────────────────────────────────────
status: ## Show PyGeoVision environment status
	pygeovision status

doctor: ## Run installation diagnostics
	pygeovision doctor

models: ## List all registered models (119 architectures)
	pygeovision models list

models-foundation: ## List foundation models (DINOv3, Prithvi)
	pygeovision models list --task foundation

datasets: ## Show dataset registry stats
	pygeovision datasets list | head -30

pipelines: ## List all 51 pipelines
	pygeovision pipeline list

indices: ## List all 22 spectral indices
	pygeovision indices list

## ── Serving ─────────────────────────────────────────────────────────────────
serve: ## Start inference server (port 8080)
	pygeovision serve start --port 8080

serve-dev: ## Start inference server in dev mode with auto-reload
	uvicorn pygeovision.serving.api:create_app --host 0.0.0.0 --port 8080 --reload --factory

## ── Build & Docs ────────────────────────────────────────────────────────────
build: ## Build PyPI distribution packages
	python -m build

docs: ## Build MkDocs documentation site
	mkdocs build

docs-serve: ## Serve docs locally with live reload
	mkdocs serve

## ── Docker ───────────────────────────────────────────────────────────────────
docker-build: ## Build PyGeoVision Docker image
	docker build -t pygeovision:2.1.2 .

docker-build-gpu: ## Build GPU Docker image
	docker build --target gpu -t pygeovision:2.1.2-gpu .

docker-run: ## Run inference server in Docker (CPU)
	docker run -p 8080:8080 -v $(PWD)/models:/models pygeovision:2.1.2

docker-compose-up: ## Start full stack (API + MLflow + Redis)
	docker-compose up -d

docker-compose-monitoring: ## Start full stack + Prometheus + Grafana
	docker-compose --profile monitoring up -d

docker-compose-down: ## Stop all services
	docker-compose down

## ── Validation & Preprocessing ──────────────────────────────────────────────
validate: ## Validate a raster file (set FILE=path/to/scene.tif)
	@[ "$(FILE)" ] && pygeovision validate run $(FILE) || echo "Usage: make validate FILE=scene.tif"

preprocess-stack: ## Stack Sentinel-2 bands (set SCENE_DIR and BANDS)
	@[ "$(SCENE_DIR)" ] && \
	    pygeovision preprocess stack $(SCENE_DIR) --bands B02,B03,B04,B08,B11,B12 \
	    --output stack.tif --validate || echo "Usage: make preprocess-stack SCENE_DIR=./data/S2/"

## ── Notebooks ───────────────────────────────────────────────────────────────
notebooks: ## Launch Jupyter with all 25 production notebooks
	jupyter lab projects/

## ── Cleanup ─────────────────────────────────────────────────────────────────
clean: ## Remove build artifacts, caches, and temp files
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache htmlcov .coverage .ruff_cache site/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.pyo" -delete 2>/dev/null || true

clean-deep: clean ## Deep clean including temp geospatial data
	rm -rf chips/ training_chips/ pipeline_output/ predictions/ results/ downloads/ raw/
	find . -name "*.tif" -not -path "*/tests/*" -delete 2>/dev/null || true

## ── Release ──────────────────────────────────────────────────────────────────
release-check: ## Verify everything before a release
	@echo "Running pre-release checks…"
	@pytest tests/ -q --tb=short
	@ruff check pygeovision/
	@black --check pygeovision/ --line-length 100
	@python -c "from pygeovision import __version__; print(f'  Version: {__version__}')"
	@echo "✓ Pre-release checks passed"

release-dry: ## Dry-run build + twine check
	python -m build
	twine check dist/*
