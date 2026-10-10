# Appendix E: Contributing

## How to Contribute

PyGeoVision welcomes contributions of all kinds:
- Bug reports and fixes
- New model integrations
- New satellite data providers
- Documentation improvements
- New use-case notebooks
- Test coverage improvements

## Development Setup

```bash
git clone https://github.com/EOCoreINT/pygeovision
cd PyGeoVision
pip install -e ".[dev]"
pre-commit install
```

## Code Style

PyGeoVision uses Ruff for linting:

```bash
ruff check pygeovision/
ruff format pygeovision/
```

Key conventions:
- Type annotations required for all public functions
- Docstrings in NumPy format
- Line length: 100 characters
- `snake_case` for functions and variables
- `PascalCase` for classes

## Testing

```bash
# Run all tests (excluding live integration tests)
pytest tests/ --ignore=tests/test_live.py -q

# Run specific module tests
pytest tests/test_viz.py tests/test_insar.py tests/test_enterprise.py -v

# Run with coverage
pytest tests/ --ignore=tests/test_live.py --cov=pygeovision --cov-report=html
```

**Target:** 451+ tests passing, 0 failures.

## Adding a New Model

1. Create `pygeovision/models/your_category/your_model.py`
2. Implement `forward()` with `(B, C, H, W)` input convention
3. Register in `pygeovision/models/__init__.py`
4. Add tests in `tests/test_your_category.py`
5. Add a notebook in `projects/`

## Adding a New SAR Tool

Extend `pygeovision/agent/tools.py`:

```python
from pygeovision.agent.tools import GeoTool, ToolResult, TOOL_REGISTRY

class MySARTool(GeoTool):
    name        = "my_sar_analysis"
    description = "Describe what this tool does..."
    parameters  = [{"name":"input_path","type":"str","required":True}]

    def run(self, input_path, **_):
        # Your implementation
        return ToolResult(tool=self.name, success=True,
                           output={"result": "..."},
                           output_path="./output.tif")

TOOL_REGISTRY["my_sar_analysis"] = MySARTool
```

## Pull Request Checklist

- [ ] Tests pass (`pytest tests/ --ignore=tests/test_live.py`)
- [ ] No syntax errors (`ruff check pygeovision/`)
- [ ] Docstrings added
- [ ] CHANGELOG.md updated
- [ ] No hard-coded paths
- [ ] No credentials in code
