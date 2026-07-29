"""Tests for pygeovision.cli.main — real bugs found and fixed:

  1. `pygeovision ai train` crashed with TypeError on EVERY invocation:
     TrainingConfig(task=task, ...) was called, but TrainingConfig is a
     plain dataclass with no `task` field at all.
  2. `--backbone` was a real, documented CLI flag with a default, but
     never referenced anywhere in the function body — a user's explicit
     backbone choice was silently discarded.
  3. `--validate` on `indices compute` and `postprocess vectorise` was
     accepted and documented but never referenced anywhere — running
     with --validate had zero effect. Also found a dead `dict(...)`
     expression statement in the same function (constructed, never used).
"""
import pytest

click = pytest.importorskip("click", reason="click not installed")
from click.testing import CliRunner


class TestAITrainCommand:
    """Regression tests for the real TrainingConfig(task=...) crash."""

    def test_does_not_crash(self, tmp_path):
        from pygeovision.cli.main import cli
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(cli, [
            "ai", "train", "segmentation",
            "--data", str(data_dir),
            "--output", str(tmp_path / "model.pth"),
        ])
        assert result.exit_code == 0, result.output
        assert result.exception is None

    def test_backbone_flag_appears_in_output(self, tmp_path):
        """Regression test for the real dead-parameter bug: --backbone
        was accepted but never referenced, so a user's explicit choice
        was silently discarded from the guidance output."""
        from pygeovision.cli.main import cli
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(cli, [
            "ai", "train", "segmentation",
            "--data", str(data_dir),
            "--output", str(tmp_path / "model.pth"),
            "--backbone", "efficientnet-b4",
        ])
        assert result.exit_code == 0, result.output
        assert "efficientnet-b4" in result.output

    def test_default_backbone_also_appears(self, tmp_path):
        from pygeovision.cli.main import cli
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        runner = CliRunner()
        result = runner.invoke(cli, [
            "ai", "train", "detection",
            "--data", str(data_dir),
            "--output", str(tmp_path / "model.pth"),
        ])
        assert result.exit_code == 0, result.output
        assert "resnet50" in result.output  # the documented default


class TestIndicesComputeValidateFlag:
    """Regression tests for the real dead --validate flag."""

    def _make_stack(self, tmp_path, bands=6, size=50):
        rasterio = pytest.importorskip("rasterio", reason="rasterio not installed")
        import numpy as np
        from rasterio.transform import from_bounds
        path = tmp_path / "stack.tif"
        transform = from_bounds(0, 0, size * 10, size * 10, size, size)
        data = np.random.randint(1, 5000, (bands, size, size)).astype("uint16")
        with rasterio.open(path, "w", driver="GTiff", height=size, width=size, count=bands,
                            dtype="uint16", crs="EPSG:32630", transform=transform) as dst:
            dst.write(data)
        return path

    def test_validate_flag_produces_real_validation_output(self, tmp_path):
        pytest.importorskip("rasterio", reason="rasterio not installed")
        from pygeovision.cli.main import cli
        img_path = self._make_stack(tmp_path)
        out_path = tmp_path / "ndvi.tif"

        runner = CliRunner()
        result = runner.invoke(cli, [
            "indices", "compute", str(img_path), "ndvi",
            "--output", str(out_path), "--validate",
        ])
        assert result.exit_code == 0, result.output
        assert "Validation:" in result.output, "the --validate flag had no effect"
        assert "PASSED" in result.output or "FIXED" in result.output

    def test_without_validate_flag_no_validation_output(self, tmp_path):
        """Confirms --validate genuinely gates the behavior (not always-on)."""
        pytest.importorskip("rasterio", reason="rasterio not installed")
        from pygeovision.cli.main import cli
        img_path = self._make_stack(tmp_path)
        out_path = tmp_path / "ndvi.tif"

        runner = CliRunner()
        result = runner.invoke(cli, [
            "indices", "compute", str(img_path), "ndvi", "--output", str(out_path),
        ])
        assert result.exit_code == 0, result.output
        assert "Validation:" not in result.output

    def test_no_dead_dict_expression_left_over(self):
        """Regression test: a bare `dict(...)` expression statement was
        constructed and immediately discarded in this function — confirm
        the source no longer contains it."""
        import inspect
        import sys
        import pygeovision.cli.main  # ensure it's loaded into sys.modules
        cli_main_mod = sys.modules["pygeovision.cli.main"]
        func = cli_main_mod.indices_compute.callback  # unwrap the click.Command
        source = inspect.getsource(func)
        for line in source.splitlines():
            stripped = line.strip()
            assert not stripped.startswith("dict("), f"dead dict() expression still present: {stripped}"


class TestPostVectoriseValidateFlag:
    """Regression tests for the real dead --validate flag on vectorise."""

    def _make_pred_raster(self, tmp_path, size=50):
        rasterio = pytest.importorskip("rasterio", reason="rasterio not installed")
        import numpy as np
        from rasterio.transform import from_bounds
        path = tmp_path / "pred.tif"
        transform = from_bounds(0, 0, size * 10, size * 10, size, size)
        data = np.zeros((1, size, size), dtype="uint8")
        data[0, 10:30, 10:30] = 1
        with rasterio.open(path, "w", driver="GTiff", height=size, width=size, count=1,
                            dtype="uint8", crs="EPSG:32630", transform=transform) as dst:
            dst.write(data)
        return path

    def test_validate_flag_checks_geometry_validity(self, tmp_path):
        pytest.importorskip("rasterio", reason="rasterio not installed")
        pytest.importorskip("shapely", reason="shapely not installed")
        from pygeovision.cli.main import cli
        pred_path = self._make_pred_raster(tmp_path)
        out_path = tmp_path / "buildings.geojson"

        runner = CliRunner()
        result = runner.invoke(cli, [
            "postprocess", "vectorise", str(pred_path),
            "--output", str(out_path), "--target-class", "1",
            "--min-area", "10", "--validate",
        ])
        assert result.exit_code == 0, result.output
        assert "Validation:" in result.output, "the --validate flag had no effect"
        assert "PASSED" in result.output
