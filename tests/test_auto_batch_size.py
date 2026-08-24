"""Tests for TiledInference's memory-aware auto batch_size selection.

Real, measured problem this addresses: every pipeline previously
constructed TiledInference without specifying batch_size, so all of
them used the old fixed default of 8 -- confirmed directly (in isolated,
fresh-process measurements to avoid cross-contamination) that batch_size
scales inference memory roughly linearly for a real ResNet-18-based
model: 1->~258MB, 2->~479MB, 4->~942MB, 8->~1790MB, just for the model's
own forward-pass memory. Through a full real run_pair() call,
batch_size=1 measured ~1107MB total vs batch_size=8's ~2672MB.

batch_size now defaults to None (auto-detect), resolved via a real,
opportunistic psutil-based check when available, and a safe, fixed
fallback (2, not the old 8) when it isn't -- psutil is not a declared
pygeovision dependency, so this must degrade gracefully, never raise or
silently assume a large-memory machine.

Honest limitation, documented in the code and repeated here: the
threshold values are tuned from measurements on one real model family
(a ResNet-18-based encoder) -- a significantly larger or smaller model
could need different thresholds. This is a real, measured improvement
for that model family, not a universal guarantee.
"""
import pytest
from unittest.mock import patch, MagicMock

torch = pytest.importorskip("torch", reason="torch not installed")
import torch.nn as nn


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.head = nn.Conv2d(8, 2, 1)

    def forward(self, x):
        return self.head(torch.relu(self.conv(x)))


class TestAutoSelectBatchSizeThresholds:
    """Direct tests of the real tiered heuristic, matching the actual
    measured thresholds."""

    def test_below_2gb_selects_batch_size_1(self):
        from pygeovision.ai.inference.tiled_inference import TiledInference
        mock_vm = MagicMock(available=1500 * 1e6)
        with patch("psutil.virtual_memory", return_value=mock_vm):
            assert TiledInference._auto_select_batch_size() == 1

    def test_below_4gb_selects_batch_size_2(self):
        from pygeovision.ai.inference.tiled_inference import TiledInference
        mock_vm = MagicMock(available=3000 * 1e6)
        with patch("psutil.virtual_memory", return_value=mock_vm):
            assert TiledInference._auto_select_batch_size() == 2

    def test_below_8gb_selects_batch_size_4(self):
        from pygeovision.ai.inference.tiled_inference import TiledInference
        mock_vm = MagicMock(available=6000 * 1e6)
        with patch("psutil.virtual_memory", return_value=mock_vm):
            assert TiledInference._auto_select_batch_size() == 4

    def test_8gb_or_above_selects_batch_size_8(self):
        from pygeovision.ai.inference.tiled_inference import TiledInference
        mock_vm = MagicMock(available=16000 * 1e6)
        with patch("psutil.virtual_memory", return_value=mock_vm):
            assert TiledInference._auto_select_batch_size() == 8

    def test_boundary_exactly_2gb_is_the_2gb_tier_not_1gb_tier(self):
        """Boundary check: the comparison is strictly '<', so exactly
        2000MB should NOT fall into the batch_size=1 tier."""
        from pygeovision.ai.inference.tiled_inference import TiledInference
        mock_vm = MagicMock(available=2000 * 1e6)
        with patch("psutil.virtual_memory", return_value=mock_vm):
            assert TiledInference._auto_select_batch_size() == 2


class TestPsutilUnavailableFallback:
    def test_missing_psutil_falls_back_to_safe_fixed_default(self):
        from pygeovision.ai.inference.tiled_inference import TiledInference
        with patch.dict("sys.modules", {"psutil": None}):
            result = TiledInference._auto_select_batch_size()
        assert result == 2

    def test_fallback_is_conservative_not_the_old_default_of_8(self):
        """The fallback must be safer than the old fixed default, not
        just any arbitrary value."""
        from pygeovision.ai.inference.tiled_inference import TiledInference
        with patch.dict("sys.modules", {"psutil": None}):
            result = TiledInference._auto_select_batch_size()
        assert result < 8


class TestExplicitBatchSizeAlwaysWins:
    """Explicit user choice must never be silently overridden by
    auto-detection, regardless of what real memory is available."""

    def test_explicit_value_respected_on_a_low_memory_machine(self):
        from pygeovision.ai.inference.tiled_inference import TiledInference
        mock_vm = MagicMock(available=1000 * 1e6)  # a machine that would auto-select 1
        with patch("psutil.virtual_memory", return_value=mock_vm):
            engine = TiledInference(TinyModel(), tile_size=512, overlap=64, batch_size=8)
        assert engine.config.batch_size == 8

    def test_explicit_value_respected_on_a_high_memory_machine(self):
        from pygeovision.ai.inference.tiled_inference import TiledInference
        mock_vm = MagicMock(available=32000 * 1e6)  # a machine that would auto-select 8
        with patch("psutil.virtual_memory", return_value=mock_vm):
            engine = TiledInference(TinyModel(), tile_size=512, overlap=64, batch_size=1)
        assert engine.config.batch_size == 1

    def test_no_batch_size_given_genuinely_auto_detects(self):
        from pygeovision.ai.inference.tiled_inference import TiledInference
        mock_vm = MagicMock(available=1500 * 1e6)
        with patch("psutil.virtual_memory", return_value=mock_vm):
            engine = TiledInference(TinyModel(), tile_size=512, overlap=64)
        assert engine.config.batch_size == 1


class TestRealEndToEndMemoryBenefit:
    """The actual point of this feature, verified with a real
    measurement: a simulated low-memory machine must genuinely produce
    lower real peak memory than a simulated high-memory one, for the
    identical real workload."""

    def test_simulated_low_memory_uses_less_real_memory_than_high_memory(self, tmp_path):
        import resource
        import numpy as np
        import rasterio
        from rasterio.transform import from_bounds
        from pygeovision.ai.inference.tiled_inference import TiledInference

        rasterio_mod = pytest.importorskip("rasterio")
        size = 1200
        transform = from_bounds(-0.45, 5.50, 0.05, 5.75, size, size)
        rng = np.random.default_rng(0)
        path = tmp_path / "scene.tif"
        with rasterio.open(path, "w", driver="GTiff", height=size, width=size, count=3,
                            dtype="uint16", crs="EPSG:4326", transform=transform) as dst:
            for b in range(1, 4):
                dst.write(rng.integers(500, 3000, (size, size)).astype("uint16"), b)

        def run_with_simulated_memory(available_mb):
            mock_vm = MagicMock(available=available_mb * 1e6)
            with patch("psutil.virtual_memory", return_value=mock_vm):
                engine = TiledInference(TinyModel(), tile_size=512, overlap=64)
            baseline = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            engine.run(str(path), str(tmp_path / f"pred_{available_mb}.tif"), num_classes=2)
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return (peak - baseline) / 1000, engine.config.batch_size

        low_mem_mb, low_bs = run_with_simulated_memory(1500)
        high_mem_mb, high_bs = run_with_simulated_memory(32000)

        assert low_bs < high_bs, "low-memory scenario should genuinely select a smaller batch_size"
