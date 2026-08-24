"""Tests for pygeovision.data.texture — real Haralick GLCM computation.

Real gap this addresses: pygeofetch's SpectralIndices.texture() is
labelled "GLCM" but its actual implementation computes local windowed
variance of a single fixed-direction pixel difference via
scipy.ndimage.uniform_filter — a fast approximation, not a genuine
co-occurrence-matrix computation (confirmed by reading its source; the
code comment itself says this was chosen to avoid the real algorithm's
cost). For a study that specifically names "grey-level co-occurrence
matrix" as a citable method, this matters.

This module computes real GLCM via skimage.feature.graycomatrix/
graycoprops — verified against actual Haralick theory, not just "runs
without crashing": a genuinely smooth region must show low contrast/high
homogeneity/high energy/high ASM relative to a genuinely disturbed
region, within a realistically shared quantization range (one scene
containing both regions, not isolated per-region test patches — an
earlier draft of this test got this wrong and produced backwards
results purely from a quantization-range artifact, not a real bug).
"""
import pytest

skimage = pytest.importorskip("skimage", reason="scikit-image not installed")
rasterio = pytest.importorskip("rasterio", reason="rasterio not installed")

import numpy as np
from rasterio.transform import from_bounds


def _make_smooth_vs_disturbed_scene(size=56, seed=0):
    """One shared scene, smooth (undisturbed) top half + disturbed
    (checkerboard, mining-pit-like) bottom half — quantized together,
    matching real usage of a single raster tile."""
    rng = np.random.default_rng(seed)
    scene = np.full((size, size), 100.0, dtype=np.float32) + rng.normal(0, 1, (size, size))
    for i in range(size // 2, size, 4):
        for j in range(0, size, 4):
            scene[i:i + 4, j:j + 4] = rng.choice([20, 180])
    return scene


class TestGLCMDiscriminatesRealTexture:
    """The core correctness tests: real GLCM feature relationships must
    match Haralick theory, not just produce some numeric output."""

    def test_contrast_higher_for_disturbed_region(self):
        from pygeovision.data.texture import compute_glcm_texture
        scene = _make_smooth_vs_disturbed_scene()
        result = compute_glcm_texture(scene, window=7, mode="block", levels=32)
        n_blocks = result["contrast"].shape[0]
        smooth = result["contrast"][: n_blocks // 2, :].mean()
        disturbed = result["contrast"][n_blocks // 2 :, :].mean()
        assert disturbed > smooth * 5

    def test_homogeneity_higher_for_smooth_region(self):
        from pygeovision.data.texture import compute_glcm_texture
        scene = _make_smooth_vs_disturbed_scene()
        result = compute_glcm_texture(scene, window=7, mode="block", levels=32)
        n_blocks = result["homogeneity"].shape[0]
        smooth = result["homogeneity"][: n_blocks // 2, :].mean()
        disturbed = result["homogeneity"][n_blocks // 2 :, :].mean()
        assert smooth > disturbed

    def test_energy_and_asm_higher_for_smooth_region(self):
        from pygeovision.data.texture import compute_glcm_texture
        scene = _make_smooth_vs_disturbed_scene()
        result = compute_glcm_texture(scene, window=7, mode="block", levels=32,
                                       features=["energy", "ASM"])
        n_blocks = result["energy"].shape[0]
        for feat in ("energy", "ASM"):
            smooth = result[feat][: n_blocks // 2, :].mean()
            disturbed = result[feat][n_blocks // 2 :, :].mean()
            assert smooth > disturbed, f"{feat}: smooth={smooth} should exceed disturbed={disturbed}"

    def test_unknown_feature_raises_clear_error(self):
        from pygeovision.data.texture import compute_glcm_texture
        scene = _make_smooth_vs_disturbed_scene(size=14)
        with pytest.raises(ValueError, match="Unknown GLCM feature"):
            compute_glcm_texture(scene, window=7, features=["not_a_real_feature"])


class TestGLCMModes:
    def test_block_mode_output_is_coarser_than_input(self):
        from pygeovision.data.texture import compute_glcm_texture
        scene = _make_smooth_vs_disturbed_scene(size=56)
        result = compute_glcm_texture(scene, window=7, mode="block")
        assert result["contrast"].shape == (8, 8)  # 56/7 = 8

    def test_sliding_mode_output_matches_input_resolution(self):
        from pygeovision.data.texture import compute_glcm_texture
        small = np.zeros((10, 10), dtype=np.float32)
        small[:5, :] = 50.0
        small[5:, :] = np.random.default_rng(3).choice([10, 200], (5, 10)).astype(np.float32)
        result = compute_glcm_texture(small, window=3, mode="sliding", levels=16,
                                       features=["contrast"])
        assert result["contrast"].shape == (10, 10)
        assert result["contrast"][7:, :].mean() > result["contrast"][:3, :].mean()

    def test_invalid_mode_raises(self):
        from pygeovision.data.texture import compute_glcm_texture
        scene = _make_smooth_vs_disturbed_scene(size=14)
        with pytest.raises(ValueError, match="mode must be"):
            compute_glcm_texture(scene, window=7, mode="bogus")

    def test_window_larger_than_input_raises_clear_error(self):
        from pygeovision.data.texture import compute_glcm_texture
        tiny = np.zeros((5, 5), dtype=np.float32)
        with pytest.raises(ValueError, match="larger than the input"):
            compute_glcm_texture(tiny, window=10, mode="block")


class TestGLCMTextureRasterIO:
    def test_real_georeferenced_output(self, tmp_path):
        from pygeovision.data.texture import GLCMTexture

        size = 28
        scene = _make_smooth_vs_disturbed_scene(size=size)
        raster_path = tmp_path / "scene.tif"
        transform = from_bounds(0, 0, size * 10, size * 10, size, size)
        with rasterio.open(raster_path, "w", driver="GTiff", height=size, width=size, count=1,
                            dtype="float32", crs="EPSG:32630", transform=transform) as dst:
            dst.write(scene, 1)

        tex = GLCMTexture()
        out_path = tmp_path / "glcm.tif"
        result = tex.compute(str(raster_path), str(out_path), window=7, mode="block",
                              features=["contrast", "homogeneity", "energy"])

        assert result["success"] is True
        with rasterio.open(out_path) as src:
            assert src.crs is not None
            assert src.count == 3
            assert list(src.descriptions) == ["contrast", "homogeneity", "energy"]
            # block mode: pixel size scales by window factor (10m source -> 70m output)
            assert abs(src.transform.a - 70.0) < 1e-6

    def test_default_output_path(self, tmp_path):
        from pygeovision.data.texture import GLCMTexture

        size = 14
        scene = _make_smooth_vs_disturbed_scene(size=size)
        raster_path = tmp_path / "scene.tif"
        transform = from_bounds(0, 0, size * 10, size * 10, size, size)
        with rasterio.open(raster_path, "w", driver="GTiff", height=size, width=size, count=1,
                            dtype="float32", crs="EPSG:32630", transform=transform) as dst:
            dst.write(scene, 1)

        tex = GLCMTexture()
        result = tex.compute(str(raster_path), window=7, mode="block")
        assert result["output_path"] == str(tmp_path / "scene_glcm.tif")
