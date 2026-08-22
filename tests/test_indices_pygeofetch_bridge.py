"""Tests for pygeovision.data.indices' real pygeofetch delegation bridge.

Real architectural finding this addresses: pygeovision maintained its own
complete 737-line SpectralIndices implementation running in parallel with
pygeofetch's own (now more capable) SpectralIndices — exactly the
duplication the pygeofetch integration guide's anti-patterns table warns
against. For file-based inputs (the common case), NDVI/NDWI now genuinely
delegate to pygeofetch's real implementation rather than recomputing the
same formula a second time.

pygeofetch's index API is file-path-only (no in-memory array mode at
all), so array inputs correctly stay on pygeovision's own formula — that
is a genuine capability gap, not leftover duplication.
"""
import pytest

rasterio = pytest.importorskip("rasterio", reason="rasterio not installed")
pygeofetch = pytest.importorskip("pygeofetch", reason="pygeofetch not installed")

import numpy as np
from rasterio.transform import from_bounds


def _make_stack(path, seed=0, size=50):
    transform = from_bounds(0, 0, size * 10, size * 10, size, size)
    rng = np.random.default_rng(seed)
    data = rng.integers(1, 5000, (4, size, size)).astype("uint16")  # blue,green,red,nir
    with rasterio.open(path, "w", driver="GTiff", height=size, width=size, count=4,
                        dtype="uint16", crs="EPSG:32630", transform=transform) as dst:
        dst.write(data)
    return data


class TestNDVIDelegation:
    def test_file_input_matches_expected_formula_exactly(self, tmp_path):
        from pygeovision.data.indices import SpectralIndices
        scene = tmp_path / "scene.tif"
        data = _make_stack(scene, seed=0)

        idx = SpectralIndices()
        out = tmp_path / "ndvi.tif"
        idx.ndvi(str(scene), red_band=3, nir_band=4, output_path=str(out))

        with rasterio.open(out) as src:
            ndvi_arr = src.read(1)
            assert src.crs is not None  # georeferencing preserved through the round-trip

        red, nir = data[2].astype(np.float32), data[3].astype(np.float32)
        expected = (nir - red) / (nir + red + 1e-10)
        assert np.abs(ndvi_arr - expected).max() < 1e-3

    def test_file_input_actually_calls_pygeofetchs_real_method(self, tmp_path, monkeypatch):
        """Proves genuine delegation, not just a coincidentally-correct
        result — the earlier test alone can't distinguish 'delegated' from
        'silently fell back but got the right answer anyway'."""
        from pygeovision.data.indices import SpectralIndices
        from pygeofetch.processing.indices import SpectralIndices as RealPgfIndices

        scene = tmp_path / "scene.tif"
        _make_stack(scene, seed=0)

        calls = []
        original_ndvi = RealPgfIndices.ndvi

        def _tracking_ndvi(self, **kwargs):
            calls.append(kwargs)
            return original_ndvi(self, **kwargs)
        monkeypatch.setattr(RealPgfIndices, "ndvi", _tracking_ndvi)

        idx = SpectralIndices()
        idx.ndvi(str(scene), red_band=3, nir_band=4)

        assert len(calls) == 1, "pygeofetch's real ndvi() was not called — delegation isn't happening"
        assert "red" in calls[0] and "nir" in calls[0]

    def test_falls_back_to_native_when_pygeofetch_call_fails(self, tmp_path, monkeypatch):
        """If the real pygeofetch call fails for any reason, this must
        fall back to pygeovision's own verified formula rather than
        propagate the failure or silently return garbage."""
        from pygeovision.data import indices as indices_mod
        from pygeovision.data.indices import SpectralIndices

        scene = tmp_path / "scene.tif"
        data = _make_stack(scene, seed=1)

        class BrokenPgfIndices:
            def ndvi(self, **kwargs):
                raise RuntimeError("simulated pygeofetch failure")

        monkeypatch.setattr(indices_mod, "_pygeofetch_indices", lambda: BrokenPgfIndices())

        idx = SpectralIndices()
        result = idx.ndvi(str(scene), red_band=3, nir_band=4)  # no output_path -> array

        red, nir = data[2].astype(np.float32), data[3].astype(np.float32)
        expected = (nir - red) / (nir + red + 1e-10)
        assert np.abs(result - expected).max() < 1e-3


class TestNDWIDelegation:
    def test_file_input_matches_expected_formula_exactly(self, tmp_path):
        from pygeovision.data.indices import SpectralIndices
        scene = tmp_path / "scene.tif"
        data = _make_stack(scene, seed=2)

        idx = SpectralIndices()
        out = tmp_path / "ndwi.tif"
        idx.ndwi(str(scene), green_band=2, nir_band=4, output_path=str(out))

        with rasterio.open(out) as src:
            ndwi_arr = src.read(1)

        green, nir = data[1].astype(np.float32), data[3].astype(np.float32)
        expected = (green - nir) / (green + nir + 1e-10)
        assert np.abs(ndwi_arr - expected).max() < 1e-3


class TestArrayInputStillWorksAndUsesNativePath:
    """pygeofetch's index API has no in-memory-array mode — confirms
    array inputs correctly stay on pygeovision's own path rather than
    attempting (and failing) to delegate."""

    def test_array_input_does_not_attempt_delegation(self, monkeypatch):
        from pygeovision.data import indices as indices_mod
        from pygeovision.data.indices import SpectralIndices

        called = {"delegated": False}

        def _fake_pgf():
            called["delegated"] = True
            return None
        monkeypatch.setattr(indices_mod, "_pygeofetch_indices", _fake_pgf)

        idx = SpectralIndices()
        arr = np.random.default_rng(0).random((10, 10)).astype(np.float32) * 5000
        idx.ndvi(arr, red_band=1, nir_band=2)

        assert called["delegated"] is False, "array input should never attempt pygeofetch delegation"


@pytest.mark.xfail(reason=(
    "Real pre-existing bug found while testing the delegation bridge, not "
    "introduced by it: _load_band() on a bare ndarray ignores band_idx "
    "entirely, so ndvi(single_2d_array, red_band=1, nir_band=2) computes "
    "(arr - arr) / (arr + arr) = 0 everywhere — the two 'bands' are "
    "actually the same array. The documented '2-element list/array "
    "[red, nir]' input format also isn't implemented at all (_load_band "
    "has no list branch). Needs a real fix as a follow-up, tracked here "
    "so it isn't silently lost."
))
def test_single_array_input_is_not_actually_meaningful():
    from pygeovision.data.indices import SpectralIndices
    idx = SpectralIndices()
    rng = np.random.default_rng(0)
    red = rng.random((10, 10)).astype(np.float32) * 5000
    nir = rng.random((10, 10)).astype(np.float32) * 5000
    real_expected = (nir - red) / (nir + red + 1e-10)

    # The documented usage pattern from the module docstring:
    # "From numpy array directly: ndvi_arr = client.indices.ndvi_array(red=red_arr, nir=nir_arr)"
    result = idx.ndvi([red, nir], red_band=1, nir_band=2)
    assert np.abs(result - real_expected).max() < 1e-3
