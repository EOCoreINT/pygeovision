"""NDRE needs a red-edge band: the default six-band stack has none, so it must not be computed from SWIR."""
import numpy as np
import pytest
from click.testing import CliRunner

rasterio = pytest.importorskip("rasterio")


@pytest.fixture()
def six_band(tmp_path):
    from rasterio.transform import from_origin
    p = tmp_path / "stack.tif"
    data = np.random.default_rng(1).uniform(0.05, 0.6, (6, 20, 20)).astype("float32")
    with rasterio.open(p, "w", driver="GTiff", height=20, width=20, count=6, dtype="float32",
                       crs="EPSG:4326", transform=from_origin(0, 1, 0.01, 0.01)) as dst:
        dst.write(data)
    return str(p)


def test_compute_all_skips_ndre_without_red_edge(six_band, tmp_path):
    from pygeovision.data.indices import SpectralIndices
    out = SpectralIndices().compute_all(six_band, indices=["ndvi", "ndre"], output_dir=str(tmp_path / "o"))
    assert "ndvi" in out and "ndre" not in out


def test_compute_all_computes_ndre_with_red_edge(six_band, tmp_path):
    from pygeovision.data.indices import SpectralIndices
    out = SpectralIndices().compute_all(six_band, indices=["ndre"], output_dir=str(tmp_path / "o"),
                                        band_map={"red_edge": 5})
    assert "ndre" in out


def test_cli_ndre_requires_red_edge(six_band):
    from pygeovision.cli.main import cli
    r = CliRunner().invoke(cli, ["indices", "compute", six_band, "ndre"])
    assert r.exit_code == 2
    assert "red-edge" in r.output
    ok = CliRunner().invoke(cli, ["indices", "compute", six_band, "ndre", "--red-edge", "5"])
    assert ok.exit_code == 0


def test_file_outputs_keep_nodata_for_every_index(tmp_path):
    """A NaN (cloud) input pixel must come out as nodata, not as a plausible index value of 0."""
    from rasterio.transform import from_origin

    from pygeovision.data.indices import SpectralIndices
    data = np.full((6, 4, 4), 0.3, dtype="float32")
    data[:, 1, 1] = np.nan
    src = tmp_path / "s.tif"
    with rasterio.open(src, "w", driver="GTiff", height=4, width=4, count=6, dtype="float32",
                       crs="EPSG:4326", transform=from_origin(0, 1, 0.01, 0.01)) as dst:
        dst.write(data)
    ix = SpectralIndices()
    for name, call in {
        "ndvi": lambda o: ix.ndvi(str(src), 3, 4, o),
        "evi": lambda o: ix.evi(str(src), 1, 3, 4, output_path=o),
        "mndwi": lambda o: ix.mndwi(str(src), 2, 5, o),
        "nbr": lambda o: ix.nbr(str(src), 4, 6, o),
    }.items():
        out = call(str(tmp_path / f"{name}.tif"))
        with rasterio.open(out) as r:
            a = r.read(1)
            assert a[1, 1] == r.nodata, name
            assert a[0, 0] != r.nodata, name


def test_array_outputs_are_finite():
    from pygeovision.data.indices import SpectralIndices
    a = np.array([[0.5, np.nan]], dtype="float32")
    out = SpectralIndices().ndvi(a)          # an array in, an array out: no data becomes 0
    assert np.isfinite(out).all()
