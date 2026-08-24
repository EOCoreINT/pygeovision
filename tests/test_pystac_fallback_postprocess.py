"""Tests for the "pystac-fallback direct download" post-processing path
(SatelliteFetcher._apply_post_process, _is_pixel_space_transform),
traced from a real production crash.

Real bugs found and fixed, in the order they were uncovered:

  1. _apply_post_process() only ever handled "reproject:" and "cog"
     steps -- "unzip" fell through unhandled ("Unknown post-process
     step 'unzip' -- skipping"), so a genuinely-still-zipped download
     got passed straight into reproject as if it were a real raster.

  2. When post-processing genuinely failed (corrupt zip, no raster
     found inside, rescue failed), the function returned the original,
     equally-unusable file (`return current`) instead of signalling
     real failure -- silently letting a broken scene continue through
     the pipeline as if it had succeeded. The caller's
     `... or out_file` fallback made this worse. Both now propagate a
     real None/failure instead.

  3. The heaviest bug: _is_pixel_space_transform()'s corruption
     heuristic used meters-only thresholds, applied to the OUTPUT
     transform after reprojection -- but EPSG:4326 (the single most
     common target in this pipeline) is geographic (degrees), where a
     genuinely valid 250m pixel is ~0.0026 degrees, always far below
     any meters-scale threshold. This flagged virtually every valid
     geographic reprojection as "corrupted". A naive CRS-aware fix
     using origin-proximity-to-zero as a corruption signal ALSO failed,
     because real geographic origins (longitude/latitude) are always
     numerically small regardless of validity. The real fix uses pixel
     size alone for geographic CRS (a near 1.0 degree/pixel, ~111km, is
     the real corruption signature -- not origin proximity).
"""
import pytest

rasterio = pytest.importorskip("rasterio", reason="rasterio not installed")

import numpy as np
from rasterio.crs import CRS
from rasterio.transform import Affine, from_bounds


class TestIsPixelSpaceTransformCRSAware:
    """The core, most important regression tests: correct discrimination
    across all four combinations of valid/corrupt x geographic/projected."""

    def test_valid_geographic_transform_not_flagged(self):
        from pygeovision.data.fetch import _is_pixel_space_transform
        t = from_bounds(-74.1, 40.6, -73.7, 40.9, 40, 40)  # ~0.01 deg/pixel, genuinely valid
        assert _is_pixel_space_transform(t, crs=CRS.from_epsg(4326)) is False

    def test_corrupt_geographic_transform_is_flagged(self):
        from pygeovision.data.fetch import _is_pixel_space_transform
        t = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 40.0)  # the documented corruption signature
        assert _is_pixel_space_transform(t, crs=CRS.from_epsg(4326)) is True

    def test_valid_projected_utm_transform_not_flagged(self):
        from pygeovision.data.fetch import _is_pixel_space_transform
        t = from_bounds(500000, 4500000, 510000, 4510000, 40, 40)  # 250m/pixel, genuinely valid
        assert _is_pixel_space_transform(t, crs=CRS.from_epsg(32618)) is False

    def test_corrupt_projected_utm_transform_is_flagged(self):
        """The original, documented Sentinel-1 corruption case -- must
        not regress."""
        from pygeovision.data.fetch import _is_pixel_space_transform
        t = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 500.0)
        assert _is_pixel_space_transform(t, crs=CRS.from_epsg(32618)) is True

    def test_no_crs_given_preserves_original_meters_based_behaviour(self):
        """Backward compatibility: omitting crs must not change behaviour
        for existing callers that only ever checked projected data."""
        from pygeovision.data.fetch import _is_pixel_space_transform
        t = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 500.0)
        assert _is_pixel_space_transform(t) is True

    def test_none_transform_is_always_flagged(self):
        from pygeovision.data.fetch import _is_pixel_space_transform
        assert _is_pixel_space_transform(None) is True


def _zip_a_real_raster(tmp_path, size=40):
    inner_tif = tmp_path / "inner_extracted" / "LE07_red.TIF"
    inner_tif.parent.mkdir(parents=True)
    with rasterio.open(inner_tif, "w", driver="GTiff", height=size, width=size, count=1,
                        dtype="uint16", crs="EPSG:32618",
                        transform=from_bounds(500000, 4500000, 510000, 4510000, size, size)) as dst:
        dst.write(np.full((size, size), 777, dtype="uint16"), 1)

    import zipfile
    zip_path = tmp_path / "LE07_L2SP_013032_20180616_02_T1_red.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(inner_tif, arcname="LE07_red.TIF")
    return zip_path


class TestApplyPostProcessRealUnzipSupport:
    def test_real_zip_with_real_raster_extracts_and_reprojects_successfully(self, tmp_path):
        from pygeovision.data.fetch import SatelliteFetcher
        fetcher = SatelliteFetcher.__new__(SatelliteFetcher)
        zip_path = _zip_a_real_raster(tmp_path)

        result = fetcher._apply_post_process(zip_path, ["unzip", "reproject:EPSG:4326"])
        assert result is not None

        with rasterio.open(result) as src:
            assert src.crs == CRS.from_epsg(4326)
            data = src.read(1)
        assert 777 in data

    def test_invalid_zip_fails_clearly_not_silently(self, tmp_path):
        from pygeovision.data.fetch import SatelliteFetcher
        fetcher = SatelliteFetcher.__new__(SatelliteFetcher)

        fake_zip = tmp_path / "not_really_a_zip.zip"
        fake_zip.write_bytes(b"this is not a real zip file")

        result = fetcher._apply_post_process(fake_zip, ["unzip", "reproject:EPSG:4326"])
        assert result is None

    def test_zip_with_no_raster_inside_fails_clearly(self, tmp_path):
        from pygeovision.data.fetch import SatelliteFetcher
        import zipfile
        fetcher = SatelliteFetcher.__new__(SatelliteFetcher)

        zip_path = tmp_path / "empty_of_rasters.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("readme.txt", "no raster here")

        result = fetcher._apply_post_process(zip_path, ["unzip", "reproject:EPSG:4326"])
        assert result is None

    def test_non_zip_file_skips_unzip_step_harmlessly(self, tmp_path):
        """A file that isn't a .zip (e.g. already a real .tif) should
        pass through the unzip step unchanged, not fail."""
        from pygeovision.data.fetch import SatelliteFetcher
        fetcher = SatelliteFetcher.__new__(SatelliteFetcher)

        size = 20
        tif_path = tmp_path / "already_a_tif.tif"
        with rasterio.open(tif_path, "w", driver="GTiff", height=size, width=size, count=1,
                            dtype="uint16", crs="EPSG:4326",
                            transform=from_bounds(-74.1, 40.6, -73.7, 40.9, size, size)) as dst:
            dst.write(np.full((size, size), 42, dtype="uint16"), 1)

        result = fetcher._apply_post_process(tif_path, ["unzip"])
        assert result == tif_path
