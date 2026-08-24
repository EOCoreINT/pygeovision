"""Tests for pygeovision.data.fetch._derive_asset_key().

Real bugs found and fixed, traced from a second real production crash
after the first multi-asset download fix:

  "run_pair: image pair is not pixel-aligned — ..._SR_B1_EPSG_4326.TIF
  is 6179x9158, ..._QA_PIXEL.TIF is 7851x7741."

Tracing this back: the key-derivation logic used to populate
DownloadResult.asset_paths had two real bugs of its own:

  1. It treated ANY trailing all-digit token as a resolution suffix
     (checking `token.rstrip("m").isdigit()`, which is True even for a
     token with no "m" to strip at all) — "...B04_10m_EPSG_4326.tif"
     (a real reproject:EPSG:4326 output) has "4326" as its last token,
     which passed this check, deriving "EPSG" as the key instead of
     "B04".
  2. It only ever took the SINGLE last underscore-segment, truncating
     real compound Landsat identifiers: "QA_PIXEL" became "PIXEL",
     "SR_B1" became "B1" — losing exactly the prefix that distinguishes
     a real reflectance band from a quality-assessment band.

Every filename tested here is a real, exact filename from one of two
separate actual production failure logs, not a synthetic approximation.
"""
import pytest


class TestDeriveAssetKeyRealSentinel2Filenames:
    """Every case here is an exact real filename from the first
    production crash log (the AOT-selected-for-inference bug)."""

    def test_aot(self):
        from pygeovision.data.fetch import _derive_asset_key
        stem = "S2A_MSIL2A_20240615T153941_R011_T18TXL_20240616T010729_T18TXL_20240615T153941_AOT_10m"
        assert _derive_asset_key(stem) == "AOT"

    def test_b02(self):
        from pygeovision.data.fetch import _derive_asset_key
        stem = "S2A_MSIL2A_20240615T153941_R011_T18TXL_20240616T010729_T18TXL_20240615T153941_B02_10m"
        assert _derive_asset_key(stem) == "B02"

    def test_b8a(self):
        from pygeovision.data.fetch import _derive_asset_key
        stem = "S2A_MSIL2A_20240615T153941_R011_T18TXL_20240616T010729_T18TXL_20240615T153941_B8A_20m"
        assert _derive_asset_key(stem) == "B8A"

    def test_scl(self):
        from pygeovision.data.fetch import _derive_asset_key
        stem = "S2A_MSIL2A_20240615T153941_R011_T18TXL_20240616T010729_T18TXL_20240615T153941_SCL_20m"
        assert _derive_asset_key(stem) == "SCL"


class TestDeriveAssetKeyRealLandsatFilenames:
    """Every case here is an exact real filename from the second
    production crash log (the QA_PIXEL-selected-for-inference bug) or
    the earlier full-asset-list log for the same real scenes."""

    def test_sr_b1_with_real_epsg_reproject_suffix(self):
        """The specific real regression: this exact filename's trailing
        '4326' token was previously mis-treated as a resolution suffix,
        deriving 'EPSG' as the key instead of 'SR_B1'."""
        from pygeovision.data.fetch import _derive_asset_key
        stem = ("LE07_L2SP_013032_20180616_02_T1_LE07_L2SP_013032_20180616_"
                "20200829_02_T1_SR_B1_EPSG_4326")
        assert _derive_asset_key(stem) == "SR_B1"

    def test_qa_pixel_not_truncated_to_pixel(self):
        from pygeovision.data.fetch import _derive_asset_key
        stem = ("LC08_L2SP_014031_20240615_02_T1_LC08_L2SP_014031_20240615_"
                "20240705_02_T1_QA_PIXEL")
        assert _derive_asset_key(stem) == "QA_PIXEL"

    def test_st_qa(self):
        from pygeovision.data.fetch import _derive_asset_key
        stem = ("LE07_L2SP_013032_20180616_02_T1_LE07_L2SP_013032_20180616_"
                "20200829_02_T1_ST_QA")
        assert _derive_asset_key(stem) == "ST_QA"

    def test_sr_b3(self):
        from pygeovision.data.fetch import _derive_asset_key
        stem = ("LE07_L2SP_013032_20180616_02_T1_LE07_L2SP_013032_20180616_"
                "20200829_02_T1_SR_B3")
        assert _derive_asset_key(stem) == "SR_B3"

    def test_st_b10(self):
        from pygeovision.data.fetch import _derive_asset_key
        stem = ("LC08_L2SP_014031_20240615_02_T1_LC08_L2SP_014031_20240615_"
                "20240705_02_T1_ST_B10")
        assert _derive_asset_key(stem) == "ST_B10"

    def test_sr_qa_aerosol(self):
        from pygeovision.data.fetch import _derive_asset_key
        stem = ("LC08_L2SP_014031_20240615_02_T1_LC08_L2SP_014031_20240615_"
                "20240705_02_T1_SR_QA_AEROSOL")
        assert _derive_asset_key(stem) == "SR_QA_AEROSOL"

    def test_sr_atmos_opacity(self):
        from pygeovision.data.fetch import _derive_asset_key
        stem = ("LE07_L2SP_013032_20180616_02_T1_LE07_L2SP_013032_20180616_"
                "20200829_02_T1_SR_ATMOS_OPACITY")
        assert _derive_asset_key(stem) == "SR_ATMOS_OPACITY"

    def test_qa_radsat(self):
        from pygeovision.data.fetch import _derive_asset_key
        stem = ("LC08_L2SP_014031_20240615_02_T1_LC08_L2SP_014031_20240615_"
                "20240705_02_T1_QA_RADSAT")
        assert _derive_asset_key(stem) == "QA_RADSAT"
