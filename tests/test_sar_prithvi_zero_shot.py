"""The zero-shot SAR adapter is a VH threshold, and its result must say so."""
import numpy as np

from pygeovision.models.adapters.sar_prithvi import SARPrithviAdapter


def test_zero_shot_reports_that_it_is_a_threshold():
    vh = np.full((64, 64), 0.6, dtype="float32")
    vv = np.full((64, 64), 0.7, dtype="float32")
    vh[:8, :8] = 0.1
    vv[:8, :8] = 0.15
    res = SARPrithviAdapter(mode="zero_shot").run(vv, vh)
    assert res["method"] == "vh_threshold"
    assert int(res["prediction"].sum()) == 64
