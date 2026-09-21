"""The reference policy: properties any correct FAO-56 water balance must have."""
import numpy as np

import reference as R


def test_crop_coefficient_follows_the_stages():
    assert R.kcb_curve(1) == R.KCB["ini"]
    mid = R.L["ini"] + R.L["dev"] + 5
    assert R.kcb_curve(mid) == R.KCB["mid"]


def test_depletion_stays_inside_the_soil():
    n = 130
    dap = np.arange(1, n + 1)
    wb = R.water_balance(np.full(n, 5.0), np.zeros(n), dap, irrigate=False)
    assert np.all(wb["Dr"] >= 0)
    assert np.all(wb["Dr"] <= wb["TAW"] + 1e-9), "depletion cannot exceed total available water"
    assert np.all(np.diff(wb["Dr"]) >= -1e-9), "with no water in, depletion cannot fall"


def test_irrigation_resets_depletion():
    n = 130
    dap = np.arange(1, n + 1)
    wb = R.water_balance(np.full(n, 5.0), np.zeros(n), dap, irrigate=True)
    days = np.flatnonzero(wb["irr"] > 0)
    assert len(days) > 0, "a dry season with no rain must trigger irrigation"
    d = days[0]
    assert wb["Dr"][d] < wb["Dr"][d - 1], "an irrigation must lower depletion on the day it is applied"


def test_thermal_time_accumulates_and_respects_the_base():
    tt = R.thermal_time(np.full(10, 20.0), np.full(10, 10.0))
    assert np.all(np.diff(tt) > 0)
    cold = R.thermal_time(np.full(10, 6.0), np.full(10, 2.0))
    assert np.allclose(cold, 0.0), "no development below the base temperature"
