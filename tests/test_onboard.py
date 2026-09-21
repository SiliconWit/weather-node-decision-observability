"""The on-node recursion and event scoring."""
import numpy as np

import onboard as O
import reference as R


def test_recursion_given_reference_inputs_reproduces_the_reference():
    rng = np.random.default_rng(0)
    n = 400
    eto = 3.5 + rng.standard_normal(n) * 0.8
    rain = np.where(rng.random(n) < 0.3, rng.gamma(0.8, 8.0, n), 0.0)
    dap = (np.arange(n) % 130) + 1
    wb = R.water_balance(np.maximum(eto, 0), rain, dap)
    ref = (wb["Dr"] > wb["RAW"]).astype(int)
    got = O.onboard_decision(np.maximum(eto, 0), rain, dap)
    assert np.array_equal(got, ref), "the node's recursion must match the reference step for step"


def test_event_scoring_counts_a_late_event_once():
    true = np.zeros(30, int); true[10] = 1
    pred = np.zeros(30, int); pred[11] = 1
    assert O.event_match(pred, true, 0)["f1"] == 0.0
    assert O.event_match(pred, true, 1)["f1"] == 1.0


def test_hargreaves_is_plausible_for_the_highlands():
    e = O.eto_hargreaves(np.full(365, 23.0), np.full(365, 11.0), np.arange(1, 366), -0.42)
    assert np.all((e > 1.0) & (e < 8.0))


def test_radiation_from_temperature_range_matches_example_15():
    """FAO-56 Example 15: Lyon, 45 degrees 43 minutes north, 200 m, 15 July,
    Tmax 26.6 C and Tmin 14.8 C, interior k_Rs 0.16, gives R_s = 22.3 MJ/m2/day."""
    rs = O.rs_from_temperature_range(26.6, 14.8, 196, 45.72, 200.0, krs=0.16)
    assert abs(float(rs) - 22.3) < 0.2


def test_radiation_from_temperature_range_matches_example_16():
    """FAO-56 Example 16: Bangkok, 13 degrees 44 minutes north, 15 April,
    Tmax 34.8 C and Tmin 25.6 C, coastal k_Rs 0.19, gives R_s = 21.9 MJ/m2/day."""
    rs = O.rs_from_temperature_range(34.8, 25.6, 105, 13.73, 0.0, krs=0.19)
    assert abs(float(rs) - 21.9) < 0.2


def test_radiation_estimate_is_capped_at_clear_sky():
    rs = O.rs_from_temperature_range(40.0, 0.0, 60, -0.42, 2062.0, krs=0.19)
    ra = O.extraterrestrial(60, -0.42)
    assert float(rs) <= (0.75 + 2e-5 * 2062.0) * float(ra) + 1e-9


def test_elevation_from_pressure_inverts_fao_eq_7():
    import eto
    for z in (0.0, 1000.0, 2062.0):
        assert abs(float(O.elevation_from_pressure(eto.pressure_from_elev(z))) - z) < 1e-6


def test_krs_calibration_recovers_the_coefficient_that_generated_the_target():
    rng = np.random.default_rng(1)
    n = 365
    tmax = 23 + 2 * rng.standard_normal(n); tmin = 11 + 1.5 * rng.standard_normal(n)
    rh = np.full(n, 70.0); u2 = np.full(n, 2.0); P = np.full(n, 79.2)
    doy = np.arange(1, n + 1)
    target = O.eto_pm_node(tmax, tmin, rh, u2, P, doy, -0.42, krs=0.175)
    k = O.calibrate_krs(target, tmax, tmin, rh, u2, P, doy, -0.42)
    assert abs(k - 0.175) < 1e-4
