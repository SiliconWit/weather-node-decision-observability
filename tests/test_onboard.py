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
