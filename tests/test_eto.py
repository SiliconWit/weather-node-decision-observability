"""Reference evapotranspiration, checked against FAO-56's own tabulated values.

Nothing here depends on the Nyeri record. If these fail, every depletion the
reference policy computes is wrong, and nothing later will say so.
"""
import numpy as np

import eto


def test_saturation_vapour_pressure_matches_the_fao_table():
    """FAO-56 Annex 2, Table 2.3: e_s at 20 C is 2.338 kPa."""
    assert abs(float(eto.es_curve(20.0)) - 2.338) < 0.002


def test_extraterrestrial_radiation_matches_example_8():
    """FAO-56 Example 8: 20 degrees south, 3 September, R_a = 32.2 MJ/m2/day."""
    assert abs(float(eto.extraterrestrial(246, -20.0)) - 32.2) < 0.2


def test_eto_is_non_negative_and_rises_with_radiation():
    doy = np.full(5, 60)
    kw = dict(Tmax=np.full(5, 24.0), Tmin=np.full(5, 11.0), RH=np.full(5, 70.0),
              u2=np.full(5, 2.0), P=np.full(5, 79.0), doy=doy, lat_deg=-0.42, z=2062.0)
    low = eto.eto_pm(Rs=np.full(5, 12.0), **kw)
    high = eto.eto_pm(Rs=np.full(5, 24.0), **kw)
    assert np.all(low >= 0)
    assert np.all(high > low), "more sunshine must mean more evaporative demand"
