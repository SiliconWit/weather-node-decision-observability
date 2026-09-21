"""FAO-56 reference evapotranspiration, Penman-Monteith (Allen et al. 1998, eq. 6)."""
import numpy as np

GSC = 0.0820          # solar constant, MJ m-2 min-1
SIGMA = 4.903e-9      # Stefan-Boltzmann, MJ K-4 m-2 day-1
ALBEDO = 0.23         # reference grass

def es_curve(T):                      # eq 11
    return 0.6108 * np.exp(17.27 * T / (T + 237.3))

def delta_slope(T):                   # eq 13
    return 4098.0 * es_curve(T) / (T + 237.3) ** 2

def psychrometric(P):                 # eq 8
    return 0.665e-3 * P

def pressure_from_elev(z):            # eq 7
    return 101.3 * ((293.0 - 0.0065 * z) / 293.0) ** 5.26

def extraterrestrial(doy, lat_deg):   # eq 21, 23, 24, 25
    phi = np.deg2rad(lat_deg)
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365.0)
    dec = 0.409 * np.sin(2 * np.pi * doy / 365.0 - 1.39)
    x = np.clip(-np.tan(phi) * np.tan(dec), -1.0, 1.0)
    ws = np.arccos(x)
    return (24 * 60 / np.pi) * GSC * dr * (
        ws * np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.sin(ws))

def net_radiation(Rs, Ra, Tmax, Tmin, ea, z):
    Rns = (1 - ALBEDO) * Rs                              # eq 38
    Rso = (0.75 + 2e-5 * z) * Ra                         # eq 37
    ratio = np.clip(Rs / np.maximum(Rso, 1e-9), 0.3, 1.0)
    Rnl = (SIGMA * ((Tmax + 273.16) ** 4 + (Tmin + 273.16) ** 4) / 2
           * (0.34 - 0.14 * np.sqrt(np.maximum(ea, 0)))
           * (1.35 * ratio - 0.35))                      # eq 39
    return Rns - Rnl

def eto_pm(Tmax, Tmin, RH, u2, Rs, P, doy, lat_deg, z, G=0.0):
    """Returns ETo in mm/day. Inputs are daily values."""
    Tmean = (Tmax + Tmin) / 2.0
    D = delta_slope(Tmean)
    g = psychrometric(P)
    es = (es_curve(Tmax) + es_curve(Tmin)) / 2.0
    ea = es * RH / 100.0                                 # eq 19 fallback
    Ra = extraterrestrial(doy, lat_deg)
    Rn = net_radiation(Rs, Ra, Tmax, Tmin, ea, z)
    num = 0.408 * D * (Rn - G) + g * (900.0 / (Tmean + 273.0)) * u2 * (es - ea)
    den = D + g * (1 + 0.34 * u2)
    return np.maximum(num / den, 0.0)
