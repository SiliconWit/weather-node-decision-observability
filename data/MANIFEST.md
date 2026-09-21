# Weather record

| | |
|---|---|
| Source | NASA POWER daily point API, version 2.9.7 |
| Point | latitude -0.4201, longitude 36.9476 (DeKUT, Nyeri) |
| Period | 2005-01-01 to 2024-12-31, local solar time |
| Community | AG (radiation in MJ per square metre per day) |

| File | Variables | MD5 |
|---|---|---|
| `nyeri_daily.json` | T2M, T2M_MAX, T2M_MIN, RH2M, WS2M, PS, ALLSKY_SFC_SW_DWN, PRECTOTCORR | `bd83dad44f684a939d189a45b4c94254` |
| `nyeri_wd.json` | WD2M | `88e645180d72ea1456b34cb9db7fa750` |

The two files are separate requests because POWER serves wind direction from a
different source product.

Both files are committed here, and every result is computed from them. POWER revises
its record, so a fresh download may not match the checksums above.
`data/fetch_power.py` fetches a fresh copy into `data/download/` for comparison and
leaves the committed files untouched. NASA POWER data are freely available; see
https://power.larc.nasa.gov.
