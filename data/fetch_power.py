"""Download the NASA POWER daily record this study uses and compare it with the manifest.

    python3 data/fetch_power.py

The record every result is computed from is already committed in data/. This
script fetches a fresh copy into data/download/, so that it never overwrites the
cached pair, and prints the checksums against those in MANIFEST.md. POWER revises
its record periodically, so a fresh download can differ from the cached copy; the
code reads the cached pair in data/.
"""
import hashlib, json, os, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "https://power.larc.nasa.gov/api/temporal/daily/point"
POINT = dict(latitude=-0.4201, longitude=36.9476, start=20050101, end=20241231,
             community="AG", format="JSON")
REQUESTS = {
    "nyeri_daily.json": "T2M,T2M_MAX,T2M_MIN,RH2M,WS2M,PS,ALLSKY_SFC_SW_DWN,PRECTOTCORR",
    "nyeri_wd.json": "WD2M",
}
EXPECTED = {"nyeri_daily.json": "bd83dad44f684a939d189a45b4c94254",
            "nyeri_wd.json": "88e645180d72ea1456b34cb9db7fa750"}


def main():
    for name, params in REQUESTS.items():
        q = "&".join(f"{k}={v}" for k, v in dict(POINT, parameters=params).items())
        os.makedirs(os.path.join(HERE, "download"), exist_ok=True)
        path = os.path.join(HERE, "download", name)
        with urllib.request.urlopen(f"{BASE}?{q}", timeout=120) as r:
            body = json.load(r)
        json.dump(body, open(path, "w"))
        got = hashlib.md5(open(path, "rb").read()).hexdigest()
        ok = "matches the cached copy" if got == EXPECTED[name] else "differs from the cached copy"
        print(f"{name}: {got}  {ok}")


if __name__ == "__main__":
    main()
