"""Build household net-load arrays from the Ausgrid 2012-13 Solar Home dataset.

Net load = general consumption (GC) + controlled load (CL) - gross PV generation (GG),
in kWh per 30-minute interval. Also computes a clear-sky irradiance feature for each
customer from solar geometry at the customer's postcode (Haurwitz model), which is a
deterministic, always-available proxy for the PV potential at any future time.

Output: data/processed/ausgrid_2012_13.npz
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "2012-2013 Solar home electricity data v2.csv"
POSTCODES = ROOT / "data" / "raw" / "Australian_Post_Codes_Lat_Lon.csv"
OUT = ROOT / "data" / "processed" / "ausgrid_2012_13.npz"

STEPS_PER_DAY = 48
LOCAL_TZ = "Australia/Sydney"  # timestamps are civil time incl. daylight saving (checked against PV peak timing)


def load_raw():
    d = pd.read_csv(RAW, skiprows=1)
    d["date"] = pd.to_datetime(d["date"], format="%d/%m/%Y")
    return d


def to_matrix(d, category, customers, days):
    """Return array [n_customers, n_days * 48]; missing customer-days are NaN."""
    vals = d[d["Consumption Category"] == category]
    cols = list(d.columns[5:5 + STEPS_PER_DAY])
    piv = vals.set_index(["Customer", "date"])[cols]
    full = pd.MultiIndex.from_product([customers, days], names=["Customer", "date"])
    piv = piv.reindex(full)
    arr = piv.to_numpy(dtype=np.float32).reshape(len(customers), len(days) * STEPS_PER_DAY)
    return arr


def clear_sky_ghi(times_utc, lat, lon):
    """Haurwitz clear-sky GHI (W/m^2) for an array of UTC times at one location."""
    doy = times_utc.dayofyear.to_numpy()
    hour = (times_utc.hour + times_utc.minute / 60.0).to_numpy()
    gamma = 2 * np.pi / 365 * (doy - 1 + (hour - 12) / 24)
    eqtime = 229.18 * (0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
                       - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma))
    decl = (0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
            - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
            - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma))
    tst = hour * 60 + eqtime + 4 * lon  # true solar time, minutes
    ha = np.deg2rad(tst / 4 - 180)
    phi = np.deg2rad(lat)
    cosz = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(ha)
    cosz = np.clip(cosz, 0, None)
    ghi = np.where(cosz > 0, 1098 * cosz * np.exp(-0.057 / np.maximum(cosz, 1e-3)), 0.0)
    return ghi.astype(np.float32)


def main():
    d = load_raw()
    customers = np.sort(d["Customer"].unique())
    days = pd.date_range(d["date"].min(), d["date"].max(), freq="D")
    meta = d.groupby("Customer")[["Generator Capacity", "Postcode"]].first().loc[customers]

    gc = to_matrix(d, "GC", customers, days)
    gg = to_matrix(d, "GG", customers, days)
    cl = to_matrix(d, "CL", customers, days)
    has_cl = ~np.all(np.isnan(cl), axis=1)
    cl = np.where(np.isnan(cl) & ~np.isnan(gc), 0.0, cl)  # customers without CL have zero CL
    cons = gc + cl
    net = cons - gg

    # interval start times (local standard time); column "0:30" is the interval 00:00-00:30
    t_local = pd.date_range(days[0], periods=len(days) * STEPS_PER_DAY, freq="30min")
    t_mid_utc = (t_local + pd.Timedelta(minutes=15)).tz_localize(
        LOCAL_TZ, ambiguous="NaT", nonexistent="shift_forward").tz_convert("UTC")
    t_mid_utc = pd.DatetimeIndex(pd.Series(t_mid_utc).ffill()).tz_localize(None)

    pc = pd.read_csv(POSTCODES).drop_duplicates("postcode").set_index("postcode")
    lat = pc.reindex(meta["Postcode"].values)["lat"].to_numpy(dtype=float)
    lon = pc.reindex(meta["Postcode"].values)["lon"].to_numpy(dtype=float)
    lat = np.where(np.isnan(lat), -33.87, lat)  # fall back to Sydney CBD if unknown
    lon = np.where(np.isnan(lon), 151.21, lon)
    csky = np.stack([clear_sky_ghi(t_mid_utc, la, lo) for la, lo in zip(lat, lon)])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT, net=net, cons=cons, gg=gg, csky=csky,
        capacity=meta["Generator Capacity"].to_numpy(np.float32),
        postcode=meta["Postcode"].to_numpy(), customer=customers, has_cl=has_cl,
        lat=lat, lon=lon, t_local=t_local.values.astype("datetime64[m]"),
    )
    print(f"customers={len(customers)} steps={net.shape[1]} missing={np.isnan(net).mean():.4%}")
    print(f"net load kWh/30min: mean={np.nanmean(net):.3f} min={np.nanmin(net):.3f} max={np.nanmax(net):.3f}")
    print(f"share of intervals with export (net<0): {np.nanmean(net < 0):.2%}")


if __name__ == "__main__":
    main()
