"""Dataset access, windowing and feature construction shared by all models."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / "data" / "processed" / "ausgrid_2012_13.npz"

STEPS_PER_DAY = 48
L = 96            # look-back: 48 h of half-hourly history
H = 48            # forecast horizon: next 24 h
QUANTILES = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95], dtype=np.float32)
Q_LO, Q_MED, Q_HI = 0, 3, 6   # indices of the 5 %, 50 % and 95 % quantiles
EVAL_H = np.array([1, 2, 4, 8, 12, 24, 36, 48])  # horizons (steps ahead) scored in detail

# Chronological split, in day indices from 2012-07-01
SPLITS = {
    "train": (0, 153),    # 2012-07-01 .. 2012-11-30
    "val": (153, 168),    # 2012-12-01 .. 2012-12-15  (early stopping)
    "cal": (168, 184),    # 2012-12-16 .. 2012-12-31  (conformal calibration)
    "test": (184, 365),   # 2013-01-01 .. 2013-06-30  (streaming evaluation)
}

# NSW public holidays inside the data period
HOLIDAYS = pd.to_datetime([
    "2012-10-01", "2012-12-25", "2012-12-26", "2013-01-01", "2013-01-28",
    "2013-03-29", "2013-04-01", "2013-04-25", "2013-06-10",
])


@dataclass
class Series:
    """Normalised panel of series, shape [N, T]."""
    net: np.ndarray
    gg: np.ndarray
    cons: np.ndarray
    csky: np.ndarray       # clear-sky PV potential, already scaled by capacity and normaliser
    scale: np.ndarray      # [N] normaliser (kWh per 30 min)
    mask: np.ndarray       # [N, T] True where net load is observed
    calendar: np.ndarray   # [4, T]: hour sin, hour cos, off-day flag, day-of-year cos
    t_local: np.ndarray

    @property
    def n(self):
        return self.net.shape[0]

    @property
    def t(self):
        return self.net.shape[1]


def complete_homes(net):
    """Homes with no missing data; the aggregate (feeder-level) series sums only these (299 of 300)."""
    return ~np.isnan(net).any(axis=1)


def load(level="household"):
    """level='household' (300 homes) or 'aggregate' (sum of all homes, a feeder proxy)."""
    z = np.load(NPZ, allow_pickle=True)
    net, gg, cons = z["net"], z["gg"], z["cons"]
    csky_kw = z["csky"] / 1000.0 * z["capacity"][:, None]  # kW-ish PV potential per home
    if level == "aggregate":
        full = complete_homes(net)
        net = net[full].sum(0, keepdims=True)
        gg = gg[full].sum(0, keepdims=True)
        cons = cons[full].sum(0, keepdims=True)
        csky_kw = csky_kw[full].sum(0, keepdims=True)
        mask = np.ones_like(net, dtype=bool)
    else:
        mask = ~np.isnan(net)
    tr = slice(0, SPLITS["train"][1] * STEPS_PER_DAY)
    scale = np.nanstd(net[:, tr], axis=1).astype(np.float32)
    s = scale[:, None]
    # clear-sky PV potential, normalised to the series' scale (0.5 h interval -> kWh)
    csky = (csky_kw * 0.5) / s

    t_local = pd.DatetimeIndex(z["t_local"])
    hod = (t_local.hour + t_local.minute / 60).to_numpy()
    off = ((t_local.dayofweek >= 5) | t_local.normalize().isin(HOLIDAYS)).astype(np.float32)
    calendar = np.stack([
        np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24), off,
        np.cos(2 * np.pi * (t_local.dayofyear.to_numpy() - 15) / 365.25),
    ]).astype(np.float32)

    def norm(a):
        return np.nan_to_num(a / s, nan=0.0).astype(np.float32)

    return Series(norm(net), norm(gg), norm(cons), csky.astype(np.float32), scale, mask,
                  calendar, t_local.values)


def origins(split, extra_left=0):
    """Forecast origins o (index of the first forecast step) whose targets lie in `split`.

    An origin o uses history [o-L, o) and forecasts [o, o+H). For training splits we require
    the whole target window inside the split; for test we score every origin in the split.
    """
    a, b = SPLITS[split]
    lo = max(a * STEPS_PER_DAY, L) - extra_left
    hi = b * STEPS_PER_DAY - H + 1
    return np.arange(lo, hi)


def batch(series: Series, homes, orig):
    """Gather model inputs for (home, origin) pairs.

    Returns past [B, 7, L], future [B, 5, H], target [B, H] (normalised), mask [B, H].
    """
    homes = np.asarray(homes)
    orig = np.asarray(orig)
    pidx = orig[:, None] + np.arange(-L, 0)[None, :]
    fidx = orig[:, None] + np.arange(0, H)[None, :]
    h2 = homes[:, None]
    cal_p = series.calendar[:, pidx].transpose(1, 0, 2)
    cal_f = series.calendar[:, fidx].transpose(1, 0, 2)
    past = np.concatenate([
        series.net[h2, pidx][:, None], series.gg[h2, pidx][:, None],
        series.cons[h2, pidx][:, None], series.csky[h2, pidx][:, None], cal_p[:, :3],
    ], axis=1)
    future = np.concatenate([
        series.csky[h2, fidx][:, None], cal_f,
    ], axis=1)
    target = series.net[h2, fidx]
    m = series.mask[h2, fidx]
    return past.astype(np.float32), future.astype(np.float32), target.astype(np.float32), m


def all_pairs(series: Series, split):
    o = origins(split)
    homes = np.repeat(np.arange(series.n), len(o))
    orig = np.tile(o, series.n)
    return homes, orig
