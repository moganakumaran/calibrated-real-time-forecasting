"""Paper figures from results/metrics.json and saved forecasts."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import data as D
from conformal import calibrate
from evaluate import CONF_NAMES, MODEL_NAMES, targets

FIG = D.ROOT / "figures"
RES = D.ROOT / "results"
plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 200, "savefig.bbox": "tight", "legend.fontsize": 7})
COL = {"persistence": "#888888", "lightgbm-q": "#1f77b4", "mlp-q": "#2ca02c",
       "lstm-gauss": "#9467bd", "pv-tcqn": "#d62728"}
CCOL = {"raw": "#888888", "scqr": "#1f77b4", "aci": "#2ca02c", "rcqr": "#9467bd", "rw-aci": "#d62728"}
W1 = 3.5  # IEEE single column width (in)


def fig_profiles():
    z = np.load(D.NPZ, allow_pickle=True)
    net = z["net"][D.complete_homes(z["net"])].sum(0)
    t = pd.DatetimeIndex(z["t_local"])
    df = pd.DataFrame({"net": net, "m": t.month, "hh": t.hour + t.minute / 60})
    fig, ax = plt.subplots(figsize=(W1, 2.0))
    cmap = plt.get_cmap("viridis")
    for i, m in enumerate([7, 9, 11, 1, 3, 5]):
        p = df[df.m == m].groupby("hh").net.mean()
        ax.plot(p.index, p.values, color=cmap(i / 6), label=pd.Timestamp(2012, m, 1).strftime("%b"))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Mean net load (kWh/30 min)")
    ax.set_xticks(range(0, 25, 4))
    ax.legend(ncol=3, loc="upper left")
    fig.savefig(FIG / "profiles.pdf")


def fig_horizon(M):
    fig, axs = plt.subplots(1, 2, figsize=(2 * W1, 1.9))
    for ax, level in zip(axs, ("aggregate", "household")):
        for m, r in M[level]["models"].items():
            ax.plot(D.EVAL_H / 2, r["crps_by_h"], marker="o", ms=3, color=COL[m], label=MODEL_NAMES[m])
        ax.set_xlabel("Horizon (h)")
        ax.set_ylabel("CRPS (kWh)")
        ax.set_title("Feeder aggregate (299 homes)" if level == "aggregate" else "Individual households")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.1))
    fig.savefig(FIG / "crps_horizon.pdf")


def fig_monthly(M):
    fig, axs = plt.subplots(1, 2, figsize=(2 * W1, 1.9), sharey=True)
    for ax, level in zip(axs, ("aggregate", "household")):
        mon = M[level]["monthly"]["pv-tcqn"]
        for meth, v in mon.items():
            ks = sorted(v, key=int)
            ax.plot(range(len(ks)), [v[k] for k in ks], marker="o", ms=3, color=CCOL[meth], label=CONF_NAMES[meth])
            ax.set_xticks(range(len(ks)), [pd.Timestamp(2013, int(k), 1).strftime("%b") for k in ks])
        ax.axhline(0.9, color="k", ls="--", lw=0.8)
        ax.set_title("Feeder aggregate" if level == "aggregate" else "Individual households")
        ax.set_ylabel("Coverage of 90 % interval")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.1))
    fig.savefig(FIG / "monthly_coverage.pdf")


def fig_fan():
    """Day-ahead (h = 12 h) view of one week of the aggregate test period."""
    y = targets("aggregate", "test")
    yc = targets("aggregate", "cal")
    q = np.load(RES / "preds" / "aggregate_pv-tcqn_s0_test.npy")
    qc = np.load(RES / "preds" / "aggregate_pv-tcqn_s0_cal.npy")
    lo, hi, _ = calibrate(q[..., 0], q[..., 6], y, qc[..., 0], qc[..., 6], yc, 0.1, "rw-aci")
    k = list(D.EVAL_H).index(24)
    o = D.origins("test")
    t = pd.DatetimeIndex(np.load(D.NPZ, allow_pickle=True)["t_local"])[o + 24 - 1]
    sel = (t >= "2013-02-04") & (t < "2013-02-11")
    fig, ax = plt.subplots(figsize=(2 * W1, 1.8))
    ax.fill_between(t[sel], q[0, sel, k, 0], q[0, sel, k, 6], color="#d62728", alpha=0.15, label="Model 5–95 %")
    ax.fill_between(t[sel], q[0, sel, k, 1], q[0, sel, k, 5], color="#d62728", alpha=0.25, label="Model 10–90 %")
    ax.plot(t[sel], lo[0, sel, k], color="#1f77b4", lw=0.8, ls="--", label="RW-ACI 90 % bounds")
    ax.plot(t[sel], hi[0, sel, k], color="#1f77b4", lw=0.8, ls="--")
    ax.plot(t[sel], y[0, sel, k], color="k", lw=0.9, label="Actual")
    ax.plot(t[sel], q[0, sel, k, 3], color="#d62728", lw=0.8, label="Median")
    ax.set_ylabel("Net load (kWh/30 min)")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.25))
    fig.autofmt_xdate()
    fig.savefig(FIG / "fan_week.pdf")


def fig_gamma(M):
    s = M["aggregate"]["sens"]
    g = sorted([k for k in s if k.startswith("gamma")], key=lambda k: float(k.split("=")[1]))
    fig, ax = plt.subplots(figsize=(W1, 1.8))
    x = [float(k.split("=")[1]) for k in g]
    ax.semilogx(x, [s[k]["picp"] for k in g], marker="o", ms=3, color="#d62728", label="Coverage")
    ax.axhline(0.9, color="k", ls="--", lw=0.8)
    ax.set_xlabel(r"ACI step size $\gamma$")
    ax.set_ylabel("Coverage")
    ax2 = ax.twinx()
    ax2.semilogx(x, [s[k]["winkler"] for k in g], marker="s", ms=3, color="#1f77b4", label="Winkler")
    ax2.set_ylabel("Winkler score (kWh)")
    ax2.grid(False)
    fig.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.05))
    fig.savefig(FIG / "gamma.pdf")


def fig_homes():
    """Distribution of per-home test coverage for each calibration method (household PV-TCQN)."""
    cache = RES / "per_home_coverage.json"
    if cache.exists():
        cov = json.load(open(cache))
    else:
        y = targets("household", "test")
        yc = targets("household", "cal")
        q = np.load(RES / "preds" / "household_pv-tcqn_s0_test.npy")
        qc = np.load(RES / "preds" / "household_pv-tcqn_s0_cal.npy")
        cov = {}
        for meth in CONF_NAMES:
            lo, hi, _ = calibrate(q[..., 0], q[..., 6], y, qc[..., 0], qc[..., 6], yc, 0.1, meth)
            inside = ((y >= lo) & (y <= hi)).astype(float)
            inside[np.isnan(y)] = np.nan
            cov[meth] = np.nanmean(inside, axis=(1, 2)).tolist()
        cache.write_text(json.dumps(cov))
    fig, ax = plt.subplots(figsize=(W1, 1.9))
    meths = list(CONF_NAMES)
    ax.boxplot([cov[m] for m in meths], whis=(1, 99), showfliers=True, flierprops={"markersize": 2},
               medianprops={"color": "k"})
    ax.set_xticks(range(1, len(meths) + 1), [CONF_NAMES[m].replace(" ", "\n", 1) for m in meths])
    ax.axhline(0.9, color="k", ls="--", lw=0.8)
    ax.set_ylabel("Per-home coverage")
    fig.savefig(FIG / "per_home_coverage.pdf")


def main():
    FIG.mkdir(exist_ok=True)
    M = json.load(open(RES / "metrics.json"))
    fig_profiles()
    fig_horizon(M)
    fig_monthly(M)
    fig_fan()
    fig_gamma(M)
    fig_homes()
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
