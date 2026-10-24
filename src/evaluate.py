"""Score all saved forecasts, run conformal calibration variants and statistical tests.

Usage: python evaluate.py   (writes results/metrics.json and results/*.csv)
"""
import glob
import json
import time

import numpy as np
import pandas as pd

import data as D
from conformal import calibrate

PRED = D.ROOT / "results" / "preds"
RES = D.ROOT / "results"
ALPHA = 0.10
MODEL_NAMES = {"persistence": "Seasonal persistence", "lightgbm-q": "LightGBM-Q",
               "mlp-q": "MLP-Q", "lstm-gauss": "LSTM-Gaussian", "pv-tcqn": "PV-TCQN"}
CONF_NAMES = {"raw": "Uncalibrated", "scqr": "Split CQR", "aci": "ACI",
              "rcqr": "Rolling CQR", "rw-aci": "RW-ACI"}


def targets(level, split):
    z = np.load(D.NPZ, allow_pickle=True)
    net = z["net"]
    if level == "aggregate":
        net = net[D.complete_homes(net)].sum(0, keepdims=True)
    o = D.origins(split)
    tgt = o[:, None] + D.EVAL_H[None, :] - 1
    return net[:, tgt]  # [N, O, K]


def test_months(level):
    o = D.origins("test")
    t = pd.DatetimeIndex(np.load(D.NPZ, allow_pickle=True)["t_local"])[o]
    return t.month.to_numpy()


def pinball_all(q, y):
    d = y[..., None] - q
    return np.maximum(D.QUANTILES * d, (D.QUANTILES - 1) * d)  # [N, O, K, NQ]


def interval_metrics(lo, hi, y, alpha=ALPHA):
    ok = ~np.isnan(y)
    cov = ((y >= lo) & (y <= hi))[ok].mean()
    w = (hi - lo)[ok].mean()
    wk = (hi - lo) + (2 / alpha) * np.maximum(lo - y, 0) + (2 / alpha) * np.maximum(y - hi, 0)
    return {"picp": float(cov), "mpiw": float(w), "winkler": float(np.nanmean(np.where(ok, wk, np.nan)))}


def point_prob_metrics(q, y):
    ok = ~np.isnan(y)
    med = q[..., D.Q_MED]
    e = (med - y)[ok]
    pb = pinball_all(q, y)
    crps = 2 * np.nanmean(np.where(ok[..., None], pb, np.nan), axis=-1)  # quantile CRPS approx
    out = {"mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())),
           "nmae": float(np.abs(e).mean() / np.abs(y[ok]).mean()),
           "crps": float(np.nanmean(crps))}
    out.update(interval_metrics(q[..., D.Q_LO], q[..., D.Q_HI], y))
    out["crps_by_h"] = [float(np.nanmean(crps[:, :, k])) for k in range(len(D.EVAL_H))]
    out["mae_by_h"] = [float(np.nanmean(np.abs(med - y)[:, :, k])) for k in range(len(D.EVAL_H))]
    return out, crps


def dm_test(l1, l2, lag):
    """Diebold-Mariano test on loss series (1 vs 2), Newey-West variance. Negative = 1 better."""
    from scipy import stats
    d = l1 - l2
    d = d[~np.isnan(d)]
    n = len(d)
    dc = d - d.mean()
    v = dc @ dc / n
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (dc[k:] @ dc[:-k]) / n
    t = d.mean() / np.sqrt(v / n)
    return float(t), float(2 * stats.norm.sf(abs(t)))


def load_preds(level, model):
    out = {}
    for split in ("cal", "test"):
        files = sorted(glob.glob(str(PRED / f"{level}_{model}_s*_{split}.npy")))
        out[split] = [np.load(f) for f in files]
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="aggregate,household")
    ap.add_argument("--out", default="metrics.json")
    args = ap.parse_args()
    results = {}
    for level in args.levels.split(","):
        y_te, y_ca = targets(level, "test"), targets(level, "cal")
        months = test_months(level)
        R = results[level] = {"models": {}, "conformal": {}, "dm": {}, "monthly": {}, "sens": {}}
        crps_series = {}
        for model in MODEL_NAMES:
            p = load_preds(level, model)
            if not p["test"]:
                continue
            per_seed = []
            for i, q in enumerate(p["test"]):
                m, crps = point_prob_metrics(q, y_te)
                per_seed.append(m)
                if i == 0:
                    crps_series[model] = np.nanmean(crps, axis=(0, 2))  # per-origin
            agg = {k: float(np.mean([m[k] for m in per_seed])) for k in per_seed[0] if not k.endswith("_by_h")}
            agg.update({k + "_std": float(np.std([m[k] for m in per_seed])) for k in ("crps", "mae", "picp")})
            agg["crps_by_h"] = list(np.mean([m["crps_by_h"] for m in per_seed], 0))
            agg["mae_by_h"] = list(np.mean([m["mae_by_h"] for m in per_seed], 0))
            agg["seeds"] = len(per_seed)
            meta = sorted(glob.glob(str(PRED / f"{level}_{model}_s*.json")))
            if meta:
                mj = [json.load(open(f)) for f in meta]
                for k in ("latency_ms_median", "latency_ms_p95", "train_time_s", "params"):
                    if k in mj[0]:
                        agg[k] = float(np.mean([j[k] for j in mj]))
            R["models"][model] = agg
            print(level, model, {k: round(v, 4) for k, v in agg.items() if isinstance(v, float)}, flush=True)

        # ablations of PV-TCQN (aggregate level, 3 seeds each)
        R["ablation"] = {}
        for ab in ("pv-tcqn", "pv-tcqn-no-csky", "pv-tcqn-no-pvhist", "pv-tcqn-no-noskip"):
            p = load_preds(level, ab)
            if not p["test"]:
                continue
            ms = [point_prob_metrics(q, y_te)[0] for q in p["test"]]
            R["ablation"][ab] = {k: (float(np.mean([m[k] for m in ms])), float(np.std([m[k] for m in ms])))
                                 for k in ("mae", "rmse", "crps", "picp", "winkler")}
            R["ablation"][ab]["seeds"] = len(ms)

        # Diebold-Mariano: proposed vs each baseline on per-origin CRPS
        if "pv-tcqn" in crps_series:
            for model, c in crps_series.items():
                if model != "pv-tcqn":
                    R["dm"][model] = dict(zip(("stat", "p"), dm_test(crps_series["pv-tcqn"], c, lag=48)))

        # conformal calibration variants on top of two base forecasters
        for base in ("pv-tcqn", "lightgbm-q"):
            p = load_preds(level, base)
            if not p["test"]:
                continue
            q, qc = p["test"][0], p["cal"][0]
            lo, hi, lo_c, hi_c = q[..., D.Q_LO], q[..., D.Q_HI], qc[..., D.Q_LO], qc[..., D.Q_HI]
            R["conformal"][base] = {}
            R["monthly"][base] = {}
            for meth in CONF_NAMES:
                t0 = time.perf_counter()
                l2, h2, _ = calibrate(lo, hi, y_te, lo_c, hi_c, y_ca, ALPHA, meth)
                dt = time.perf_counter() - t0
                m = interval_metrics(l2, h2, y_te)
                m["us_per_series_step"] = 1e6 * dt / (lo.shape[0] * lo.shape[1])
                inside = ((y_te >= l2) & (y_te <= h2)).astype(float)
                inside[np.isnan(y_te)] = np.nan
                mon = {int(mm): float(np.nanmean(inside[:, months == mm])) for mm in np.unique(months)}
                m["max_monthly_gap"] = float(max(abs(v - (1 - ALPHA)) for v in mon.values()))
                if level == "household":
                    # share of (home, month) cells whose coverage falls below 85 %
                    cells = [np.nanmean(inside[:, months == mm], axis=(1, 2)) for mm in np.unique(months)]
                    m["under85_share"] = float(np.mean(np.stack(cells) < 0.85))
                m["by_h"] = [interval_metrics(l2[:, :, k], h2[:, :, k], y_te[:, :, k]) for k in range(len(D.EVAL_H))]
                R["conformal"][base][meth] = m
                R["monthly"][base][meth] = mon
                print(level, base, meth, {k: round(v, 4) for k, v in m.items() if isinstance(v, float)}, flush=True)

            if level == "aggregate" and base == "pv-tcqn":
                for g in (0.001, 0.002, 0.005, 0.01, 0.02, 0.05):
                    l2, h2, _ = calibrate(lo, hi, y_te, lo_c, hi_c, y_ca, ALPHA, "rw-aci", gamma=g)
                    R["sens"][f"gamma={g}"] = interval_metrics(l2, h2, y_te)
                for w in (192, 336, 720):
                    l2, h2, _ = calibrate(lo, hi, y_te, lo_c, hi_c, y_ca, ALPHA, "rw-aci", window=w)
                    R["sens"][f"window={w}"] = interval_metrics(l2, h2, y_te)

    (RES / args.out).write_text(json.dumps(results, indent=1))
    print("saved", RES / args.out)


if __name__ == "__main__":
    main()
