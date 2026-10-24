"""Classical baselines: seasonal persistence with empirical quantiles, and LightGBM quantile.

Usage: python baselines.py --level household
"""
import argparse
import warnings
import json
import time

import lightgbm as lgb
import numpy as np

import data as D

warnings.filterwarnings("ignore", category=DeprecationWarning)
PRED = D.ROOT / "results" / "preds"


def persistence(s, split, resid_q):
    """y_hat(o+h-1) = y(o+h-1-48) (same time yesterday), plus per-series residual quantiles."""
    o = D.origins(split)
    tgt = o[:, None] + D.EVAL_H[None, :] - 1          # [O, K]
    point = s.net[:, tgt - D.STEPS_PER_DAY]           # [N, O, K] normalised
    q = point[..., None] + resid_q[:, None, :, :]      # [N, O, K, NQ]
    return (q * s.scale[:, None, None, None]).astype(np.float32)


def persistence_resid_q(s):
    """Residual quantiles of seasonal persistence on train+val, per series and horizon."""
    a, b = 0, D.SPLITS["val"][1] * D.STEPS_PER_DAY
    t = np.arange(a + D.STEPS_PER_DAY, b)
    r = s.net[:, t] - s.net[:, t - D.STEPS_PER_DAY]
    r = np.where(s.mask[:, t] & s.mask[:, t - D.STEPS_PER_DAY], r, np.nan)
    rq = np.nanquantile(r, D.QUANTILES, axis=1).T      # [N, NQ]; same for every horizon
    return np.repeat(rq[:, None, :], len(D.EVAL_H), axis=1)


def gbm_features(s, homes, orig, h):
    """Tabular features for (series, origin, horizon) rows. h is 1-based steps ahead."""
    tgt = orig + h - 1
    last_day = np.stack([s.net[homes, orig - k] for k in range(1, 49)], 1).mean(1)
    cols = [
        s.net[homes, orig - 1], s.net[homes, orig - 2], s.net[homes, orig - 3],
        s.net[homes, orig - 4],
        last_day,
        s.net[homes, tgt - 48], s.net[homes, tgt - 96],
        s.gg[homes, tgt - 48], s.cons[homes, tgt - 48],
        s.csky[homes, tgt], s.csky[homes, tgt - 48],
        s.gg[homes, orig - 1] / np.maximum(s.csky[homes, orig - 1], 1e-3),  # recent clearness proxy
        s.calendar[0, tgt], s.calendar[1, tgt], s.calendar[2, tgt], s.calendar[3, tgt],
        h.astype(np.float32),
    ]
    return np.stack(cols, 1).astype(np.float32)


def sample_rows(s, split, n, rng):
    hh, oo = D.all_pairs(s, split)
    i = rng.randint(0, len(hh), n)
    h = D.EVAL_H[rng.randint(0, len(D.EVAL_H), n)] if split != "train" else rng.randint(1, D.H + 1, n)
    homes, orig = hh[i], oo[i]
    y = s.net[homes, orig + h - 1]
    m = s.mask[homes, orig + h - 1]
    x = gbm_features(s, homes, orig, h)
    return x[m], y[m]


def gbm(s, level, rng, n_train):
    xtr, ytr = sample_rows(s, "train", n_train, rng)
    xva, yva = sample_rows(s, "val", min(n_train // 5, 100000), rng)
    models = []
    t0 = time.time()
    for tau in D.QUANTILES:
        m = lgb.LGBMRegressor(objective="quantile", alpha=float(tau), n_estimators=600,
                              learning_rate=0.05, num_leaves=63, min_child_samples=50,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.9,
                              n_jobs=4, verbose=-1)
        m.fit(xtr, ytr, eval_set=[(xva, yva)], callbacks=[lgb.early_stopping(50, verbose=False)])
        models.append(m)
    train_time = time.time() - t0

    out = {}
    for split in ("cal", "test"):
        o = D.origins(split)
        res = np.zeros((s.n, len(o), len(D.EVAL_H), len(D.QUANTILES)), np.float32)
        for ki, h in enumerate(D.EVAL_H):
            homes = np.repeat(np.arange(s.n), len(o))
            orig = np.tile(o, s.n)
            x = gbm_features(s, homes, orig, np.full(len(homes), h))
            for qi, m in enumerate(models):
                res[:, :, ki, qi] = m.predict(x, num_threads=4).reshape(s.n, len(o))
        res = np.sort(res, axis=-1)  # remove quantile crossing
        out[split] = res * s.scale[:, None, None, None]

    # latency: one forecast for one series, all 48 horizons, 7 quantiles
    x1 = gbm_features(s, np.zeros(48, int), np.full(48, D.origins("test")[0]), np.arange(1, 49))
    ts = []
    for _ in range(100):
        t1 = time.perf_counter()
        for m in models:
            m.predict(x1, num_threads=1)
        ts.append(time.perf_counter() - t1)
    meta = {"train_time_s": train_time, "n_train_rows": int(len(ytr)),
            "trees": [int(m.best_iteration_ or m.n_estimators) for m in models],
            "latency_ms_median": 1000 * float(np.median(ts)),
            "latency_ms_p95": 1000 * float(np.percentile(ts, 95))}
    # Reproduction-only persistence: occurs after fitting, prediction and timing.
    # It does not alter model parameters, predictions or the experimental protocol.
    for qi, model in enumerate(models):
        model.booster_.save_model(str(PRED / f"{level}_lightgbm-q_s0_q{qi}.txt"))
    provenance = D.ROOT / "provenance"
    provenance.mkdir(exist_ok=True)
    (provenance / f"lightgbm_{level}_configuration.json").write_text(
        json.dumps([m.get_params() for m in models], indent=2))
    return out, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="household")
    a = ap.parse_args()
    s = D.load(a.level)
    PRED.mkdir(parents=True, exist_ok=True)
    rq = persistence_resid_q(s)
    for split in ("cal", "test"):
        np.save(PRED / f"{a.level}_persistence_s0_{split}.npy", persistence(s, split, rq))
    rng = np.random.RandomState(0)
    n_train = 600000 if a.level == "household" else 120000
    out, meta = gbm(s, a.level, rng, n_train)
    for split, q in out.items():
        np.save(PRED / f"{a.level}_lightgbm-q_s0_{split}.npy", q.astype(np.float32))
    (PRED / f"{a.level}_lightgbm-q_s0.json").write_text(json.dumps(meta, indent=1))
    print(meta)


if __name__ == "__main__":
    main()
