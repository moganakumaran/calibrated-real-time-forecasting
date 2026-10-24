"""Train a neural quantile forecaster and save test/calibration predictions.

Usage: python train_deep.py --level household --model pv-tcqn --seed 0
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import data as D
from models import MODELS, gauss_nll, pinball

PRED = D.ROOT / "results" / "preds"


def to_t(*arrs):
    return [torch.from_numpy(a) for a in arrs]


ABLATE = set()


def ablate(past, fut):
    """Zero out input groups for ablation studies (see --ablate)."""
    if "csky" in ABLATE:
        past[:, 3] = 0
        fut[:, 0] = 0
    if "pvhist" in ABLATE:
        past[:, 1:3] = 0
    return past, fut


def forward_q(model, past, fut):
    past, fut = ablate(past.clone(), fut.clone())
    if hasattr(model, "quantiles"):
        return model.quantiles(past, fut)
    return model(past, fut)


def loss_fn(model, past, fut, y, m):
    past, fut = ablate(past, fut)
    if hasattr(model, "quantiles"):
        mu, sd = model(past, fut)
        return gauss_nll(mu, sd, y, m)
    return pinball(model(past, fut), y, m)


@torch.no_grad()
def predict(model, s, homes, orig, keep_h=None, bs=4096):
    """Quantile forecasts in kWh, [B, len(keep_h) or H, NQ]."""
    model.eval()
    out = []
    for i in range(0, len(homes), bs):
        p, f, _, _ = D.batch(s, homes[i:i + bs], orig[i:i + bs])
        q = forward_q(model, *to_t(p, f)).numpy()
        if keep_h is not None:
            q = q[:, keep_h - 1, :]
        out.append(q * s.scale[homes[i:i + bs]][:, None, None])
    return np.concatenate(out)


def val_loss(model, s, homes, orig, bs=4096):
    model.eval()
    tot, cnt = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(homes), bs):
            p, f, y, m = D.batch(s, homes[i:i + bs], orig[i:i + bs])
            q = forward_q(model, *to_t(p, f))
            tot += pinball(q, torch.from_numpy(y), torch.from_numpy(m.astype(np.float32))).item() * len(p)
            cnt += len(p)
    return tot / cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="household")
    ap.add_argument("--model", default="pv-tcqn")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps", type=int, default=300, help="batches per epoch")
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--ablate", default="", help="comma list: csky, pvhist, noskip")
    a = ap.parse_args()
    ABLATE.update(x for x in a.ablate.split(",") if x)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(a.threads)

    s = D.load(a.level)
    th, to = D.all_pairs(s, "train")
    vh, vo = D.all_pairs(s, "val")
    if len(vh) > 40000:
        sel = np.random.RandomState(0).choice(len(vh), 40000, replace=False)
        vh, vo = vh[sel], vo[sel]
    model = MODELS[a.model]()
    if "noskip" in ABLATE:
        model.seasonal.requires_grad_(False)
    nparams = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    best, best_state, bad, log = np.inf, None, 0, []
    t0 = time.time()
    rng = np.random.RandomState(a.seed)
    for ep in range(a.epochs):
        model.train()
        tr = 0.0
        for _ in range(a.steps):
            idx = rng.randint(0, len(th), a.bs)
            p, f, y, m = D.batch(s, th[idx], to[idx])
            p, f, y = to_t(p, f, y)
            loss = loss_fn(model, p, f, y, torch.from_numpy(m.astype(np.float32)))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr += loss.item()
        sched.step()
        vl = val_loss(model, s, vh, vo)
        log.append({"epoch": ep, "train": tr / a.steps, "val_pinball": vl, "t": time.time() - t0})
        print(f"ep {ep:2d} train {tr / a.steps:.4f} val {vl:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                break
    train_time = time.time() - t0
    model.load_state_dict(best_state)

    PRED.mkdir(parents=True, exist_ok=True)
    name = a.model + ("-no-" + "-".join(sorted(ABLATE)) if ABLATE else "")
    tag = f"{a.level}_{name}_s{a.seed}"
    for split in ("cal", "test"):
        h, o = D.all_pairs(s, split)
        q = predict(model, s, h, o, keep_h=D.EVAL_H)
        q = q.reshape(s.n, -1, len(D.EVAL_H), len(D.QUANTILES)).astype(np.float32)
        np.save(PRED / f"{tag}_{split}.npy", q)

    # single-forecast latency on CPU (one series, all 48 horizons)
    torch.set_num_threads(1)
    p, f, _, _ = D.batch(s, np.array([0]), np.array([D.origins("test")[0]]))
    p, f = to_t(p, f)
    model.eval()
    with torch.no_grad():
        for _ in range(20):
            forward_q(model, p, f)
        ts = []
        for _ in range(300):
            t1 = time.perf_counter()
            forward_q(model, p, f)
            ts.append(time.perf_counter() - t1)
    torch.save(model.state_dict(), PRED / f"{tag}.pt")
    meta = {"params": nparams, "train_time_s": train_time, "best_val": best,
            "latency_ms_median": 1000 * float(np.median(ts)),
            "latency_ms_p95": 1000 * float(np.percentile(ts, 95)), "log": log, "args": vars(a)}
    (PRED / f"{tag}.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "log"}))


if __name__ == "__main__":
    main()
