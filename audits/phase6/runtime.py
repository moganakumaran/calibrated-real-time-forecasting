"""Matched one-thread component and full-cycle runtime measurements."""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import platform
import time
from pathlib import Path

from common import PHASE4, REPRO, RESULTS, file_record, read, write

import lightgbm as lgb
import numpy as np
import torch

import baselines as B
import data as D
import models as M
from conformal import _qlevel, _replace_sorted

CONFIG = read(Path(__file__).with_name("config.json"))


def summarize(samples):
    values = np.asarray(samples, dtype=float)
    return {
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
        "samples_ms": values.tolist(),
    }


def time_call(function, warmups=20, calls=100):
    for _ in range(warmups):
        function()
    samples = []
    for _ in range(calls):
        start = time.perf_counter_ns()
        function()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return summarize(samples)


def neural_components(series, model, n_series, origin):
    homes = np.arange(n_series)
    origins = np.full(n_series, origin)
    holder = {}

    def features():
        p, f, _, _ = D.batch(series, homes, origins)
        holder["x"] = (torch.from_numpy(p), torch.from_numpy(f))
        return holder["x"]

    features()

    def inference():
        p, f = holder["x"]
        with torch.no_grad():
            holder["q"] = model(p, f).numpy() * series.scale[homes, None, None]
        return holder["q"]

    def combined():
        features()
        return inference()

    return features, inference, combined


def tree_components(series, boosters, n_series, origin, horizon_count):
    homes = np.repeat(np.arange(n_series), horizon_count)
    origins = np.full(n_series * horizon_count, origin)
    horizons = np.tile(np.arange(1, horizon_count + 1), n_series)
    holder = {}

    def features():
        holder["x"] = B.gbm_features(series, homes, origins, horizons)
        return holder["x"]

    features()

    def inference():
        holder["q"] = np.stack([m.predict(holder["x"], num_threads=1) for m in boosters], axis=-1)
        return holder["q"]

    def combined():
        features()
        return inference()

    return features, inference, combined


class CalibrationState:
    def __init__(self, scores8, n_series, horizon_count, adaptive=True):
        source = scores8[:n_series]
        if horizon_count == 48:
            source = np.take(source, np.arange(48) % source.shape[1], axis=1)
        self.source = source.astype(np.float64, copy=True)
        self.n, self.k, self.w = self.source.shape
        self.buf = self.source.copy()
        self.srt = np.sort(self.buf, axis=-1)
        self.pos = np.zeros((self.n, self.k), dtype=int)
        self.alpha = np.full((self.n, self.k), 0.1)
        self.q_previous = np.take_along_axis(self.srt, np.full((self.n, self.k, 1), int(0.9 * self.w)), axis=-1)[..., 0]
        self.ar_n = np.arange(self.n)[:, None]
        self.ar_k = np.arange(self.k)[None, :]
        self.call = 0
        self.adaptive = adaptive

    def score_maturity(self):
        j = self.call % self.w
        lo = -0.5 * np.ones((self.n, self.k))
        hi = 0.5 * np.ones((self.n, self.k))
        y = self.source[..., j]
        return np.maximum(lo - y, y - hi)

    def update(self):
        # Offset the incoming score stream so replacement exercises actual value
        # movement rather than writing back the value currently at the ring position.
        new = self.source[..., (self.call + 37) % self.w]
        old = self.buf[self.ar_n, self.ar_k, self.pos]
        if self.adaptive:
            error = (new > self.q_previous).astype(float)
            self.alpha += 0.005 * (0.1 - error)
        self.buf[self.ar_n, self.ar_k, self.pos] = new
        self.pos = (self.pos + 1) % self.w
        self.srt = _replace_sorted(self.srt, old, new)
        level = _qlevel(self.alpha, self.w)
        index = np.clip(np.ceil(level * self.w).astype(int) - 1, 0, self.w - 1)
        self.q_previous = self.srt[self.ar_n, self.ar_k, index]
        self.call += 1
        return self.q_previous


def calibration_scores():
    pred = np.load(REPRO / "results/preds/household_pv-tcqn_s0_cal.npy", mmap_mode="r")
    z = np.load(REPRO / "data/processed/ausgrid_2012_13.npz", allow_pickle=True)
    origins = np.arange(168 * 48, 184 * 48 - 48 + 1)
    target_index = origins[:, None] + D.EVAL_H[None, :] - 1
    y = z["net"][:, target_index]
    score = np.maximum(np.asarray(pred[..., D.Q_LO]) - y, y - np.asarray(pred[..., D.Q_HI]))
    with np.errstate(all="ignore"):
        fill = np.nanmedian(score, axis=1, keepdims=True)
        fill = np.where(np.isnan(fill), np.nanmedian(score, axis=(0, 1), keepdims=True), fill)
    score = np.where(np.isnan(score), fill, score)
    return score.transpose(0, 2, 1)


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    warmups = CONFIG["runtime_scope"]["warmups"]
    calls = CONFIG["runtime_scope"]["timed_calls"]
    origin = int(D.origins("test")[4320])
    series = D.load("household")
    model_path = REPRO / "results/preds/household_pv-tcqn_s0.pt"
    model = M.PVTCQN()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    tree_paths = [REPRO / f"results/preds/household_lightgbm-q_s0_q{i}.txt" for i in range(7)]
    boosters = [lgb.Booster(model_file=str(path)) for path in tree_paths]
    scores8 = calibration_scores()
    runs = []

    for n_series in (1, 300):
        nf, ni, nc = neural_components(series, model, n_series, origin)
        runs.extend([
            {"component": "feature_assembly", "model": "PV-TCQN", "series": n_series, "horizons": 48, **time_call(nf, warmups, calls)},
            {"component": "model_inference", "model": "PV-TCQN", "series": n_series, "horizons": 48, **time_call(ni, warmups, calls)},
            {"component": "features_plus_inference", "model": "PV-TCQN", "series": n_series, "horizons": 48, **time_call(nc, warmups, calls)},
        ])
        for horizon_count in (8, 48):
            tf, ti, tc = tree_components(series, boosters, n_series, origin, horizon_count)
            runs.extend([
                {"component": "feature_assembly", "model": "LightGBM-Q", "series": n_series, "horizons": horizon_count, **time_call(tf, warmups, calls)},
                {"component": "model_inference", "model": "LightGBM-Q", "series": n_series, "horizons": horizon_count, **time_call(ti, warmups, calls)},
                {"component": "features_plus_inference", "model": "LightGBM-Q", "series": n_series, "horizons": horizon_count, **time_call(tc, warmups, calls)},
            ])

        for horizon_count in (8, 48):
            state_score = CalibrationState(scores8, n_series, horizon_count)
            state_update = CalibrationState(scores8, n_series, horizon_count)
            runs.extend([
                {"component": "mature_score_computation", "model": "RW-ACI", "series": n_series, "horizons": horizon_count, **time_call(state_score.score_maturity, warmups, calls)},
                {"component": "buffer_level_update", "model": "RW-ACI", "series": n_series, "horizons": horizon_count, **time_call(state_update.update, warmups, calls)},
            ])

        for horizon_count in (8, 48):
            state = CalibrationState(scores8, n_series, horizon_count)
            nf2, ni2, nc2 = neural_components(series, model, n_series, origin)
            def full_cycle():
                nc2()
                state.score_maturity()
                return state.update()
            runs.append({"component": "measured_compute_cycle", "model": "PV-TCQN+RW-ACI", "series": n_series, "horizons": horizon_count, **time_call(full_cycle, warmups, calls)})

    environment = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "lightgbm": lgb.__version__,
        "logical_cpus": os.cpu_count(),
        "cpu": "Apple M1 Pro",
        "memory_gib": 16,
        "threads": 1,
    }
    output = {
        "status": "complete",
        "environment": environment,
        "protocol": {
            "warmups": warmups,
            "timed_calls": calls,
            "origin": origin,
            "model_state": "in-memory, evaluation mode",
            "neural_inference": "PV-TCQN intrinsically emits all 48 horizons; eight-horizon rows differ only in calibration-state timing.",
            "calibration_48_note": "The implementation was historically evaluated at eight horizons. For timing only, its observed eight-horizon score-buffer distributions are repeated to create 48 state rows; no 48-horizon reliability result is inferred.",
            "exclusions": ["disk/network ingestion", "service serialization", "scheduler", "database writes", "meter transport", "model loading"],
        },
        "inputs": [file_record(model_path), *[file_record(path) for path in tree_paths]],
        "runs": runs,
    }
    write(RESULTS / "runtime.json", output)
    print({"runs": len(runs), "status": "complete"})


if __name__ == "__main__":
    main()
