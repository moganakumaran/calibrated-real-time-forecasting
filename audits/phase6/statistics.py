"""Phase 6 paired QS7 inference from saved forecasts; no model fitting."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy import stats

from common import EVIDENCE, PHASE4, REPRO, RESULTS, file_record, read, write

TAU = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95], dtype=np.float64)
EVAL8 = np.array([1, 2, 4, 8, 12, 24, 36, 48])
CONFIG = read(Path(__file__).with_name("config.json"))


def target_array(level: str, horizons: np.ndarray):
    z = np.load(REPRO / "data/processed/ausgrid_2012_13.npz", allow_pickle=True)
    net = z["net"]
    if level == "aggregate":
        net = net[~np.isnan(net).any(axis=1)].sum(axis=0, keepdims=True)
    origins = np.arange(184 * 48, 365 * 48 - 48 + 1)
    indices = origins[:, None] + horizons[None, :] - 1
    return net[:, indices].astype(np.float64)


def per_origin_qs7(pred_path: Path, targets: np.ndarray, home_chunk: int = 12):
    q = np.load(pred_path, mmap_mode="r")
    if q.shape[:3] != targets.shape or q.shape[-1] != len(TAU):
        raise ValueError(f"shape mismatch: {pred_path}: {q.shape} versus {targets.shape}")
    n_origins = targets.shape[1]
    total = np.zeros(n_origins, dtype=np.float64)
    count = np.zeros(n_origins, dtype=np.int64)
    for start in range(0, targets.shape[0], home_chunk):
        stop = min(start + home_chunk, targets.shape[0])
        y = targets[start:stop]
        qq = np.asarray(q[start:stop], dtype=np.float64)
        valid = np.isfinite(y)
        delta = y[..., None] - qq
        pinball = np.maximum(TAU * delta, (TAU - 1.0) * delta)
        record_qs7 = 2.0 * pinball.mean(axis=-1)
        total += np.where(valid, record_qs7, 0.0).sum(axis=(0, 2))
        count += valid.sum(axis=(0, 2))
    if np.any(count == 0):
        raise ValueError("origin without a valid target")
    return total / count, count


def hac(d: np.ndarray, lag: int):
    d = np.asarray(d, dtype=np.float64)
    n = d.size
    centered = d - d.mean()
    variance = float(centered @ centered / n)
    for k in range(1, min(lag, n - 1) + 1):
        covariance = float(centered[k:] @ centered[:-k] / n)
        variance += 2.0 * (1.0 - k / (lag + 1.0)) * covariance
    if variance <= 0:
        return {"lag": lag, "long_run_variance": variance, "stat": None, "p_two_sided": None}
    statistic = float(d.mean() / math.sqrt(variance / n))
    return {
        "lag": lag,
        "long_run_variance": variance,
        "stat": statistic,
        "p_two_sided": float(2.0 * stats.norm.sf(abs(statistic))),
    }


def circular_block_means(values: np.ndarray, block: int, reps: int, rng: np.random.Generator):
    values = np.asarray(values, dtype=np.float64)
    n = values.size
    blocks = math.ceil(n / block)
    out = np.empty(reps, dtype=np.float64)
    offsets = np.arange(block)
    for begin in range(0, reps, 100):
        end = min(begin + 100, reps)
        starts = rng.integers(0, n, size=(end - begin, blocks, 1))
        indices = (starts + offsets[None, None, :]) % n
        sampled = values[indices].reshape(end - begin, -1)[:, :n]
        out[begin:end] = sampled.mean(axis=1)
    return out


def block_bootstrap(d: np.ndarray, block: int, reps: int, seed: int):
    rng = np.random.default_rng(seed + block)
    raw = circular_block_means(d, block, reps, rng)
    centered = circular_block_means(d - d.mean(), block, reps, rng)
    observed = abs(float(d.mean()))
    p_null = float((np.count_nonzero(np.abs(centered) >= observed) + 1) / (reps + 1))
    return {
        "block": block,
        "repetitions": reps,
        "mean_ci95": [float(np.percentile(raw, 2.5)), float(np.percentile(raw, 97.5))],
        "centered_null_p_two_sided": p_null,
    }


def holm_adjust(p_values):
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    m = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p_values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def mean_loss(paths, targets):
    losses = []
    reference_count = None
    for path in paths:
        loss, count = per_origin_qs7(path, targets)
        if reference_count is None:
            reference_count = count
        elif not np.array_equal(reference_count, count):
            raise AssertionError("seed denominators differ")
        losses.append(loss)
    return np.mean(losses, axis=0), reference_count


def comparison(label, pv_paths, base_paths, targets, horizons, level, pv_seed, base_model):
    pv_paths = [pv_paths] if isinstance(pv_paths, Path) else list(pv_paths)
    base_paths = [base_paths] if isinstance(base_paths, Path) else list(base_paths)
    pv, counts = mean_loss(pv_paths, targets)
    baseline, counts2 = mean_loss(base_paths, targets)
    if not np.array_equal(counts, counts2):
        raise AssertionError("paired denominators differ")
    d = pv - baseline
    lags = CONFIG["statistical_scope"]["hac_lags"]
    blocks = CONFIG["statistical_scope"]["bootstrap_blocks"]
    reps = CONFIG["statistical_scope"]["bootstrap_repetitions"]
    seed = CONFIG["statistical_scope"]["bootstrap_seed"]
    return {
        "label": label,
        "level": level,
        "horizons": horizons.tolist(),
        "pv_seed": pv_seed,
        "pv_seed_count": len(pv_paths),
        "baseline_seed_count": len(base_paths),
        "baseline": base_model,
        "origins": int(d.size),
        "records": int(counts.sum()),
        "pv_mean_qs7": float(pv.mean()),
        "baseline_mean_qs7": float(baseline.mean()),
        "paired_mean_difference_pv_minus_baseline": float(d.mean()),
        "relative_pv_reduction_percent": float(100.0 * (baseline.mean() - pv.mean()) / baseline.mean()),
        "naive_iid_se": float(d.std(ddof=1) / math.sqrt(d.size)),
        "hac": [hac(d, lag) for lag in lags],
        "block_bootstrap": [block_bootstrap(d, block, reps, seed + pv_seed * 1000) for block in blocks],
        "inputs": [file_record(path) for path in pv_paths + base_paths],
    }


def main():
    pred = REPRO / "results/preds"
    replay = PHASE4 / "results/replayed"
    y_agg8 = target_array("aggregate", EVAL8)
    y_agg48 = target_array("aggregate", np.arange(1, 49))
    y_hh8 = target_array("household", EVAL8)
    manuscript_comparisons = []
    historical_seed0_comparisons = []

    for baseline in ("persistence", "lightgbm-q", "mlp-q", "lstm-gauss"):
        base_seeds = (0, 1, 2) if baseline in ("mlp-q", "lstm-gauss") else (0,)
        manuscript_comparisons.append(comparison(
            f"aggregate_8_seed_mean_pv_vs_{baseline}",
            [pred / f"aggregate_pv-tcqn_s{seed}_test.npy" for seed in (0, 1, 2)],
            [pred / f"aggregate_{baseline}_s{seed}_test.npy" for seed in base_seeds],
            y_agg8, EVAL8, "aggregate", 0, baseline,
        ))
        historical_seed0_comparisons.append(comparison(
            f"aggregate_8_historical_seed0_pv_vs_{baseline}",
            pred / "aggregate_pv-tcqn_s0_test.npy",
            pred / f"aggregate_{baseline}_s0_test.npy",
            y_agg8, EVAL8, "aggregate", 0, baseline,
        ))
    manuscript_comparisons.append(comparison(
        "aggregate_48_seed_mean_pv_vs_lightgbm-q",
        [replay / f"aggregate_pv-tcqn_s{seed}_test.npy" for seed in (0, 1, 2)],
        replay / "aggregate_lightgbm-q_s0_test.npy",
        y_agg48, np.arange(1, 49), "aggregate", 0, "lightgbm-q",
    ))
    manuscript_comparisons.append(comparison(
        "household_8_pv_s0_vs_lightgbm-q",
        pred / "household_pv-tcqn_s0_test.npy",
        pred / "household_lightgbm-q_s0_test.npy",
        y_hh8, EVAL8, "household", 0, "lightgbm-q",
    ))

    seed_effects_8 = []
    seed_effects_48 = []
    for seed in (0, 1, 2):
        seed_effects_8.append(comparison(
            f"aggregate_8_pv_s{seed}_vs_lightgbm-q",
            pred / f"aggregate_pv-tcqn_s{seed}_test.npy",
            pred / "aggregate_lightgbm-q_s0_test.npy",
            y_agg8, EVAL8, "aggregate", seed, "lightgbm-q",
        ))
        seed_effects_48.append(comparison(
            f"aggregate_48_pv_s{seed}_vs_lightgbm-q",
            replay / f"aggregate_pv-tcqn_s{seed}_test.npy",
            replay / "aggregate_lightgbm-q_s0_test.npy",
            y_agg48, np.arange(1, 49), "aggregate", seed, "lightgbm-q",
        ))
    primary = manuscript_comparisons[:4]
    p48 = [next(x for x in row["hac"] if x["lag"] == 48)["p_two_sided"] for row in primary]
    adjusted = holm_adjust(p48)
    for row, value in zip(primary, adjusted):
        row["holm_adjusted_p_lag48_four_aggregate8_comparisons"] = value

    output = {
        "status": "complete",
        "metric": CONFIG["statistical_scope"]["metric"],
        "difference_sign": "negative paired difference means PV-TCQN has lower QS7",
        "aggregation": "Each time-series observation is one forecast origin after averaging valid homes, selected horizons, and seven quantiles. Overlapping targets induce serial dependence.",
        "manuscript_seed_mean_comparisons": manuscript_comparisons,
        "historical_seed0_comparisons": historical_seed0_comparisons,
        "aggregate_lightgbm_seed_sensitivity_8": seed_effects_8,
        "aggregate_lightgbm_seed_sensitivity_48": seed_effects_48,
        "limitations": [
            "Household neural results have one seed.",
            "The test period and horizon analyses were already inspected; inference is retrospective.",
            "Normal-reference HAC and block bootstrap address serial dependence but not dataset or model-selection uncertainty.",
            "For manuscript seed-mean comparisons, the fitted-seed mean is treated as fixed and inference is over forecast origins only.",
            "Seed variability across three fitted aggregate networks is descriptive and is not a random-effects population analysis."
        ],
    }
    write(RESULTS / "statistics.json", output)
    print({"manuscript_comparisons": len(manuscript_comparisons), "historical_seed0": len(historical_seed0_comparisons), "seed_effects": len(seed_effects_8) + len(seed_effects_48), "status": "complete"})


if __name__ == "__main__":
    main()
