"""Audit a completed new run against its records and original summary values.

Writes comparison evidence only; never changes scientific results or the paper.
"""
from pathlib import Path
import csv
import hashlib
import json
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import data as D
import train_deep as T


def sha(path):
    out = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            out.update(chunk)
    return out.hexdigest()


def numeric_leaves(value, path=""):
    if isinstance(value, dict):
        return {k: v for name, item in value.items()
                for k, v in numeric_leaves(item, path + "/" + name).items()}
    if isinstance(value, list):
        return {k: v for i, item in enumerate(value)
                for k, v in numeric_leaves(item, path + "/" + str(i)).items()}
    if isinstance(value, (int, float)):
        return {path: value}
    return {}


def main():
    state = json.loads((ROOT / "provenance/run_status.json").read_text())
    if state["status"] != "complete":
        raise SystemExit("The run is not complete; refusing a final comparison.")
    old = json.loads((ROOT / "reference_results/metrics.json").read_text())
    new = json.loads((ROOT / "results/metrics.json").read_text())
    a, b = numeric_leaves(old), numeric_leaves(new)
    report = {"label": "New independent run versus original saved summaries",
              "checks": [], "numeric_fields": len(a), "missing_numeric_paths": sorted(set(a) - set(b)),
              "additional_numeric_paths": sorted(set(b) - set(a))}

    def check(name, passed, detail=None):
        report["checks"].append({"name": name, "passed": bool(passed), "detail": detail})

    check("All 26 protocol stages completed", len(state["completed"]) == 26 and
          all(x["returncode"] == 0 for x in state["completed"]))
    check("Numeric summary schema preserved", set(a) == set(b))
    check("No source changed during execution", all(sha(ROOT / p) == h for p, h in state["source_hashes"].items()))
    protected = json.loads((ROOT / "provenance/pre_run_integrity.json").read_text())
    check("Revised manuscript package preserved", all(sha(ROOT.parent / p) == h
          for p, h in protected["revision_snapshot"].items()))
    original = json.loads((ROOT.parent / "revision/validation/original_manifest.json").read_text())["files"]
    check("Original 56 files preserved", all(sha(ROOT.parent / p) == h for p, h in original.items()))
    inputs = json.loads((ROOT / "provenance/inputs_sha256.json").read_text())
    check("Raw data and reference summaries preserved", all(sha(ROOT / p) == h for p, h in inputs.items()
          if p.startswith(("data/raw/", "reference_results/"))))

    rows = []
    for path in sorted(set(a) & set(b)):
        timing = any(s in path for s in ("latency_ms", "train_time_s", "us_per_series_step"))
        rows.append({"path": path, "kind": "timing" if timing else "scientific",
                     "original": a[path], "new_run": b[path], "difference": b[path] - a[path],
                     "exact_match": a[path] == b[path]})
    with (ROOT / "provenance/summary_comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    science = [r for r in rows if r["kind"] == "scientific"]
    report["scientific_values"] = {"total": len(science), "exact_matches": sum(r["exact_match"] for r in science),
                                    "different": sum(not r["exact_match"] for r in science)}
    report["timing_note"] = "New hardware/environment timings are recorded separately and are not expected to equal the original."
    report["primary_comparisons"] = {}
    for level in ("aggregate", "household"):
        report["primary_comparisons"][level] = {}
        for label, metrics in (("original", old), ("new_run", new)):
            p, l = [metrics[level]["models"][m] for m in ("pv-tcqn", "lightgbm-q")]
            report["primary_comparisons"][level][label] = {
                "pv_qs7": p["crps"], "lightgbm_qs7": l["crps"],
                "qs7_reduction_percent": 100 * (1 - p["crps"] / l["crps"]),
                "shortest_horizon_reduction_percent": 100 * (1 - p["crps_by_h"][0] / l["crps_by_h"][0]),
                "pv_rw_aci_pooled_coverage": metrics[level]["conformal"]["pv-tcqn"]["rw-aci"]["picp"],
                "pv_rw_aci_max_monthly_gap": metrics[level]["conformal"]["pv-tcqn"]["rw-aci"]["max_monthly_gap"],
                "pv_rw_aci_under85_share": metrics[level]["conformal"]["pv-tcqn"]["rw-aci"].get("under85_share"),
                "dm_vs_lightgbm": metrics[level]["dm"]["lightgbm-q"],
            }

    forecasts = sorted((ROOT / "results/preds").glob("*.npy"))
    check("All 50 forecast arrays recorded", len(forecasts) == 50)
    report["forecast_arrays"] = []
    for p in forecasts:
        q = np.load(p, mmap_mode="r")
        level = p.name.split("_")[0]
        split = p.stem.split("_")[-1]
        expected = (300 if level == "household" else 1, len(D.origins(split)), 8, 7)
        finite, monotone = True, True
        for i in range(0, q.shape[0], 8):
            block = q[i:i + 8]
            finite = finite and bool(np.isfinite(block).all())
            monotone = monotone and bool((np.diff(block, axis=-1) >= 0).all())
        report["forecast_arrays"].append({"file": p.name, "shape": list(q.shape),
                                            "expected_shape": list(expected), "finite": finite,
                                            "noncrossing": monotone})
        check(p.name + " shape/finite/noncrossing", q.shape == expected and finite and monotone)

    # Replay fixed observations from every neural checkpoint, not a new evaluation.
    torch.set_num_threads(1)
    checkpoints = sorted((ROOT / "results/preds").glob("*.pt"))
    check("All 21 neural checkpoints recorded", len(checkpoints) == 21)
    report["checkpoint_replay"] = []
    for p in checkpoints:
        meta = json.loads(p.with_suffix(".json").read_text())
        args = meta["args"]
        series = D.load(args["level"])
        model = T.MODELS[args["model"]]()
        model.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))
        T.ABLATE.clear()
        T.ABLATE.update(x for x in args["ablate"].split(",") if x)
        for split in ("cal", "test"):
            origin_positions = np.array([0, len(D.origins(split)) // 2, len(D.origins(split)) - 1])
            homes = np.repeat(np.array([0, series.n - 1]), 3)
            positions = np.tile(origin_positions, 2)
            q = T.predict(model, series, homes, D.origins(split)[positions], keep_h=D.EVAL_H)
            saved = np.load(ROOT / "results/preds" / f"{p.stem}_{split}.npy", mmap_mode="r")[homes, positions]
            delta = float(np.max(np.abs(q - saved)))
            ok = np.allclose(q, saved, rtol=1e-4, atol=1e-5)
            record = {"checkpoint": p.name, "split": split,
                "max_absolute_difference": delta, "small_batch_passed": bool(ok),
                "tolerance": "rtol=1e-4, atol=1e-5; small-batch diagnostic"}
            if not ok:
                # A small batch/one thread can select different floating-point
                # kernels. Resolve by replaying the original complete batches
                # and thread count, without relaxing the comparison tolerance.
                torch.set_num_threads(args["threads"])
                hh, oo = D.all_pairs(series, split)
                flat_indices = homes * len(D.origins(split)) + positions
                batch_starts = sorted(set((flat_indices // 4096 * 4096).tolist()))
                full_saved = np.load(ROOT / "results/preds" / f"{p.stem}_{split}.npy", mmap_mode="r").reshape(-1, 8, 7)
                exact, largest, values = True, 0., 0
                for start in batch_starts:
                    stop = min(start + 4096, len(hh))
                    replay = T.predict(model, series, hh[start:stop], oo[start:stop], keep_h=D.EVAL_H, bs=4096)
                    expected = full_saved[start:stop]
                    exact = exact and bool(np.array_equal(replay, expected))
                    largest = max(largest, float(np.max(np.abs(replay - expected))))
                    values += replay.size
                record["original_settings_replay"] = {"threads": args["threads"], "batch_size": 4096,
                    "scalar_forecasts": int(values), "max_absolute_difference": largest, "bitwise_equal": exact}
                ok = exact
                torch.set_num_threads(1)
            record["passed"] = bool(ok)
            report["checkpoint_replay"].append(record)
            check(f"{p.name}/{split} replay", ok, record.get("original_settings_replay"))
    check("Fourteen LightGBM quantile-model files recorded", len(list((ROOT / "results/preds").glob("*_q*.txt"))) == 14)
    report["failed_checks"] = [c for c in report["checks"] if not c["passed"]]
    (ROOT / "provenance/reproduction_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("scientific_values", "primary_comparisons", "failed_checks")}, indent=2))
    if report["failed_checks"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
