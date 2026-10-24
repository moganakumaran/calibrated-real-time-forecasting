"""Run the supplied protocol in this separate directory, recording every stage.

No cloud/model API calls, no manuscript edits and no writes to reference_results.
The only added model-code behavior saves fitted LightGBM models/configurations.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protocol():
    jobs = [("prepare", ["prepare_data.py"]),
            ("baseline_aggregate", ["baselines.py", "--level", "aggregate"]),
            ("baseline_household", ["baselines.py", "--level", "household"])]
    for seed in range(3):
        for model in ("pv-tcqn", "mlp-q", "lstm-gauss"):
            jobs.append((f"aggregate_{model}_s{seed}", ["train_deep.py", "--level", "aggregate",
                         "--model", model, "--seed", str(seed), "--epochs", "25", "--steps", "150"]))
    for ablation in ("csky", "pvhist", "noskip"):
        for seed in range(3):
            jobs.append((f"aggregate_no_{ablation}_s{seed}", ["train_deep.py", "--level", "aggregate",
                         "--model", "pv-tcqn", "--seed", str(seed), "--epochs", "25", "--steps", "150",
                         "--ablate", ablation]))
    for model in ("pv-tcqn", "mlp-q", "lstm-gauss"):
        jobs.append((f"household_{model}_s0", ["train_deep.py", "--level", "household", "--model", model,
                     "--seed", "0", "--epochs", "20", "--steps", "300"]))
    jobs.extend([("evaluate", ["evaluate.py"]), ("figures", ["make_figures.py"])])
    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Run all unchanged protocol stages.")
    args = parser.parse_args()
    omp = Path(importlib.util.find_spec("torch").origin).parent / "lib"
    if not (omp / "libomp.dylib").is_file():
        raise SystemExit("Required OpenMP runtime is unavailable")
    env = os.environ.copy()
    env.update({"PYTHONPATH": str(ROOT / "vendor"), "PYTHONDONTWRITEBYTECODE": "1",
                "DYLD_LIBRARY_PATH": str(omp), "MPLCONFIGDIR": str(ROOT / "cache/matplotlib"),
                "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
    jobs = protocol()
    plan = {"created_utc": now(), "label": "Independent reproduction, not recovered original run",
            "python": sys.executable, "platform": platform.platform(), "cpu_threads": 4,
            "free_gib": shutil.disk_usage(ROOT).free / 2 ** 30,
            "openmp_library": str(omp / "libomp.dylib"), "openmp_sha256": sha(omp / "libomp.dylib"),
            "training_protocol_source": "../netload-forecasting-main/run_all.sh",
            "stages": [{"name": name, "arguments": cmd} for name, cmd in jobs],
            "external_model_service_cost": 0}
    (ROOT / "provenance/run_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    check = subprocess.run([sys.executable, "-B", "-c",
                            "import torch,lightgbm,numpy,pandas,scipy,sklearn,matplotlib; "
                            "import importlib.metadata,json; print(json.dumps({p:importlib.metadata.version(p) "
                            "for p in ['torch','lightgbm','numpy','pandas','scipy','scikit-learn','matplotlib']}))"],
                           env=env, cwd=ROOT, text=True, capture_output=True)
    (ROOT / "provenance/dependency_check.txt").write_text(check.stdout + check.stderr)
    if check.returncode:
        raise SystemExit("Dependency import failed; see provenance/dependency_check.txt")
    (ROOT / "provenance/environment_versions.json").write_text(check.stdout)
    freeze = subprocess.run([sys.executable, "-B", "-m", "pip", "freeze"], env=env, cwd=ROOT,
                            text=True, capture_output=True, check=True)
    (ROOT / "provenance/environment_freeze.txt").write_text(freeze.stdout)
    if not args.execute:
        print(f"Preflight passed; {len(jobs)} stages prepared. No training performed.")
        return
    if plan["free_gib"] < 15:
        raise SystemExit("Less than 15 GiB free; stopping before generating forecasts")
    state_path = ROOT / "provenance/run_status.json"
    if state_path.exists() or (ROOT / "results/preds").exists():
        raise SystemExit("Run/output already exists; refusing to overwrite or silently rerun")
    state = {"started_utc": now(), "pid": os.getpid(), "status": "running", "completed": [],
             "source_hashes": {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / "src").glob("*.py"))}}

    def save():
        state_path.write_text(json.dumps(state, indent=2) + "\n")

    save()
    for index, (name, cmd) in enumerate(jobs, 1):
        state["current_stage"] = name
        state["stage_started_utc"] = now()
        save()
        started = time.monotonic()
        log = ROOT / "logs" / f"{index:02d}_{name}.log"
        print(f"[{index}/{len(jobs)}] {name} started {now()}", flush=True)
        with log.open("w") as stream:
            proc = subprocess.run([sys.executable, "-B", "-u", *cmd], cwd=ROOT / "src", env=env,
                                  stdout=stream, stderr=subprocess.STDOUT)
        record = {"name": name, "returncode": proc.returncode, "seconds": time.monotonic() - started,
                  "ended_utc": now(), "log": str(log.relative_to(ROOT))}
        state["completed"].append(record)
        if proc.returncode:
            state["status"] = "failed"
            save()
            raise SystemExit(f"{name} failed; no later stages executed. See {log}")
        save()
        print(f"[{index}/{len(jobs)}] {name} finished in {record['seconds']:.1f}s", flush=True)
    state["status"] = "complete"
    state["ended_utc"] = now()
    state["current_stage"] = None
    save()
    outputs = {str(p.relative_to(ROOT)): sha(p) for folder in ["data/processed", "results", "figures"]
               for p in sorted((ROOT / folder).rglob("*")) if p.is_file()}
    (ROOT / "provenance/outputs_sha256.json").write_text(json.dumps(outputs, indent=2) + "\n")
    print("All protocol stages complete; outputs saved separately from original manuscript/results.")


if __name__ == "__main__":
    main()
