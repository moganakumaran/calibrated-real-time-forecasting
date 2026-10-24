"""Wait for the separately running protocol, then audit and write a local report.

No manuscript changes, uploads or messages. This records incomplete/failed runs
honestly instead of treating the presence of intermediate files as success.
"""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def author_status():
    confirmation = ROOT.parent / "AUTHOR_CONFIRMATION_2026-10-22.json"
    if not confirmation.is_file():
        return "Authorship, funding and AI assistance before the revision still require author confirmation. "
    details = json.loads(confirmation.read_text())
    return (f"The author confirmed sole authorship as {details['author']}, no funding and no AI assistance "
            "before this revision. The actual Codex audit/editing/reproducibility assistance remains disclosed "
            "in the separate author-confirmed manuscript copy. ")


def status_note(status, **extra):
    path = ROOT / "provenance/completion_status.json"
    path.write_text(json.dumps({"updated_utc": datetime.now(timezone.utc).isoformat(),
                               "status": status, "monitor_pid": os.getpid(), **extra}, indent=2) + "\n")


def main():
    status_note("waiting_for_protocol")
    previous = None
    while True:
        try:
            state = json.loads((ROOT / "provenance/run_status.json").read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(20)
            continue
        key = (state["status"], state["current_stage"])
        if key != previous:
            print("Protocol:", key, flush=True)
            previous = key
        if state["status"] == "failed":
            stage = state["current_stage"]
            status_note("protocol_failed", failed_stage=stage)
            (ROOT / "REPRODUCTION_REPORT.md").write_text(
                "# Independent reproduction status\n\n"
                f"The protocol stopped at `{stage}`. No later stages were executed. "
                "This is an incomplete reproduction, not evidence that the manuscript results were replicated. "
                "See `provenance/run_status.json` and the corresponding stage log. "
                "Original results and the revised manuscript remain preserved.\n")
            return
        # The controller writes output hashes after setting status complete.
        if state["status"] == "complete" and (ROOT / "provenance/outputs_sha256.json").is_file():
            break
        time.sleep(20)
    status_note("auditing")
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "PYTHONDONTWRITEBYTECODE": "1"})
    with (ROOT / "logs/27_reproduction_audit.log").open("w") as stream:
        p = subprocess.run([sys.executable, "-B", str(ROOT / "tools/audit_reproduction.py")],
                           cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    audit_path = ROOT / "provenance/reproduction_audit.json"
    if not audit_path.is_file():
        status_note("audit_failed", returncode=p.returncode)
        (ROOT / "REPRODUCTION_REPORT.md").write_text(
            "# Independent reproduction status\n\nThe 26 protocol stages completed, but the final audit failed "
            "before producing its report. Inspect `logs/27_reproduction_audit.log`. "
            "Do not treat the run as validated or replace manuscript values.\n")
        return
    report = json.loads(audit_path.read_text())
    counts = report["scientific_values"]
    checks = report["checks"]
    lines = ["# Independent reproduction report", "",
             "This is a new run of the supplied protocol on the recorded Mac environment, not recovered original records.", "",
             f"All 26 protocol stages completed. Final audit: {sum(c['passed'] for c in checks)}/{len(checks)} checks passed.",
             f"Scientific numeric fields exactly matching the saved summaries: {counts['exact_matches']}/{counts['total']}; "
             f"different fields: {counts['different']}. Timing fields are separated because hardware/environment differ.", "",
             "## Principal comparisons", "",
             "| Level | Quantity | Original summary | New run |", "| --- | --- | ---: | ---: |"]
    labels = {"pv_qs7": "PV-TCQN QS7", "lightgbm_qs7": "LightGBM QS7", "qs7_reduction_percent": "QS7 reduction vs LightGBM (%)",
              "shortest_horizon_reduction_percent": "Shortest-horizon reduction (%)",
              "pv_rw_aci_pooled_coverage": "PV RW-ACI pooled coverage",
              "pv_rw_aci_max_monthly_gap": "PV RW-ACI maximum pooled monthly gap",
              "pv_rw_aci_under85_share": "PV RW-ACI home-month share below 85%"}
    for level, comparison in report["primary_comparisons"].items():
        for key, label in labels.items():
            a, b = comparison["original"][key], comparison["new_run"][key]
            if a is not None:
                lines.append(f"| {level} | {label} | {a:.10g} | {b:.10g} |")
    lines += ["", "## Interpretation and remaining work", "",
              "The comparison includes all saved scientific numeric fields, without tuning or selecting reruns to obtain agreement. "
              "Exact agreement of summary values alone does not establish identical original model weights. "
              "The new prediction files, checkpoints, logs, configuration, versions and checksums provide auditable evidence for this run.", "",
              "Full deltas are in `provenance/summary_comparison.csv`; checkpoint replay, array checks and integrity results are in "
              "`provenance/reproduction_audit.json`. Inspect any discrepancies and failures before deciding how to cite this reproduction. "
              "The original paper/results have not been changed.", "",
              author_status() + "No automatic submission-readiness certification, manuscript replacement, push or publication was performed."]
    if report["failed_checks"]:
        lines += ["", "## Failed audit checks", ""]
        lines += ["- " + c["name"] for c in report["failed_checks"]]
    (ROOT / "REPRODUCTION_REPORT.md").write_text("\n".join(lines) + "\n")
    status_note("audited" if p.returncode == 0 else "audit_checks_failed",
                report="REPRODUCTION_REPORT.md", scientific_counts=counts, returncode=p.returncode)
    print("Completion report written; manuscript unchanged.", flush=True)


if __name__ == "__main__":
    main()
