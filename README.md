# Calibrated real-time net-load forecasting

This repository is the public reproducibility core for **PV-Aware Multi-Horizon Probabilistic Net-Load Forecasting with Horizon-Delayed Online Conformal Calibration**.

## Evidence boundary

This is a cleaned snapshot of the audited October 2026 reproduction, not a recovery of the unavailable historical original run. It preserves the executable source, declared configurations, compact result summaries, logs, figures, checksums, and the statistical/runtime audit used by the manuscript. It does not change or rerun any experiment.

The bounded empirical result is a 6.82% lower aggregate QS7 for PV-TCQN than the original LightGBM-Q configuration on the original eight selected horizons. The all-48 result reverses numerically by 0.50% and is statistically inconclusive; the household comparison is also inconclusive. See `docs/final_claim_evidence_matrix.md` before citing results.

## Data

The study uses the public Ausgrid Solar Home Electricity dataset, indexed by the IEEE PES Task Force on Data Sharing:

https://ieee-pes-data-sharing.org/datasets/detail/5795a399-f06e-4747-b518-cb9d961616a2

Raw and processed data are not redistributed here. Place the provider-supplied `2012-2013 Solar home electricity data v2.csv` and the postcode file at the paths documented in `src/prepare_data.py`. Expected input hashes are recorded in `provenance/inputs_sha256.json`.

## Reproduction

Create an isolated Python environment, install `requirements.txt`, review `run_all.sh`, and run it only in a fresh working copy. The complete CPU reproduction is computationally expensive. The original evidence directory deliberately refused to overwrite completed outputs.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash run_all.sh
```

The saved compact outputs are in `results/`; reference summaries are in `reference_results/`. Statistical and runtime audit code and saved outputs are under `audits/phase6/`.

## Large artifacts intentionally omitted

Approximately 3.1 GB of generated prediction arrays, fitted checkpoints, processed data, and third-party raw inputs are not placed in GitHub. Their original relative paths, sizes, and SHA-256 values are listed in `LARGE_ARTIFACTS.json`; the complete generated-output checksum record remains in `provenance/outputs_sha256.json`. These omissions avoid redistributing source data and GitHub-inappropriate binary files. They do not imply that the omitted artifacts were unavailable to the audit.

## Scope and limitations

- The January--June 2013 test period has informed retrospective analyses and is not an untouched confirmation period.
- Household PV-TCQN results use one neural seed; no all-48 household evaluation is reported.
- The 48-row calibration runtime state replicates observed eight-horizon score distributions and is not an all-48 calibration-quality result.
- The finite-rank clipped delayed-feedback implementation carries no claimed conditional, simultaneous, exchangeable-CQR, or arbitrary-sequence guarantee.
- This repository contains no credentials, private dataset, or submission-system material.

## License

Code is released under the MIT License. Dataset use remains subject to the data provider's terms. Published papers and third-party materials retain their respective rights.
