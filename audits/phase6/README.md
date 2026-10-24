# Phase 6 execution record

This directory contains dependence-aware paired QS7 inference and matched local runtime measurements from preserved October 22 reproduction and Phase 4 artifacts. No fitting, parameter selection, manuscript edit, commit, push or publication is performed.

Run from the workspace root with the existing research environment:

```sh
pv_python=scholarforge/papers_v3/ECF/artifact/.venv/bin/python
phase6=scholarforge/papers_v3/calibrated-real-time-forecasting/docs/review/phase6
"$pv_python" -B "$phase6/statistics.py"
DYLD_LIBRARY_PATH=/opt/homebrew/lib/python3.13/site-packages/torch/lib MPLCONFIGDIR=/private/tmp/pv_phase6_mpl "$pv_python" -B "$phase6/runtime.py"
"$pv_python" -B "$phase6/test_phase6.py"
"$pv_python" -B "$phase6/present_results.py"
```

The runtime protocol deliberately uses one thread. The 48-horizon calibration timing repeats observed eight-horizon score-buffer distributions to measure state-scaling cost only; it is not a new reliability evaluation. See `evidence/protocol_correction.json` for the pre-report correction to the incoming score stream.

The authoritative phase report is `../phase6_statistics_runtime.md`. Machine-readable outputs are under `results/`.
