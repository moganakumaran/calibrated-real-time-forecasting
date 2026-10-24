# Phase 2 — Runtime and calibration semantics

**Completed:** 23 October 2026. **Working copy:** `phase2_runtime_calibration_revision_2026-10-23/`. The Phase 1 deliverables remain byte-identical to their recorded hashes. No forecasting, calibration or scoring implementation was changed.

## 1. Runtime audit

### Evidence inspected

- `docs/review/phase6/runtime.py`
- `docs/review/phase6/config.json`
- `docs/review/phase6/results/runtime.json`
- `docs/review/phase6/results/runtime.csv`
- `docs/review/phase6/evidence/protocol_correction.json`
- Saved reproduction checkpoint and LightGBM boosters listed with hashes in `runtime.json`

### Confirmed measurement semantics

- Hardware: Apple M1 Pro, 10 logical CPUs, 16 GiB; Python 3.13.5, NumPy 2.4.3, PyTorch 2.10.0 and LightGBM 4.6.0.
- Threading: PyTorch intra/inter-op and OMP/OpenBLAS/MKL/vecLib are constrained to one thread.
- Each reported row uses 20 untimed warmups followed by 100 timed calls measured with `time.perf_counter_ns()`.
- Median is `numpy.median`; p95 is `numpy.percentile(samples, 95)`. Recalculation from the 100 saved 48-row cycle samples gives exactly 624.4692 ms.
- `full_cycle()` directly calls feature assembly plus neural inference, matured-score computation and state update inside the timed function. The 300-home combined medians, 380.204 ms for eight rows and 519.609 ms for 48 rows, are therefore measured combined calls.
- The combined medians are not arithmetic sums: component-median sums are 370.414 ms and 490.548 ms respectively.
- For 48 rows, `CalibrationState` maps row indices through `np.arange(48) % 8`; it repeats the observed eight-horizon score-buffer distributions. This tests state-update scaling only and is not an all-48 calibration-quality evaluation.
- A pre-report protocol correction offset the incoming score stream by 37 positions so buffer replacement moves actual values. The full declared timing protocol was rerun; the superseded timing JSON was not interpreted or cited.

### Supported runtime numbers

| Operation, 300 homes | Median | p95 | Status |
|---|---:|---:|---|
| PV-TCQN feature assembly plus all-48 inference | 346.692 ms | 377.576 ms | MEASURED |
| RW-ACI buffer/level update, eight rows | 23.712 ms | 26.316 ms | MEASURED |
| RW-ACI buffer/level update, replicated 48-row state | 143.790 ms | 169.793 ms | MEASURED SCALING ONLY |
| Combined call, eight calibration rows | 380.204 ms | 410.634 ms | MEASURED |
| Combined call, replicated 48-row state | 519.609 ms | 624.469 ms | MEASURED SCALING ONLY |

Excluded from every timing claim: meter ingestion and transport, persistence/database writes, model loading, serialization, scheduling and network/service orchestration. The benchmark is local in-memory compute evidence, not production, end-to-end or edge-device latency.

## 2. Signed conformal-set audit

### Rank and endpoint convention

The manuscript equation selects

`k = min(n, max(1, ceil((n+1)(1-alpha))))`

and code `_rank_index(alpha, n)` implements the same one-based clipped rank followed by a zero-based array index. The returned set uses endpoints `lo-Q` and `hi+Q`, inclusive in the manuscript membership expression. The focused scalar-reference tests cover finite ranks, extreme alpha, duplicate FIFO replacement and singleton windows.

### Negative correction and empty sets

The signed CQR score is `max(lo-y, y-hi)` and is negative for an outcome inside the raw interval. A sufficiently negative selected correction can make `lo-Q > hi+Q`. Code returns those signed endpoints without sorting or widening them. The declared scoring policy is:

- empty set: guaranteed miss;
- set length: `max(upper-lower, 0)`, hence zero for an empty set;
- PICP and length retain the full observed-population denominator;
- conventional full-population Winkler is unavailable if any observed set is empty;
- restricted rolling-CQR/RW-ACI Winkler comparisons use explicitly stated common-nonempty masks and do not prove full-population proper-score dominance.

Table II remains consistent with the corrected Phase 5 audit: all aggregate rows have zero empty sets; affected household rows show `--` for full-population Winkler. The largest count remains 391 of 20,738,400 records for household PV-TCQN ACI. No endpoint or metric output was regenerated because no code defect was found.

### Delayed feedback

For a forecast issued at origin `t` and horizon `h`, the code first makes its score available at origin `t+h`. The vectorized update uses `j-h`, and Algorithm 1 uses the forecast at `t-h` with target `y[t-1]`, so the equation, algorithm and implementation agree under the paper's availability convention. Missing outcomes skip both adaptive and rolling updates. This convention assumes the newly completed interval reading is available at the next origin; actual meter transport and service delay were not measured.

The adaptive level itself is not clipped; only the empirical rank is clipped to a stored finite score. The manuscript correctly denies automatic inheritance of standard ACI arbitrary-sequence, exchangeable CQR, conditional, joint or simultaneous coverage guarantees.

## 3. Manuscript changes

- `paper/sections/abstract.tex`: identifies replicated eight-horizon calibration-state scaling and enumerates excluded operational costs.
- `paper/sections/problem.tex`: distinguishes the evaluated availability convention from unmeasured meter/service delay.
- `paper/sections/setup.tex`: defines empirical p95, 100-call denominator, 48-row scaling, and empty-set miss/zero-length semantics.
- `paper/sections/results.tex`: states that combined cycles are directly timed calls rather than sums, and separates local compute from production/end-to-end/edge latency.
- `paper/sections/conclusion.tex`: identifies the 48-row replication, excludes operational stages, and requests a genuine all-48 calibration study as future work.
- `README.md`: labels the separately versioned Phase 2 copy.

Regenerated outputs:

- `phase2_runtime_calibration_revision_2026-10-23/calibrated_net_load_revised.pdf`
- `phase2_runtime_calibration_revision_2026-10-23/word/calibrated_net_load_IEEE.docx`
- `phase2_runtime_calibration_revision_2026-10-23/word/calibrated_net_load_regular.docx`
- Word intermediate source, AST, compilation files and structural manifests
- `validation/validate_phase2.py`, `validation_results.json` and refreshed `correctness_tests.json`

## 4. Validation

| Check | Result |
|---|---|
| `docs/review/phase6/test_phase6.py` | **PASS**, 6/6; samples, summaries, protocol rows and manuscript effects reconcile. |
| `validation/test_correctness.py` | **PASS**, 34/34; includes rank, delayed feedback, negative correction, empty sets, endpoint scoring and no-future-information tests. The deliberate all-missing QS7 test emits the expected `Mean of empty slice` warning. |
| `validation/validate_fixes.py` | **PASS**; preprocessing 11, set metrics 20, replay 8. |
| `validation/validate_phase2.py` | **PASS**; evidence tokens, combined-call source, repeated state, exclusions, direct rank, empty-set policy, Winkler denominator and delay boundary verified. |
| Tectonic cached build | **PASS**, 12 pages, 30 resolved citations. Existing minor box and `algorithm.sty` encoding warnings are retained for Phase 4 inspection. |
| Word reconstruction | **PASS**; identical scientific content between formats, 3 tables, 8 figures, 30 references, 8 editable display equations and 181 verified numeric table cells. |
| Visual inspection of revised LaTeX PDF pages 1--7 and 10--11 where changed/semantic elements occur | **PASS**; rank equation, Algorithm 1, metrics, Tables I--II, abstract, runtime and conclusion are readable. |
| Fresh Microsoft Word PDF and every-page inspection | **NOT RUN**; mandatory Phase 4 work remains. |

## 5. Scientific impact and gate

No runtime observation, calibration output, interval endpoint, empty-set count, denominator or score changed. No experiment or model rerun is needed. The changes prevent the 0.520/0.624 s values from being read as end-to-end deployment measurements or validated all-48 calibration, and make the signed-set scoring convention explicit.

**Phase 2 gate: PASS.** Readers can now distinguish the local scaling benchmark from deployment/calibration-quality evidence and can trace the finite-rank, delayed-feedback and empty-set semantics from equation to code to reported metrics. Phase 3 may be assessed for feasibility; if skipped, mandatory Phase 4 can proceed.
