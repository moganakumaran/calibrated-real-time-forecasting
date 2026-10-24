# Phase 5 — Final independent referee report and go/no-go

Completed: 2026-10-23  
Manuscript reviewed: `phase4_visual_math_reference_revision_2026-10-23/`  
Commit recorded at review: `fc6d94cf02991958a7a1e4fa7298336bd626d61a`  
Scope: skeptical IEEE referee, forecasting/online-conformal review, and reproducibility audit. No manuscript or numerical result was changed in Phase 5.

## Executive assessment

The revised paper is internally consistent and materially more defensible than the earlier version. Its main result is now correctly bounded to eight selected horizons, while the equally weighted all-48 reversal, household near-tie, conditional-regime failures, empty sets, unfavorable ablation and runtime limitations remain visible. The work is framed as controlled application-specific evidence rather than a new TCN--CQR architecture or delay theorem.

The evidence supports submission of a retrospective empirical paper with explicit limitations. It does **not** establish independent confirmation, general 24-hour superiority, population-level household superiority, conditional or joint coverage, end-to-end deployment latency, or field reliability.

## Required final checklist

| Item | Verdict | Precise evidence |
|---|---|---|
| Abstract fairly distinguishes selected-eight vs all-48 forecasts | **PASS** | Abstract reports 6.82% lower QS7 on eight selected horizons, four short leads and unknown selection chronology; it separately reports 0.50% numerically worse over all 48 with an interval including zero. `reproduction_2026-10-22/results/metrics.json`; `docs/review/phase6/results/statistics.json`. |
| Novelty is empirical and prior art is fairly cited | **PASS** | Introduction and Related Work expressly deny invention of TCN--CQR and delayed ACI and cite Hu et al., Hallberg Szabadváry, and the September 2026 El Halabi--Brandt preprint. The contributions are application-specific implementation, causal update handling, matched controls and audits. |
| Household near-tie and mixed all-48 metrics are retained | **PASS** | Results VI-A retains household QS7 0.120118 vs 0.120014 as inconclusive; all-48 PV-TCQN has lower RMSE but higher MAE and 0.50% higher QS7. `docs/review/phase6/results/statistics.json`; `docs/review/phase4/evidence/benchmark.csv`. |
| High-PV and ramp undercoverage, regime hindsight and overlap are clear | **PASS** | Abstract, Results VI-B, Fig. 6 caption and Conclusion retain 78.64%/71.98%, state overlapping retrospective target-outcome labels, and deny deployable predictors or conditional guarantees. `docs/review/phase5/results/group_metrics.csv`. |
| Rolling CQR vs RW-ACI coverage-tail/width trade-off is accurate | **PASS** | Results states 4.18--4.43% household mean-length cost, separately ranked tails, rolling-CQR sharpness and mixed aggregate behavior. `docs/review/phase5/results/rolling_vs_rw_aci.json`. |
| Empty sets and Winkler limitations are mathematically consistent | **PASS** | Method Eq. (6), scoring prose, Table II and Results treat empty sets as misses of zero length, retain full denominators, suppress full-population conventional Winkler for affected rows and label common-nonempty comparisons. Correctness tests pass 34/34. |
| Historical vs freshly reproduced vs post hoc experiments are distinguishable | **PASS** | Introduction discloses missing historical run records; Results identifies preserved reproduction, supplementary controls and retrospective analyses; saved reproduction, Phase 4--6 and Phase 3 matched-control artifacts are separately located and labeled. |
| Dataset quality caveats and screening cohorts are correctly grounded | **PASS** | Setup and Results distinguish 299 completeness-selected homes, 54 three-year screened customers and 187 single-year customers; substituted readings, absent controlled-load zero fills, zero-PV periods, timestamp/DST, approximate location and real-time availability uncertainty remain explicit. |
| The 48-row runtime scaling experiment is not presented as full 48-horizon calibration validation | **PASS** | Abstract, Results VI-E and Conclusion state that 48 rows repeat eight-horizon score distributions, identify combined-call timing and exclude ingestion, transport, persistence, loading, serialization, scheduling and orchestration. `docs/review/phase6/results/runtime.json`. |
| Remaining LightGBM validation-weighting and tuning-budget differences are either tested or qualified | **PASS** | Phase 3 aligns validation weighting and produces the same tree counts and bitwise-identical forecasts. Results retains unequal representation, budget units, historical tuning and optimal-tuning limitations. `docs/review_final/phase3_matched_baseline_v1/results.json`. |
| No unobserved validation/test split is falsely described as independent | **PASS** | Setup labels all supplementary controls and regime diagnostics retrospective and says the current reproduction does not make the test period untouched. Phase 3 is called a retrospective control, not independent confirmation. |
| Every figure, table, equation, citation and numeric abstract statement renders correctly and has evidence | **PASS** | Phase 4 checks 8 equations, Algorithm 1, Tables I--III, Figs. 1--8, 30 references and all 48 pages across three renderers. `docs/review_final/04_visual_math_reference_QA.md`; Phase 4 validator PASS 49/49. |
| No operational, edge-compute, conditional-coverage or theoretical guarantees are invented | **PASS** | Abstract, Method, Results and Conclusion consistently deny deployed reliability, conditional/simultaneous coverage, arbitrary-sequence guarantees, broad architectural superiority and edge-device conclusions. |

## Residual risks

### Critical

**None for the current bounded claims.** A critical issue would reappear if the title, abstract, response letter or submission metadata were changed to claim general 24-hour superiority, independent confirmation, conditional coverage, or deployment readiness.

### Major

1. **No genuinely untouched confirmatory period.** January--June 2013 has informed supplementary analyses. HAC and moving-block calculations handle within-period origin dependence, not model-selection, training-seed or cross-year uncertainty. Resolution requires a prospectively locked later year, independently reserved chronology or external dataset.
2. **Household evidence has one PV-TCQN seed and no all-48 household evaluation.** The near-tie cannot establish equivalence or superiority, and household training-seed uncertainty remains unknown. Resolution requires additional household seeds and saved all-48 household forecasts under a predetermined protocol.
3. **Baseline fairness is only partially aligned.** Phase 3 removes validation-horizon weighting as an observed cause for this particular fitted LightGBM baseline, but tree rows and neural windows represent information/budget differently and historical tuning budgets are not commensurate. Resolution requires prospectively matched model-selection budgets and a new untouched evaluation period.
4. **All-48 calibration quality is not measured.** The 48-row runtime state is replicated scaling evidence only. Resolution requires a genuine all-48 calibration run with horizon-specific observed scores and predefined evaluation.
5. **Operational validity remains untested.** The data are a retrospective completeness-selected portfolio proxy, not a feeder or meter-quality-clean deployment cohort. End-to-end service timing, data arrival, persistence, failures and decision outcomes were not measured. Resolution requires prospective service instrumentation and field evaluation.

### Minor

1. Original eight-horizon and gamma-selection chronology remain unknown and must stay disclosed.
2. The cached external `algorithm.sty` produces an encoding warning and the compiler reports underfull boxes; Phase 4 confirmed no visible defect.
3. The AI-assistance statement is policy-aligned based on the documented revision, but final wording and venue-specific declarations remain the author's responsibility at submission.
4. The final IEEE Word reference page is intentionally not balanced to fill both columns; it is legible and not a correctness defect.

## Submission-readiness verdict

### A. Editorial/typesetting readiness: **READY**

The IEEE DOCX, regular DOCX, LaTeX PDF and Word PDFs have consistent scientific content. Native equations, tables, figures, citations, page boundaries and disclosure passed structural and page-by-page checks. No editorial/typesetting blocker remains.

### B. Scientific evidence readiness: **PARTIAL — READY FOR PEER REVIEW ONLY WITH THE CURRENT BOUNDED CLAIMS**

The evidence is sufficient for peer review as a retrospective, configuration-specific empirical study. It is not ready to support general model superiority, independent validation, conditional/joint coverage, population reliability, or deployment claims. The current manuscript correctly avoids those claims, so the residual evidence limitations are transparent rather than submission-blocking.

## Go/no-go

**GO for submission preparation under the current text and claims.**  
**NO-GO for any stronger superiority, guarantee or deployment wording without new experiments.**

Before upload, the author should verify the target venue's page/template/declaration requirements, approve the final disclosure, and ensure the uploaded DOCX/PDF hashes match the Phase 4 handoff. Those are submission-control steps, not scientific reruns.
