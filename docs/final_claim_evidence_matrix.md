# Final claim--evidence matrix

Final manuscript: `phase4_visual_math_reference_revision_2026-10-23/`  
Evidence labels: **REPRODUCED**, **VERIFIED**, **MEASURED**, **POST HOC**, **REPORTED/NOT INDEPENDENTLY VERIFIED**, and **UNKNOWN** retain the definitions established in Phase 0.

| Claim | Exact result/configuration artifact | Status and statistical qualifier | Final manuscript location |
|---|---|---|---|
| Study evaluates 300 homes and a 299-home portfolio proxy, not a physical feeder | `reproduction_2026-10-22/data/processed/ausgrid_2012_13.npz`; `docs/review/phase3/evidence/data_audit.json`; Phase 3 experiment manifest | **REPRODUCED + VERIFIED** construction; completeness-selected, not quality-clean or feeder-identified | Abstract; Setup V-A |
| Chronology is July--November fitting, Dec. 1--15 validation, Dec. 16--31 calibration, Jan.--June testing | Reproduction configuration and saved origin arrays; `reproduction_2026-10-22/provenance/run_plan.json` | **REPRODUCED + VERIFIED**; later supplementary use makes test analyses retrospective | Abstract; Setup V-B |
| PV-TCQN predicts seven ordered quantiles for 48 half-hour horizons | saved model configuration/checkpoints; `src/models.py`; correctness tests | **CODE-VERIFIED**; describes evaluated implementation, not general architecture superiority | Abstract; Method IV-B |
| Selected-eight aggregate QS7 is 10.761 vs 11.549, 6.82% lower | `reproduction_2026-10-22/results/metrics.json`; `docs/review/phase6/results/statistics.json` | **REPRODUCED + VERIFIED**; bounded to original selected horizons and original LightGBM-Q configuration | Abstract; Table I; Results VI-A; Conclusion |
| Four selected horizons are within four hours; their selection chronology is unknown | horizon set in setup/config `{1,2,4,8,12,24,36,48}`; historical recovery audit | Count **VERIFIED**; selection rationale **UNKNOWN** | Abstract; Setup V-B; Results VI-A; Conclusion |
| Selected-eight paired QS7 difference is -0.787794, with HAC and moving-block support | `docs/review/phase6/results/statistics.json`; `statistical_comparison.csv` | **VERIFIED, POST HOC** within-period inference; HAC p=0.000363 at lag 48, p=0.0213 at lag 336; one-week interval [-1.483325,-0.139583]; no training-seed/cross-year inference | Abstract; Setup V-D; Results VI-A; Conclusion |
| All-48 QS7 reverses: 11.669 vs 11.610, PV-TCQN 0.50% higher | `docs/review/phase4/evidence/benchmark.csv`; `docs/review/phase6/results/statistics.json` | **REPRODUCED + VERIFIED, supplementary/POST HOC**; p=0.797 at lag 48 and interval includes zero | Abstract; Table I; Fig. 3; Results VI-A; Conclusion |
| All-48 point metrics are mixed: PV-TCQN lower RMSE but higher MAE | `docs/review/phase4/evidence/benchmark.csv`; Table I source | **REPRODUCED + VERIFIED**; no uniform metric winner | Results VI-A; Conclusion qualitative scope |
| Household selected-eight QS7 is a near-tie, 0.120118 vs 0.120014 | saved household forecasts; `docs/review/phase6/results/statistics.json`; reproduction metrics | **REPRODUCED + VERIFIED** for one neural seed; p=0.772 and interval includes zero; neither superiority nor equivalence | Abstract; Table I; Results VI-A; Conclusion |
| Phase 3 all-48 LightGBM validation weighting selects the same trees and identical forecasts | `docs/review_final/phase3_matched_baseline_v1/config.json`; `artifacts/metadata.json`; `results.json`; saved predictions | **VERIFIED retrospective sensitivity**, preregistered before fitting; predictions bitwise identical; not independent confirmation | Setup V-E; Results VI-A |
| Remaining LightGBM representation/tuning-budget differences are unresolved | Phase 3 protocol/results; original neural/tree configs | **VERIFIED limitation**; validation weighting tested, representation and commensurate historical tuning not established | Setup V-E; Results VI-A |
| Direct finite integer rank and delayed score availability match code | `src/conformal.py`; `validation/test_correctness.py`; `docs/review/phase2/evidence/rank_impact.json` | **CODE-VERIFIED**, 34/34 focused tests; does not inherit general exchangeable/ACI/conditional/joint guarantees | Problem III; Method IV-C; Eq. (6); Algorithm 1 |
| Negative correction may make an empty set; maximum reported count is 391/20,738,400 | `docs/review/phase5/results/calibration_audit.json`; `main_metrics.csv`; corrected bounds | **REPRODUCED + VERIFIED**; empty sets are misses with zero length; full-population conventional Winkler unavailable for affected rows | Setup V-D; Table II; Results VI-B |
| Common-nonempty household Winkler comparisons exclude 99/38 records | `docs/review/phase5/results/rolling_vs_rw_aci.json`; calibration records | **VERIFIED on stated masks**; conditional comparison, not full-population proper-score dominance | Results VI-B |
| RW-ACI improves household worst-decile home--month--horizon coverage with 4.18--4.43% longer mean sets | `docs/review/phase5/results/rolling_vs_rw_aci.json` | **VERIFIED, POST HOC diagnostic**; separately ranked tails; mixed aggregate behavior; no universal advantage | Abstract; Results VI-B; Conclusion |
| Household PV-TCQN RW-ACI coverage is 78.64% in high PV and 71.98% in large ramps | `docs/review/phase5/results/group_metrics.csv`; `calibration_audit.json` | **REPRODUCED + VERIFIED, POST HOC**; outcome-conditioned overlapping regimes, repeated forecast--target records, not deployable predictors or conditional guarantee | Abstract; Fig. 6; Results VI-B; Conclusion |
| Removing past clear-sky improves QS7 while removing future clear-sky worsens it | `docs/review/phase4/evidence/benchmark.csv`; `contrasts.json`; saved variant forecasts | **REPRODUCED + VERIFIED, POST HOC**; all declared variants retained; no favorable variant promoted to primary | Table III; Results VI-C; Conclusion |
| Matched net-load/calendar TCN is worse, supporting joint PV-related information in this architecture | `docs/review/phase4/evidence/benchmark.csv`; `contrasts.json` | **REPRODUCED + VERIFIED, POST HOC**; matched-information control, not architectural novelty or general superiority | Introduction contributions; Results VI-C; Conclusion |
| 431 category-days have literal non-actual labels; conservative exclusions affect 16,647 household and 12,039 aggregate records | `docs/review/phase3/evidence/data_audit.json`; `quality_sensitivity.json`; customer flags | **REPRODUCED + VERIFIED under declared rules**; issued forecasts/training unchanged, not clean-data retraining | Setup V-A; Results VI-D |
| Ratnam cohorts are 54 across three clean years and 187 for the single July 2012--June 2013 year | Ratnam et al. DOI `10.1080/14786451.2015.1100196`; publisher note; Phase 1 primary-source audit | **VERIFIED from primary source**; different from 299 completeness cohort; 54/53-home checks are frozen-forecast POST HOC sensitivity | Results VI-D |
| 0.520 s median / 0.624 s p95 300-home combined cycle uses replicated 48-row calibration state | `docs/review/phase6/results/runtime.json`; `runtime.csv`; `runtime.py`; config/protocol correction | **MEASURED + VERIFIED scaling only**; one thread, 20 warmups, 100 calls; directly timed combined call | Abstract; Setup V-D; Results VI-E; Conclusion |
| Runtime excludes operational stages and is not edge/end-to-end latency | Phase 6 runtime protocol; code inspection | **VERIFIED limitation**; excludes ingestion, transport, persistence, model loading, serialization, scheduling and service/network orchestration | Abstract; Results VI-E; Conclusion |
| All-48 calibration quality is unavailable | Phase 6 runtime state mapping; absence of horizon-specific all-48 calibration scores | **UNKNOWN/NOT RUN** as scientific calibration experiment; runtime scaling must not be used as quality evidence | Abstract; Results VI-E; Conclusion/future work |
| Original historical weights, forecasts, tuning records and exact losses are unavailable | historical recovery and reproduction audits; `docs/review/phase3/evidence/reproduction_audit.json` | **UNKNOWN / REPORTED, NOT INDEPENDENTLY VERIFIED**; reproduction is preserved but not represented as identical historical recovery | Introduction; Results VI-D; Conclusion |
| Equations, tables, figures, citations and final disclosures render correctly | `docs/review_final/04_visual_math_reference_QA.md`; `phase4.../validation/render_checks.json`; `phase4_validation_results.json`; final PDF diff | **STRUCTURALLY + VISUALLY VERIFIED**; Phase 4 PASS 49/49 and every page inspected | Entire final deliverable |

## Frozen claim boundary

Without new experiments, the final manuscript must not be revised to assert:

- uniform 24-hour or all-horizon PV-TCQN superiority;
- household superiority or equivalence;
- independent confirmation on January--June 2013;
- conditional, simultaneous, joint or arbitrary-sequence coverage guarantees;
- full-population Winkler dominance where any signed conformal set is empty;
- validated all-48 calibration quality from the replicated-state timing benchmark;
- production, end-to-end, edge-device or field-deployment latency;
- physical-feeder or meter-quality-clean population scope;
- a novel TCN--CQR method or delayed-feedback theorem.

New evidence needed to expand that boundary is a prospectively locked model-selection protocol, genuinely untouched later year or external dataset, additional household seeds, saved all-48 household forecasts, genuine horizon-specific all-48 calibration, and instrumented end-to-end field timing.
