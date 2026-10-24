# Phase 3 — Matched LightGBM validation-weighting sensitivity

Date: 2026-10-23  
Experiment: `phase3_lgbm_all48_validation_v1_20261023`  
Status: **complete retrospective sensitivity; not independent confirmation**

## Evidence boundary

This experiment answers one narrow fairness question: whether replacing the original LightGBM-Q validation sample over eight selected horizons with equal weighting over all 48 horizons changes early stopping or the saved test forecasts. It does not align feature representation, training-budget units, historical tuning effort, or model-search budgets. January--June 2013 had already been examined, so none of the results below are independent test evidence.

The protocol was locked before model fitting in `03_matched_baseline_protocol.md`. The run reused the exact original 120,000 aggregate LightGBM training draws, fixed the original seven quantile models and hyperparameters, and changed only validation to the exhaustive Cartesian product of 673 complete-window validation origins and 48 horizons (32,304 rows). No test result was used for early stopping or model selection.

## Result

The all-48 validation weighting selected the same seven tree counts as the original sampled-eight validation:

`[132, 108, 72, 168, 80, 90, 138]` for quantiles `[0.025, 0.10, 0.25, 0.50, 0.75, 0.90, 0.975]`.

The regenerated all-48 LightGBM test predictions are bitwise identical to the preserved original replay (`array_equal=True`; maximum absolute difference `0`; differing values `0`). Therefore the existing comparisons are unchanged:

| Evaluation | PV-TCQN QS7 | Matched-validation LightGBM-Q QS7 | PV-TCQN minus LightGBM-Q | Interpretation |
|---|---:|---:|---:|---|
| Selected eight horizons, primary | 10.761149 | 11.548943 | -0.787794 | 6.821% lower for PV-TCQN in this retrospective evaluation |
| All 48 equally weighted horizons, secondary | 11.668506 | 11.610077 | +0.058429 | 0.503% higher for PV-TCQN; ordering reverses |

Dependence-sensitive checks preserve the same distinction. For selected eight horizons, HAC two-sided p-values are 0.000363 at lag 48 and 0.021272 at lag 336; the 336-origin circular-block 95% interval is [-1.519389, -0.154474]. For all 48 horizons, the lag-48 HAC p-value is 0.796949 and the 336-origin block interval is [-0.688841, 0.696049], so the all-48 comparison remains inconclusive. These values use the preregistered Phase 3 resampling seed and are retained as sensitivity outputs rather than substituted into the manuscript's historical interval.

Point metrics also remain unchanged. On the selected eight horizons, LightGBM-Q MAE/RMSE are 18.855148/31.322112 and the PV-TCQN seed-mean values are 17.880683/27.609537. Across all 48 horizons, LightGBM-Q MAE/RMSE are 19.018169/31.641244 and PV-TCQN values are 19.433268/30.607609, preserving the mixed all-horizon metric ordering.

Household results were not rerun under this control. The existing selected-eight household comparison remains a near-tie in QS7 (PV-TCQN 0.120118 versus original LightGBM-Q 0.120014), not an equivalence result. Household all-48 forecasts remain unavailable.

## Scientific conclusion

Validation-horizon weighting is not an observed explanation for the fitted original LightGBM baseline's selected-eight/all-48 reversal: changing that weighting left the fitted stopping points and every test prediction unchanged. This is useful reproducibility evidence, but it does not establish equal information representation, commensurate learning budgets, optimal LightGBM tuning, or prospective superiority. A decisive comparison still requires prospectively locked model-selection budgets and a genuinely unexamined later period or external dataset.

The manuscript was changed only to report this bounded sensitivity result and its residual limitations. No abstract, conclusion, table value, or headline numerical claim was changed.

## Artifacts

- Protocol: `docs/review_final/03_matched_baseline_protocol.md`
- Configuration: `docs/review_final/phase3_matched_baseline_v1/config.json`
- Fit/replay code: `docs/review_final/phase3_matched_baseline_v1/run_experiment.py`
- Scoring code: `docs/review_final/phase3_matched_baseline_v1/score_experiment.py`
- Pre-run checks: `docs/review_final/phase3_matched_baseline_v1/test_protocol.py`
- Post-run checks: `docs/review_final/phase3_matched_baseline_v1/test_results.py`
- Machine-readable summary: `docs/review_final/phase3_matched_baseline_v1/results.json`
- Per-horizon scores: `docs/review_final/phase3_matched_baseline_v1/per_horizon.csv`
- Saved models, keys, masks and predictions: `docs/review_final/phase3_matched_baseline_v1/artifacts/`
- Separately versioned manuscript: `phase3_matched_baseline_revision_2026-10-23/`

SHA-256 values recorded at handoff:

- `config.json`: `415301ebb1c2779f2bfa859230cb1da48086691e07f892f8f1d7e472e5008d2c`
- `run_experiment.py`: `f177ac5bab8c08790719f853b2f61ac45078cae6321e089eabf34b74af03f673`
- `score_experiment.py`: `e169fbf2b7f0ff58a3763f63e3ca088457d22f88d2c5e104f1d5e2b1b116d48a`
- `results.json`: `9b0e652902c07937c028065c660ff6ff2a23d2c2e20e3f012fb89896863c986f`
- `per_horizon.csv`: `7554bf67d9946a9843cbbb7c555b1d1ee7c5bce7e66e7af78312c179dbf0ebc3`
- Phase 3 LaTeX PDF: `a942071e8420fa701dc2c5ec318609d4a5f6f4558826b3c61b988123f9aef4a9`

## Validation log

Pre-run protocol checks ultimately passed 5/5: locked horizons, absent output, explicit retrospective label, exact reconstruction of the original seed-0 training draw, and exhaustive/equal validation weighting. Before any fit existed, the first check exposed an incorrect expected validation count of 34,608; the protocol and test were corrected to 32,304 because 47 incomplete-window origins are excluded. The first execution attempt then stopped at import because the preserved LightGBM vendor path was absent; no model or result had been produced, and the runner was amended to use the preserved reproduction environment. Both amendments are timestamped in the protocol.

The completed fit/scoring run and checks were:

```text
.../ECF/artifact/.venv/bin/python -B test_protocol.py       PASS 5/5
.../ECF/artifact/.venv/bin/python -B run_experiment.py      PASS
.../ECF/artifact/.venv/bin/python -B score_experiment.py    PASS
.../ECF/artifact/.venv/bin/python -B test_results.py        PASS 6/6
python -B validation/test_correctness.py                     PASS 34/34
python -B validation/validate_fixes.py                       PASS 11/11, 20/20, 8/8
python -B validation/validate_phase3.py                      PASS
tectonic paper/main.tex                                      PASS (12 pages, 30 references)
python -B word/build_editable_word.py                        PASS
```

The post-run checks cover the retrospective label, balanced validation, finite/noncrossing prediction shapes, both horizon comparisons, all 48 per-horizon rows, and exact baseline identity. The correctness suite emitted its expected `Mean of empty slice` warning in the deliberate all-missing test. Tectonic retained a pre-existing invalid-UTF-8 warning in `algorithm.sty` and underfull box warnings; these are Phase 4 visual/typesetting QA items, not evidence of changed numerical output.

Visual inspection of the changed LaTeX pages confirmed that the added protocol and result paragraphs are legible and remain within the two-column layout. Fresh Word-rendered PDF inspection is intentionally deferred to mandatory Phase 4.
