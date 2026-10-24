# Phase 3 preregistration — all-48 validation-weighting control

**Protocol locked:** 2026-10-23T14:51:07Z, before creation or execution of the experiment script.  
**Experiment ID:** `phase3_lgbm_all48_validation_v1_20261023`  
**Status at lock:** no new model fitted; no new validation or test score computed.  
**Scope:** retrospective aggregate sensitivity only, not independent confirmation.

**Pre-run factual amendment, 2026-10-23T14:54:01Z:** the validation row count was made explicit as 673 complete-window origins $\times$ 48 = 32,304. The initial test incorrectly expected 721 origins without accounting for the 47 starts excluded to keep 48-step targets inside the validation split. The test failed before fitting; no outcome, model or decision rule was inspected or changed.

**Pre-run environment amendment, 2026-10-23T14:54:57Z:** the first launch stopped at `import lightgbm` because the library is stored in the reproduction's preserved `vendor/` directory rather than installed inside the base ECF environment. No artifact directory or fit was created. The runner was amended only to prepend that recorded vendor path; execution uses the same preserved LightGBM 4.6.0 and OpenMP library path as Phase 4.

## Research question

Holding the original aggregate LightGBM training draws, features, quantile objectives, hyperparameters and stopping rule fixed, how does replacing its selected-eight validation sampling with equally weighted all-48 validation affect its comparison with the already saved PV-TCQN forecasts?

This control isolates validation-horizon weighting. It does **not** establish equal information representation, equal training examples, equal compute, equal historical tuning effort or independent test evidence.

## Locked inputs

- Processed data: `reproduction_2026-10-22/data/processed/ausgrid_2012_13.npz`.
- LightGBM training keys: first 120,000 rows of `docs/review/phase4/results/trees/aggregate_2x/training_keys.npy`. The full file SHA-256 is `7d475ad53284396b8b8a8dd94c88186d14e81a8249b4827acd13249c37e2e551`; Phase 4 records that its first block is the original seed-0 sample.
- Saved PV-TCQN selected-eight test predictions, seeds 0/1/2, SHA-256:
  - `c487a8588308ef136fe702a0b0fa43ce92eb036a4dffb2019776b54bcf6aaf5a`
  - `8e358b872c2a5aa808a235f7bf8bf10433e4997d28c93e24ed8013bf11c39ea0`
  - `491d09d7a4dd29475c284cc10672ba9543bbc669d2b10936837baf03323854af`
- Saved replayed PV-TCQN all-48 test predictions, seeds 0/1/2, SHA-256:
  - `49a976e6b4eee1da26a68903d05b4ad0789b87cc4f92f77c778f0758ed5cf265`
  - `9fb365851b8e05a18e8e9a602b33a6053436194fe7c904580bb8022bf5f4207e`
  - `5546b9498ddb83f57311d634ab3ccc3fb9031c96ccee622a1850c22e2f66280a`

## Locked model and fitting budget

- Level: 299-home aggregate only.
- One fixed LightGBM configuration; no hyperparameter search and no test-informed refit.
- Seven independently fitted quantile objectives at `0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95`.
- Maximum 600 boosting iterations per quantile; early stopping patience 50.
- `learning_rate=0.05`, `num_leaves=63`, `min_child_samples=50`, `subsample=0.8`, `subsample_freq=1`, `colsample_bytree=0.9`, `n_jobs=4`, with the original default random state.
- Training budget: exactly the original 120,000 aggregate row draws. One row carries one randomly selected target horizon; the keys are frozen above.
- Validation budget: all 673 aggregate validation origins whose complete 48-step target windows remain within the split, crossed with horizons 1--48 (32,304 rows), with each horizon equally weighted. This aligns horizon weighting with neural validation but not representation: a neural validation window emits 48 horizons jointly, while a tree validation row contains one horizon.
- Existing PV-TCQN comparator: three already saved seeds, each trained under the documented 25-epoch, 150-step-per-epoch cap with patience 4. No neural retraining or selection is permitted.
- Budget interpretation: each family has one fixed declared configuration in this control, but fit units and historical tuning effort are not commensurate. “Matched” below means validation weighting only.

## Locked predictions and outcomes

- Save new LightGBM boosters and complete all-48 validation/calibration/test predictions under the experiment ID. Do not overwrite reproduction or Phase 4 artifacts.
- Test period remains 1 January--30 June 2013 and has already been examined. The analysis must always be labeled **retrospective sensitivity**.
- No model, horizon subset, hyperparameter, seed, or reporting rule may be changed after inspecting test results.

## Locked outcomes

### Primary

- Aggregate QS7 over original horizons `{1,2,4,8,12,24,36,48}`.
- Comparator: mean per-origin loss across saved PV-TCQN seeds 0--2 versus the new LightGBM prediction.
- Report paired mean difference `PV-TCQN - LightGBM`, relative PV reduction, HAC sensitivity at lags `0,8,24,48,96,336`, and circular moving-block 95% intervals at blocks `48,96,336` with 2,000 repetitions and seed 32026.

### Secondary

- Aggregate all-48 QS7 with the same paired inference.
- Per-horizon QS7 for all 48 horizons.
- Aggregate selected-eight and all-48 MAE and RMSE from median forecasts.
- Validation pinball loss and selected tree iteration for each quantile.
- Existing household selected-eight MAE/QS7 will be restated from the audited reproduction, explicitly not re-estimated under matched validation weighting. No household all-48 result exists.

## Locked decision and reporting rules

1. Report all results regardless of direction or significance.
2. Do not promote the new tree model to the manuscript's primary baseline.
3. Do not describe the result as independent, confirmatory, prospective or a fair comparison in every respect.
4. If selected-eight PV-TCQN QS7 remains lower, say only that the advantage persists in this retrospective validation-weighting sensitivity.
5. If it disappears or reverses, preserve that result and weaken the manuscript accordingly.
6. The all-48 result remains a required secondary result and may not be omitted.
7. No test result may determine a new horizon subset, model setting or preferred seed.
8. A genuinely decisive comparison still requires an untouched later period or dataset and prospectively locked selection budgets.

## Feasibility gate

**GO for the narrow validation-weighting control.** Prior Phase 4 aggregate tree fits took seconds per quantile, and this experiment fits seven models once. The experiment is methodologically defensible as a retrospective sensitivity but cannot resolve the unknown historical tuning budget or create independent evidence.
