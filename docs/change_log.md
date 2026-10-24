# Final audited change log

Audit period: 2026-10-23  
Repository commit observed throughout: `fc6d94cf02991958a7a1e4fa7298336bd626d61a`  
The forecasting project remains untracked as a whole. No reset, checkout, commit, push, submission or modification of unrelated repository work was performed.

## Preserved baseline

The protected starting point remains `final_revision_2026-10-22/`. Phase 0 recorded its hashes in `docs/review_final/00_baseline_and_evidence_map.md`. The requested `calibrated_net_load_regular(2).docx` was absent; the repository-designated current counterpart was used. Older 18-page Word-era manuscripts were excluded.

Each revision was made in a new versioned directory:

1. `phase1_claims_revision_2026-10-23/`
2. `phase2_runtime_calibration_revision_2026-10-23/`
3. `phase3_matched_baseline_revision_2026-10-23/`
4. `phase4_visual_math_reference_revision_2026-10-23/`

## Phase 0 — baseline and evidence map

### Exact action

- Located the authoritative LaTeX source, current DOCX/PDF outputs, Word-generation path, audited reproduction, saved checkpoints/forecasts, Phase 2--7 evidence and figure/table provenance.
- Recorded commit, dirty/untracked status and baseline hashes.
- Distinguished the current 12/13-page manuscript from older 18-page versions.
- Inspected native source/OMML and actual PDFs to distinguish parser extraction artifacts from visual defects.

### Files added

- `docs/review_final/00_baseline_and_evidence_map.md`

### Scientific outputs

- None changed; inspection only.

## Phase 1 — scientific claim precision and novelty

### Authored text edits

- `paper/sections/abstract.tex`: bounded the main result to eight selected horizons; added the all-48 reversal, household inconclusiveness, regime hindsight and runtime scope.
- `paper/sections/introduction.tex`: reframed novelty as controlled application-specific evidence and acknowledged established TCN--CQR, ACI and delayed-feedback work.
- `paper/sections/setup.tex`: clarified selected-horizon concentration, unknown chronology and retrospective test use.
- `paper/sections/results.tex`: retained mixed/unfavorable evidence, qualified equivalence/superiority and distinguished the 54/187 Ratnam cohorts.
- `paper/sections/conclusion.tex`: synchronized the bounded selected-eight claim, all-48 reversal, household near-tie, unfavorable past-clear-sky finding and limitations.
- `paper/tables/models.tex`: changed duplicate-looking headers to `QS7 (selected 8)` and `QS7 (all 48; aggregate)` and clarified metric/seed scope.
- `README.md`: labeled the version and rebuild boundaries.

### Regenerated outputs

- LaTeX PDF; IEEE and regular DOCX files; editable Word source/AST; compilation records; DOCX structural manifests.

### Scientific impact

- Editorial precision only. No training, tuning, forecast, checkpoint, table number, figure data, p-value, confidence interval or calibration result changed.

## Phase 2 — runtime and signed-calibration semantics

### Authored text edits

- `paper/sections/abstract.tex`: identified repeated eight-horizon distributions in the 48-row runtime state and enumerated excluded operational costs.
- `paper/sections/problem.tex`: separated the evaluated next-origin availability convention from unmeasured meter/service delays.
- `paper/sections/setup.tex`: defined empirical p95, call count, replicated scaling state, empty-set miss/zero-length semantics and Winkler denominators.
- `paper/sections/results.tex`: documented directly measured combined calls, not sums, and denied production/end-to-end/edge interpretation.
- `paper/sections/conclusion.tex`: requested genuine all-48 calibration and retained deployment exclusions.
- `README.md`: labeled the version.

### Code decision

- No implementation change. Code/equation review found the direct integer rank, endpoint convention, delayed update order and signed empty-set handling consistent. Therefore no outputs were regenerated as a scientific bug fix.

### Regenerated outputs

- LaTeX PDF; IEEE/regular DOCX files and associated Word/source validation outputs.

## Phase 3 — matched LightGBM validation-weighting sensitivity

### Preregistered experiment

- Added `docs/review_final/03_matched_baseline_protocol.md` before fitting.
- Added `docs/review_final/phase3_matched_baseline_v1/config.json`, fit/scoring scripts and pre/post tests.
- Reused the exact original 120,000 aggregate LightGBM training draws and fixed hyperparameters.
- Replaced only validation sampling with all 673 complete-window origins crossed with 48 horizons (32,304 equally weighted rows).
- Retained the already examined January--June 2013 test period and labeled the result retrospective/not independent.

### Experiment outputs

- Saved seven boosters, metadata, train/validation keys and masks, validation/calibration/test predictions, selected-eight slice, `results.json` and `per_horizon.csv` under `docs/review_final/phase3_matched_baseline_v1/`.
- Result: the same tree counts `[132,108,72,168,80,90,138]` and bitwise-identical test forecasts. Existing 6.82% selected-eight result and 0.50% all-48 reversal did not change.

### Authored manuscript edits

- `paper/sections/setup.tex`: documented the exact retrospective matched-validation control and remaining representation/budget differences.
- `paper/sections/results.tex`: reported identical stopping points/predictions and denied equal tuning, optimality or independent confirmation.
- `README.md`: labeled the version.

### Reports

- `docs/review_final/03_matched_baseline_results.md`

## Phase 4 — mathematics, references and rendered-deliverable QA

### Authored manuscript edit

- `paper/sections/acknowledgment.tex`: made the AI-assistance disclosure specific to OpenAI Codex, the Abstract and Sections I, II, V--VII, the assistance type and author verification.
- `README.md`: labeled the final visual/reference version.

### QA tooling and records

- Added `validation/make_pdf_diff.py` and saved page images, a PDF diff and manifest under `validation/pdf_diff/`.
- Added `validation/validate_phase4.py` and `phase4_validation_results.json`.
- Refreshed `validation/render_checks.json` and page/contact images.
- Added `docs/review_final/04_visual_math_reference_QA.md`.

### Regenerated deliverables

- `phase4_visual_math_reference_revision_2026-10-23/calibrated_net_load_revised.pdf`
- `phase4_visual_math_reference_revision_2026-10-23/word/calibrated_net_load_IEEE.docx`
- `phase4_visual_math_reference_revision_2026-10-23/word/calibrated_net_load_regular.docx`
- Fresh Microsoft Word PDFs under `word/proofs/`.
- Word source, AST, native-table/math reconstruction and structural manifests.

### Visual result

- Inspected 12 LaTeX, 13 IEEE Word and 23 regular Word pages.
- No missing symbol, clipping, overflow, unresolved citation, malformed table heading or blank page.
- Fresh Phase 3 vs Phase 4 IEEE diff changed only pages 12--13 because of the disclosure and reference reflow; pages 1--11 are pixel-identical.

## Phase 5 — independent referee pass

### Files added

- `docs/review_final/05_final_referee_report.md`
- `docs/review_final/final_claim_evidence_matrix.md`
- `docs/review_final/change_log.md`
- `docs/review_final/validate_final_phase5.py`
- `docs/review_final/phase5_validation_results.json`

### Manuscript impact

- None. Phase 5 found no new inconsistency requiring a manuscript edit or scientific rerun.

## Validation record

The following final/inherited validations were actually run:

| Validation | Final result |
|---|---|
| Phase 3 protocol checks | PASS 5/5 |
| Phase 3 result checks | PASS 6/6 |
| `validation/test_correctness.py` | PASS 34/34; expected warning in deliberate all-missing QS7 test |
| `validation/validate_fixes.py` | PASS: preprocessing 11, set metrics 20, replay 8 |
| `validation/validate_phase3.py` | PASS all inherited claim/evidence checks |
| `validation/render_audit.py` | PASS for LaTeX, IEEE Word and regular Word PDFs |
| `validation/make_pdf_diff.py` | PASS; 13 pages, changes on 12--13 only |
| `validation/validate_phase4.py` | PASS 49/49 |
| Tectonic cached final build | PASS; 12 pages, 30 resolved citations |
| Word reconstruction | PASS; identical content, 3 tables, 181 numeric cells, 8 figures, 30 references, 158 native math objects |
| Microsoft Word IEEE export | PASS; fresh 13-page proof |
| Microsoft Word regular export | PASS; fresh 23-page proof |
| Manual page inspection | PASS; all 48 final pages inspected |
| `docs/review_final/validate_final_phase5.py` | PASS; final report and deliverable assertions |

An inherited-test invocation once used a workspace-relative Python path while already inside the Phase 4 directory and failed before executing tests; the same commands were rerun with the absolute environment path and passed. A baseline comparison Word export initially encountered a duplicate document name; the protected file was not changed, and a uniquely named temporary copy produced the final diff.

## Final deliverables and hashes

| Deliverable | SHA-256 |
|---|---|
| LaTeX PDF | `d06df3c3b0b0d9b177b1c9e523d15b6adf18f38691adc2d3dd3e72c5b1f4b40d` |
| IEEE DOCX | `ac2f7c3f1646bd2ac0ac5136d7ab1f2ead1133610b5c6ffd4c503c3795ae3a86` |
| IEEE Word PDF | `9668a86d405bab1d70d3104d2a265f6010325c75c339761c9d82ae70f6e1c426` |
| Regular DOCX | `26e5658f4703701c4a2d2b4ea74fb859cb456a6aa09158074101b38e7cc5798c` |
| Regular Word PDF | `ab5c8e44c65dc39fa984ad05ca44255676900cfe007eb53af43071929b56a7b7` |
| Before/after IEEE PDF diff | `f72cdf12bc1bfbfb76178ead6148db920b6670c9082f3fef26b7d1a249b4d77e` |

## Historical findings deliberately unchanged

- Selected-eight aggregate advantage and all-48 reversal.
- Household near-tie and lower LightGBM household MAE.
- Mixed all-48 MAE/RMSE rankings.
- High-PV and large-ramp undercoverage.
- Empty conformal sets and restricted Winkler denominators.
- Rolling-CQR sharpness versus RW-ACI household tail/width trade-off.
- Past-clear-sky removal improvement and future-clear-sky removal degradation.
- One household neural seed and unavailable household all-48 results.
- Data substitutions, missing calibration, absent-CL, zero-PV, DST/location and information-availability limitations.
- Missing original historical weights/logs and unknown horizon/gamma selection chronology.
- Replicated 48-row runtime state is scaling evidence, not calibration-quality or deployment validation.

## Work that requires new evidence

- Untouched later year or external dataset with prospectively locked selection.
- Additional household seeds and saved household all-48 forecasts.
- Prospectively matched tuning/model-selection budgets.
- Genuine horizon-specific all-48 calibration study.
- End-to-end service and field timing including data transport, persistence and failure handling.
