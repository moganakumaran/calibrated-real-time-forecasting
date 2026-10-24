# Independent reproduction report

This is a new run of the supplied protocol on the recorded Mac environment, not recovered original records.

All 26 protocol stages completed. Final audit: 101/101 checks passed.
Scientific numeric fields exactly matching the saved summaries: 511/1064; different fields: 553. Timing fields are separated because hardware/environment differ.

## Principal comparisons

| Level | Quantity | Original summary | New run |
| --- | --- | ---: | ---: |
| aggregate | PV-TCQN QS7 | 10.76207161 | 10.76114877 |
| aggregate | LightGBM QS7 | 11.54894257 | 11.54894257 |
| aggregate | QS7 reduction vs LightGBM (%) | 6.813359335 | 6.821350013 |
| aggregate | Shortest-horizon reduction (%) | 24.32913492 | 24.34321105 |
| aggregate | PV RW-ACI pooled coverage | 0.9000983682 | 0.9001273001 |
| aggregate | PV RW-ACI maximum pooled monthly gap | 0.01476934524 | 0.01476934524 |
| household | PV-TCQN QS7 | 0.1201110259 | 0.1201183796 |
| household | LightGBM QS7 | 0.1200141162 | 0.1200141162 |
| household | QS7 reduction vs LightGBM (%) | -0.08074858602 | -0.08687596776 |
| household | Shortest-horizon reduction (%) | 8.629324571 | 8.620415402 |
| household | PV RW-ACI pooled coverage | 0.9000134533 | 0.9000119585 |
| household | PV RW-ACI maximum pooled monthly gap | 0.001211226852 | 0.001201678241 |
| household | PV RW-ACI home-month share below 85% | 0 | 0 |

## Interpretation and remaining work

The comparison includes all saved scientific numeric fields, without tuning or selecting reruns to obtain agreement. Exact agreement of summary values alone does not establish identical original model weights. The new prediction files, checkpoints, logs, configuration, versions and checksums provide auditable evidence for this run.

Full deltas are in `provenance/summary_comparison.csv`; checkpoint replay, array checks and integrity results are in `provenance/reproduction_audit.json`. Inspect any discrepancies and failures before deciding how to cite this reproduction. The original paper/results have not been changed.

The author confirmed sole authorship as Mogana Kumaran Sivaraman, no funding and no AI assistance before this revision. The actual Codex audit/editing/reproducibility assistance remains disclosed in the separate author-confirmed manuscript copy. No automatic submission-readiness certification, manuscript replacement, push or publication was performed.
