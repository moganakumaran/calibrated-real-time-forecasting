# Review of the completed reproduction — 22 October 2026

All 26 protocol stages completed without retuning or selecting a replacement run. The final audit passed 101/101 checks. This is a new local CPU execution of the supplied protocol, not recovered original experiment records. The original 56 protected files and the earlier 66-file revision snapshot remain unchanged. The new author-confirmed paper also retains the original primary numbers.

## Findings and differences

The aggregate PV-TCQN QS7 is 10.76114877 versus the original 10.76207161. Its reduction relative to LightGBM-Q is 6.82135% versus 6.81336%, preserving the rounded 6.8% finding. Household PV-TCQN QS7 is 0.12011838 versus 0.12011103; both are slightly higher than LightGBM-Q's 0.12001412. Neither run establishes household QS7 superiority or statistical equivalence to LightGBM-Q.

The baselines' overall QS7 values match exactly. Neural results do not all match: aggregate LSTM QS7 changes by about 1.53%, aggregate MLP by −0.102%, and aggregate PV-TCQN by −0.00857%. Household neural QS7 changes range from 0.00612% to 0.135%. These differences cannot be attributed to a particular original software or hardware component because the original environment and checkpoints are unavailable.

Nominal Diebold–Mariano p-values also change. The household comparison with LightGBM-Q changes from 0.78780 to 0.77235; the comparison with MLP-Q changes from 0.02746 to 0.00640. These retain their respective decisions at an unadjusted 0.05 threshold, but are not interchangeable estimates. The paper's normal-reference, seed-0, 48-lag HAC and nonstationarity limitations still apply. No stronger inferential claim is added.

PV-TCQN RW-ACI pooled coverage changes from 0.90009837 to 0.90012730 at the aggregate level and from 0.90001345 to 0.90001196 at the household level. The household home-month share below 85% remains zero. This supports the reported empirical calibration pattern, without establishing conditional coverage, trajectory coverage or operational reliability. Ablation mean-QS7 ordering remains unchanged.

Across the entire saved metric schema, 511 of 1064 scientific numeric fields match exactly and 553 differ. These fields include nested summaries of related observations and are not 1064 independent experiments. The full comparison is retained in `provenance/summary_comparison.csv`. Timing measurements are separate because the environment differs; they do not validate the historical CPU timings or an end-to-end deployment claim.

## Replay discrepancy resolved

The first audit passed 100/101 checks. One small-batch checkpoint replay, using one CPU thread, differed from its saved forecast by at most 0.000041008 kWh. The original inference configuration for this new run used four threads and batches of 4096. Replaying the complete affected test array under those settings reproduced all 483,896 scalar forecasts bit-for-bit, with maximum absolute difference zero. No model was retrained, prediction replaced or tolerance loosened.

The final audit retains the small-batch discrepancy and the exact original-settings replay. Other checkpoint checks sample fixed observations; the 101 checks do not claim bitwise full-array replay of every checkpoint. Initial audit evidence is preserved under `provenance/initial_audit/`, and the full affected-array diagnostic is `provenance/checkpoint_replay_diagnostic.json`.

## Evidence boundary

The new run now has 50 forecast arrays, 21 neural checkpoints, 14 LightGBM model files, execution logs, configuration, environment versions and hashes. These close the lack of an auditable executed reproduction. They do not establish the exact provenance of historical manuscript values. The current paper retains that limitation. No original result or manuscript score was replaced, and no files were pushed or published.
