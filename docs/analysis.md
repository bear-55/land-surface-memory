# Analysis and numerical reproduction

The analysis uses all ten paired seeds: 11, 29, 47, 65, 83, 103, 119, 137, 155, and 173. All three models and all 107 held-out basins are retained. Run commands from the repository root after preparing the data, training the 30 runs, and evaluating both periods as described in the main README.

```bash
python scripts/analysis_core.py
python scripts/diagnose_gates.py --device cuda
python scripts/analysis_gates.py
python scripts/analysis_temporal.py
python scripts/analysis_plots.py
python scripts/analysis_verify.py
```

`diagnose_gates.py` also supports `--device cpu`; the recurrent diagnostic is considerably slower on a CPU. It reuses the validation-selected M1 checkpoints and checks its predictions against the saved evaluation. It never trains or selects a checkpoint. The remaining commands operate on predictions or derived diagnostics.

## Inputs and outputs

| Step | Inputs | Main outputs under `runs/analysis` |
| --- | --- | --- |
| `analysis_core.py` | `runs/evaluation_test/{model}/seed_{seed}/predictions_test.npz`; frozen split; physical attributes | Overall NSE; paired NSE; all regime MAE/MSE; error budgets; both trade-off counts; all 16 attribute groups and correlations |
| `diagnose_gates.py` | `runs/formal_pub_527/M1/seed_{seed}/best.pt`; prepared cache; M1 test predictions | Daily target-day gate summaries in `runs/gates` |
| `analysis_gates.py` | Daily gate summaries; core analysis; prepared meteorology; HUC2 assignments | Gate contrasts and error correlations; 24 rainfall association tests; adjusted quintile curves |
| `analysis_temporal.py` | Test and historical predictions from the same checkpoints; core groups | Both threshold schemes; basin changes; group summaries; seed and joint-pattern persistence |
| `analysis_plots.py` | These numerical tables | Five SVG and 600 dpi PNG figure pairs in `runs/figures` |

Each prediction NPZ contains one aligned vector per key: `basin_ids`, `dates`, `q_observed_mm_day`, and `q_predicted_mm_day`. The analysis checks exact identity of basin, date, and observed discharge across the 30 runs. Missing observations are not replaced, and poor model performance never changes the evaluation mask. The command-line `--help` documents path overrides.

## Scientific definitions

Flow regimes use each basin's observed 20th and 95th percentiles, calculated with linear interpolation. High flow is selected first (`Q >= Q95`); low flow is the remaining `Q <= Q20`; medium flow contains the other days. Differences are the first model minus the second: negative ΔMAE is a gain. Predictions are not ensembled. Differences are paired within a basin and seed, averaged across seeds, and then summarized across basins. Day-weighted contributions add back to each basin's full-period error change; medians are not additive.

Physical attributes are grouped using terciles of the 370 training basins. Ties stay together and missing physical values are excluded feature by feature. The group table separately reports medium-flow gains with high-flow costs and the reverse combination. A group median after seed averaging, a regime-specific direction count, and a simultaneous two-regime count are different estimands.

Net gate modulation is `sigmoid(u_f + delta_f) - sigmoid(u_f)` evaluated at the same actual M1 hidden state. It is a local gate comparison, not a second rollout with the adapter disabled. Unit indices are not aligned across seeds. Daily unit means are averaged within each basin and regime before seed and group aggregation.

P3 sums precipitation from `t-2` to `t`; AP30 sums `t-32` to `t-3`. To match the published calculation, precipitation and temperature are recovered from the normalized float32 cache using its training-only scaler. For each basin, seed, and flow selection, partial Spearman correlations control for the other precipitation window, observed discharge, seven-day mean temperature, and annual sine/cosine terms. Average ranks preserve ties. Correlations use at least 30 valid days. Basin and seed means receive equal weights. Two-sided exact sign flips operate on HUC2-region sums, and Benjamini–Hochberg adjustment includes all 24 cells: four basin groups, three flow selections, and two predictors. The 2,000 HUC2/seed bootstrap replicates use seed 20260928.

The rainfall figure uses adjusted precipitation-rank residuals, standardized by the population standard deviation. Its quintile probabilities are generated with `np.linspace(0, 1, 6)`; internal-boundary ties enter the higher bin. Predictor bins are shared across seeds. Standardized modulation residuals are averaged within bins and then equally across seeds and basins. The 61 nested bands are empirical between-basin quantile ranges, not confidence intervals. Panel correlations use daily residuals rather than the five plotted points.

The cross-period analysis compares 1 October 2003–30 September 2008 with 1 October 1989–30 September 1999 on the same held-out basins. Both periods use the same selected model weights. `period_relative` calculates flow thresholds within each period; `original_absolute` applies the later period's thresholds to the earlier period. This is retrospective persistence of the fitted models, not evidence from retraining or independent future model selection.

## Reference results and checks

`reference_results` contains compact numerical snapshots, not raw CAMELS records, model weights, or daily predictions. It permits a quick inspection of reported effects and a plotting smoke check:

```bash
python scripts/analysis_plots.py --analysis reference_results --output runs/reference_figures
```

This command replots archived summaries; it does not reproduce model training. The public plotting script reproduces the numerical content of Figures 3–7 in a portable layout. It does not reproduce the manuscript's manual vector-graphics styling, its map, or its framework illustration.

`analysis_verify.py` checks regenerated tables against the archived summaries at explicit tolerances. Retraining on different software or GPU arithmetic can produce differences, especially in direction counts near zero. Inspect these differences and report the actual environment and seed-level outcomes. Never replace newly generated outputs with reference values to obtain a passing check.

The gate extractor was also checked on eight 365-day windows using the existing seed-11 checkpoint on a CPU. Its prediction matched the public model forward pass exactly for that batch, and its six gate summaries differed from the archived diagnostic by at most `5.96e-8`. `gate_batch_check.json` records this bounded check; the complete ten-checkpoint extraction was not rerun during packaging.

All CSV values retain working precision. `mid` is the machine-readable code for medium flow. The reference file manifest gives SHA-256 checksums for the distributed snapshots.
