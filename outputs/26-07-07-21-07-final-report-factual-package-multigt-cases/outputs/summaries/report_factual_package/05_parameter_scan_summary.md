# Parameter Scan Summary

Source files:

- `outputs/26-07-07-18-52-spectral-param-scan/outputs/summaries/param_scan_all_runs.csv`
- `outputs/26-07-07-18-52-spectral-param-scan/outputs/summaries/one_factor_sensitivity_summary.csv`
- `outputs/26-07-07-18-52-spectral-param-scan/outputs/summaries/pareto_frontier_runs.csv`

The scan evaluated 172 rows in the all-run table. Parameter ranges are defined in `scripts/run_spectral_param_scan.py::ONE_FACTOR_SPACE` and `COMBO_SPACE`.

| parameter | values_scanned | GT_coverage_range | predicted_GT_fraction_range | supportable_gt_coverage_range | unsupportable_gt_coverage_range | predicted_duration_ratio_range | stricter_balanced_score_range | recommended_value_from_one_factor |
|---|---|---|---|---|---|---|---|---|
| trend_threshold | 0.5,0.55,0.6,0.65,0.7 | 0.062 | 0.002 | 0.065 | 0.015 | 0.046 | 0.029 | 0.5 |
| residual_weight | 0.1,0.2,0.25,0.35,0.45 | 0.035 | 0.035 | 0.030 | 0.191 | 0.061 | 0.016 | 0.1 |
| trend_window | 30,50,100,150,300 | 0.033 | 0.002 | 0.033 | 0.000 | 0.026 | 0.015 | 30 |
| fusion_threshold | 0.25,0.3,0.35,0.4,0.45,0.5 | 0.055 | 0.051 | 0.050 | 0.209 | 0.090 | 0.014 | 0.5 |
| airpls_lambda | 100,1000,10000,100000 | 0.011 | 0.011 | 0.010 | 0.049 | 0.019 | 0.006 | 10000 |
| sg_weight | 0.0,0.1,0.2,0.3 | 0.020 | 0.021 | 0.017 | 0.070 | 0.036 | 0.006 | 0.0 |
| low_residual_penalty_weight | 0.05,0.1,0.15,0.2 | 0.005 | 0.009 | 0.003 | 0.034 | 0.013 | 0.005 | 0.2 |
| peak_mad_k | 1.5,2.0,2.5,3.0 | 0.014 | 0.002 | 0.014 | 0.042 | 0.013 | 0.004 | 1.5 |
| sg_window_length | 9,17,31,51 | 0.004 | 0.006 | 0.004 | 0.016 | 0.010 | 0.002 | 31 |
| length_penalty_weight | 0.05,0.1,0.15,0.2,0.3 | 0.009 | 0.009 | 0.008 | 0.018 | 0.017 | 0.001 | 0.3 |
| trend_weight | 0.1,0.15,0.25,0.35,0.45 | 0.011 | 0.007 | 0.010 | 0.020 | 0.016 | 0.001 | 0.35 |
| sg_polyorder | 2,3 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 2 |

## Interpretation

- Most influential parameters by stricter-score range: trend_threshold, residual_weight, trend_window, fusion_threshold, airpls_lambda.
- Lower-sensitivity parameters in this one-factor scan: sg_window_length, length_penalty_weight, trend_weight, sg_polyorder.
- Bayesian optimization is not continued here because `AGENT.md` asks for factual packaging only, and current runs already expose the main recall/strict trade-off.
- Pareto frontier rows are retained in `pareto_frontier_runs.csv`; final Recall and Strict points are manually selected operating points from pipeline-level comparisons, not new optimization claims.
- Default parameter baseline: `{'fusion_threshold': 0.35, 'trend_window': 100, 'trend_threshold': 0.6, 'airpls_lambda': 1000, 'peak_mad_k': 3.0, 'sg_window_length': 17, 'sg_polyorder': 2, 'length_penalty_weight': 0.15, 'low_residual_penalty_weight': 0.1, 'residual_weight': 0.25, 'trend_weight': 0.15, 'sg_weight': 0.2, 'raw_weight': 0.25, 'peak_count_weight': 0.1, 'residual_mad_k': 3.0, 'peak_stop_ratio': 0.25, 'merge_gap_frames': 48}`.
- Combo space: `{'fusion_threshold': [0.3, 0.35, 0.4, 0.45], 'trend_window': [50, 100, 150], 'trend_weight': [0.15, 0.25, 0.35], 'length_penalty_weight': [0.1, 0.2, 0.3]}`.
