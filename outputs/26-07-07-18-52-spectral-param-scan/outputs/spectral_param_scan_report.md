# Spectral Parameter Scan Report

## Executive summary

- Runs evaluated: 172; Pareto frontier runs: 36.
- Default Full Spectral-Fusion-Refined: GT_coverage=0.729, predicted_GT_fraction=0.506, supportable_gt_coverage=0.742, unsupportable_gt_coverage=0.247, predicted_duration_ratio=0.562.
- Best stricter-balanced run: `trend_threshold_0.5` with stricter_balanced_score=0.468.
- Recommended operating point: `trend_threshold_0.5`. It is chosen from the Pareto frontier when possible and penalizes over-wide / unsupportable coverage.
- The balanced scores are auxiliary ranking tools, not absolute accuracy measures.

## Scan design

This stage deliberately uses ablation, one-factor sensitivity, and a small combo grid instead of Bayesian optimization. The goal is to explain which parameters move the coverage-purity trade-off, not to find a black-box optimum before the objective and validation split are settled.

- Ablation toggles SG, airPLS residual, and trend evidence around the current default.
- One-factor scans change one parameter at a time while holding the default fixed.
- Combo scan covers the main operating controls: fusion threshold, trend window, trend weight, and length penalty.

## Ablation results

- `Peak-Aware-Refined`: GT=0.700, purity=0.519, supportable=0.705, unsupportable=0.441, duration=0.541, strict=0.418.
- `Spectral-Fusion without SG`: GT=0.705, purity=0.526, supportable=0.720, unsupportable=0.187, duration=0.525, strict=0.452.
- `Spectral-Fusion without airPLS`: GT=0.650, purity=0.591, supportable=0.668, unsupportable=0.071, duration=0.433, strict=0.459.
- `Spectral-Fusion without trend`: GT=0.682, purity=0.513, supportable=0.695, unsupportable=0.218, duration=0.520, strict=0.433.
- `Spectral-Fusion SG only`: GT=0.276, purity=0.595, supportable=0.282, unsupportable=0.039, duration=0.185, strict=0.280.
- `Spectral-Fusion airPLS only`: GT=0.504, purity=0.522, supportable=0.516, unsupportable=0.081, duration=0.381, strict=0.365.
- `Spectral-Fusion trend only`: GT=0.064, purity=0.591, supportable=0.064, unsupportable=0.019, duration=0.043, strict=0.177.
- `Spectral-Fusion SG + airPLS`: GT=0.682, purity=0.513, supportable=0.695, unsupportable=0.218, duration=0.520, strict=0.433.
- `Spectral-Fusion SG + trend`: GT=0.650, purity=0.591, supportable=0.668, unsupportable=0.071, duration=0.433, strict=0.459.
- `Spectral-Fusion airPLS + trend`: GT=0.705, purity=0.526, supportable=0.720, unsupportable=0.187, duration=0.525, strict=0.452.
- `Full Spectral-Fusion-Refined`: GT=0.729, purity=0.506, supportable=0.742, unsupportable=0.247, duration=0.562, strict=0.450.

## One-factor sensitivity

- `fusion_threshold`: GT range 0.055, purity range 0.051, stricter-score range 0.014; best value by stricter score = `0.5`.
- `trend_window`: GT range 0.033, purity range 0.002, stricter-score range 0.015; best value by stricter score = `30`.
- `trend_threshold`: GT range 0.062, purity range 0.002, stricter-score range 0.029; best value by stricter score = `0.5`.
- `airpls_lambda`: GT range 0.011, purity range 0.011, stricter-score range 0.006; best value by stricter score = `10000`.
- `peak_mad_k`: GT range 0.014, purity range 0.002, stricter-score range 0.004; best value by stricter score = `1.5`.
- `sg_window_length`: GT range 0.004, purity range 0.006, stricter-score range 0.002; best value by stricter score = `31`.
- `sg_polyorder`: GT range 0.000, purity range 0.000, stricter-score range 0.000; best value by stricter score = `2`.
- `length_penalty_weight`: GT range 0.009, purity range 0.009, stricter-score range 0.001; best value by stricter score = `0.3`.
- `low_residual_penalty_weight`: GT range 0.005, purity range 0.009, stricter-score range 0.005; best value by stricter score = `0.2`.
- `residual_weight`: GT range 0.035, purity range 0.035, stricter-score range 0.016; best value by stricter score = `0.1`.
- `trend_weight`: GT range 0.011, purity range 0.007, stricter-score range 0.001; best value by stricter score = `0.35`.
- `sg_weight`: GT range 0.020, purity range 0.021, stricter-score range 0.006; best value by stricter score = `0.0`.

## Combo scan and Pareto frontier

- Pareto frontier size: 36.
- Pareto objectives maximize GT_coverage, predicted_GT_fraction, and supportable_gt_coverage, while minimizing predicted_duration_ratio and unsupportable_gt_coverage.
- See `summaries/pareto_frontier_runs.csv` and the coverage-purity figure for runs that improve one axis without being dominated on the others.

## Default vs best comparison

| selection | run | GT_coverage | predicted_GT_fraction | supportable_gt_coverage | unsupportable_gt_coverage | predicted_duration_ratio | stricter_balanced_score |
|---|---|---:|---:|---:|---:|---:|---:|
| Peak-Aware-Refined | `Peak-Aware-Refined` | 0.700 | 0.519 | 0.705 | 0.441 | 0.541 | 0.418 |
| default Spectral-Fusion-Refined | `Full Spectral-Fusion-Refined` | 0.729 | 0.506 | 0.742 | 0.247 | 0.562 | 0.450 |
| best_recall_oriented | `trend_threshold_0.5` | 0.767 | 0.507 | 0.782 | 0.258 | 0.590 | 0.468 |
| best_purity_oriented | `Spectral-Fusion SG only` | 0.276 | 0.595 | 0.282 | 0.039 | 0.185 | 0.280 |
| best_supportable_oriented | `trend_threshold_0.5` | 0.767 | 0.507 | 0.782 | 0.258 | 0.590 | 0.468 |
| best_duration_controlled | `combo_fusion_threshold0.45_trend_window50_trend_weight0.35_length_penalty_weight0.3` | 0.721 | 0.524 | 0.737 | 0.123 | 0.538 | 0.465 |
| best_balanced | `trend_threshold_0.5` | 0.767 | 0.507 | 0.782 | 0.258 | 0.590 | 0.468 |
| recommended_operating_point | `trend_threshold_0.5` | 0.767 | 0.507 | 0.782 | 0.258 | 0.590 | 0.468 |

## Recommendation

- Use `trend_threshold_0.5` as the recommended operating point for the current report if the validation protocol accepts parameter selection on this dataset.
- Keep recall-oriented, purity-oriented, and supportable-oriented picks as supplementary operating points rather than replacing the main result silently.
- Do not run Bayesian optimization yet. The objective function is still being interpreted, many controls are discrete module switches, and no train/validation/test split has been enforced.

## Limitations

- Without a train/validation/test split, parameter scanning may overfit the current dataset.
- `balanced_score` and `stricter_balanced_score` use human-set weights and are not absolute accuracy.
- Human GT and VDA scores are not absolute truth and can disagree temporally.
- Over-tuning post-processing can hide limitations in VDA scoring.
- Score-unsupported GT cannot be restored reliably by interval post-processing alone.

## Next steps

- Split a validation set before any further optimization.
- Consider Bayesian optimization only after the key parameters and objective are fixed.
- Prioritize error taxonomy and case studies for missed score-unsupported GT.