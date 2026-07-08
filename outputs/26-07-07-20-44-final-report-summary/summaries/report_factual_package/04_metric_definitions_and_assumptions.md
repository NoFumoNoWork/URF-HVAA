# Metric Definitions And Assumptions

All metric formulas are implemented in `scripts/evaluate_interval_methods.py::evaluate_one_video` and `aggregate_rows`.

- `GT_coverage`: intersection duration between prediction and merged GT divided by total merged GT duration.
- `predicted_GT_fraction` / purity: intersection duration divided by predicted duration.
- `supportable_gt_coverage`: coverage over GT rows whose support group is `supportable`.
- `unsupportable_gt_coverage`: coverage over GT rows whose support group is `unsupportable`.
- `predicted_duration_ratio`: predicted duration divided by video duration.
- `balanced_score`: `0.4*GT_coverage + 0.3*predicted_GT_fraction + 0.2*supportable_gt_coverage - 0.1*predicted_duration_ratio`.
- `stricter_balanced_score`: `0.30*GT_coverage + 0.25*predicted_GT_fraction + 0.25*supportable_gt_coverage - 0.10*predicted_duration_ratio - 0.10*unsupportable_gt_coverage`.
- `num_predicted_intervals`: aggregate predicted interval count after per-video merging.
- `mean_interval_length`: same reported value as `mean_predicted_interval_length`.
- `median_interval_length`: same reported value as `median_predicted_interval_length`.

## Supportable / Unsupportable Assumptions

- Supportability labels come from `outputs/26-07-07-15-14-gt-score-alignment-analysis/outputs/gt_support_classification.csv`, loaded by `load_gt_rows`.
- The evaluator does not classify supportability live from `score_threshold=0.6`; it reads the precomputed support group.
- `score_threshold=0.6` is used for fixed-window score-positive methods in `evaluate_interval_methods.py::add_window_methods`, for default spectral trend thresholding, and in earlier score-alignment artifacts.
- Score-unsupported GT is not equivalent to annotation error.
- Unsupported coverage is a diagnostic constraint, not an independent objective to minimize.
- If unsupported coverage rises with duration while purity drops, the result suggests over-extension.
- If unsupported coverage is moderate while GT/supportable coverage increases, it may reflect human event context or score-weak anomaly evidence.
