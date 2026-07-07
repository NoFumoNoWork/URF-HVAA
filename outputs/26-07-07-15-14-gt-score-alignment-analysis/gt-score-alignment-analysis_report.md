# GT-Score Alignment Analysis Report

## Method

This analysis diagnoses consistency between human event-level GT and score-level anomaly evidence. It does not assume either GT or score is absolutely correct. The goal is to separate post-processing error, score-supported misses, score-unsupported GT intervals, and score-positive GT-negative regions.

## Inputs

- gt_stats_csv: `outputs\26-07-07-14-43-gt-score-window-curves\outputs\gt_interval_score_stats.csv`
- video_inventory_csv: `outputs\26-07-07-14-43-gt-score-window-curves\outputs\video_score_curve_inventory.csv`
- GT intervals: 1394
- videos: 640
- score JSON paths: 640
- warnings/skipped score files: 0

## Thresholds And Label Definitions

The following thresholds were used in this run:

- `score_positive_threshold`: 0.6. A score point or fixed window is treated as score-positive at this threshold.
- `strong_mean_threshold`: 0.6. A GT interval is strongly supported if its mean score reaches this value, provided it has enough score samples.
- `strong_max_threshold`: 0.8. A GT interval is strongly supported if its max score reaches this value, provided it has enough score samples.
- `weak_max_threshold`: 0.5. A GT interval is weakly supported if its max score reaches this value but it is not strongly supported.
- `unsupported_max_threshold`: 0.4. A sufficiently sampled GT interval with max score below this value is score-unsupported.
- `min_sparse_points`: 2. Intervals with fewer than this many score points are `barely_sampled`.
- `min_well_sampled_points`: 5. Intervals with at least `min_sparse_points` but fewer than this many score points are `sparsely_sampled`.
- `window_sizes`: 30, 100, 300 frames.
- `threshold_sweep`: 0.4,0.5,0.6,0.7,0.8. Used only for sensitivity analysis.

`support_type` is assigned in this priority order:

- `unobserved_or_missing_score`: `score_point_count == 0`, missing mean/max score, or missing score file.
- `barely_sampled`: `0 < score_point_count < min_sparse_points`.
- `sparsely_sampled`: `min_sparse_points <= score_point_count < min_well_sampled_points`. This rule is evaluated before strong/weak support, so short intervals with high scores can still be marked sparse.
- `strongly_score_supported`: enough samples and (`max_score >= strong_max_threshold` or `mean_score >= strong_mean_threshold`).
- `weakly_score_supported`: enough samples and `max_score >= weak_max_threshold`, but not strongly supported.
- `score_unsupported`: enough samples and `max_score < unsupported_max_threshold`.
- `ambiguous_mid_score`: all remaining middle-score cases.

`response_shape` is assigned as:

- `sustained_response`: `mean_score >= strong_mean_threshold` and `max_score >= strong_max_threshold`.
- `localized_response`: `max_score >= strong_max_threshold` and `mean_score < strong_mean_threshold`.
- `weak_or_no_response`: `max_score < weak_max_threshold`.
- `sparse_or_unknown`: too few score points or missing score values.
- `moderate_response`: all remaining response-shape cases.

`recoverable_by_postprocessing` is `True` for `strongly_score_supported` and `weakly_score_supported`, `False` for `score_unsupported`, `unobserved_or_missing_score`, and `barely_sampled`, and `uncertain` for `sparsely_sampled` and `ambiguous_mid_score`.

Window-level score-positive logic uses `max(window_scores) >= score_positive_threshold` or `mean(window_scores) >= strong_mean_threshold`.

Post-processing upper bound treats an interval as recoverable if it is strongly/weakly score-supported or has a `localized_response`/`sustained_response`; sparse intervals remain uncertain unless this response-shape evidence is present.

## GT Support Summary

- `gt_support_classification.csv` contains original GT rows plus support classification fields.
- `fig_gt_support_by_dataset.png` visualizes support counts by dataset.
- strongly_score_supported: 809 (58.03%)
- weakly_score_supported: 145 (10.40%)
- ambiguous_mid_score: 26 (1.87%)
- score_unsupported: 55 (3.95%)
- sparsely_sampled: 326 (23.39%)
- barely_sampled: 32 (2.30%)
- unobserved_or_missing_score: 1 (0.07%)

## Label-wise Summary

Labels with high unsupported ratios among the current top rows:
- UCF-Crime / Abuse: unsupported=50.00%, n=2
- UCF-Crime / Shoplifting: unsupported=48.00%, n=25
- UCF-Crime / Robbery: unsupported=20.00%, n=5
- UCF-Crime / Vandalism: unsupported=12.50%, n=8
- UCF-Crime / Explosion: unsupported=9.09%, n=22

## Duration-Score Relationship

Duration bins are summarized in `duration_score_summary.csv`; scatter plots are `fig_duration_vs_max_score.png` and `fig_duration_vs_mean_score.png`.

## Window-level Consistency

Highest GT+Score+ ratio: XD-Violence window=300 with 92.82%.
Full results are in `window_confusion_summary.csv` and `label_window_confusion_summary.csv`.

## Outside-GT High Score Analysis

Detected outside-GT high-score intervals: 5578. Per-video ratios are in `video_outside_gt_score_summary.csv`; intervals are in `outside_gt_high_score_intervals.csv`.

## Post-processing Upper Bound

- UCF-Crime: recoverable=73.72%, uncertain=12.82%, unrecoverable=13.46%
- XD-Violence: recoverable=80.45%, uncertain=14.14%, unrecoverable=5.41%

Peak-aware refinement can only recover score-supported local anomalies; it cannot recover GT intervals where the upstream scorer provides no abnormal signal.

## Limitations

- Score stride may undersample short GT intervals.
- Multi-label videos do not uniquely attribute each interval to one label.
- Score thresholds are diagnostic and need sensitivity analysis.
- GT and VLM score may differ in definition and temporal granularity.
- This analysis does not prove whether human annotation or VLM score is wrong.