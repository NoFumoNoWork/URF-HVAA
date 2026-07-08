# Interval Evaluation Summary

- Unit: frame index / frame duration.
- Main recall denominator: `GT_eval = s-GT union uc-GT`.
- `us-GT` is excluded from main TP/FN and reported only as diagnostic coverage.
- FP is the part of PI outside all GT (`s-GT union uc-GT union us-GT`).
- Reproduction command: `python scripts\run_interval_evaluation_summary.py`.

| Method | PI Duration | PI Ratio | s-GT Recall | uc-GT Recall | Eval Recall | GT Precision | Eval Precision | us-GT Coverage | FP Duration | FP Ratio in PI |
| ------ | ----------: | -------: | ----------: | -----------: | ----------: | -----------: | -------------: | -------------: | ----------: | -------------: |
| Original | 897336 | 0.562 | 0.742 | 0.553 | 0.735 | 0.506 | 0.503 | 0.247 | 443659 | 0.494 |
| Peak baseline | 839328 | 0.541 | 0.705 | 0.643 | 0.703 | 0.519 | 0.514 | 0.441 | 403785 | 0.481 |
| Strict | 837816 | 0.527 | 0.725 | 0.491 | 0.717 | 0.527 | 0.525 | 0.187 | 396126 | 0.473 |
| Recall-first | 901180 | 0.566 | 0.768 | 0.537 | 0.760 | 0.520 | 0.518 | 0.225 | 432484 | 0.480 |
| Precision-first / Low-FP | 778532 | 0.491 | 0.705 | 0.443 | 0.696 | 0.550 | 0.549 | 0.107 | 350564 | 0.450 |

## Target Assessment

- Low-FP target eval recall: 0.717.
- Low-FP observed eval recall: 0.696.
- Target met: no.
- Compared with Strict, Low-FP changes FP Duration by -45562 frames and GT Precision by 0.023.
- Compared with Recall-first, Low-FP changes FP Duration by -81920 frames and GT Precision by 0.030.
- Interpretation: Low-FP substantially reduces FP and improves precision, but it falls slightly below the chosen eval-recall target; it is a precision-first operating point rather than a no-regret replacement for Strict or Recall-first.

## Precision-first / Low-FP Configuration

The low-FP configuration was selected from a small precision-oriented candidate set, not a broad parameter scan.

```json
{
  "fusion_threshold": 0.38,
  "trend_window": 100,
  "trend_threshold": 0.6,
  "airpls_lambda": 1000,
  "peak_mad_k": 3.0,
  "sg_window_length": 17,
  "sg_polyorder": 2,
  "length_penalty_weight": 0.22,
  "low_residual_penalty_weight": 0.15,
  "residual_weight": 0.25,
  "trend_weight": 0.2,
  "sg_weight": 0.0,
  "raw_weight": 0.25,
  "peak_count_weight": 0.0,
  "residual_mad_k": 3.0,
  "peak_stop_ratio": 0.25,
  "merge_gap_frames": 32,
  "enable_sg": true,
  "enable_airpls": true,
  "enable_trend": true,
  "run_name": "Low-FP mild threshold0.38 gap32 min32 raw0.60",
  "enabled_components": "raw, sg, residual, trend, peak_count, length_penalty, low_residual_penalty",
  "disabled_components": "",
  "configuration_type": "precision-first",
  "sg_smoothing_enabled": true,
  "sg_candidate_generation_enabled": true,
  "residual_candidate_generation_enabled": true,
  "trend_candidate_generation_enabled": true,
  "notes": "precision-first candidate",
  "post_min_duration": 32,
  "post_min_raw_max": 0.6
}
```

Selection objective:

`score = GT_precision - 4.0 * max(0, target_eval_recall - eval_recall) - 0.25 * PI_duration_ratio`

The target eval recall is `max(strict_eval_recall, 0.75 * original_eval_recall)`.

## Low-FP Search Candidates

| candidate | objective | eval_recall | GT_precision | PI_ratio | FP_ratio |
|---|---:|---:|---:|---:|---:|
| Low-FP mild threshold0.38 gap32 min32 raw0.60 | 0.342 | 0.696 | 0.550 | 0.491 | 0.450 |
| Low-FP balanced threshold0.40 gap24 min48 raw0.60 | 0.190 | 0.644 | 0.589 | 0.426 | 0.411 |
| Low-FP balanced threshold0.42 gap24 min48 raw0.65 | -0.309 | 0.502 | 0.629 | 0.313 | 0.371 |
| Low-FP threshold0.45 gap24 min64 raw0.7 | -1.196 | 0.261 | 0.665 | 0.155 | 0.335 |
| Low-FP threshold0.50 gap16 min96 raw0.75 | -1.890 | 0.079 | 0.674 | 0.046 | 0.326 |
| Low-FP threshold0.55 gap16 min128 raw0.80 mean0.30 | -1.989 | 0.049 | 0.689 | 0.028 | 0.311 |
| Low-FP threshold0.60 gap0 min128 raw0.85 | -2.032 | 0.038 | 0.690 | 0.022 | 0.310 |