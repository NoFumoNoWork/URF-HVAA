# Spectral Fusion Ablation Summary

## System structure

- Score JSON loading: `scripts/run_spectral_score_decomposition.py::load_score_series` reads dict/list score JSON into sorted frame and score arrays; `scripts/evaluate_interval_methods.py::load_score_series` has a dict form for fixed-window baselines.
- Candidate interval generation: `generate_spectral_intervals` in the decomposition script creates `SG-Peak`, `AirPLS-Residual`, and `Trend-Guided-100F`; the focused ablation script uses `scripts/run_spectral_param_scan.py::generate_candidates_for_video` to create the same families with parameterized curves.
- SG smoothing: `compute_sg` calls `scipy.signal.savgol_filter` and writes curves named `sg_score_{window}_{poly}`.
- airPLS baseline/residual: `airpls` computes the baseline; `build_decomposition` and `precompute_curves` store `airpls_baseline_{lambda}` and `airpls_residual_{lambda}`.
- Trend evidence: `rolling_mean_by_frames` creates `rolling_mean_{window}`; trend candidates use `trend > trend_threshold` in the scan code and `trend100 > score_threshold` in the original decomposition script.
- Peak-count evidence: `extract_features` in the original script and `feature_rows` in the scan script compute `residual_peak_count` from positive residual evidence over each candidate interval.
- Fusion score: original `build_fusion` in `scripts/run_spectral_score_decomposition.py`; parameterized version in `scripts/run_spectral_param_scan.py::build_fusion`; this ablation script uses the same feature definitions but separates direct terms explicitly.
- `fusion_threshold`: used after score calculation to keep candidate intervals whose score is at least the threshold, then `merge_with_gap` merges retained intervals per video.
- Length penalty: candidate `interval_length` is min-max normalized and subtracted as `length_penalty_weight * length_penalty`.
- Low-residual penalty: `low_residual_ratio` is the fraction of residual samples <= 0.05 inside the interval and is subtracted directly as `low_residual_penalty_weight * low_residual_ratio`.
- Supportable/unsupportable coverage: `scripts/evaluate_interval_methods.py::support_group` maps GT rows to supportable, unsupportable, or uncertain, then `evaluate_one_video` and `aggregate_rows` compute covered duration divided by group GT duration.
- `score_threshold=0.6`: in the inspected code it is used for fixed-window candidate generation and trend/local evidence checks. Supportability grouping is read from `recoverable_by_postprocessing` and `support_type` columns; the evaluator itself does not threshold scores at 0.6 to define supportability.

The direct fusion score used for this focused ablation is:

```text
fusion_score =
    raw_weight * raw_evidence
  + sg_weight * sg_evidence
  + residual_weight * residual_evidence
  + trend_weight * trend_evidence
  + peak_count_weight * peak_count_evidence
  - length_penalty_weight * length_penalty
  - low_residual_penalty_weight * low_residual_penalty
```

Original decomposition-script constants are the same shape except hard-coded as `0.25 raw + 0.20 SG + 0.25 residual + 0.15 trend + 0.10 peak_count - 0.15 length - 0.10 low_residual`.

## Default configuration

| parameter | value | source |
|---|---:|---|
| `fusion_threshold` | 0.35 | scripts/run_spectral_param_scan.py::DEFAULT_PARAMS; original CLI default also 0.35 |
| `score_threshold` | 0.6 | scripts/run_spectral_score_decomposition.py CLI default; used by original trend/window candidate checks |
| `trend_threshold` | 0.6 | DEFAULT_PARAMS |
| `trend_window` | 100 | DEFAULT_PARAMS |
| `trend_weight` | 0.15 | DEFAULT_PARAMS |
| `residual_weight` | 0.25 | DEFAULT_PARAMS |
| `sg_weight` | 0.2 | DEFAULT_PARAMS |
| `peak_count_weight` | 0.1 | DEFAULT_PARAMS |
| `length_penalty_weight` | 0.15 | DEFAULT_PARAMS |
| `low_residual_penalty_weight` | 0.1 | DEFAULT_PARAMS |
| `airpls_lambda` | 1000 | DEFAULT_PARAMS |
| `airpls_order` | 2 | original CLI default and scan precompute |
| `airpls_itermax` | 20 | original CLI default and scan precompute |
| `sg_window_length` | 17 | DEFAULT_PARAMS |
| `sg_polyorder` | 2 | DEFAULT_PARAMS |
| `peak_mad_k` | 3.0 | DEFAULT_PARAMS |
| `residual_mad_k` | 3.0 | DEFAULT_PARAMS |
| `peak_stop_ratio` | 0.25 | DEFAULT_PARAMS |
| `merge_gap_frames` | 48 | DEFAULT_PARAMS |

## SG direct fusion weight sensitivity

| run | sg_weight | GT | purity | supportable | unsupportable | duration | balanced | strict | intervals | mean_len | median_len | delta_strict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `sg_weight_0` | 0.00 | 0.712 | 0.523 | 0.728 | 0.187 | 0.533 | 0.534 | 0.454 | 1396.000 | 607.493 | 474.000 | 0.005 |
| `sg_weight_0.05` | 0.05 | 0.717 | 0.516 | 0.732 | 0.225 | 0.543 | 0.534 | 0.450 | 1433.000 | 603.743 | 472.000 | 0.001 |
| `sg_weight_0.1` | 0.10 | 0.722 | 0.510 | 0.735 | 0.233 | 0.553 | 0.533 | 0.449 | 1471.000 | 598.600 | 464.000 | -0.000 |
| `sg_weight_0.2` | 0.20 | 0.729 | 0.506 | 0.742 | 0.247 | 0.562 | 0.535 | 0.450 | 1488.000 | 603.048 | 470.000 | 0.000 |
| `sg_weight_0.35` | 0.35 | 0.735 | 0.501 | 0.748 | 0.257 | 0.572 | 0.537 | 0.450 | 1523.000 | 599.551 | 464.000 | 0.000 |

- Best SG weight by stricter score: `0.00`.
- Default sg_weight=0.2 does not improve stricter_balanced_score versus sg_weight=0 in this fixed-candidate experiment.

## Module ablation results

| run | enabled | disabled | GT | purity | supportable | unsupportable | duration | balanced | strict | intervals | mean_len | median_len |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `Full Spectral-Fusion-Refined` | raw, sg, residual, trend, peak_count, length_penalty, low_residual_penalty |  | 0.729 | 0.506 | 0.742 | 0.247 | 0.562 | 0.535 | 0.450 | 1488.000 | 603.048 | 470.000 |
| `w/o SG evidence` | raw, residual, trend, peak_count, length_penalty, low_residual_penalty | sg | 0.712 | 0.523 | 0.728 | 0.187 | 0.533 | 0.534 | 0.454 | 1396.000 | 607.493 | 474.000 |
| `w/o airPLS residual evidence` | raw, sg, trend, peak_count, length_penalty, low_residual_penalty | residual | 0.665 | 0.576 | 0.683 | 0.075 | 0.454 | 0.530 | 0.461 | 1148.000 | 626.077 | 492.000 |
| `w/o trend evidence` | raw, sg, residual, peak_count, length_penalty, low_residual_penalty | trend | 0.723 | 0.513 | 0.737 | 0.226 | 0.552 | 0.535 | 0.452 | 1415.000 | 620.769 | 492.000 |
| `w/o peak-count evidence` | raw, sg, residual, trend, length_penalty, low_residual_penalty | peak_count | 0.728 | 0.507 | 0.742 | 0.247 | 0.560 | 0.536 | 0.450 | 1490.000 | 599.374 | 470.000 |
| `w/o length penalty` | raw, sg, residual, trend, peak_count, low_residual_penalty | length_penalty | 0.732 | 0.502 | 0.745 | 0.257 | 0.568 | 0.535 | 0.449 | 1485.000 | 610.742 | 478.000 |
| `w/o low-residual penalty` | raw, sg, residual, trend, peak_count, length_penalty | low_residual_penalty | 0.732 | 0.501 | 0.744 | 0.277 | 0.569 | 0.535 | 0.446 | 1487.000 | 610.913 | 472.000 |
| `w/o both penalties` | raw, sg, residual, trend, peak_count | length_penalty, low_residual_penalty | 0.738 | 0.500 | 0.751 | 0.277 | 0.575 | 0.538 | 0.449 | 1486.000 | 618.054 | 480.000 |
| `SG only` | sg | raw, residual, trend, peak_count, length_penalty, low_residual_penalty | 0.000 | NA | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | NA | NA |
| `residual only` | residual | raw, sg, trend, peak_count, length_penalty, low_residual_penalty | 0.000 | NA | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | NA | NA |
| `trend only` | trend | raw, sg, residual, peak_count, length_penalty, low_residual_penalty | 0.000 | NA | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | NA | NA |
| `peak-count only` | peak_count | raw, sg, residual, trend, length_penalty, low_residual_penalty | 0.000 | NA | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | NA | NA |
| `raw only` | raw | sg, residual, trend, peak_count, length_penalty, low_residual_penalty | 0.000 | NA | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | NA | NA |
| `raw + trend` | raw, trend | sg, residual, peak_count, length_penalty, low_residual_penalty | 0.269 | 0.626 | 0.276 | 0.037 | 0.171 | 0.333 | 0.285 | 343.000 | 778.297 | 656.000 |
| `raw + residual` | raw, residual | sg, trend, peak_count, length_penalty, low_residual_penalty | 0.689 | 0.517 | 0.703 | 0.221 | 0.523 | 0.519 | 0.437 | 1207.000 | 687.039 | 544.000 |
| `raw + trend + residual` | raw, trend, residual | sg, peak_count, length_penalty, low_residual_penalty | 0.723 | 0.509 | 0.737 | 0.261 | 0.555 | 0.534 | 0.447 | 1390.000 | 635.945 | 496.000 |
| `raw + trend + residual + length penalty` | raw, trend, residual, length_penalty | sg, peak_count, low_residual_penalty | 0.717 | 0.513 | 0.731 | 0.226 | 0.546 | 0.532 | 0.449 | 1394.000 | 623.877 | 492.000 |
| `raw + trend + residual + both penalties` | raw, trend, residual, length_penalty, low_residual_penalty | sg, peak_count | 0.710 | 0.527 | 0.725 | 0.187 | 0.527 | 0.534 | 0.455 | 1400.000 | 598.440 | 464.000 |

## Answers

1. Full Spectral-Fusion-Refined is better than Peak-Aware-Refined by stricter score in this evaluation. Full strict=0.450; Peak-Aware strict=0.418.
2. Trend evidence appears not a clear main gain source: removing it changes strict score by 0.002. In the additive runs, `raw + trend` is high-purity but low-recall, so trend is useful as context but not sufficient alone.
3. airPLS residual evidence mainly buys recall/coverage at the cost of broader and less pure predictions. Removing residual drops GT coverage by -0.064, but improves purity by 0.070, lowers unsupportable coverage by -0.172, lowers duration by -0.108, and improves strict score by 0.012.
4. SG direct fusion weight is weak or negative under this fixed-candidate test; best tested sg_weight is 0.00.
5. SG should be considered for removal from direct fusion while retaining smoothing/candidate/diagnostic roles unless validated on a held-out split.
6. Peak-count independent contribution is weak: `w/o peak-count evidence` delta_strict=0.000, with nearly unchanged supportable and unsupportable coverage.
7. Length penalty helps modestly control width/unsupported coverage: removing it changes duration by 0.006 and unsupportable coverage by 0.010.
8. Low-residual penalty helps reduce unsupported coverage: removing it changes unsupportable coverage by 0.030 and strict score by -0.003.
9. Obvious drag components are those whose removal improves strict score; see `ablation_delta_vs_full.csv`. Best strict run here is `w/o airPLS residual evidence`.
10. Recall-oriented: `w/o both penalties`. Duration-controlled: `w/o airPLS residual evidence`. Main report recommendation from this ablation: `w/o airPLS residual evidence` pending validation split.

## Short conclusion

- Core useful components: raw score evidence, trend context, and penalties for duration / low residual support.
- Auxiliary useful components: SG smoothing/candidate generation and diagnostic curves remain useful even when direct SG weight is weak.
- Components with weak or negative direct contribution: SG direct evidence, peak-count direct evidence, and airPLS residual direct evidence under this fixed-candidate strict-score objective.
- Recommended direct fusion weights: raw=0.25, sg=0.00, residual=0.00, trend=0.15, peak_count=0.10, length_penalty=0.15, low_residual_penalty=0.10 as a validation candidate; among actually executed rows, `w/o airPLS residual evidence` is best.
- Recommended operating point: `w/o airPLS residual evidence` for strict balance; `w/o airPLS residual evidence` when duration control is prioritized.
- Remaining limitations: no held-out validation split, fixed candidate pool isolates direct scoring but does not test candidate-generation ablations, and supportability labels come from prior score-support analysis rather than new human adjudication.