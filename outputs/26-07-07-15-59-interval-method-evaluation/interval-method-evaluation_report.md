# Interval Method Evaluation Report

## Executive summary

- Raw balanced_score leader: `Random-Same-Length`; it is treated as a sanity/broad-coverage baseline, not as the recommended detector.
- Recommended balanced current method: `Peak-Aware-Refined`.
- GT coverage: 0.700.
- predicted_GT_fraction: 0.519.
- supportable_gt_coverage: 0.705.
- unsupportable_gt_coverage: 0.441.
- Evidence of gaining coverage by broadening intervals: limited/not primary; predicted_duration_ratio=0.541.

This report does not use ordinary frame-level accuracy as the main metric because normal frames dominate anomaly videos and can make accuracy look artificially high.

## Method descriptions

- `Adaptive`: Adaptive: selects intervals using adaptive score/spacing rules tuned for long multi-anomaly videos.
- `Hierarchical-Merged`: Merged intervals: adjacent or overlapping micro fragments are merged into coarser event-level intervals.
- `Hierarchical-Micro-100F`: Micro intervals: fine-grained abnormal fragments selected from score-thresholded windows at a specific scale.
- `Hierarchical-Micro-300F`: Micro intervals: fine-grained abnormal fragments selected from score-thresholded windows at a specific scale.
- `Hierarchical-Micro-30F`: Micro intervals: fine-grained abnormal fragments selected from score-thresholded windows at a specific scale.
- `Multiscale`: Multiscale: collects top candidate windows across several temporal scales.
- `Peak-Aware-Refined`: Peak-aware refined: uses local peak evidence to rescue, split, and refine hierarchical intervals.
- `Peak-Expanded-Baseline`: Peak-expanded baseline: baseline interval expanded around score peaks.
- `Random-Same-Length`: Random same-length baseline: random intervals with matched length, useful as a sanity baseline.
- `TopK-1`: Top-K: selects the highest-scoring candidate windows per video.
- `TopK-10`: Top-K: selects the highest-scoring candidate windows per video.
- `TopK-2`: Top-K: selects the highest-scoring candidate windows per video.
- `TopK-3`: Top-K: selects the highest-scoring candidate windows per video.
- `TopK-5`: Top-K: selects the highest-scoring candidate windows per video.
- `Window-1000F`: Window method: fixed-size windows are marked abnormal when their mean anomaly score passes the score threshold.
- `Window-100F`: Window method: fixed-size windows are marked abnormal when their mean anomaly score passes the score threshold.
- `Window-300F`: Window method: fixed-size windows are marked abnormal when their mean anomaly score passes the score threshold.
- `Window-30F`: Window method: fixed-size windows are marked abnormal when their mean anomaly score passes the score threshold.
- `Wmax-Baseline`: Wmax baseline: the original best-scoring window-like interval per video.

## Metrics definition

- `predicted_GT_fraction`: fraction of predicted abnormal duration that overlaps human GT; this is interval purity.
- `GT_coverage`: fraction of GT abnormal duration covered by predictions; this is duration-level recall.
- `GT_uncovered_ratio`: one minus GT coverage.
- `supportable_gt_coverage`: coverage on GT intervals whose scores are strongly or weakly supported.
- `unsupportable_gt_coverage`: coverage on GT intervals without score evidence; high values can come from broad intervals and are not automatically good.
- `predicted_duration_ratio`: predicted abnormal duration divided by total video duration.
- `event hit ratio`: fraction of GT events with any prediction overlap.
- `IoU-threshold hit ratio`: stricter event hit ratio requiring interval IoU above the threshold.

## Overall comparison

- CSV: `method_overall_metrics.csv`
- Figure: `fig_method_gt_coverage_vs_purity.png`
- Figure: `fig_method_predicted_duration_ratio.png`

| method | GT_coverage | predicted_GT_fraction | supportable_gt_coverage | unsupportable_gt_coverage | predicted_duration_ratio | balanced_score |
|---|---:|---:|---:|---:|---:|---:|
| `Random-Same-Length` | 0.994 | 0.406 | 0.994 | 0.992 | 0.983 | 0.620 |
| `TopK-10` | 0.858 | 0.401 | 0.859 | 0.753 | 0.786 | 0.556 |
| `Peak-Aware-Refined` | 0.700 | 0.519 | 0.705 | 0.441 | 0.541 | 0.522 |
| `Hierarchical-Merged` | 0.696 | 0.491 | 0.702 | 0.438 | 0.551 | 0.511 |
| `Window-30F` | 0.514 | 0.670 | 0.529 | 0.044 | 0.308 | 0.482 |
| `Hierarchical-Micro-300F` | 0.638 | 0.492 | 0.644 | 0.375 | 0.505 | 0.481 |
| `Window-100F` | 0.470 | 0.703 | 0.483 | 0.050 | 0.268 | 0.469 |
| `Adaptive` | 0.685 | 0.383 | 0.692 | 0.426 | 0.605 | 0.467 |
| `Window-300F` | 0.423 | 0.705 | 0.437 | 0.041 | 0.241 | 0.444 |
| `TopK-5` | 0.601 | 0.444 | 0.605 | 0.372 | 0.509 | 0.444 |
| `Hierarchical-Micro-100F` | 0.409 | 0.580 | 0.414 | 0.155 | 0.282 | 0.392 |
| `Window-1000F` | 0.315 | 0.669 | 0.325 | 0.035 | 0.189 | 0.373 |

## Supportability-aware comparison

- Best supportable GT coverage: `Random-Same-Length`.
- Unsupportable coverage is reported separately because it may reflect over-wide predictions rather than true score-supported detection.

## Coverage-purity trade-off

- High GT coverage with low predicted_GT_fraction indicates broad intervals that cover GT plus much normal/non-GT time.
- High predicted_GT_fraction with high GT_uncovered_ratio indicates a conservative method with good purity but many missed GT events.
- The current operating point should be selected from the coverage-purity frontier rather than by maximizing a single score.

## Best method recommendation

- Best recall-oriented method excluding random sanity baseline: `TopK-10`.
- Best purity-oriented method: `Window-300F`.
- Best raw balanced_score method: `Random-Same-Length`; this is diagnostic and may select over-broad baselines.
- Recommended current method: `Peak-Aware-Refined`.

## Limitations

- This is an offline evaluation and does not modify VDA, VLM, LLM scoring, interval extraction, or peak refinement.
- Supportable and unsupportable groups depend on the existing score-support classification.
- Human GT and VDA score may differ in definition and temporal granularity.
- Window methods can increase coverage by widening intervals, often reducing purity.
- IoU is sensitive to interval length and can penalize short/long interval mismatch.

## Next steps

- Do not directly choose 300F/1000F windows as the final method only because they improve coverage.
- Choose an operating point on the coverage-purity frontier.
- To exceed the post-processing upper bound, revisit VDA: VLM descriptions, LLM scoring, and annotation consistency.

## Parsed and skipped inputs

- Parsed methods: 19.
- Candidate/score files with warnings: 0.