# Original Pipeline Limitations From Results

The limitations below are grounded in aggregate metrics, case-study rows, and the final comparison table.

| configuration | GT_coverage | predicted_GT_fraction | supportable_gt_coverage | unsupportable_gt_coverage | predicted_duration_ratio | stricter_balanced_score |
|---|---|---|---|---|---|---|
| Peak-Aware | 0.700 | 0.519 | 0.705 | 0.441 | 0.541 | 0.418 |
| Full | 0.729 | 0.506 | 0.742 | 0.247 | 0.562 | 0.450 |
| SG0 | 0.712 | 0.523 | 0.728 | 0.187 | 0.533 | 0.454 |
| Recall | 0.753 | 0.520 | 0.768 | 0.225 | 0.566 | 0.469 |
| Strict | 0.710 | 0.527 | 0.725 | 0.187 | 0.527 | 0.455 |

## Findings

- Temporal mismatch: human GT intervals are event-level ranges, while anomaly/VAD scores are frame/score-level curves. The evaluator measures duration overlap in `evaluate_one_video`, so wide GT context and local score peaks can disagree.
- Local peak mismatch: Peak-Aware reaches GT=0.700 and strict=0.418, but Full/SG0/Recall/Strict change supportable and unsupported coverage by using interval-level features.
- Missing joint modeling: baseline peak or threshold generation does not explicitly combine raw evidence, residual evidence, trend, length penalty, and low-residual penalty; those terms are implemented in `score_fusion`.
- No explicit supportability split in generation: supportable/unsupportable is added by evaluation from `gt_support_classification.csv`, not by the original interval generator.
- Over-merge, over-extension, and fragmentation are visible through duration/purity trade-offs. Full default has duration=0.562; SG0 lowers unsupported coverage and duration while slightly lowering GT.
- Peak-Aware differs from final operating points: Recall raises GT/supportable coverage, while Strict improves conservative score relative to Peak-Aware without treating unsupported coverage as a standalone objective.
