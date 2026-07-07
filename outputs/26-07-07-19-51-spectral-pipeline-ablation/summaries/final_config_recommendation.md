# Final Spectral Fusion Configuration Recommendation

## New Direct Fusion Candidate: Spectral-Fusion-SG0

- Changed parameter: `sg_weight` from 0.20 to 0.00.
- SG smoothing still runs; cached `sg_score_{window}_{poly}` curves remain available.
- SG candidate intervals still generate by default through `SG-Peak` unless explicitly disabled in pipeline ablations.
- `sg_evidence` can still be computed as a feature, but it contributes zero direct positive score in `Spectral-Fusion-SG0`.

```text
fusion_score =
    raw_weight * raw_evidence
  + residual_weight * residual_evidence
  + trend_weight * trend_evidence
  + peak_count_weight * peak_count_evidence
  - length_penalty_weight * length_penalty
  - low_residual_penalty_weight * low_residual_penalty
```

## Pipeline SG Candidate Ablation

- Full default: GT=0.729, purity=0.506, supportable=0.742, unsupported=0.247, duration=0.562, strict=0.450.
- Spectral-Fusion-SG0: GT=0.712, purity=0.523, supportable=0.728, unsupported=0.187, duration=0.533, strict=0.454.
- Removing SG candidates with SG0: GT=0.705, purity=0.526, supportable=0.720, unsupported=0.187, duration=0.525, strict=0.452.
- Interpretation: `sg_weight=0` tests direct evidence; removing SG candidates tests candidate-generation contribution. Compare the CSV deltas to decide whether SG is useful as source generation even when it is not useful as direct score.

## Pipeline Residual Candidate Ablation

- w/o residual direct evidence: GT=0.665, purity=0.576, supportable=0.683, unsupported=0.075, duration=0.454, strict=0.461.
- w/o residual candidates: GT=0.722, purity=0.510, supportable=0.736, unsupported=0.247, duration=0.553, strict=0.448.
- Interpretation: residual direct evidence trades coverage against duration/unsupported coverage. With the old SG-positive score, setting residual direct evidence to zero improves strict score by shrinking intervals. Under the SG0 candidate family, however, completely zeroing residual direct evidence can collapse coverage, so residual direct weight should be tuned rather than blindly removed.

## Final Candidate Comparison

| run | type | GT | purity | supportable | unsupported | duration | strict |
|---|---|---:|---:|---:|---:|---:|---:|
| `Peak-Aware-Refined baseline` | baseline | 0.700 | 0.519 | 0.705 | 0.441 | 0.541 | 0.418 |
| `Full Spectral-Fusion-Refined default` | default | 0.729 | 0.506 | 0.742 | 0.247 | 0.562 | 0.450 |
| `Spectral-Fusion-SG0` | SG0 | 0.712 | 0.523 | 0.728 | 0.187 | 0.533 | 0.454 |
| `Strict SG0 residual0 peak_count_keep` | strict-oriented | 0.091 | 0.613 | 0.092 | 0.019 | 0.060 | 0.196 |
| `Strict SG0 residual0 peak_count0` | strict-oriented | 0.085 | 0.611 | 0.086 | 0.019 | 0.056 | 0.192 |
| `Duration combo SG0 residual0` | duration-controlled | 0.311 | 0.630 | 0.318 | 0.045 | 0.197 | 0.306 |
| `Duration combo SG0 residual0.10` | duration-controlled | 0.548 | 0.623 | 0.564 | 0.053 | 0.349 | 0.421 |
| `Recall trend0.5 SG0 residual0.10` | recall-oriented | 0.638 | 0.594 | 0.655 | 0.061 | 0.425 | 0.455 |
| `Recall trend0.5 SG0 residual0.25` | recall-oriented | 0.753 | 0.520 | 0.768 | 0.225 | 0.566 | 0.469 |
| `Raw trend residual penalties SG0 residual0.25 peak0` | strict-oriented | 0.710 | 0.527 | 0.725 | 0.187 | 0.527 | 0.455 |
| `Raw trend residual penalties SG0 residual0.10 peak0` | strict-oriented | 0.557 | 0.616 | 0.573 | 0.053 | 0.359 | 0.423 |

## Recall-Oriented Configuration

- Configuration name: `Recall trend0.5 SG0 residual0.25`.
- Candidate sources: Peak-Aware-Refined, Hierarchical-Merged, SG-Peak, AirPLS-Residual, Trend-Guided.
- Direct weights: raw=0.25, sg=0.0, residual=0.25, trend=0.15, peak_count=0.1, length_penalty=0.15, low_residual_penalty=0.1.
- Key thresholds: fusion_threshold=0.35, trend_threshold=0.5, trend_window=100.
- Metrics: GT=0.753, purity=0.520, supportable=0.768, unsupported=0.225, duration=0.566, strict=0.469.
- Relative to Full default: GT 0.024, supportable 0.026, purity 0.015, unsupported -0.022, duration 0.004.
- Relative to Peak-Aware: GT 0.053, supportable 0.063, purity 0.001, unsupported -0.216, duration 0.025.
- Why recall-oriented: it prioritizes GT/supportable coverage and accepts some loss in purity or duration control.

## Strict-Oriented / Duration-Controlled Configuration

- Configuration name: `Raw trend residual penalties SG0 residual0.25 peak0`.
- Candidate sources: Peak-Aware-Refined, Hierarchical-Merged, SG-Peak, AirPLS-Residual, Trend-Guided.
- Direct weights: raw=0.25, sg=0.0, residual=0.25, trend=0.15, peak_count=0.0, length_penalty=0.15, low_residual_penalty=0.1.
- Key thresholds: fusion_threshold=0.35, trend_threshold=0.6, trend_window=100.
- Metrics: GT=0.710, purity=0.527, supportable=0.725, unsupported=0.187, duration=0.527, strict=0.455.
- Viable duration-controlled alternative: `Duration combo SG0 residual0.10` with GT=0.548, purity=0.623, supportable=0.564, unsupported=0.053, duration=0.349, strict=0.421.
- Relative to Full default: GT -0.019, supportable -0.017, purity 0.022, unsupported -0.061, duration -0.035.
- Relative to Peak-Aware: GT 0.010, supportable 0.020, purity 0.008, unsupported -0.254, duration -0.014.
- Why strict-oriented: it prefers higher purity, lower unsupported coverage, and lower duration even if GT coverage drops.

## Final Judgments

1. `sg_weight` should be set to 0 for the next default direct-fusion candidate.
2. SG smoothing should be retained.
3. SG candidate generation should be retained if removing it lowers GT/supportable coverage without enough strict-score gain; use `pipeline_sg_candidate_ablation.csv` for the exact trade-off.
4. Residual direct fusion evidence should not be removed globally. For the SG0 strict-balanced candidate, a positive residual direct weight preserves coverage; for duration-controlled variants, lower residual weight such as 0.10 can be tested.
5. Residual candidate generation should be retained. Removing residual candidates slightly lowers coverage and did not produce a better strict operating point in the final comparison.
6. Trend evidence is a stabilizing context term; it is more important in strict/duration-controlled configurations where residual is downweighted.
7. Penalties should be retained because they control duration and unsupported coverage.
8. Peak-count direct evidence is weak but cheap; keep it only if validation confirms no harm, otherwise set it to 0 in strict mode.
9. Two operating points are recommended: one strict/main-report configuration and one supplementary recall-oriented configuration.
10. Main report candidate: `Raw trend residual penalties SG0 residual0.25 peak0` pending validation. Supplementary recall-oriented result: `Recall trend0.5 SG0 residual0.25`.

## Limitations

- These are offline post-processing ablations on the same data, not held-out validation results.
- Candidate-generation ablations still reuse existing Peak-Aware and Hierarchical candidates.
- Supportable/unsupportable labels come from prior score-support classification.