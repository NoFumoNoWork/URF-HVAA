# Spectral Score Decomposition Report

## Motivation

The anomaly score curve is treated as a spectroscopy-like signal containing noise, baseline/trend, local peaks, and broad bands. The goal is not to train a new model, but to test interpretable preprocessing and interval fusion rules.

## Literature-inspired design

- Savitzky-Golay smoothing uses local least-squares polynomial fitting to smooth noise while preserving peak shape better than a simple moving average.
- airPLS estimates an adaptive baseline with iteratively reweighted penalized least squares, reducing the influence of high peak points on the baseline.
- Multiple preprocessing permutations are compared downstream instead of selecting one curve by intuition.

## Parameters

- SG windows: [9, 17, 31]; SG polyorders: [2, 3].
- airPLS lambdas: [100.0, 1000.0, 10000.0]; order=2; itermax=20.
- trend windows: [30, 100, 300] frames.
- score_threshold: 0.6; fusion_threshold: 0.35.

## Method

- `SG-Peak`: detects peaks on the primary SG-smoothed curve using median + k*MAD, then expands boundaries until residual evidence decays.
- `AirPLS-Residual`: detects contiguous high positive residual regions after baseline subtraction.
- `Trend-Guided-100F`: selects 100F trend-positive regions and requires raw/residual support.
- `Spectral-Fusion-Refined`: fuses existing peak-aware/hierarchical intervals with spectral candidates using normalized raw, SG, residual, trend, peak-count, length, and low-residual evidence.

## Results

- Videos processed: 640; GT intervals: 1394; warnings: 0.
- Best new spectral method by balanced_score: `Spectral-Fusion-Refined`.

| method | GT_coverage | predicted_GT_fraction | supportable_gt_coverage | unsupportable_gt_coverage | predicted_duration_ratio | balanced_score |
|---|---:|---:|---:|---:|---:|---:|
| `Peak-Aware-Refined` | 0.700 | 0.519 | 0.705 | 0.441 | 0.541 | 0.522 |
| `SG-Peak` | 0.087 | 0.501 | 0.088 | 0.053 | 0.069 | 0.195 |
| `AirPLS-Residual` | 0.064 | 0.359 | 0.060 | 0.116 | 0.072 | 0.138 |
| `Trend-Guided-100F` | 0.469 | 0.708 | 0.483 | 0.051 | 0.266 | 0.470 |
| `Spectral-Fusion-Refined` | 0.731 | 0.506 | 0.744 | 0.247 | 0.564 | 0.536 |
| `Window-100F` | 0.470 | 0.703 | 0.483 | 0.050 | 0.268 | 0.469 |
| `Window-300F` | 0.423 | 0.705 | 0.437 | 0.041 | 0.241 | 0.444 |
| `TopK-10` | 0.858 | 0.401 | 0.859 | 0.753 | 0.786 | 0.556 |

## Recommendation

`Spectral-Fusion-Refined` beat `Peak-Aware-Refined` by the current balanced_score. If it failed to improve, the likely reason is that the VDA score evidence and peak-aware rules already capture most recoverable signal, while extra residual/trend intervals add duration or miss sparse GT intervals.

Random-Same-Length is retained only as a sanity baseline and must not be recommended as a detector.

## Limitations

- GT intervals with no VDA score response cannot be recovered by post-processing alone.
- airPLS lambda strongly affects baseline and residual estimates.
- Large SG windows can smooth away short anomalies.
- Fusion weights are fixed rules, not learned on a validation set.
- Score stride and sparse labels can distort short-interval evidence.

## Next steps

- Sweep fusion weights and residual thresholds on a validation split.
- Keep the coverage-purity frontier as the operating-point selection tool.
- If spectral decomposition cannot beat peak-aware refinement, return to VDA score generation and annotation consistency rather than only post-processing.