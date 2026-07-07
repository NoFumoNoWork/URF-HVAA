# Spectral Fusion / Interval Reconstruction Final Project Report

## 1. Motivation

Human GT is event-level, while the anomaly/VAD score is a temporal score curve. The two have natural temporal mismatch: GT can include context, lead-in, and aftermath, while the score often peaks around visually salient frames. The goal of this project is score-level interval reconstruction: use the score curve and derived evidence to produce interpretable abnormal intervals.

## 2. Method Overview

The system decomposes the score curve into raw score, SG-smoothed score, airPLS residual, trend evidence, and peak-count evidence. Candidate intervals come from Peak-Aware, Hierarchical-Merged, SG-Peak, AirPLS-Residual, and Trend-Guided sources. Fusion scoring then combines evidence and penalties before final interval merging.

The key correction is SG0: SG smoothing and SG-Peak candidates are retained, but direct SG positive fusion weight is set to zero. The final recommendation keeps two operating points: a recall-oriented point and a strict/conservative point.

## 3. Baseline Comparison

| config | GT | purity | supportable | unsupported | duration | strict |
|---|---:|---:|---:|---:|---:|---:|
| Peak-Aware | 0.700 | 0.519 | 0.705 | 0.441 | 0.541 | 0.418 |
| Full | 0.729 | 0.506 | 0.742 | 0.247 | 0.562 | 0.450 |
| SG0 | 0.712 | 0.523 | 0.728 | 0.187 | 0.533 | 0.454 |
| Recall | 0.753 | 0.520 | 0.768 | 0.225 | 0.566 | 0.469 |
| Strict | 0.710 | 0.527 | 0.725 | 0.187 | 0.527 | 0.455 |

## 4. Parameter Scan Summary

- `fusion_threshold` controls the recall-purity trade-off.
- `trend_threshold` is a key lever for recall-oriented operation.
- Length and low-residual penalties control duration and unsupported coverage.
- Low-level SG, airPLS lambda, and peak MAD parameters were less decisive than operating-point controls in the current scans.

## 5. Ablation Summary

- SG direct weight should be 0 for the next direct-fusion default candidate.
- SG smoothing and SG candidate generation should be retained.
- Residual candidate generation should be retained.
- Residual direct weight controls recall-strict trade-off; it should not be blindly removed.
- Peak-count direct evidence is weak and can be set to 0 in strict mode.
- Penalties should be retained because they reduce duration and unsupported over-extension.

## 6. Final Configurations

- Recall-oriented: `Recall trend0.5 SG0 residual0.25` with GT=0.753, supportable=0.768, purity=0.520, unsupported=0.225, duration=0.566.
- Strict-oriented: `Raw trend residual penalties SG0 residual0.25 peak0` with GT=0.710, supportable=0.725, purity=0.527, unsupported=0.187, duration=0.527.

## 7. Case Studies

- `over-wide full corrected by SG0/Strict`: `XD-Violence / v=15wDrZJQpsw__#00-00-00_00-00-51_label_B6-0-0`. Full default predicts a broader interval footprint than SG0/Strict while the conservative variants retain some GT overlap.
- `recall-oriented recovers supportable GT`: `XD-Violence / The.Hurt.Locker.2008__#0-19-22_0-22-32_label_B2-B1-0`. Recall-oriented configuration improves supportable GT coverage relative to Full default.
- `residual direct evidence over-extension`: `XD-Violence / v=9Jk2sIp5MRQ__#1_label_G-0-0`. Full default expands more than SG0, illustrating how direct evidence can increase duration and unsupported coverage.
- `score-unsupported GT partially covered`: `XD-Violence / Crank.Dircut.2006__#0-27-42_0-29-01_label_B1-0-0`. The video has score-unsupported GT that is nevertheless partially covered, useful for explaining unsupported coverage as diagnostic rather than purely negative.
- `score no-response failure`: `UCF-Crime / Abuse028_x264`. All operating points have very low GT coverage and/or the score curve has weak response, so post-processing cannot reliably recover the event.

## 8. Failure Taxonomy

Main failure modes are score-supported misses, score-supported fragmentation, over-merged predictions, score-unsupported GT, and boundary mismatch. See `failure_taxonomy.md` for details.

## 9. Unsupported Coverage Interpretation

Score-unsupported GT coverage should not be interpreted as purely negative, because human annotations may include event-level context or weakly expressed anomalies that are not fully reflected in the score curve. However, since the current method only operates on the anomaly score signal, excessive unsupported coverage together with high predicted duration ratio and low predicted GT fraction indicates likely over-extension. Therefore, unsupported coverage is treated as a diagnostic constraint rather than an objective to be minimized independently.

## 10. Limitations

- No held-out validation split was used.
- This is offline post-processing only.
- Score-unsupported GT cannot be reliably recovered by score-only post-processing.
- Supportable/unsupportable labels come from prior score-support classification.
- Human GT and score evidence may encode different temporal semantics.

## 11. Conclusion

The project moved from simple threshold/peak post-processing toward score-level temporal evidence decomposition and interval reconstruction. The current dataset supports two operating points rather than one universal optimum: a recall-oriented configuration for broader event recovery and a strict-oriented configuration for conservative reporting.