# Ablation Summary Facts

## Direct Fusion Ablation

Source: `outputs/26-07-07-19-33-spectral-ablation-study/summaries/ablation_module_results.csv`.

| run_name | GT_coverage | predicted_GT_fraction | supportable_gt_coverage | unsupportable_gt_coverage | predicted_duration_ratio | stricter_balanced_score |
|---|---|---|---|---|---|---|
| Full Spectral-Fusion-Refined | 0.729 | 0.506 | 0.742 | 0.247 | 0.562 | 0.450 |
| w/o SG evidence | 0.712 | 0.523 | 0.728 | 0.187 | 0.533 | 0.454 |
| w/o airPLS residual evidence | 0.665 | 0.576 | 0.683 | 0.075 | 0.454 | 0.461 |
| w/o trend evidence | 0.723 | 0.513 | 0.737 | 0.226 | 0.552 | 0.452 |
| w/o peak-count evidence | 0.728 | 0.507 | 0.742 | 0.247 | 0.560 | 0.450 |
| w/o length penalty | 0.732 | 0.502 | 0.745 | 0.257 | 0.568 | 0.449 |
| w/o low-residual penalty | 0.732 | 0.501 | 0.744 | 0.277 | 0.569 | 0.446 |
| w/o both penalties | 0.738 | 0.500 | 0.751 | 0.277 | 0.575 | 0.449 |
| SG only | 0.000 | NA | 0.000 | 0.000 | 0.000 | 0.000 |
| residual only | 0.000 | NA | 0.000 | 0.000 | 0.000 | 0.000 |
| trend only | 0.000 | NA | 0.000 | 0.000 | 0.000 | 0.000 |
| peak-count only | 0.000 | NA | 0.000 | 0.000 | 0.000 | 0.000 |
| raw only | 0.000 | NA | 0.000 | 0.000 | 0.000 | 0.000 |
| raw + trend | 0.269 | 0.626 | 0.276 | 0.037 | 0.171 | 0.285 |
| raw + residual | 0.689 | 0.517 | 0.703 | 0.221 | 0.523 | 0.437 |
| raw + trend + residual | 0.723 | 0.509 | 0.737 | 0.261 | 0.555 | 0.447 |
| raw + trend + residual + length penalty | 0.717 | 0.513 | 0.731 | 0.226 | 0.546 | 0.449 |
| raw + trend + residual + both penalties | 0.710 | 0.527 | 0.725 | 0.187 | 0.527 | 0.455 |

## Pipeline-Level Ablation

Source: `outputs/26-07-07-19-51-spectral-pipeline-ablation/summaries/`.

SG ablation:

| run_name | GT_coverage | predicted_GT_fraction | supportable_gt_coverage | unsupportable_gt_coverage | predicted_duration_ratio | stricter_balanced_score |
|---|---|---|---|---|---|---|
| Full default | 0.729 | 0.506 | 0.742 | 0.247 | 0.562 | 0.450 |
| Spectral-Fusion-SG0 | 0.712 | 0.523 | 0.728 | 0.187 | 0.533 | 0.454 |
| Pipeline w/o SG candidates | 0.726 | 0.508 | 0.740 | 0.240 | 0.558 | 0.450 |
| Pipeline w/o SG candidates + SG0 | 0.705 | 0.526 | 0.720 | 0.187 | 0.525 | 0.452 |

Residual ablation:

| run_name | GT_coverage | predicted_GT_fraction | supportable_gt_coverage | unsupportable_gt_coverage | predicted_duration_ratio | stricter_balanced_score |
|---|---|---|---|---|---|---|
| Full default | 0.729 | 0.506 | 0.742 | 0.247 | 0.562 | 0.450 |
| w/o residual direct evidence | 0.665 | 0.576 | 0.683 | 0.075 | 0.454 | 0.461 |
| w/o residual candidates | 0.722 | 0.510 | 0.736 | 0.247 | 0.553 | 0.448 |
| w/o residual candidates + w/o residual direct evidence | 0.654 | 0.584 | 0.672 | 0.071 | 0.441 | 0.459 |
| SG0 + w/o residual direct evidence | 0.091 | 0.613 | 0.092 | 0.019 | 0.060 | 0.196 |
| SG0 + w/o residual candidates + w/o residual direct evidence | 0.080 | 0.610 | 0.080 | 0.019 | 0.053 | 0.189 |

## Required Answers

1. SG direct weight should be set to 0 for the selected final operating points.
2. SG smoothing should be retained because SG-Peak candidates remain enabled in SG0/Recall/Strict.
3. SG candidate generation should be retained; pipeline ablation separates it from direct SG weight.
4. Residual candidate generation should be retained.
5. Residual direct weight should not be globally deleted; selected Recall and Strict both use residual_weight=0.25.
6. Residual direct weight controls the recall/strict trade-off together with threshold and peak-count settings.
7. Peak-count direct evidence does not dominate the selected final configuration; Strict sets peak_count_weight=0.
8. Penalties are retained to control duration and low-residual interval expansion.
9. Full default uses SG direct weight; SG0 removes direct SG weight; Recall lowers trend threshold; Strict removes direct SG and peak-count weight while retaining residual and penalties.
