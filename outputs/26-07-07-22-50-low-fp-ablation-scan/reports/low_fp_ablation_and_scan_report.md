# Low-FP Ablation And Scan Report

## 1. Final Configuration

The fixed final configuration is `low_fp_with_valley_cut_final`: a Low-FP base configuration plus negative-evidence / valley post-processing. It should be interpreted as a precision-first operating point, not as the recall-optimal method.

Key parameters: `fusion_threshold=0.38`, `merge_gap_frames=32`, `post_min_duration=32`, `post_min_raw_max=0.60`, `length_penalty_weight=0.22`, `low_residual_penalty_weight=0.15`, `residual_weight=0.25`, `trend_weight=0.20`, `min_normal_duration=96`, `protect_sgt_ucgt=True`.

## 2. Main Result

- s-GT Recall: 0.705.
- uc-GT Recall: 0.443.
- Eval Recall: 0.696.
- GT Precision: 0.552.
- FP Duration: 347264.
- FP Ratio in PI: 0.448.

This operating point keeps FP lower than the looser settings, but it gives up recall compared with recall-oriented configurations. That is the intended trade-off.

## 3. Ablation Study

Valley cut is a lightweight refinement: compared with `no_valley_cut`, it removes 3300 FP frames, changes GT Precision by 0.002, and does not change Eval Recall at the displayed precision.

The main FP-control modules are the length penalty and low residual penalty. Removing length penalty raises FP Duration from 347264 to 383679; removing low residual penalty raises it to 386156.

Trend and residual components are recall-bearing evidence. The `no_trend_component` and `no_residual_component` rows have lower FP mostly because they predict far fewer intervals; their Eval Recall collapses, so they should not be interpreted as useful FP improvements.

| Variant | PI Ratio | s-GT Recall | uc-GT Recall | Eval Recall | GT Precision | FP Duration | FP Ratio in PI | Delta Eval Recall vs final_full | Delta GT Precision vs final_full | Delta FP Duration vs final_full |
|---|---|---|---|---|---|---|---|---|---|---|
| final_full | 0.489 | 0.705 | 0.443 | 0.696 | 0.552 | 347264 | 0.448 | 0.000 | 0.000 | 0 |
| no_valley_cut | 0.491 | 0.705 | 0.443 | 0.696 | 0.550 | 350564 | 0.450 | 0.000 | -0.002 | 3300 |
| no_length_penalty | 0.518 | 0.723 | 0.479 | 0.714 | 0.534 | 383679 | 0.466 | 0.018 | -0.018 | 36415 |
| no_low_residual_penalty | 0.519 | 0.719 | 0.490 | 0.711 | 0.532 | 386156 | 0.468 | 0.016 | -0.020 | 38892 |
| no_trend_component | 0.177 | 0.235 | 0.149 | 0.232 | 0.517 | 133298 | 0.483 | -0.463 | -0.035 | -213966 |
| no_residual_component | 0.052 | 0.078 | 0.098 | 0.078 | 0.601 | 32116 | 0.399 | -0.617 | 0.049 | -315148 |
| no_sg_candidate | 0.480 | 0.695 | 0.436 | 0.686 | 0.555 | 338201 | 0.445 | -0.010 | 0.003 | -9063 |
| no_post_min_duration | 0.491 | 0.707 | 0.443 | 0.698 | 0.551 | 349742 | 0.449 | 0.002 | -0.001 | 2478 |
| large_merge_gap | 0.498 | 0.716 | 0.444 | 0.706 | 0.550 | 355376 | 0.450 | 0.011 | -0.002 | 8112 |
| small_merge_gap | 0.483 | 0.700 | 0.437 | 0.691 | 0.554 | 341653 | 0.446 | -0.005 | 0.002 | -5611 |

## 4. Parameter Scan

The local scans vary one parameter around the final setting while keeping the rest fixed. They are explanatory local checks, not a global search.

The `fusion_threshold` scan shows the clearest recall-precision trade-off: increasing the threshold monotonically lowers FP Duration and raises GT Precision, while Eval Recall drops.

`valley_low_score_threshold` is insensitive in the tested range under the current constraints: all rows produce the same final PI metrics at the displayed precision.

`min_normal_duration=48` removes more FP than the final value, but it has a higher NI-over-sGT Ratio. The final `min_normal_duration=96` is therefore the more conservative safety choice.

### fusion_threshold

| Parameter | Value | PI Ratio | s-GT Recall | uc-GT Recall | Eval Recall | GT Precision | FP Duration | FP Ratio in PI | us-GT Coverage | FP Removed by NI | TP Lost by NI | NI-over-sGT Ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fusion_threshold | 0.340 | 0.520 | 0.724 | 0.486 | 0.715 | 0.532 | 386678 | 0.468 | 0.149 | 4664.000 | 0.000 | 0.071 |
| fusion_threshold | 0.360 | 0.506 | 0.715 | 0.467 | 0.707 | 0.541 | 369151 | 0.459 | 0.107 | 3920.000 | 0.000 | 0.071 |
| fusion_threshold | 0.380 | 0.489 | 0.705 | 0.443 | 0.696 | 0.552 | 347264 | 0.448 | 0.104 | 3300.000 | 0.000 | 0.071 |
| fusion_threshold | 0.400 | 0.468 | 0.689 | 0.437 | 0.680 | 0.564 | 322879 | 0.436 | 0.086 | 2344.000 | 0.000 | 0.071 |
| fusion_threshold | 0.420 | 0.445 | 0.672 | 0.415 | 0.663 | 0.580 | 294817 | 0.420 | 0.083 | 1776.000 | 0.000 | 0.071 |

### merge_gap_frames

| Parameter | Value | PI Ratio | s-GT Recall | uc-GT Recall | Eval Recall | GT Precision | FP Duration | FP Ratio in PI | us-GT Coverage | FP Removed by NI | TP Lost by NI | NI-over-sGT Ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| merge_gap_frames | 0 | 0.483 | 0.700 | 0.437 | 0.691 | 0.554 | 341653 | 0.446 | 0.102 | 3276.000 | 0.000 | 0.071 |
| merge_gap_frames | 16 | 0.486 | 0.702 | 0.440 | 0.693 | 0.554 | 343736 | 0.446 | 0.104 | 3284.000 | 0.000 | 0.071 |
| merge_gap_frames | 32 | 0.489 | 0.705 | 0.443 | 0.696 | 0.552 | 347264 | 0.448 | 0.104 | 3300.000 | 0.000 | 0.071 |
| merge_gap_frames | 64 | 0.494 | 0.711 | 0.442 | 0.702 | 0.551 | 352363 | 0.449 | 0.104 | 3364.000 | 0.000 | 0.071 |
| merge_gap_frames | 96 | 0.498 | 0.716 | 0.444 | 0.706 | 0.550 | 355376 | 0.450 | 0.132 | 3380.000 | 0.000 | 0.071 |

### post_min_duration

| Parameter | Value | PI Ratio | s-GT Recall | uc-GT Recall | Eval Recall | GT Precision | FP Duration | FP Ratio in PI | us-GT Coverage | FP Removed by NI | TP Lost by NI | NI-over-sGT Ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| post_min_duration | 16 | 0.491 | 0.707 | 0.443 | 0.698 | 0.551 | 349726 | 0.449 | 0.104 | 3292.000 | 0.000 | 0.071 |
| post_min_duration | 32 | 0.489 | 0.705 | 0.443 | 0.696 | 0.552 | 347264 | 0.448 | 0.104 | 3300.000 | 0.000 | 0.071 |
| post_min_duration | 48 | 0.487 | 0.703 | 0.443 | 0.694 | 0.552 | 345954 | 0.448 | 0.104 | 3340.000 | 0.000 | 0.071 |
| post_min_duration | 64 | 0.485 | 0.701 | 0.443 | 0.692 | 0.553 | 343768 | 0.447 | 0.102 | 3396.000 | 0.000 | 0.071 |
| post_min_duration | 96 | 0.482 | 0.698 | 0.438 | 0.689 | 0.554 | 340988 | 0.446 | 0.102 | 3644.000 | 5.000 | 0.071 |

### post_min_raw_max

| Parameter | Value | PI Ratio | s-GT Recall | uc-GT Recall | Eval Recall | GT Precision | FP Duration | FP Ratio in PI | us-GT Coverage | FP Removed by NI | TP Lost by NI | NI-over-sGT Ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| post_min_raw_max | 0.500 | 0.489 | 0.705 | 0.443 | 0.696 | 0.552 | 347264 | 0.448 | 0.104 | 3300.000 | 0.000 | 0.071 |
| post_min_raw_max | 0.550 | 0.489 | 0.705 | 0.443 | 0.696 | 0.552 | 347264 | 0.448 | 0.104 | 3300.000 | 0.000 | 0.071 |
| post_min_raw_max | 0.600 | 0.489 | 0.705 | 0.443 | 0.696 | 0.552 | 347264 | 0.448 | 0.104 | 3300.000 | 0.000 | 0.071 |
| post_min_raw_max | 0.650 | 0.489 | 0.705 | 0.443 | 0.695 | 0.552 | 347232 | 0.448 | 0.104 | 3300.000 | 0.000 | 0.071 |
| post_min_raw_max | 0.700 | 0.489 | 0.705 | 0.443 | 0.695 | 0.552 | 347232 | 0.448 | 0.104 | 3300.000 | 0.000 | 0.071 |

### valley_low_score_threshold

| Parameter | Value | PI Ratio | s-GT Recall | uc-GT Recall | Eval Recall | GT Precision | FP Duration | FP Ratio in PI | us-GT Coverage | FP Removed by NI | TP Lost by NI | NI-over-sGT Ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| low_score_threshold | 0.250 | 0.489 | 0.705 | 0.443 | 0.696 | 0.552 | 347264 | 0.448 | 0.104 | 3300.000 | 0.000 | 0.072 |
| low_score_threshold | 0.300 | 0.489 | 0.705 | 0.443 | 0.696 | 0.552 | 347264 | 0.448 | 0.104 | 3300.000 | 0.000 | 0.071 |
| low_score_threshold | 0.350 | 0.489 | 0.705 | 0.443 | 0.696 | 0.552 | 347264 | 0.448 | 0.104 | 3300.000 | 0.000 | 0.071 |
| low_score_threshold | 0.400 | 0.489 | 0.705 | 0.443 | 0.696 | 0.552 | 347264 | 0.448 | 0.104 | 3300.000 | 0.000 | 0.071 |
| low_score_threshold | 0.450 | 0.489 | 0.705 | 0.443 | 0.696 | 0.552 | 347264 | 0.448 | 0.104 | 3300.000 | 0.000 | 0.071 |

### valley_min_normal_duration

| Parameter | Value | PI Ratio | s-GT Recall | uc-GT Recall | Eval Recall | GT Precision | FP Duration | FP Ratio in PI | us-GT Coverage | FP Removed by NI | TP Lost by NI | NI-over-sGT Ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| min_normal_duration | 48 | 0.482 | 0.705 | 0.443 | 0.696 | 0.560 | 336534 | 0.440 | 0.096 | 14030.000 | 0.000 | 0.086 |
| min_normal_duration | 72 | 0.487 | 0.705 | 0.443 | 0.696 | 0.554 | 344832 | 0.446 | 0.104 | 5732.000 | 0.000 | 0.070 |
| min_normal_duration | 96 | 0.489 | 0.705 | 0.443 | 0.696 | 0.552 | 347264 | 0.448 | 0.104 | 3300.000 | 0.000 | 0.071 |
| min_normal_duration | 128 | 0.490 | 0.705 | 0.443 | 0.696 | 0.551 | 348680 | 0.449 | 0.104 | 1884.000 | 0.000 | 0.070 |
| min_normal_duration | 160 | 0.490 | 0.705 | 0.443 | 0.696 | 0.551 | 349356 | 0.449 | 0.107 | 1208.000 | 0.000 | 0.086 |

## 5. Near/Far FP Diagnostic

Near/far FP splits `PI intersect nonGT` into FP close to any GT interval versus FP outside a buffered GT neighborhood.

| method | margin | PI Duration | FP Duration | near_GT_FP_duration | far_GT_FP_duration | near_GT_FP_ratio_in_FP | far_GT_FP_ratio_in_FP |
|---|---|---|---|---|---|---|---|
| Low-FP with valley cut final | 32 | 775204 | 347264 | 52643 | 294621 | 0.152 | 0.848 |
| Low-FP with valley cut final | 64 | 775204 | 347264 | 96485 | 250779 | 0.278 | 0.722 |
| Low-FP with valley cut final | 128 | 775204 | 347264 | 157359 | 189905 | 0.453 | 0.547 |

At the main margin of 64 frames, near-GT FP is 96485 frames (0.278 of FP), while far-GT FP is 250779 frames (0.722 of FP). The high FP Ratio is therefore dominated by far-GT FP, not just GT boundary spillover.

## 6. Limitations

- FP Ratio remains nontrivial even after valley cut.
- GT/VAD evidence is layered and boundary-uncertain.
- Valley detector can touch s-GT; `protect_sgt_ucgt=True` is required.
- Low-FP sacrifices Eval Recall, so it is suited for low-false-positive reporting rather than maximum recall.