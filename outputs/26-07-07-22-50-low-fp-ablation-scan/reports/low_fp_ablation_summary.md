# Low-FP Ablation Summary

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
