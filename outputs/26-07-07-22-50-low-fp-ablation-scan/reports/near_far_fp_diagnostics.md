# Near/Far FP Diagnostics

Near/far FP is computed only on false-positive duration, i.e. `PI intersect nonGT`. `near_GT_FP` is the part of FP lying inside `buffer(GT, margin)`, and `far_GT_FP` is the remaining FP outside that buffer.

Margin 64 frames is the main diagnostic setting; margins 32 and 128 frames are included as sensitivity checks.

| method | margin | PI Duration | FP Duration | near_GT_FP_duration | far_GT_FP_duration | near_GT_FP_ratio_in_FP | far_GT_FP_ratio_in_FP |
|---|---|---|---|---|---|---|---|
| Low-FP with valley cut final | 32 | 775204 | 347264 | 52643 | 294621 | 0.152 | 0.848 |
| Low-FP with valley cut final | 64 | 775204 | 347264 | 96485 | 250779 | 0.278 | 0.722 |
| Low-FP with valley cut final | 128 | 775204 | 347264 | 157359 | 189905 | 0.453 | 0.547 |

## Main Reading

At margin 64, near-GT FP is 96485 frames (0.278 of FP), while far-GT FP is 250779 frames (0.722 of FP).
Therefore the current high FP ratio is dominated by far-from-GT false positives rather than only boundary spillover around GT intervals.
