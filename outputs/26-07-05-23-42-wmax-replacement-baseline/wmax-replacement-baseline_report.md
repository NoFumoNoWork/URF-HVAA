# Wmax Replacement Baseline Experiment

This experiment compares three single-interval strategies with the same window length as the original Wmax:

- `wmax`: original highest-average sliding window.
- `random_same_length`: random interval with the same length, repeated multiple times.
- `peak_expand`: expand a same-length interval around the maximum score peak; if multiple max peaks exist, choose the expanded interval with the highest mean anomaly score.

## Results

| Method | Segment miss rate | Video any miss rate | Mean coverage |
|---|---:|---:|---:|
| wmax | 61.98% | 59.22% | 0.255 |
| peak_expand | 56.67% | 51.88% | 0.295 |
| random_same_length mean (100 runs) | 65.49% ± 0.76% | 62.28% ± 1.14% | 0.212 ± 0.006 |

## Interpretation

- Random same-length intervals estimate how much coverage comes from window length and anomaly density alone.
- Peak-expanded intervals test whether anchoring on the strongest score point is enough, independent of averaging all candidate windows.
- If peak expansion approaches Wmax, the original averaging window mainly selects around score peaks. If it underperforms Wmax, the surrounding score distribution matters.