# Adaptive Interval Selection Report

## Method

The adaptive method slides over fixed windows `300, 600, 1200` and adaptive windows `max_frame//20, max_frame//10`, then scores each candidate with:

`proposal_score = 0.6 * mean_score + 0.3 * top20_mean_score + 0.1 * max_score`

Candidates above the within-video 85th percentile are retained, then overlapping or near-adjacent intervals are merged with IoU >= 0.3 or gap <= 150 frames. Final intervals use the highest-proposal candidate in each cluster as their score carrier.

## Parameters

- threshold_percentile: 85
- merge_iou: 0.3
- merge_gap: 150 frames
- min_duration: 60 frames
- post_filter_percentile: 75

## Coverage And Output Trade-Off

| Method | Segment miss rate | Video any miss rate | Mean coverage | Mean intervals/video |
|---|---:|---:|---:|---:|
| original_wmax | 61.98% | 59.22% | 0.255 | 1.000 |
| topk_k5 | 22.38% | 23.28% | 0.687 | 5.000 |
| topk_k10 | 5.67% | 7.81% | 0.902 | 10.000 |
| multiscale_k10 | 22.88% | 27.03% | 0.547 | 6.995 |
| adaptive | 21.74% | 24.53% | 0.707 | 1.541 |

## Adaptive Output Statistics

- mean_pred_intervals_per_video: 1.540625
- median_pred_intervals_per_video: 1.0
- max_pred_intervals_per_video: 5
- mean_pred_interval_duration: 1128.278905
- redundancy: `{'mean_pred_intervals_covering_each_gt': 0.850789, 'max_pred_intervals_covering_each_gt': 5, 'mean_gt_intervals_covered_by_each_pred': 1.20284, 'max_gt_intervals_covered_by_each_pred': 9}`

## Fragmentation

The mean interval count and redundancy statistics are the main fragmentation indicators. High interval count with low miss rate indicates recall-oriented behavior; high redundancy means multiple predicted intervals overlap the same GT event.

## Typical Cases

- XD-Violence `v=38GQ9L2meyE__#1_label_B6-0-0`: ok outputs\adaptive_timeline_plots\XD-Violence\v_38GQ9L2meyE___1_label_B6-0-0.png
- XD-Violence `v=uQY15O3LKI0__#1_label_B6-0-0`: ok outputs\adaptive_timeline_plots\XD-Violence\v_uQY15O3LKI0___1_label_B6-0-0.png
- UCF-Crime `Assault010_x264`: ok outputs\adaptive_timeline_plots\UCF-Crime\Assault010_x264.png

## Interpretation

- Adaptive intervals are not fixed-K; they trade more flexible event count for coverage.
- Compare adaptive against Wmax and Top-K to decide whether adaptive thresholding gives better recall per emitted interval.