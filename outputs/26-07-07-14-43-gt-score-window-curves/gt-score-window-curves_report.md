# GT Interval Score Statistics And Window Curves

## Method

- For each manually annotated abnormal interval, score points satisfying `gt_start <= frame < gt_end` are collected.
- Statistics use the raw anomaly score values inside each GT interval.
- Variance is population variance, computed with `numpy.var`.
- Sliding-average curves are centered frame-window means over 300F, 100F, and 30F windows, plotted together with the raw curve.
- Curves are drawn in this order: 300F, 100F, 30F, then raw. The raw curve is light blue and placed on top so it remains visible even when close to the 30F curve.
- GT intervals are shown as light orange background spans in each plot.

## Outputs

- `outputs/gt_interval_score_stats.csv`
- `outputs/gt_interval_score_stats.json`
- `outputs/video_score_curve_inventory.csv`
- `outputs/score_curve_plot_manifest.json`
- `outputs/score_curve_plots/`
- `outputs/multi_gt_score_curve_plots/`

## Summary

- videos with GT abnormal intervals: 640
- videos with both GT and scores: 640
- GT abnormal intervals: 1394
- GT intervals with score points: 1393
- GT intervals without score points: 1
- plots generated: 640
- single-GT plots: 364
- multi-GT plots: 276
- plot misses: 0
- max_plots argument: all

## Dataset Breakdown

### XD-Violence

- videos with GT: 500
- videos with GT and scores: 500
- GT intervals: 1238
- GT intervals with scores: 1237
- GT intervals without scores: 1

### UCF-Crime

- videos with GT: 140
- videos with GT and scores: 140
- GT intervals: 156
- GT intervals with scores: 156
- GT intervals without scores: 0
