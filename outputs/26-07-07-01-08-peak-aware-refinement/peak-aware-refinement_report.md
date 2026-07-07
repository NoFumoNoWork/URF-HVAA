# Peak-Aware Refinement Report

## Method

This run preserves raw anomaly scores for peak height and local maxima. The baseline is used only as local background; residual is `raw_score - baseline`.

## Config

- baseline: `median`, window=101, quantile=0.3
- peak detection: mad_k=2.5, min_width=3, min_distance=10
- peak expansion: stop_ratio=0.2, min_len=3
- rescue: min_prominence=0.2, min_area=0.3
- split: min_gap_len=50, low_score_quantile=0.4

## Summary

- videos processed: 1090
- total peaks: 1205
- total rescued intervals: 104
- total split gaps: 0
- videos with no peaks: 585

## Visualizations

- XD-Violence `v=38GQ9L2meyE__#1_label_B6-0-0`: ok outputs\26-07-07-01-08-peak-aware-refinement\outputs\peak_refinement\visualizations\XD-Violence\v_38GQ9L2meyE___1_label_B6-0-0_peak_refined.png
- XD-Violence `v=uQY15O3LKI0__#1_label_B6-0-0`: ok outputs\26-07-07-01-08-peak-aware-refinement\outputs\peak_refinement\visualizations\XD-Violence\v_uQY15O3LKI0___1_label_B6-0-0_peak_refined.png
- UCF-Crime `Assault010_x264`: ok outputs\26-07-07-01-08-peak-aware-refinement\outputs\peak_refinement\visualizations\UCF-Crime\Assault010_x264_peak_refined.png