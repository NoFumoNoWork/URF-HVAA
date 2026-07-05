# Adaptive Parameter Grid Sweep

Grid:

- threshold_percentile: 75, 80, 85, 90
- post_filter_percentile: none, 50, 75
- merge_gap: 150, 300, 600
- merge_iou: 0.3, 0.5

## Selected Points

| Role | method_config | miss_rate | mean_coverage | mean_intervals/video | mean_duration | redundancy |
|---|---|---:|---:|---:|---:|---|
| conservative_adaptive | tp80_postnone_gap150_iou0p3 (strict target satisfied) | 16.64% | 0.774 | 1.484 | 1359.5 | gt_pred_mean=0.893831;pred_gt_mean=1.311579 |
| balanced_adaptive | tp75_postnone_gap150_iou0p3 (nearest feasible; no grid point has 2-4 mean intervals/video) | 11.19% | 0.837 | 1.428 | 1573.0 | gt_pred_mean=0.939742;pred_gt_mean=1.43326 |
| recall_adaptive | tp75_postnone_gap600_iou0p3 (nearest feasible by miss rate; no grid point has 5-7 mean intervals/video) | 9.04% | 0.876 | 1.155 | 2025.8 | gt_pred_mean=0.926829;pred_gt_mean=1.748309 |
| best_overall | tp75_postnone_gap600_iou0p3 (lowest miss rate in scanned grid) | 9.04% | 0.876 | 1.155 | 2025.8 | gt_pred_mean=0.926829;pred_gt_mean=1.748309 |

## Feasibility Notes

- Scanned configs produced mean intervals/video in `1.155 - 1.617`.
- Therefore this grid does not contain strict balanced points with 2-4 mean intervals/video.
- It also does not contain strict recall points with 5-7 mean intervals/video.
- To reach those output-count regimes, the next sweep should reduce merging strength, lower or disable gap merge, or emit clusters before final merging.

## Full Grid

| method_config | miss_rate | mean_coverage | mean_intervals/video | mean_duration | redundancy |
|---|---:|---:|---:|---:|---|
| tp75_postnone_gap600_iou0p3 | 9.04% | 0.876 | 1.155 | 2025.8 | gt_pred_mean=0.926829;pred_gt_mean=1.748309 |
| tp75_postnone_gap600_iou0p5 | 9.04% | 0.876 | 1.155 | 2025.8 | gt_pred_mean=0.926829;pred_gt_mean=1.748309 |
| tp75_post50_gap600_iou0p3 | 9.04% | 0.876 | 1.155 | 2025.8 | gt_pred_mean=0.926829;pred_gt_mean=1.748309 |
| tp75_post50_gap600_iou0p5 | 9.04% | 0.876 | 1.155 | 2025.8 | gt_pred_mean=0.926829;pred_gt_mean=1.748309 |
| tp75_post75_gap600_iou0p3 | 9.04% | 0.876 | 1.155 | 2025.8 | gt_pred_mean=0.926829;pred_gt_mean=1.748309 |
| tp75_post75_gap600_iou0p5 | 9.04% | 0.876 | 1.155 | 2025.8 | gt_pred_mean=0.926829;pred_gt_mean=1.748309 |
| tp80_postnone_gap600_iou0p3 | 14.71% | 0.809 | 1.203 | 1755.7 | gt_pred_mean=0.880918;pred_gt_mean=1.594805 |
| tp80_postnone_gap600_iou0p5 | 14.71% | 0.809 | 1.203 | 1755.7 | gt_pred_mean=0.880918;pred_gt_mean=1.594805 |
| tp80_post50_gap600_iou0p3 | 14.71% | 0.809 | 1.203 | 1755.7 | gt_pred_mean=0.880918;pred_gt_mean=1.594805 |
| tp80_post50_gap600_iou0p5 | 14.71% | 0.809 | 1.203 | 1755.7 | gt_pred_mean=0.880918;pred_gt_mean=1.594805 |
| tp80_post75_gap600_iou0p3 | 14.71% | 0.809 | 1.203 | 1755.7 | gt_pred_mean=0.880918;pred_gt_mean=1.594805 |
| tp80_post75_gap600_iou0p5 | 14.71% | 0.809 | 1.203 | 1755.7 | gt_pred_mean=0.880918;pred_gt_mean=1.594805 |
| tp85_postnone_gap600_iou0p3 | 19.37% | 0.750 | 1.228 | 1503.4 | gt_pred_mean=0.833572;pred_gt_mean=1.478372 |
| tp85_postnone_gap600_iou0p5 | 19.37% | 0.750 | 1.228 | 1503.4 | gt_pred_mean=0.833572;pred_gt_mean=1.478372 |
| tp85_post50_gap600_iou0p3 | 19.37% | 0.750 | 1.228 | 1503.4 | gt_pred_mean=0.833572;pred_gt_mean=1.478372 |
| tp85_post50_gap600_iou0p5 | 19.37% | 0.750 | 1.228 | 1503.4 | gt_pred_mean=0.833572;pred_gt_mean=1.478372 |
| tp85_post75_gap600_iou0p3 | 19.37% | 0.750 | 1.228 | 1503.4 | gt_pred_mean=0.833572;pred_gt_mean=1.478372 |
| tp85_post75_gap600_iou0p5 | 19.37% | 0.750 | 1.228 | 1503.4 | gt_pred_mean=0.833572;pred_gt_mean=1.478372 |
| tp90_postnone_gap600_iou0p3 | 25.97% | 0.669 | 1.255 | 1216.7 | gt_pred_mean=0.771879;pred_gt_mean=1.339975 |
| tp90_postnone_gap600_iou0p5 | 25.97% | 0.669 | 1.255 | 1216.7 | gt_pred_mean=0.771879;pred_gt_mean=1.339975 |
| tp90_post50_gap600_iou0p3 | 25.97% | 0.669 | 1.255 | 1216.7 | gt_pred_mean=0.771879;pred_gt_mean=1.339975 |
| tp90_post50_gap600_iou0p5 | 25.97% | 0.669 | 1.255 | 1216.7 | gt_pred_mean=0.771879;pred_gt_mean=1.339975 |
| tp90_post75_gap600_iou0p3 | 25.97% | 0.669 | 1.255 | 1216.7 | gt_pred_mean=0.771879;pred_gt_mean=1.339975 |
| tp90_post75_gap600_iou0p5 | 25.97% | 0.669 | 1.255 | 1216.7 | gt_pred_mean=0.771879;pred_gt_mean=1.339975 |
| tp75_postnone_gap300_iou0p3 | 10.11% | 0.856 | 1.295 | 1756.8 | gt_pred_mean=0.934003;pred_gt_mean=1.570567 |
| tp75_postnone_gap300_iou0p5 | 10.11% | 0.856 | 1.295 | 1756.8 | gt_pred_mean=0.934003;pred_gt_mean=1.570567 |
| tp75_post50_gap300_iou0p3 | 10.11% | 0.856 | 1.295 | 1756.8 | gt_pred_mean=0.934003;pred_gt_mean=1.570567 |
| tp75_post50_gap300_iou0p5 | 10.11% | 0.856 | 1.295 | 1756.8 | gt_pred_mean=0.934003;pred_gt_mean=1.570567 |
| tp75_post75_gap300_iou0p3 | 10.11% | 0.856 | 1.295 | 1756.8 | gt_pred_mean=0.934003;pred_gt_mean=1.570567 |
| tp75_post75_gap300_iou0p5 | 10.11% | 0.856 | 1.295 | 1756.8 | gt_pred_mean=0.934003;pred_gt_mean=1.570567 |
| tp80_postnone_gap300_iou0p3 | 16.00% | 0.787 | 1.355 | 1511.2 | gt_pred_mean=0.883788;pred_gt_mean=1.420992 |
| tp80_postnone_gap300_iou0p5 | 16.00% | 0.787 | 1.355 | 1511.2 | gt_pred_mean=0.883788;pred_gt_mean=1.420992 |
| tp80_post50_gap300_iou0p3 | 16.00% | 0.787 | 1.355 | 1511.2 | gt_pred_mean=0.883788;pred_gt_mean=1.420992 |
| tp80_post50_gap300_iou0p5 | 16.00% | 0.787 | 1.355 | 1511.2 | gt_pred_mean=0.883788;pred_gt_mean=1.420992 |
| tp80_post75_gap300_iou0p3 | 16.00% | 0.787 | 1.355 | 1511.2 | gt_pred_mean=0.883788;pred_gt_mean=1.420992 |
| tp80_post75_gap300_iou0p5 | 16.00% | 0.787 | 1.355 | 1511.2 | gt_pred_mean=0.883788;pred_gt_mean=1.420992 |
| tp85_postnone_gap300_iou0p3 | 20.95% | 0.723 | 1.397 | 1267.0 | gt_pred_mean=0.840746;pred_gt_mean=1.310962 |
| tp85_postnone_gap300_iou0p5 | 20.95% | 0.723 | 1.397 | 1267.0 | gt_pred_mean=0.840746;pred_gt_mean=1.310962 |
| tp85_post50_gap300_iou0p3 | 20.95% | 0.723 | 1.397 | 1267.0 | gt_pred_mean=0.840746;pred_gt_mean=1.310962 |
| tp85_post50_gap300_iou0p5 | 20.95% | 0.723 | 1.397 | 1267.0 | gt_pred_mean=0.840746;pred_gt_mean=1.310962 |
| tp85_post75_gap300_iou0p3 | 20.95% | 0.723 | 1.397 | 1267.0 | gt_pred_mean=0.840746;pred_gt_mean=1.310962 |
| tp85_post75_gap300_iou0p5 | 20.95% | 0.723 | 1.397 | 1267.0 | gt_pred_mean=0.840746;pred_gt_mean=1.310962 |
| tp90_postnone_gap300_iou0p3 | 27.55% | 0.640 | 1.417 | 1028.3 | gt_pred_mean=0.775466;pred_gt_mean=1.191841 |
| tp90_postnone_gap300_iou0p5 | 27.55% | 0.640 | 1.417 | 1028.3 | gt_pred_mean=0.775466;pred_gt_mean=1.191841 |
| tp90_post50_gap300_iou0p3 | 27.55% | 0.640 | 1.417 | 1028.3 | gt_pred_mean=0.775466;pred_gt_mean=1.191841 |
| tp90_post50_gap300_iou0p5 | 27.55% | 0.640 | 1.417 | 1028.3 | gt_pred_mean=0.775466;pred_gt_mean=1.191841 |
| tp90_post75_gap300_iou0p3 | 27.55% | 0.640 | 1.417 | 1028.3 | gt_pred_mean=0.775466;pred_gt_mean=1.191841 |
| tp90_post75_gap300_iou0p5 | 27.55% | 0.640 | 1.417 | 1028.3 | gt_pred_mean=0.775466;pred_gt_mean=1.191841 |
| tp75_postnone_gap150_iou0p3 | 11.19% | 0.837 | 1.428 | 1573.0 | gt_pred_mean=0.939742;pred_gt_mean=1.43326 |
| tp75_postnone_gap150_iou0p5 | 11.19% | 0.837 | 1.428 | 1573.0 | gt_pred_mean=0.939742;pred_gt_mean=1.43326 |
| tp75_post50_gap150_iou0p3 | 11.19% | 0.837 | 1.428 | 1573.0 | gt_pred_mean=0.939742;pred_gt_mean=1.43326 |
| tp75_post50_gap150_iou0p5 | 11.19% | 0.837 | 1.428 | 1573.0 | gt_pred_mean=0.939742;pred_gt_mean=1.43326 |
| tp75_post75_gap150_iou0p3 | 11.19% | 0.837 | 1.428 | 1573.0 | gt_pred_mean=0.939742;pred_gt_mean=1.43326 |
| tp75_post75_gap150_iou0p5 | 11.19% | 0.837 | 1.428 | 1573.0 | gt_pred_mean=0.939742;pred_gt_mean=1.43326 |
| tp80_postnone_gap150_iou0p3 | 16.64% | 0.774 | 1.484 | 1359.5 | gt_pred_mean=0.893831;pred_gt_mean=1.311579 |
| tp80_postnone_gap150_iou0p5 | 16.64% | 0.774 | 1.484 | 1359.5 | gt_pred_mean=0.893831;pred_gt_mean=1.311579 |
| tp80_post50_gap150_iou0p3 | 16.64% | 0.774 | 1.484 | 1359.5 | gt_pred_mean=0.893831;pred_gt_mean=1.311579 |
| tp80_post50_gap150_iou0p5 | 16.64% | 0.774 | 1.484 | 1359.5 | gt_pred_mean=0.893831;pred_gt_mean=1.311579 |
| tp80_post75_gap150_iou0p3 | 16.64% | 0.774 | 1.484 | 1359.5 | gt_pred_mean=0.893831;pred_gt_mean=1.311579 |
| tp80_post75_gap150_iou0p5 | 16.64% | 0.774 | 1.484 | 1359.5 | gt_pred_mean=0.893831;pred_gt_mean=1.311579 |
| tp85_postnone_gap150_iou0p3 | 21.74% | 0.707 | 1.541 | 1128.3 | gt_pred_mean=0.850789;pred_gt_mean=1.20284 |
| tp85_postnone_gap150_iou0p5 | 21.74% | 0.707 | 1.541 | 1128.3 | gt_pred_mean=0.850789;pred_gt_mean=1.20284 |
| tp85_post50_gap150_iou0p3 | 21.74% | 0.707 | 1.541 | 1128.3 | gt_pred_mean=0.850789;pred_gt_mean=1.20284 |
| tp85_post50_gap150_iou0p5 | 21.74% | 0.707 | 1.541 | 1128.3 | gt_pred_mean=0.850789;pred_gt_mean=1.20284 |
| tp85_post75_gap150_iou0p3 | 21.74% | 0.707 | 1.541 | 1128.3 | gt_pred_mean=0.850789;pred_gt_mean=1.20284 |
| tp85_post75_gap150_iou0p5 | 21.74% | 0.707 | 1.541 | 1128.3 | gt_pred_mean=0.850789;pred_gt_mean=1.20284 |
| tp90_postnone_gap150_iou0p3 | 28.19% | 0.618 | 1.617 | 874.1 | gt_pred_mean=0.804161;pred_gt_mean=1.083092 |
| tp90_postnone_gap150_iou0p5 | 28.19% | 0.618 | 1.617 | 874.1 | gt_pred_mean=0.804161;pred_gt_mean=1.083092 |
| tp90_post50_gap150_iou0p3 | 28.19% | 0.618 | 1.617 | 874.1 | gt_pred_mean=0.804161;pred_gt_mean=1.083092 |
| tp90_post50_gap150_iou0p5 | 28.19% | 0.618 | 1.617 | 874.1 | gt_pred_mean=0.804161;pred_gt_mean=1.083092 |
| tp90_post75_gap150_iou0p3 | 28.19% | 0.618 | 1.617 | 874.1 | gt_pred_mean=0.804161;pred_gt_mean=1.083092 |
| tp90_post75_gap150_iou0p5 | 28.19% | 0.618 | 1.617 | 874.1 | gt_pred_mean=0.804161;pred_gt_mean=1.083092 |