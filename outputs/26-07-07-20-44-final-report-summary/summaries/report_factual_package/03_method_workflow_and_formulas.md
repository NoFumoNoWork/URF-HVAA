# Method Workflow And Formulas

## Workflow

1. Score loading: `scripts/run_spectral_param_scan.py::precompute_curves` reads score JSON paths from the inventory, using `run_spectral_score_decomposition.py::load_score_series`.
2. Score preprocessing: SG curves are created by `compute_sg`, airPLS baselines/residuals by `airpls`, and rolling trends by `rolling_mean_by_frames`.
3. Spectral evidence extraction: raw, SG, residual, trend, residual peak-count, interval length, and low-residual ratio are computed in `feature_rows`.
4. Candidate interval generation: `generate_candidates_for_video` adds `SG-Peak`, `AirPLS-Residual`, and `Trend-Guided`; existing `Peak-Aware-Refined` and `Hierarchical-Merged` are included by `make_candidates_for_config`.
5. Fusion scoring: `scripts/run_spectral_ablation_study.py::score_fusion` normalizes feature columns and applies direct weights and penalties.
6. Filtering and merging: candidates with `fusion_score >= fusion_threshold` are retained and then merged by `merge_with_gap`.
7. Evaluation: final intervals are evaluated by `evaluate_methods`, `evaluate_one_video`, and `aggregate_rows`.
8. Operating-point selection: `build_final_configs` defines recall-oriented and strict-oriented candidates, summarized in final comparison tables.

## Evidence And Penalties

- `raw_evidence`: normalized `raw_max` from interval raw score values.
- `sg_evidence`: normalized `sg_max` from selected SG curve; SG smoothing and SG-Peak candidates are retained, but final recommended direct SG weight is 0.
- `residual_evidence`: normalized `airpls_residual_max`.
- `trend_evidence`: normalized `trend_mean` over the selected rolling window.
- `peak_count_evidence`: normalized `residual_peak_count`.
- `length_penalty`: normalized `interval_length`.
- `low_residual_penalty`: unnormalized `low_residual_ratio`, subtracted directly.

## Current Fusion Formula

`fusion_score = raw_weight*N(raw_max) + sg_weight*N(sg_max) + residual_weight*N(airpls_residual_max) + trend_weight*N(trend_mean) + peak_count_weight*N(residual_peak_count) - length_penalty_weight*N(interval_length) - low_residual_penalty_weight*low_residual_ratio`

Final recommendation facts:

- SG smoothing is retained.
- SG-Peak candidate generation is retained.
- Direct SG positive fusion weight is set to 0 in SG0, Recall, and Strict.
- Residual candidate generation is retained.
- Residual direct weight is not globally deleted; it is 0.25 in Recall and Strict selected points.
- Peak-count direct evidence is set to 0 in the selected Strict operating point.
- Length and low-residual penalties are retained.

See also `summaries/report_factual_package/03_method_workflow.mmd`.
