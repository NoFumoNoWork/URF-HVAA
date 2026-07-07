# Final Two Operating Points

These configurations are recommendations for the current dataset, not claims of held-out generalization.

## Recall trend0.5 SG0 residual0.25

- Candidate sources enabled: Peak-Aware-Refined, Hierarchical-Merged, SG-Peak, AirPLS-Residual, Trend-Guided.
- Candidate sources disabled: none.
- Direct fusion formula: `raw*raw + residual*residual + trend*trend + peak_count*peak_count - length_penalty*length - low_residual_penalty*low_residual; SG term computed but sg_weight=0`.
- Weights: raw=0.25, sg=0.0, residual=0.25, trend=0.15, peak_count=0.1, length_penalty=0.15, low_residual_penalty=0.1.
- Thresholds/windows: fusion_threshold=0.35, trend_threshold=0.5, trend_window=100, score_threshold=0.6.
- Signal parameters: SG window=17, SG polyorder=2, airPLS lambda=1000, airPLS order=2, airPLS itermax=20, peak_mad_k=3.0, residual_mad_k=3.0.
- Interval parameters: merge_gap_frames=48, peak_stop_ratio=0.25.
- Run command: `python scripts\run_spectral_pipeline_ablation.py`.
- Output directory: `outputs\26-07-07-20-14-spectral-final-materials`.
- Final metrics: GT=0.753, purity=0.520, supportable=0.768, unsupported=0.225, duration=0.566, strict=0.469.

## Raw trend residual penalties SG0 residual0.25 peak0

- Candidate sources enabled: Peak-Aware-Refined, Hierarchical-Merged, SG-Peak, AirPLS-Residual, Trend-Guided.
- Candidate sources disabled: none.
- Direct fusion formula: `raw*raw + residual*residual + trend*trend + peak_count*peak_count - length_penalty*length - low_residual_penalty*low_residual; SG term computed but sg_weight=0`.
- Weights: raw=0.25, sg=0.0, residual=0.25, trend=0.15, peak_count=0.0, length_penalty=0.15, low_residual_penalty=0.1.
- Thresholds/windows: fusion_threshold=0.35, trend_threshold=0.6, trend_window=100, score_threshold=0.6.
- Signal parameters: SG window=17, SG polyorder=2, airPLS lambda=1000, airPLS order=2, airPLS itermax=20, peak_mad_k=3.0, residual_mad_k=3.0.
- Interval parameters: merge_gap_frames=48, peak_stop_ratio=0.25.
- Run command: `python scripts\run_spectral_pipeline_ablation.py`.
- Output directory: `outputs\26-07-07-20-14-spectral-final-materials`.
- Final metrics: GT=0.710, purity=0.527, supportable=0.725, unsupported=0.187, duration=0.527, strict=0.455.
