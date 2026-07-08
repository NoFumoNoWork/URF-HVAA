# Experiment Artifact Layout

New experiment artifacts should be archived under:

`outputs/yy-mm-dd-hh-min-test-name/`

Each archive folder should contain:

- `<test-name>_report.md`
- `programs/`
- `outputs/`
- `MANIFEST.md`

## Current Archives

- `outputs/26-07-05-17-46-multi-anomaly-miss/`
- `outputs/26-07-05-21-46-long-video-diagnosis/`
- `outputs/26-07-05-23-42-wmax-replacement-baseline/`
- `outputs/26-07-06-00-21-adaptive-interval-selection/`
- `outputs/26-07-06-00-37-adaptive-param-grid/`
- `outputs/26-07-06-08-39-hierarchical-intervals/`
- `outputs/26-07-07-01-08-peak-aware-refinement/`
- `outputs/26-07-07-14-43-gt-score-window-curves/`
- `outputs/26-07-07-15-14-gt-score-alignment-analysis/`
- `outputs/26-07-07-15-59-interval-method-evaluation/`
- `outputs/26-07-07-spectral-score-decomposition/`
- `outputs/26-07-07-18-52-spectral-param-scan/`
- `outputs/26-07-07-19-33-spectral-ablation-study/`
- `outputs/26-07-07-19-51-spectral-pipeline-ablation/`
- `outputs/26-07-07-20-14-spectral-final-materials/`
- `outputs/26-07-07-20-44-final-report-summary/`
- `outputs/26-07-07-21-07-final-report-factual-package-multigt-cases/`
- `outputs/26-07-07-21-20-support-labeled-multigt-cases/`
- `outputs/26-07-07-22-08-interval-eval-low-fp/`
- `outputs/26-07-07-22-31-negative-evidence-valley/`
- `outputs/26-07-07-22-50-low-fp-ablation-scan/`: final Low-FP with valley cut local ablation/scan archive, including root report copy, reports, diagnostics, plots, manifest, and generator copy.
- `outputs/26-07-08-11-15-low-fp-visualization/`: low-FP final per-video visualization archive with four-row plots, index CSV, summary JSON, manifest, and generator copy.

## Non-Archive Report Packages

- `summaries/report_factual_package/`: final report factual package requested by `AGENT.md`; intentionally not a timestamped experiment archive because the instruction requires this exact path.
- `figures/final_report/`: figures referenced by the factual package; intentionally not a timestamped archive for the same reason.
