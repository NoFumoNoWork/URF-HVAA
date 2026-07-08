# MANIFEST

- `final-report-factual-package-multigt-cases_report.md`: archive report.
- `outputs/summaries/report_factual_package/`: copied factual package.
- `outputs/figures/final_report/`: copied final report figures.
- `outputs/summaries/multigt_case_study_index.csv`: full multi-GT case index.
- `outputs/summaries/multigt_case_study_index.md`: readable multi-GT case index.
- `outputs/figures/multigt_case_studies/*.png`: per-video support-labeled multi-GT timeline plots.
- GT track colors: green=supportable, red=unsupportable, purple=uncertain.
- `programs/scripts/archive_report_package_and_plot_multigt_cases.py`: generator script.
- `programs/scripts/run_report_factual_package.py`: copied factual-package generator script when available.

## Summary JSON

```json
{
  "archive_dir": "outputs/26-07-07-21-20-support-labeled-multigt-cases",
  "source_package": "outputs/26-07-07-20-44-final-report-summary",
  "copied_package": {
    "summary_file_count": 21,
    "figure_file_count": 12
  },
  "multigt_video_count": 276,
  "multigt_plot_count": 276,
  "missing_plot_count": 0,
  "prediction_warning_count": 24,
  "gt_track_supportability_colors": {
    "supportable": "green",
    "unsupportable": "red",
    "uncertain": "purple"
  }
}
```