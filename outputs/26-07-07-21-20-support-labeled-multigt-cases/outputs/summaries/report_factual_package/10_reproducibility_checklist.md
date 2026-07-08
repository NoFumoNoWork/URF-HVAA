# Reproducibility Checklist

- Key scripts: `scripts/run_spectral_param_scan.py`, `scripts/run_spectral_ablation_study.py`, `scripts/run_spectral_pipeline_ablation.py`, `scripts/run_spectral_final_materials.py`, `scripts/run_report_factual_package.py`.
- Key inputs: GT stats CSV, support classification CSV, video score inventory CSV, cached decomposition curves.
- Key outputs: parameter scan, ablation study, pipeline ablation, final materials, and `summaries/report_factual_package/`.
- Commands: `python scripts\run_spectral_param_scan.py`, `python scripts\run_spectral_ablation_study.py`, `python scripts\run_spectral_pipeline_ablation.py`, `python scripts\run_spectral_final_materials.py`, `python scripts\run_report_factual_package.py`.
- Dependencies: Python standard library, numpy, matplotlib, and project scripts.
- Determinism: combo scan sampling uses `random.seed(20260707)` when capped; factual package performs no random sampling.
- HISTORY.md records this package under the 2026-07-07 20:37+ entries.
