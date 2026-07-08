# Report Factual Package Index

Purpose: collect reproducible, checkable factual material for later final-report writing. This is not a polished narrative report and does not introduce new large-scale tuning.

## Files

| path | description |
|---|---|
| summaries/report_factual_package/01_original_pipeline.md | factual package artifact |
| summaries/report_factual_package/01_original_pipeline_params.csv | factual package artifact |
| summaries/report_factual_package/02_original_limitations_from_results.md | factual package artifact |
| summaries/report_factual_package/03_method_workflow.mmd | factual package artifact |
| summaries/report_factual_package/03_method_workflow_and_formulas.md | factual package artifact |
| summaries/report_factual_package/04_metric_definitions_and_assumptions.md | factual package artifact |
| summaries/report_factual_package/05_parameter_scan_summary.md | factual package artifact |
| summaries/report_factual_package/05_parameter_sensitivity_table.csv | factual package artifact |
| summaries/report_factual_package/06_ablation_delta_table.csv | factual package artifact |
| summaries/report_factual_package/06_ablation_results_table.csv | factual package artifact |
| summaries/report_factual_package/06_ablation_summary_facts.md | factual package artifact |
| summaries/report_factual_package/07_final_config_table.csv | factual package artifact |
| summaries/report_factual_package/07_final_config_table.md | factual package artifact |
| summaries/report_factual_package/08_case_study_index.csv | factual package artifact |
| summaries/report_factual_package/08_case_study_index.md | factual package artifact |
| summaries/report_factual_package/09_warnings_taxonomy.csv | factual package artifact |
| summaries/report_factual_package/09_warnings_taxonomy.md | factual package artifact |
| summaries/report_factual_package/10_reproducibility_checklist.md | factual package artifact |
| summaries/report_factual_package/11_failure_taxonomy.md | factual package artifact |
| summaries/report_factual_package/package_summary.json | factual package artifact |

## Figures

| path | description |
|---|---|
| figures/final_report/fig_01_system_workflow.mmd | final-report figure or Mermaid source |
| figures/final_report/fig_02_original_vs_improved_workflow.mmd | final-report figure or Mermaid source |
| figures/final_report/fig_03_final_config_metrics.png | final-report figure or Mermaid source |
| figures/final_report/fig_04_recall_strict_tradeoff.png | final-report figure or Mermaid source |
| figures/final_report/fig_05_sg_ablation.png | final-report figure or Mermaid source |
| figures/final_report/fig_06_residual_ablation.png | final-report figure or Mermaid source |
| figures/final_report/fig_07_parameter_sensitivity_summary.png | final-report figure or Mermaid source |
| figures/final_report/fig_08_case_01.png | final-report figure or Mermaid source |
| figures/final_report/fig_08_case_02.png | final-report figure or Mermaid source |
| figures/final_report/fig_08_case_03.png | final-report figure or Mermaid source |
| figures/final_report/fig_08_case_04.png | final-report figure or Mermaid source |
| figures/final_report/fig_08_case_05.png | final-report figure or Mermaid source |

## Recommended For Web GPT Writing

- Start with `01_original_pipeline.md`, `03_method_workflow_and_formulas.md`, `04_metric_definitions_and_assumptions.md`, `07_final_config_table.md`, and `08_case_study_index.md`.
- Use `02_original_limitations_from_results.md`, `05_parameter_scan_summary.md`, and `06_ablation_summary_facts.md` for evidence-backed claims.
- Use `09_warnings_taxonomy.md` and `10_reproducibility_checklist.md` to qualify reproducibility.

## Appendix-Only Material

- Full sensitivity table, ablation delta table, and detailed case-study rows are better as appendix material.

## Cautious Claims

- Do not claim held-out generalization.
- Do not equate score-unsupported GT with annotation error.
- Do not state unsupported coverage is always bad in isolation.
- Treat the two final operating points as current-dataset recommendations.
