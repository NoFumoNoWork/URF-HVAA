import csv
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_spectral_param_scan import COMBO_SPACE, DEFAULT_PARAMS, ONE_FACTOR_SPACE, strict_score
from scripts.run_spectral_pipeline_ablation import build_final_configs


PACKAGE_DIR = Path("summaries/report_factual_package")
FIG_DIR = Path("figures/final_report")
PARAM_DIR = Path("outputs/26-07-07-18-52-spectral-param-scan/outputs")
ABLATION_DIR = Path("outputs/26-07-07-19-33-spectral-ablation-study")
PIPELINE_DIR = Path("outputs/26-07-07-19-51-spectral-pipeline-ablation")
FINAL_DIR = Path("outputs/26-07-07-20-14-spectral-final-materials")

CORE_NAMES = [
    ("Peak-Aware-Refined baseline", "Peak-Aware"),
    ("Full Spectral-Fusion-Refined default", "Full"),
    ("Spectral-Fusion-SG0", "SG0"),
    ("Recall trend0.5 SG0 residual0.25", "Recall"),
    ("Raw trend residual penalties SG0 residual0.25 peak0", "Strict"),
]
METRICS = [
    "GT_coverage",
    "predicted_GT_fraction",
    "supportable_gt_coverage",
    "unsupportable_gt_coverage",
    "predicted_duration_ratio",
    "stricter_balanced_score",
]


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def fnum(value, digits: int = 3) -> str:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return "NA"
    if math.isnan(out) or math.isinf(out):
        return "NA"
    return f"{out:.{digits}f}"


def as_float(value, default=math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(out) or math.isinf(out) else out


def rel(path: Path | str) -> str:
    return str(path).replace("\\", "/")


def metric_row(row: dict) -> dict:
    return {m: fnum(row.get(m)) for m in METRICS}


def md_table(rows: list[dict], fields: list[str]) -> str:
    out = ["| " + " | ".join(fields) + " |", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return "\n".join(out)


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def bar_metrics(rows: list[dict], labels: list[str], metrics: list[str], path: Path, title: str) -> None:
    x = np.arange(len(labels))
    width = min(0.14, 0.76 / max(1, len(metrics)))
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.7), 6.8))
    colors = ["#4C78A8", "#59A14F", "#F28E2B", "#E15759", "#7F7F7F", "#B07AA1"]
    for idx, metric in enumerate(metrics):
        values = [as_float(row.get(metric), 0.0) for row in rows]
        ax.bar(x + (idx - (len(metrics) - 1) / 2) * width, values, width=width, label=metric, color=colors[idx % len(colors)])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def workflow_mmd() -> str:
    return """flowchart LR
    A["VAD/anomaly score curve"] --> B["spectral evidence extraction"]
    B --> B1["raw"]
    B --> B2["SG smoothing"]
    B --> B3["airPLS residual"]
    B --> B4["trend"]
    B --> B5["peak-count"]
    B --> C["candidate interval generation"]
    C --> C1["Peak-Aware"]
    C --> C2["Hierarchical-Merged"]
    C --> C3["SG-Peak"]
    C --> C4["AirPLS-Residual"]
    C --> C5["Trend-Guided"]
    C --> D["interval feature extraction"]
    D --> E["fusion scoring"]
    E --> E1["SG direct weight = 0"]
    E --> E2["residual weight controls recall/strict"]
    E --> E3["length and low-residual penalties retained"]
    E --> F["fusion threshold filtering"]
    F --> G["final interval merging"]
    G --> H["evaluation"]
    H --> I["recall-oriented operating point"]
    H --> J["strict-oriented operating point"]"""


def original_vs_improved_mmd() -> str:
    return """flowchart LR
    subgraph L["original / baseline pipeline"]
      A1["score curve"] --> A2["peak-aware or threshold-based interval generation"]
      A2 --> A3["interval merge"]
      A3 --> A4["final prediction"]
    end
    subgraph R["improved pipeline"]
      B1["score curve"] --> B2["spectral decomposition"]
      B2 --> B3["multi-source candidate generation"]
      B3 --> B4["interval-level feature extraction"]
      B4 --> B5["fusion score with penalties"]
      B5 --> B6["two operating points"]
    end
    A4 -. lacks explicit supportable/unsupported semantics .-> B2
    B5 -. multi-layer evidence and penalties .-> B6"""


def write_original_pipeline(final_core: list[dict]) -> None:
    params = [
        {"pipeline": "Peak-Aware-Refined", "parameter": "baseline_window", "value": "101", "code_source": "scripts/run_peak_refinement.py::main"},
        {"pipeline": "Peak-Aware-Refined", "parameter": "baseline_method", "value": "median", "code_source": "scripts/run_peak_refinement.py::main"},
        {"pipeline": "Peak-Aware-Refined", "parameter": "peak_mad_k", "value": "2.5", "code_source": "scripts/run_peak_refinement.py::main"},
        {"pipeline": "Peak-Aware-Refined", "parameter": "peak_min_width", "value": "3", "code_source": "scripts/run_peak_refinement.py::main"},
        {"pipeline": "Peak-Aware-Refined", "parameter": "peak_min_distance", "value": "10", "code_source": "scripts/run_peak_refinement.py::main"},
        {"pipeline": "Peak-Aware-Refined", "parameter": "peak_stop_ratio", "value": "0.2", "code_source": "scripts/run_peak_refinement.py::main"},
        {"pipeline": "Hierarchical-Merged", "parameter": "micro_windows", "value": "30,100,300", "code_source": "scripts/generate_hierarchical_intervals.py::DEFAULT_MICRO_WINDOWS"},
        {"pipeline": "Hierarchical-Merged", "parameter": "context_windows", "value": "600,1200", "code_source": "scripts/generate_hierarchical_intervals.py::DEFAULT_CONTEXT_WINDOWS"},
        {"pipeline": "Hierarchical-Merged", "parameter": "scale_thresholds", "value": "30:90,100:85,300:75", "code_source": "scripts/generate_hierarchical_intervals.py::DEFAULT_SCALE_THRESHOLDS"},
        {"pipeline": "Hierarchical-Merged", "parameter": "merge_iou", "value": "0.3", "code_source": "scripts/generate_hierarchical_intervals.py::main"},
        {"pipeline": "Hierarchical-Merged", "parameter": "merge_gap", "value": "150", "code_source": "scripts/generate_hierarchical_intervals.py::main"},
        {"pipeline": "Fixed window baseline", "parameter": "window_sizes", "value": "30,100,300,1000", "code_source": "scripts/evaluate_interval_methods.py::main/add_window_methods"},
        {"pipeline": "Evaluation", "parameter": "score_positive_threshold", "value": "0.6", "code_source": "scripts/evaluate_interval_methods.py::main"},
    ]
    write_csv(PACKAGE_DIR / "01_original_pipeline_params.csv", params)
    peak = next(r for r in final_core if r["run_name"] == "Peak-Aware-Refined baseline")
    text = f"""# Original / Baseline Pipeline Facts

## Inputs

- Ground-truth interval statistics: `outputs/26-07-07-14-43-gt-score-window-curves/outputs/gt_interval_score_stats.csv`.
- Supportability labels: `outputs/26-07-07-15-14-gt-score-alignment-analysis/outputs/gt_support_classification.csv`.
- Score-curve inventory and score JSON paths: `outputs/26-07-07-14-43-gt-score-window-curves/outputs/video_score_curve_inventory.csv`.
- Existing interval artifacts are auto-scanned from `outputs/` by `scripts/evaluate_interval_methods.py::auto_scan_methods`.

## Code Facts

- Score JSON loading for evaluation uses `scripts/evaluate_interval_methods.py::load_score_series`.
- Candidate interval files are parsed by `parse_hierarchical`, `parse_refined`, `parse_wmax_baselines`, and `parse_csv_intervals`.
- Peak-Aware refinement is implemented in `scripts/run_peak_refinement.py::refine_video`; it converts hierarchical predictions and local score peaks to refined intervals.
- Hierarchical-Merged is generated by `scripts/generate_hierarchical_intervals.py::generate_video`, with scale candidates from `generate_scale_candidates`, proposal selection from `select_micro_proposals`, and merging from `cluster_micro_proposals`.
- Fixed-window baselines are added during evaluation by `scripts/evaluate_interval_methods.py::add_window_methods`.
- Aggregate metrics are computed by `scripts/evaluate_interval_methods.py::evaluate_one_video` and `aggregate_rows`.

## Main Baseline Result

Peak-Aware-Refined baseline on the final comparison table:

{md_table([{**{"run_name": peak["run_name"]}, **metric_row(peak)}], ["run_name"] + METRICS)}

## Outputs

- Peak-Aware archive: `outputs/26-07-07-01-08-peak-aware-refinement/`.
- Hierarchical archive: `outputs/26-07-06-08-39-hierarchical-intervals/`.
- Interval-method evaluation archive: `outputs/26-07-07-15-59-interval-method-evaluation/`.
- Final comparison source table: `outputs/26-07-07-19-51-spectral-pipeline-ablation/summaries/final_candidate_config_comparison.csv`.

This file is descriptive: it records code paths, defaults, and measured outputs rather than arguing for a method.
"""
    write_text(PACKAGE_DIR / "01_original_pipeline.md", text)


def write_limitations(final_core: list[dict]) -> None:
    rows = [{**{"configuration": label}, **metric_row(row)} for row, label in [(r, dict(CORE_NAMES)[r["run_name"]]) for r in final_core]]
    text = f"""# Original Pipeline Limitations From Results

The limitations below are grounded in aggregate metrics, case-study rows, and the final comparison table.

{md_table(rows, ["configuration", "GT_coverage", "predicted_GT_fraction", "supportable_gt_coverage", "unsupportable_gt_coverage", "predicted_duration_ratio", "stricter_balanced_score"])}

## Findings

- Temporal mismatch: human GT intervals are event-level ranges, while anomaly/VAD scores are frame/score-level curves. The evaluator measures duration overlap in `evaluate_one_video`, so wide GT context and local score peaks can disagree.
- Local peak mismatch: Peak-Aware reaches GT={fnum(final_core[0]['GT_coverage'])} and strict={fnum(final_core[0]['stricter_balanced_score'])}, but Full/SG0/Recall/Strict change supportable and unsupported coverage by using interval-level features.
- Missing joint modeling: baseline peak or threshold generation does not explicitly combine raw evidence, residual evidence, trend, length penalty, and low-residual penalty; those terms are implemented in `score_fusion`.
- No explicit supportability split in generation: supportable/unsupportable is added by evaluation from `gt_support_classification.csv`, not by the original interval generator.
- Over-merge, over-extension, and fragmentation are visible through duration/purity trade-offs. Full default has duration={fnum(next(r for r in final_core if r['run_name'].startswith('Full'))['predicted_duration_ratio'])}; SG0 lowers unsupported coverage and duration while slightly lowering GT.
- Peak-Aware differs from final operating points: Recall raises GT/supportable coverage, while Strict improves conservative score relative to Peak-Aware without treating unsupported coverage as a standalone objective.
"""
    write_text(PACKAGE_DIR / "02_original_limitations_from_results.md", text)


def write_method_workflow(final_core: list[dict]) -> None:
    mmd = workflow_mmd()
    write_text(PACKAGE_DIR / "03_method_workflow.mmd", mmd)
    text = f"""# Method Workflow And Formulas

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
"""
    write_text(PACKAGE_DIR / "03_method_workflow_and_formulas.md", text)


def write_metric_defs() -> None:
    text = """# Metric Definitions And Assumptions

All metric formulas are implemented in `scripts/evaluate_interval_methods.py::evaluate_one_video` and `aggregate_rows`.

- `GT_coverage`: intersection duration between prediction and merged GT divided by total merged GT duration.
- `predicted_GT_fraction` / purity: intersection duration divided by predicted duration.
- `supportable_gt_coverage`: coverage over GT rows whose support group is `supportable`.
- `unsupportable_gt_coverage`: coverage over GT rows whose support group is `unsupportable`.
- `predicted_duration_ratio`: predicted duration divided by video duration.
- `balanced_score`: `0.4*GT_coverage + 0.3*predicted_GT_fraction + 0.2*supportable_gt_coverage - 0.1*predicted_duration_ratio`.
- `stricter_balanced_score`: `0.30*GT_coverage + 0.25*predicted_GT_fraction + 0.25*supportable_gt_coverage - 0.10*predicted_duration_ratio - 0.10*unsupportable_gt_coverage`.
- `num_predicted_intervals`: aggregate predicted interval count after per-video merging.
- `mean_interval_length`: same reported value as `mean_predicted_interval_length`.
- `median_interval_length`: same reported value as `median_predicted_interval_length`.

## Supportable / Unsupportable Assumptions

- Supportability labels come from `outputs/26-07-07-15-14-gt-score-alignment-analysis/outputs/gt_support_classification.csv`, loaded by `load_gt_rows`.
- The evaluator does not classify supportability live from `score_threshold=0.6`; it reads the precomputed support group.
- `score_threshold=0.6` is used for fixed-window score-positive methods in `evaluate_interval_methods.py::add_window_methods`, for default spectral trend thresholding, and in earlier score-alignment artifacts.
- Score-unsupported GT is not equivalent to annotation error.
- Unsupported coverage is a diagnostic constraint, not an independent objective to minimize.
- If unsupported coverage rises with duration while purity drops, the result suggests over-extension.
- If unsupported coverage is moderate while GT/supportable coverage increases, it may reflect human event context or score-weak anomaly evidence.
"""
    write_text(PACKAGE_DIR / "04_metric_definitions_and_assumptions.md", text)


def write_param_scan() -> list[dict]:
    all_rows = read_csv(PARAM_DIR / "summaries/param_scan_all_runs.csv")
    one = [r for r in all_rows if r.get("scan_mode") == "one_factor"]
    sens = []
    for param, rows in sorted(defaultdict(list, {p: [r for r in one if r.get("changed_param") == p] for p in ONE_FACTOR_SPACE}).items()):
        if not rows:
            continue
        out = {"parameter": param, "values_scanned": ",".join(str(v) for v in ONE_FACTOR_SPACE.get(param, []))}
        for metric in METRICS:
            vals = [as_float(r.get(metric)) for r in rows if not math.isnan(as_float(r.get(metric)))]
            out[f"{metric}_min"] = fnum(min(vals)) if vals else "NA"
            out[f"{metric}_max"] = fnum(max(vals)) if vals else "NA"
            out[f"{metric}_range"] = fnum(max(vals) - min(vals)) if vals else "NA"
        best = max(rows, key=lambda r: as_float(r.get("stricter_balanced_score"), -999))
        out["recommended_value_from_one_factor"] = best.get("changed_value", "")
        sens.append(out)
    write_csv(PACKAGE_DIR / "05_parameter_sensitivity_table.csv", sens)
    ranked = sorted(sens, key=lambda r: as_float(r.get("stricter_balanced_score_range"), 0), reverse=True)
    text = f"""# Parameter Scan Summary

Source files:

- `outputs/26-07-07-18-52-spectral-param-scan/outputs/summaries/param_scan_all_runs.csv`
- `outputs/26-07-07-18-52-spectral-param-scan/outputs/summaries/one_factor_sensitivity_summary.csv`
- `outputs/26-07-07-18-52-spectral-param-scan/outputs/summaries/pareto_frontier_runs.csv`

The scan evaluated {len(all_rows)} rows in the all-run table. Parameter ranges are defined in `scripts/run_spectral_param_scan.py::ONE_FACTOR_SPACE` and `COMBO_SPACE`.

{md_table([{k: r.get(k, "") for k in ["parameter", "values_scanned", "GT_coverage_range", "predicted_GT_fraction_range", "supportable_gt_coverage_range", "unsupportable_gt_coverage_range", "predicted_duration_ratio_range", "stricter_balanced_score_range", "recommended_value_from_one_factor"]} for r in ranked], ["parameter", "values_scanned", "GT_coverage_range", "predicted_GT_fraction_range", "supportable_gt_coverage_range", "unsupportable_gt_coverage_range", "predicted_duration_ratio_range", "stricter_balanced_score_range", "recommended_value_from_one_factor"])}

## Interpretation

- Most influential parameters by stricter-score range: {", ".join(r["parameter"] for r in ranked[:5])}.
- Lower-sensitivity parameters in this one-factor scan: {", ".join(r["parameter"] for r in ranked[-4:])}.
- Bayesian optimization is not continued here because `AGENT.md` asks for factual packaging only, and current runs already expose the main recall/strict trade-off.
- Pareto frontier rows are retained in `pareto_frontier_runs.csv`; final Recall and Strict points are manually selected operating points from pipeline-level comparisons, not new optimization claims.
- Default parameter baseline: `{DEFAULT_PARAMS}`.
- Combo space: `{COMBO_SPACE}`.
"""
    write_text(PACKAGE_DIR / "05_parameter_scan_summary.md", text)
    return sens


def write_ablation() -> None:
    direct = read_csv(ABLATION_DIR / "summaries/ablation_module_results.csv")
    direct_delta = read_csv(ABLATION_DIR / "summaries/ablation_delta_vs_full.csv")
    sg = read_csv(PIPELINE_DIR / "summaries/pipeline_sg_candidate_ablation.csv")
    residual = read_csv(PIPELINE_DIR / "summaries/pipeline_residual_candidate_ablation.csv")
    final = read_csv(PIPELINE_DIR / "summaries/final_candidate_config_comparison.csv")
    rows = []
    for source, items in [("direct_fusion", direct), ("pipeline_sg", sg), ("pipeline_residual", residual), ("final_candidate", final)]:
        for row in items:
            out = {"ablation_group": source, "run_name": row.get("run_name", "")}
            out.update({m: fnum(row.get(m)) for m in METRICS})
            rows.append(out)
    write_csv(PACKAGE_DIR / "06_ablation_results_table.csv", rows)
    delta_rows = []
    for source, items in [("direct_fusion", direct_delta), ("pipeline_sg", sg), ("pipeline_residual", residual)]:
        for row in items:
            out = {"ablation_group": source, "run_name": row.get("run_name", "")}
            for metric in METRICS:
                out[f"delta_{metric}"] = fnum(row.get(f"delta_{metric}"))
            delta_rows.append(out)
    write_csv(PACKAGE_DIR / "06_ablation_delta_table.csv", delta_rows)
    text = f"""# Ablation Summary Facts

## Direct Fusion Ablation

Source: `outputs/26-07-07-19-33-spectral-ablation-study/summaries/ablation_module_results.csv`.

{md_table([{**{"run_name": r["run_name"]}, **metric_row(r)} for r in direct], ["run_name"] + METRICS)}

## Pipeline-Level Ablation

Source: `outputs/26-07-07-19-51-spectral-pipeline-ablation/summaries/`.

SG ablation:

{md_table([{**{"run_name": r["run_name"]}, **metric_row(r)} for r in sg], ["run_name"] + METRICS)}

Residual ablation:

{md_table([{**{"run_name": r["run_name"]}, **metric_row(r)} for r in residual], ["run_name"] + METRICS)}

## Required Answers

1. SG direct weight should be set to 0 for the selected final operating points.
2. SG smoothing should be retained because SG-Peak candidates remain enabled in SG0/Recall/Strict.
3. SG candidate generation should be retained; pipeline ablation separates it from direct SG weight.
4. Residual candidate generation should be retained.
5. Residual direct weight should not be globally deleted; selected Recall and Strict both use residual_weight=0.25.
6. Residual direct weight controls the recall/strict trade-off together with threshold and peak-count settings.
7. Peak-count direct evidence does not dominate the selected final configuration; Strict sets peak_count_weight=0.
8. Penalties are retained to control duration and low-residual interval expansion.
9. Full default uses SG direct weight; SG0 removes direct SG weight; Recall lowers trend threshold; Strict removes direct SG and peak-count weight while retaining residual and penalties.
"""
    write_text(PACKAGE_DIR / "06_ablation_summary_facts.md", text)


def write_final_configs(final_rows: list[dict]) -> None:
    cfgs = {c["run_name"]: c for c in build_final_configs() if not c.get("baseline_method")}
    selected = ["Recall trend0.5 SG0 residual0.25", "Raw trend residual penalties SG0 residual0.25 peak0"]
    by_name = {r["run_name"]: r for r in final_rows}
    rows = []
    for name in selected:
        cfg = cfgs[name]
        metrics = by_name[name]
        out = {
            "configuration_name": name,
            "candidate_sources_enabled": metrics.get("enabled_candidate_sources", ""),
            "candidate_sources_disabled": metrics.get("disabled_candidate_sources", ""),
            "fusion_formula": "raw*N(raw_max)+sg*N(sg_max)+residual*N(residual_max)+trend*N(trend_mean)+peak_count*N(residual_peak_count)-length_penalty*N(length)-low_residual_penalty*low_residual_ratio",
            "run_command": "python scripts\\run_spectral_pipeline_ablation.py",
            "output_directory": "outputs/26-07-07-19-51-spectral-pipeline-ablation/",
        }
        for key in ["raw_weight", "sg_weight", "residual_weight", "trend_weight", "peak_count_weight", "length_penalty_weight", "low_residual_penalty_weight", "fusion_threshold", "trend_threshold", "trend_window", "sg_window_length", "sg_polyorder", "airpls_lambda", "residual_mad_k", "peak_mad_k", "peak_stop_ratio", "merge_gap_frames"]:
            out[key] = cfg.get(key, "")
        out["score_threshold"] = 0.60
        out["airpls_order"] = 2
        out["airpls_itermax"] = 20
        out.update(metric_row(metrics))
        rows.append(out)
    write_csv(PACKAGE_DIR / "07_final_config_table.csv", rows)
    write_text(PACKAGE_DIR / "07_final_config_table.md", "# Final Config Table\n\n" + md_table(rows, list(rows[0].keys())))


def write_case_index() -> None:
    src = FINAL_DIR / "summaries/case_studies.csv"
    rows = read_csv(src)
    out_rows = []
    for idx, row in enumerate(rows, start=1):
        old_fig = Path(row.get("figure_path") or row.get("figure") or "")
        new_fig = FIG_DIR / f"fig_08_case_{idx:02d}.png"
        if old_fig.is_file():
            copy_file(old_fig, new_fig)
        out = {
            "case_id": f"{idx:02d}",
            "video_id": row.get("video_id", ""),
            "dataset": row.get("dataset", ""),
            "case_type": row.get("case_type", ""),
            "why_selected": row.get("why_selected", ""),
            "GT intervals summary": row.get("GT_intervals", ""),
            "score curve summary": row.get("score_summary", ""),
            "Peak-Aware prediction summary": row.get("Peak_Aware_prediction", ""),
            "Full default prediction summary": row.get("Full_default_prediction", ""),
            "SG0 prediction summary": row.get("SG0_prediction", ""),
            "Recall-oriented prediction summary": row.get("Recall_prediction") or row.get("Recall_oriented_prediction", ""),
            "Strict-oriented prediction summary": row.get("Strict_prediction") or row.get("Strict_oriented_prediction", ""),
            "figure_path": rel(new_fig),
            "one_sentence_conclusion": row.get("one_sentence_conclusion") or row.get("explanation") or row.get("why_selected", ""),
        }
        out_rows.append(out)
    write_csv(PACKAGE_DIR / "08_case_study_index.csv", out_rows)
    blocks = ["# Case Study Index", ""]
    for row in out_rows:
        blocks.extend([f"## Case {row['case_id']}: {row['case_type']}", "", md_table([row], list(row.keys())), ""])
    write_text(PACKAGE_DIR / "08_case_study_index.md", "\n".join(blocks))


def write_warnings_and_repro() -> None:
    copy_file(FINAL_DIR / "summaries/warnings_taxonomy.csv", PACKAGE_DIR / "09_warnings_taxonomy.csv")
    src_md = (FINAL_DIR / "summaries/warnings_taxonomy.md").read_text(encoding="utf-8")
    pipeline_json = json.loads((PIPELINE_DIR / "pipeline_ablation_summary.json").read_text(encoding="utf-8"))
    text = src_md + f"""

## Cross-Run Check

- Pipeline ablation summary warnings: {pipeline_json.get("warning_count", "NA")}.
- Pipeline script: `scripts/run_spectral_pipeline_ablation.py`.
- Final materials script: `scripts/run_spectral_final_materials.py`.
- Re-run needed: no, because final-materials taxonomy found only non-fatal SG raw-fallback warnings and no skipped videos/GT/prediction files.
"""
    write_text(PACKAGE_DIR / "09_warnings_taxonomy.md", text)
    repro = """# Reproducibility Checklist

- Key scripts: `scripts/run_spectral_param_scan.py`, `scripts/run_spectral_ablation_study.py`, `scripts/run_spectral_pipeline_ablation.py`, `scripts/run_spectral_final_materials.py`, `scripts/run_report_factual_package.py`.
- Key inputs: GT stats CSV, support classification CSV, video score inventory CSV, cached decomposition curves.
- Key outputs: parameter scan, ablation study, pipeline ablation, final materials, and `summaries/report_factual_package/`.
- Commands: `python scripts\\run_spectral_param_scan.py`, `python scripts\\run_spectral_ablation_study.py`, `python scripts\\run_spectral_pipeline_ablation.py`, `python scripts\\run_spectral_final_materials.py`, `python scripts\\run_report_factual_package.py`.
- Dependencies: Python standard library, numpy, matplotlib, and project scripts.
- Determinism: combo scan sampling uses `random.seed(20260707)` when capped; factual package performs no random sampling.
- HISTORY.md records this package under the 2026-07-07 20:37+ entries.
"""
    write_text(PACKAGE_DIR / "10_reproducibility_checklist.md", repro)


def write_failure_taxonomy() -> None:
    src = FINAL_DIR / "summaries/failure_taxonomy.md"
    base = src.read_text(encoding="utf-8") if src.exists() else "# Failure Taxonomy\n\nmissing data"
    extra = """

## Required Categories

- score-supported but missed: supportable GT remains uncovered; may require candidate generation or score generation changes.
- score-supported but fragmented: several short intervals cover one event; merging or higher-level event grouping is needed.
- over-merged prediction: adjacent candidates merge into broad predictions; penalties and stricter thresholds can reduce this.
- score-unsupported GT: human interval has weak score evidence; not automatically an annotation error.
- boundary mismatch: frame-level score peaks do not align with event-level human boundaries.
- low-score context coverage: predictions cover contextual parts around score-supported events.
- candidate generation miss: no candidate exists for a GT segment; fusion cannot recover it.
- fusion scoring miss: candidate exists but falls below threshold; weights/penalties govern the failure.
"""
    write_text(PACKAGE_DIR / "11_failure_taxonomy.md", base + extra)


def write_figures(final_rows: list[dict], sens_rows: list[dict]) -> None:
    write_text(FIG_DIR / "fig_01_system_workflow.mmd", workflow_mmd())
    write_text(FIG_DIR / "fig_02_original_vs_improved_workflow.mmd", original_vs_improved_mmd())
    labels = [label for _, label in CORE_NAMES]
    by_name = {r["run_name"]: r for r in final_rows}
    core = [by_name[name] for name, _ in CORE_NAMES]
    bar_metrics(core, labels, METRICS[:-1], FIG_DIR / "fig_03_final_config_metrics.png", "Final five configuration metrics")

    all_final = read_csv(PIPELINE_DIR / "summaries/final_candidate_config_comparison.csv")
    fig, ax = plt.subplots(figsize=(9, 6.5))
    xs = [as_float(r.get("stricter_balanced_score"), 0) for r in all_final]
    ys = [as_float(r.get("GT_coverage"), 0) for r in all_final]
    cs = [as_float(r.get("unsupportable_gt_coverage"), 0) for r in all_final]
    ss = [80 + 650 * as_float(r.get("predicted_duration_ratio"), 0) for r in all_final]
    ax.scatter(xs, ys, c="#BBBBBB", s=ss, alpha=0.45, edgecolor="none")
    core_colors = ["#4C78A8", "#59A14F", "#F28E2B", "#E15759", "#B07AA1"]
    for (_, label), color in zip(CORE_NAMES, core_colors):
        r = by_name[_]
        sc = ax.scatter([as_float(r["stricter_balanced_score"])], [as_float(r["GT_coverage"])], c=[as_float(r["unsupportable_gt_coverage"])], s=180, cmap="viridis", edgecolor=color, linewidth=2)
        ax.annotate(label, (as_float(r["stricter_balanced_score"]), as_float(r["GT_coverage"])), xytext=(6, 5), textcoords="offset points", fontsize=9)
    fig.colorbar(sc, ax=ax, label="unsupportable_gt_coverage")
    ax.set_xlabel("stricter_balanced_score")
    ax.set_ylabel("GT_coverage")
    ax.set_title("Recall vs strict trade-off")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_04_recall_strict_tradeoff.png", dpi=180)
    plt.close(fig)

    sg = read_csv(PIPELINE_DIR / "summaries/pipeline_sg_candidate_ablation.csv")
    bar_metrics(sg, [r["run_name"] for r in sg], METRICS, FIG_DIR / "fig_05_sg_ablation.png", "SG direct weight / SG candidate ablation")
    residual = read_csv(PIPELINE_DIR / "summaries/pipeline_residual_candidate_ablation.csv")
    bar_metrics(residual, [r["run_name"] for r in residual], METRICS, FIG_DIR / "fig_06_residual_ablation.png", "Residual direct / candidate ablation")

    fig, ax = plt.subplots(figsize=(11, 5.8))
    params = [r["parameter"] for r in sens_rows]
    vals = [as_float(r.get("stricter_balanced_score_range"), 0) for r in sens_rows]
    ax.bar(params, vals, color="#4C78A8")
    ax.set_ylabel("stricter_balanced_score range")
    ax.set_title("Parameter sensitivity summary")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_07_parameter_sensitivity_summary.png", dpi=180)
    plt.close(fig)


def write_index() -> None:
    files = sorted(PACKAGE_DIR.glob("*"))
    figs = sorted(FIG_DIR.glob("*"))
    file_rows = [{"path": rel(p), "description": "factual package artifact"} for p in files if p.name != "00_index.md"]
    fig_rows = [{"path": rel(p), "description": "final-report figure or Mermaid source"} for p in figs]
    text = f"""# Report Factual Package Index

Purpose: collect reproducible, checkable factual material for later final-report writing. This is not a polished narrative report and does not introduce new large-scale tuning.

## Files

{md_table(file_rows, ["path", "description"])}

## Figures

{md_table(fig_rows, ["path", "description"])}

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
"""
    write_text(PACKAGE_DIR / "00_index.md", text)


def main() -> None:
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    final_rows = read_csv(PIPELINE_DIR / "summaries/final_candidate_config_comparison.csv")
    by_name = {r["run_name"]: r for r in final_rows}
    final_core = [by_name[name] for name, _ in CORE_NAMES]

    write_original_pipeline(final_core)
    write_limitations(final_core)
    write_method_workflow(final_core)
    write_metric_defs()
    sens = write_param_scan()
    write_ablation()
    write_final_configs(final_rows)
    write_case_index()
    write_warnings_and_repro()
    write_failure_taxonomy()
    write_figures(final_rows, sens)
    summary = {
        "package_dir": rel(PACKAGE_DIR),
        "figure_dir": rel(FIG_DIR),
        "package_file_count": len(list(PACKAGE_DIR.glob("*"))),
        "figure_file_count": len(list(FIG_DIR.glob("*"))),
    }
    write_text(PACKAGE_DIR / "package_summary.json", json.dumps(summary, indent=2))
    write_index()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
