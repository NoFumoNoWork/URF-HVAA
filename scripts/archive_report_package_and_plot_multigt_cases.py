import argparse
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.anomaly_utils import write_json
from scripts.evaluate_interval_methods import add_window_methods, auto_scan_methods, group_gt, load_gt_rows, load_inventory, write_csv
from scripts.run_spectral_final_materials import (
    CORE_RUNS,
    DEFAULT_CACHE_SOURCE,
    PIPELINE_DIR,
    build_config_map,
    build_predictions,
    gt_str,
    interval_str,
    per_video_metrics,
    prediction_by_video,
    score_summary_for_video,
)
from scripts.run_spectral_param_scan import DEFAULT_PARAMS, precompute_curves, safe_float
from scripts.run_spectral_score_decomposition import load_score_series, safe_name


DEFAULT_OUTPUT_DIR = Path("outputs/26-07-07-21-20-support-labeled-multigt-cases")
DEFAULT_SOURCE_PACKAGE = Path("outputs/26-07-07-20-44-final-report-summary")
SUPPORT_COLORS = {
    "supportable": "#2CA02C",
    "unsupportable": "#D62728",
    "uncertain": "#9467BD",
    "": "#7F7F7F",
}
PRED_COLORS = {
    "Peak": "#4C78A8",
    "Full": "#59A14F",
    "SG0": "#F28E2B",
    "Recall": "#E15759",
    "Strict": "#7F7F7F",
}


def fmt(value, digits: int = 3) -> str:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return "NA"
    if math.isnan(out) or math.isinf(out):
        return "NA"
    return f"{out:.{digits}f}"


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def copytree_fresh(src: Path, dst: Path) -> int:
    if not src.exists():
        return 0
    shutil.copytree(src, dst, dirs_exist_ok=True)
    return sum(1 for p in dst.rglob("*") if p.is_file())


def copy_existing_package(source: Path, out: Path) -> dict:
    outputs = out / "outputs"
    summary_src = source / "summaries" / "report_factual_package"
    figure_src = source / "figures"
    summary_count = copytree_fresh(summary_src, outputs / "summaries" / "report_factual_package")
    figure_count = copytree_fresh(figure_src, outputs / "figures" / "final_report")
    return {"summary_file_count": summary_count, "figure_file_count": figure_count}


def metric_fields(prefix: str, row: dict) -> dict:
    return {
        f"{prefix}_GT_coverage": fmt(row.get("GT_coverage")),
        f"{prefix}_purity": fmt(row.get("predicted_GT_fraction")),
        f"{prefix}_supportable": fmt(row.get("supportable_gt_coverage")),
        f"{prefix}_unsupported": fmt(row.get("unsupportable_gt_coverage")),
        f"{prefix}_duration": fmt(row.get("predicted_duration_ratio")),
    }


def interval_start(row: dict) -> int:
    return int(float(row.get("start", row.get("gt_start", 0))))


def interval_end(row: dict) -> int:
    return int(float(row.get("end", row.get("gt_end", 0))))


def draw_intervals(ax, rows: list[dict], color: str | None = None, support_labeled: bool = False) -> None:
    for row in rows:
        start = interval_start(row)
        end = interval_end(row)
        if end <= start:
            continue
        row_color = color
        if support_labeled:
            row_color = SUPPORT_COLORS.get(row.get("support_group", ""), SUPPORT_COLORS[""])
        ax.axvspan(start, end, ymin=0.15, ymax=0.85, color=row_color, alpha=0.78)


def plot_support_labeled_case(path: Path, dataset: str, video_id: str, gt_rows: list[dict], pred_rows: dict[str, list[dict]], score_path: Path) -> None:
    frames, scores = load_score_series(score_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tracks = [
        ("GT", gt_rows, None, True),
        ("Peak", pred_rows["Peak-Aware-Refined baseline"], PRED_COLORS["Peak"], False),
        ("Full", pred_rows["Full Spectral-Fusion-Refined default"], PRED_COLORS["Full"], False),
        ("SG0", pred_rows["Spectral-Fusion-SG0"], PRED_COLORS["SG0"], False),
        ("Recall", pred_rows["Recall trend0.5 SG0 residual0.25"], PRED_COLORS["Recall"], False),
        ("Strict", pred_rows["Raw trend residual penalties SG0 residual0.25 peak0"], PRED_COLORS["Strict"], False),
    ]
    fig, axes = plt.subplots(len(tracks) + 1, 1, figsize=(15, 8.6), sharex=True, gridspec_kw={"height_ratios": [2.4] + [0.62] * len(tracks)})
    axes[0].plot(frames, scores, color="#333333", linewidth=1.0)
    axes[0].axhline(0.6, color="#999999", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("score")
    axes[0].set_title(f"{dataset} | {video_id}")
    axes[0].grid(True, alpha=0.2)
    for ax, (label, rows, color, support_labeled) in zip(axes[1:], tracks):
        draw_intervals(ax, rows, color=color, support_labeled=support_labeled)
        ax.set_ylabel(label, rotation=0, ha="right", va="center")
        ax.set_yticks([])
        ax.grid(axis="x", alpha=0.08)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.78) for key, color in SUPPORT_COLORS.items() if key]
    labels = [key for key in SUPPORT_COLORS if key]
    axes[1].legend(handles, labels, loc="upper right", ncol=3, fontsize=8, frameon=False, title="GT support")
    axes[-1].set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def build_multigt_cases(args: argparse.Namespace, out: Path) -> tuple[list[dict], list[dict], int]:
    warnings = []
    gt_rows = load_gt_rows(args.gt_stats_csv, args.gt_support_csv)
    inventory = load_inventory(args.video_inventory_csv, gt_rows)

    pre_args = argparse.Namespace(output_dir=Path("outputs/26-07-07-18-52-spectral-param-scan/outputs"), reuse_cached_curves=True)
    if not DEFAULT_CACHE_SOURCE.exists():
        pre_args = argparse.Namespace(output_dir=out / "outputs", reuse_cached_curves=args.reuse_cached_curves)
    decomp = precompute_curves(pre_args, inventory, warnings)

    existing = []
    auto_scan_methods(args.existing_interval_root, existing)
    add_window_methods(existing, inventory, [100, 300], DEFAULT_PARAMS["trend_threshold"], Path.cwd())
    configs = build_config_map()
    predictions, _core_overall, _all_intervals = build_predictions(configs, existing, decomp, gt_rows, inventory, warnings)
    pv_metrics = per_video_metrics(predictions, gt_rows, inventory)
    pred_by_video = prediction_by_video(predictions)
    gt_by_video = group_gt(gt_rows)

    figure_dir = out / "outputs" / "figures" / "multigt_case_studies"
    rows = []
    missing_plot_rows = []
    multi_keys = sorted(key for key, rows_for_video in gt_by_video.items() if len(rows_for_video) > 1)
    for idx, key in enumerate(multi_keys, start=1):
        dataset, video_id = key
        meta = inventory.get(key, {})
        score_path = Path(meta.get("score_json_path", ""))
        if score_path and not score_path.is_absolute():
            score_path = Path.cwd() / score_path
        plot_name = f"{idx:04d}_{dataset}_{safe_name(video_id)}.png"
        plot_path = figure_dir / plot_name
        support_counts = Counter(row.get("support_group", "") for row in gt_by_video[key])
        score_info = {}
        if score_path.is_file():
            pred_rows = {run: pred_by_video[run].get(key, []) for run, _label in CORE_RUNS}
            plot_support_labeled_case(plot_path, dataset, video_id, gt_by_video[key], pred_rows, score_path)
            score_info = score_summary_for_video(score_path)
        else:
            missing_plot_rows.append({"dataset": dataset, "video_id": video_id, "reason": f"missing score file: {score_path}"})
        item = {
            "case_id": f"{idx:04d}",
            "dataset": dataset,
            "video_id": video_id,
            "gt_interval_count": len(gt_by_video[key]),
            "supportable_gt_count": support_counts.get("supportable", 0),
            "unsupportable_gt_count": support_counts.get("unsupportable", 0),
            "uncertain_gt_count": support_counts.get("uncertain", 0),
            "gt_intervals": gt_str(gt_by_video[key], limit=20),
            "score_points": fmt(score_info.get("score_points"), 0),
            "max_score": fmt(score_info.get("max_score")),
            "mean_score": fmt(score_info.get("mean_score")),
            "frac_score_ge_0.6": fmt(score_info.get("frac_score_ge_0.6")),
            "figure_path": str(plot_path).replace("\\", "/") if plot_path.exists() else "",
            "Peak_Aware_prediction": interval_str(pred_by_video["Peak-Aware-Refined baseline"].get(key, []), limit=12),
            "Full_prediction": interval_str(pred_by_video["Full Spectral-Fusion-Refined default"].get(key, []), limit=12),
            "SG0_prediction": interval_str(pred_by_video["Spectral-Fusion-SG0"].get(key, []), limit=12),
            "Recall_prediction": interval_str(pred_by_video["Recall trend0.5 SG0 residual0.25"].get(key, []), limit=12),
            "Strict_prediction": interval_str(pred_by_video["Raw trend residual penalties SG0 residual0.25 peak0"].get(key, []), limit=12),
        }
        for run_name, label in CORE_RUNS:
            item.update(metric_fields(label, pv_metrics[run_name].get(key, {})))
        rows.append(item)
    return rows, missing_plot_rows, len(warnings)


def write_markdown_index(path: Path, rows: list[dict], missing: list[dict]) -> None:
    lines = [
        "# Multi-GT Case Visualization Index",
        "",
        f"- Total multi-GT videos: {len(rows)}.",
        f"- Plotted videos: {sum(1 for row in rows if row.get('figure_path'))}.",
        f"- Missing plots: {len(missing)}.",
        "",
        "Each figure contains raw anomaly score, GT intervals, and predictions from Peak-Aware, Full, SG0, Recall, and Strict operating points.",
        "The GT track is color-coded by supportability: green=supportable, red=unsupportable, purple=uncertain.",
        "",
        "## Selection Hints",
        "",
        "- Prefer cases with several GT intervals and visible score response if you want visually reliable examples.",
        "- Use `Recall_GT_coverage` and `Strict_purity` together instead of relying on one metric.",
        "- Treat `unsupportable_gt_count` as a diagnostic field, not a label-error claim.",
        "",
    ]
    if rows:
        preview_fields = ["case_id", "dataset", "video_id", "gt_interval_count", "supportable_gt_count", "unsupportable_gt_count", "max_score", "Full_GT_coverage", "Recall_GT_coverage", "Strict_purity", "figure_path"]
        lines.extend(["## Preview", "", "| " + " | ".join(preview_fields) + " |", "|" + "|".join(["---"] * len(preview_fields)) + "|"])
        for row in rows[:40]:
            lines.append("| " + " | ".join(str(row.get(field, "")) for field in preview_fields) + " |")
        if len(rows) > 40:
            lines.append(f"\nPreview truncated to 40 rows. Full table: `outputs/summaries/multigt_case_study_index.csv`.")
    if missing:
        lines.extend(["", "## Missing Plot Rows", ""])
        for row in missing:
            lines.append(f"- `{row['dataset']}::{row['video_id']}`: {row['reason']}")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_archive_report(path: Path, summary: dict) -> None:
    lines = [
        "# Final Report Factual Package + Multi-GT Cases",
        "",
        "This archive repackages the final report factual package under the standard `outputs/yy-mm-dd-hh-mm-*` naming convention and adds support-labeled visualizations for every video with more than one GT interval.",
        "",
        "## Counts",
        "",
        f"- Factual package summary files copied: {summary['copied_package'].get('summary_file_count', 0)}.",
        f"- Final report figure files copied: {summary['copied_package'].get('figure_file_count', 0)}.",
        f"- Multi-GT videos indexed: {summary['multigt_video_count']}.",
        f"- Multi-GT plots generated: {summary['multigt_plot_count']}.",
        "- GT supportability colors: green=supportable, red=unsupportable, purple=uncertain.",
        f"- Warning count from prediction rebuild: {summary['prediction_warning_count']}.",
        "",
        "## Main Paths",
        "",
        "- `outputs/summaries/report_factual_package/00_index.md`",
        "- `outputs/summaries/multigt_case_study_index.csv`",
        "- `outputs/summaries/multigt_case_study_index.md`",
        "- `outputs/figures/final_report/`",
        "- `outputs/figures/multigt_case_studies/`",
        "",
        "No algorithm changes or new parameter scans are introduced by this archive script.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(path: Path, summary: dict) -> None:
    lines = [
        "# MANIFEST",
        "",
        "- `final-report-factual-package-multigt-cases_report.md`: archive report.",
        "- `outputs/summaries/report_factual_package/`: copied factual package.",
        "- `outputs/figures/final_report/`: copied final report figures.",
        "- `outputs/summaries/multigt_case_study_index.csv`: full multi-GT case index.",
        "- `outputs/summaries/multigt_case_study_index.md`: readable multi-GT case index.",
        "- `outputs/figures/multigt_case_studies/*.png`: per-video support-labeled multi-GT timeline plots.",
        "- GT track colors: green=supportable, red=unsupportable, purple=uncertain.",
        "- `programs/scripts/archive_report_package_and_plot_multigt_cases.py`: generator script.",
        "- `programs/scripts/run_report_factual_package.py`: copied factual-package generator script when available.",
        "",
        "## Summary JSON",
        "",
        "```json",
        json.dumps(summary, indent=2, ensure_ascii=False),
        "```",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source_package", type=Path, default=DEFAULT_SOURCE_PACKAGE)
    parser.add_argument("--gt_stats_csv", type=Path, default=Path("outputs/26-07-07-14-43-gt-score-window-curves/outputs/gt_interval_score_stats.csv"))
    parser.add_argument("--gt_support_csv", type=Path, default=Path("outputs/26-07-07-15-14-gt-score-alignment-analysis/outputs/gt_support_classification.csv"))
    parser.add_argument("--video_inventory_csv", type=Path, default=Path("outputs/26-07-07-14-43-gt-score-window-curves/outputs/video_score_curve_inventory.csv"))
    parser.add_argument("--existing_interval_root", type=Path, default=Path("outputs"))
    parser.add_argument("--reuse_cached_curves", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    copied = copy_existing_package(args.source_package, args.output_dir)
    rows, missing, warning_count = build_multigt_cases(args, args.output_dir)
    summary_dir = args.output_dir / "outputs" / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    write_csv(summary_dir / "multigt_case_study_index.csv", rows)
    write_csv(summary_dir / "multigt_case_study_missing_plots.csv", missing, ["dataset", "video_id", "reason"])
    write_markdown_index(summary_dir / "multigt_case_study_index.md", rows, missing)

    program_dir = args.output_dir / "programs" / "scripts"
    program_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), program_dir / Path(__file__).name)
    factual_script = Path("scripts/run_report_factual_package.py")
    if factual_script.exists():
        shutil.copy2(factual_script, program_dir / factual_script.name)

    summary = {
        "archive_dir": str(args.output_dir).replace("\\", "/"),
        "source_package": str(args.source_package).replace("\\", "/"),
        "copied_package": copied,
        "multigt_video_count": len(rows),
        "multigt_plot_count": sum(1 for row in rows if row.get("figure_path")),
        "missing_plot_count": len(missing),
        "prediction_warning_count": warning_count,
        "gt_track_supportability_colors": {
            "supportable": "green",
            "unsupportable": "red",
            "uncertain": "purple",
        },
    }
    write_json(args.output_dir / "outputs" / "archive_summary.json", summary)
    write_archive_report(args.output_dir / "final-report-factual-package-multigt-cases_report.md", summary)
    write_manifest(args.output_dir / "MANIFEST.md", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
