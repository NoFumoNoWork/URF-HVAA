import argparse
import copy
import json
import math
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.anomaly_utils import write_json
from scripts.evaluate_interval_methods import add_window_methods, auto_scan_methods, group_gt, write_csv
from scripts.run_interval_evaluation_summary import (
    DEFAULT_CACHE_SOURCE,
    DISPLAY,
    NEGATIVE_FIELDS,
    detect_negative_intervals,
    evaluate_method,
    group_predictions,
    load_gt_rows,
    load_inventory,
    load_score_stats,
    method_slug,
    negative_diagnostics_rows,
    pi_diagnostics,
    plot_negative_example,
    postprocess_by_negative_intervals,
    ranges_for_eval,
    ranges_for_group,
    run_config,
    safe_float,
    video_length_for,
)
from scripts.run_spectral_param_scan import DEFAULT_PARAMS, precompute_curves
from scripts.run_spectral_score_decomposition import interval_query


DEFAULT_OUTPUT_DIR = Path("outputs/26-07-07-22-50-low-fp-ablation-scan")
FINAL_METHOD = "low_fp_with_valley_cut_final"
BASE_METHOD = "low_fp_final"
DISPLAY.update(
    {
        FINAL_METHOD: "Low-FP with valley cut final",
        BASE_METHOD: "Low-FP final",
    }
)

FINAL_CONFIG = {
    "fusion_threshold": 0.38,
    "trend_window": 100,
    "trend_threshold": 0.6,
    "airpls_lambda": 1000,
    "peak_mad_k": 3.0,
    "sg_window_length": 17,
    "sg_polyorder": 2,
    "length_penalty_weight": 0.22,
    "low_residual_penalty_weight": 0.15,
    "residual_weight": 0.25,
    "trend_weight": 0.2,
    "sg_weight": 0.0,
    "raw_weight": 0.25,
    "peak_count_weight": 0.0,
    "residual_mad_k": 3.0,
    "peak_stop_ratio": 0.25,
    "merge_gap_frames": 32,
    "enable_sg": True,
    "enable_airpls": True,
    "enable_trend": True,
    "run_name": "low_fp_with_valley_cut_final",
    "enabled_components": "raw, sg, residual, trend, peak_count, length_penalty, low_residual_penalty",
    "disabled_components": "",
    "configuration_type": "final-precision-first",
    "sg_smoothing_enabled": True,
    "sg_candidate_generation_enabled": True,
    "residual_candidate_generation_enabled": True,
    "trend_candidate_generation_enabled": True,
    "post_min_duration": 32,
    "post_min_raw_max": 0.6,
    "enable_negative_evidence_postprocess": True,
    "low_score_threshold": 0.35,
    "raw_low_threshold": 0.35,
    "residual_low_threshold": 0.05,
    "trend_low_threshold": 0.45,
    "min_normal_duration": 96,
    "max_normal_std": 0.12,
    "normal_anchor_margin": 8,
    "protect_sgt_ucgt": True,
    "allow_cut_inside_usgt": True,
    "cut_pi_by_normal_anchor": True,
    "block_merge_across_normal_anchor": True,
    "shrink_pi_boundary_by_normal_anchor": True,
    "require_no_peaks": True,
}

CORE_METRICS = [
    "PI Duration",
    "PI Ratio",
    "s-GT Recall",
    "uc-GT Recall",
    "Eval Recall",
    "GT Precision",
    "Eval Precision",
    "us-GT Coverage",
    "FP Duration",
    "FP Ratio in PI",
]


def fmt(value, digits: int = 3) -> str:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return ""
    if math.isnan(out) or math.isinf(out):
        return ""
    return f"{out:.{digits}f}"


def clone_config(**updates) -> dict:
    cfg = copy.deepcopy(FINAL_CONFIG)
    cfg.update(updates)
    return cfg


def setup_inputs(args):
    warnings = []
    gt_rows = load_gt_rows(args.gt_stats_csv, args.gt_support_csv)
    inventory = load_inventory(args.video_inventory_csv, gt_rows)
    pre_args = argparse.Namespace(output_dir=Path("outputs/26-07-07-18-52-spectral-param-scan/outputs"), reuse_cached_curves=True)
    if not DEFAULT_CACHE_SOURCE.exists():
        pre_args = argparse.Namespace(output_dir=args.output_dir / "outputs", reuse_cached_curves=args.reuse_cached_curves)
    decomp = precompute_curves(pre_args, inventory, warnings)
    existing = []
    auto_scan_methods(args.existing_interval_root, existing)
    add_window_methods(existing, inventory, [100, 300], DEFAULT_PARAMS["trend_threshold"], Path.cwd())
    return gt_rows, inventory, decomp, existing, warnings


def evaluate_config(method_key: str, cfg: dict, existing: list[dict], decomp: dict, gt_rows: list[dict], inventory: dict, warnings: list[dict]) -> tuple[list[dict], dict, dict, list[dict], dict]:
    base_cfg = copy.deepcopy(cfg)
    base_cfg["enable_negative_evidence_postprocess"] = False
    base_cfg["run_name"] = method_key
    base_rows = run_config(method_key, base_cfg, existing, decomp, warnings)
    base_metrics = evaluate_method(method_key, base_rows, gt_rows, inventory)
    if not cfg.get("enable_negative_evidence_postprocess", False):
        return base_rows, base_metrics, {}, [], {}
    negative_by_video = detect_negative_intervals(decomp, cfg)
    final_rows, valley_diag, events = postprocess_by_negative_intervals(method_key, base_rows, negative_by_video, gt_rows, inventory, cfg)
    final_metrics = evaluate_method(method_key, final_rows, gt_rows, inventory)
    final_metrics.update({field: valley_diag.get(field, "") for field in NEGATIVE_FIELDS})
    return final_rows, final_metrics, valley_diag, events, negative_by_video


def add_deltas(rows: list[dict], final_row: dict) -> list[dict]:
    out = []
    for row in rows:
        item = dict(row)
        item["Delta Eval Recall vs final_full"] = safe_float(item.get("Eval Recall"), 0) - safe_float(final_row.get("Eval Recall"), 0)
        item["Delta GT Precision vs final_full"] = safe_float(item.get("GT Precision"), 0) - safe_float(final_row.get("GT Precision"), 0)
        item["Delta FP Duration vs final_full"] = safe_float(item.get("FP Duration"), 0) - safe_float(final_row.get("FP Duration"), 0)
        out.append(item)
    return out


def row_for_table(name: str, metrics: dict, cfg: dict | None = None) -> dict:
    row = {"Variant": name, "Parameter": "", "Value": ""}
    for key in CORE_METRICS:
        row[key] = metrics.get(key, "")
    for key in [
        "Delta Eval Recall vs final_full",
        "Delta GT Precision vs final_full",
        "Delta FP Duration vs final_full",
        "FP Removed by NI",
        "TP Lost by NI",
        "NI-over-sGT Ratio",
        "NI-over-ucGT Ratio",
        "NI-over-usGT Ratio",
        "PI Cut Count",
        "Merge Blocked Count",
    ]:
        row[key] = metrics.get(key, "")
    if cfg:
        row["config_json"] = json.dumps(cfg, ensure_ascii=False, sort_keys=True)
    return row


def format_csv_rows(rows: list[dict]) -> list[dict]:
    formatted = []
    for row in rows:
        out = {}
        for key, value in row.items():
            if isinstance(value, float):
                out[key] = fmt(value)
            elif key in {"PI Duration", "FP Duration", "Delta FP Duration vs final_full", "FP Removed by NI", "TP Lost by NI", "PI Cut Count", "Merge Blocked Count"} and value != "":
                out[key] = int(float(value))
            else:
                out[key] = value
        formatted.append(out)
    return formatted


def markdown_table(rows: list[dict], fields: list[str]) -> str:
    lines = ["| " + " | ".join(fields) + " |", "|" + "|".join(["---"] * len(fields)) + "|"]
    for row in rows:
        vals = []
        for field in fields:
            val = row.get(field, "")
            if isinstance(val, float):
                vals.append(fmt(val))
            elif field in {"PI Duration", "FP Duration", "Delta FP Duration vs final_full", "FP Removed by NI", "TP Lost by NI", "PI Cut Count", "Merge Blocked Count"} and val != "":
                vals.append(str(int(float(val))))
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def select_fields(rows: list[dict], fields: list[str]) -> list[dict]:
    return [{field: row.get(field, "") for field in fields} for row in rows]


def build_ablation_variants() -> list[tuple[str, dict]]:
    return [
        ("final_full", clone_config()),
        ("no_valley_cut", clone_config(enable_negative_evidence_postprocess=False)),
        ("no_length_penalty", clone_config(length_penalty_weight=0.0)),
        ("no_low_residual_penalty", clone_config(low_residual_penalty_weight=0.0)),
        ("no_trend_component", clone_config(trend_weight=0.0, enable_trend=False, trend_candidate_generation_enabled=False)),
        ("no_residual_component", clone_config(residual_weight=0.0, enable_airpls=False, residual_candidate_generation_enabled=False)),
        ("no_sg_candidate", clone_config(sg_candidate_generation_enabled=False)),
        ("no_post_min_duration", clone_config(post_min_duration=0)),
        ("large_merge_gap", clone_config(merge_gap_frames=96)),
        ("small_merge_gap", clone_config(merge_gap_frames=0)),
    ]


def run_variant_set(variants, existing, decomp, gt_rows, inventory, warnings):
    rows_by_name = {}
    preds_by_name = {}
    events_by_name = {}
    ni_by_name = {}
    for name, cfg in variants:
        DISPLAY[name] = name
        cfg = copy.deepcopy(cfg)
        cfg["run_name"] = name
        preds, metrics, _diag, events, ni = evaluate_config(name, cfg, existing, decomp, gt_rows, inventory, warnings)
        rows_by_name[name] = row_for_table(name, metrics, cfg)
        preds_by_name[name] = preds
        events_by_name[name] = events
        ni_by_name[name] = ni
    final_row = rows_by_name["final_full"]
    rows = add_deltas([rows_by_name[name] for name, _cfg in variants], final_row)
    return rows, preds_by_name, events_by_name, ni_by_name


def write_ablation(reports: Path, rows: list[dict]) -> None:
    fields = [
        "Variant",
        "Parameter",
        "Value",
        "PI Duration",
        "PI Ratio",
        "s-GT Recall",
        "uc-GT Recall",
        "Eval Recall",
        "GT Precision",
        "Eval Precision",
        "us-GT Coverage",
        "FP Duration",
        "FP Ratio in PI",
        "Delta Eval Recall vs final_full",
        "Delta GT Precision vs final_full",
        "Delta FP Duration vs final_full",
        "FP Removed by NI",
        "TP Lost by NI",
        "NI-over-sGT Ratio",
        "NI-over-ucGT Ratio",
        "NI-over-usGT Ratio",
        "PI Cut Count",
        "Merge Blocked Count",
        "config_json",
    ]
    write_csv(reports / "low_fp_ablation_summary.csv", select_fields(format_csv_rows(rows), fields), fields)
    md_fields = ["Variant", "PI Ratio", "s-GT Recall", "uc-GT Recall", "Eval Recall", "GT Precision", "FP Duration", "FP Ratio in PI", "Delta Eval Recall vs final_full", "Delta GT Precision vs final_full", "Delta FP Duration vs final_full"]
    (reports / "low_fp_ablation_summary.md").write_text("# Low-FP Ablation Summary\n\n" + markdown_table(rows, md_fields) + "\n", encoding="utf-8")


def scan_specs() -> list[tuple[str, str, list]]:
    return [
        ("fusion_threshold", "fusion_threshold", [0.34, 0.36, 0.38, 0.40, 0.42]),
        ("merge_gap_frames", "merge_gap_frames", [0, 16, 32, 64, 96]),
        ("post_min_duration", "post_min_duration", [16, 32, 48, 64, 96]),
        ("post_min_raw_max", "post_min_raw_max", [0.50, 0.55, 0.60, 0.65, 0.70]),
        ("valley_low_score_threshold", "low_score_threshold", [0.25, 0.30, 0.35, 0.40, 0.45]),
        ("valley_min_normal_duration", "min_normal_duration", [48, 72, 96, 128, 160]),
    ]


def write_scan(reports: Path, scan_name: str, param: str, values: list, existing, decomp, gt_rows, inventory, warnings) -> list[dict]:
    rows = []
    for value in values:
        cfg = clone_config(**{param: value})
        method_key = f"scan_{scan_name}_{str(value).replace('.', 'p')}"
        DISPLAY[method_key] = method_key
        cfg["run_name"] = method_key
        _preds, metrics, _diag, _events, _ni = evaluate_config(method_key, cfg, existing, decomp, gt_rows, inventory, warnings)
        row = row_for_table(method_key, metrics, cfg)
        row["Parameter"] = param
        row["Value"] = value
        rows.append(row)
    fields = [
        "Variant",
        "Parameter",
        "Value",
        "PI Ratio",
        "s-GT Recall",
        "uc-GT Recall",
        "Eval Recall",
        "GT Precision",
        "FP Duration",
        "FP Ratio in PI",
        "us-GT Coverage",
        "FP Removed by NI",
        "TP Lost by NI",
        "NI-over-sGT Ratio",
        "NI-over-ucGT Ratio",
        "NI-over-usGT Ratio",
        "config_json",
    ]
    write_csv(reports / f"scan_{scan_name}.csv", select_fields(format_csv_rows(rows), fields), fields)
    md = "# Scan " + scan_name + "\n\n" + markdown_table(rows, fields[:-1]) + "\n"
    (reports / f"scan_{scan_name}.md").write_text(md, encoding="utf-8")
    return rows


PI_DIAGNOSTIC_FIELDS = [
    "method",
    "dataset",
    "video_id",
    "pi_index",
    "start",
    "end",
    "duration",
    "video_duration",
    "overlap_sGT",
    "overlap_ucGT",
    "overlap_usGT",
    "overlap_nonGT",
    "main_category",
    "is_major_fp",
    "max_score",
    "mean_score",
    "peak_count",
    "merged_from_count",
    "cut_by_valley",
    "blocked_by_valley",
    "s_gt_overlap",
    "uc_gt_overlap",
    "us_gt_overlap",
    "non_gt_overlap",
    "primary_category",
]


def standardized_pi_diagnostics(method_key: str, pred_rows: list[dict], gt_rows: list[dict], inventory: dict, decomp: dict, events: list[dict] | None = None) -> list[dict]:
    rows = pi_diagnostics(method_key, pred_rows, gt_rows, inventory, decomp)
    cut_events = [row for row in (events or []) if row.get("action") == "cut_pi"]
    out = []
    for row in rows:
        item = dict(row)
        item["overlap_sGT"] = item.get("s_gt_overlap", "")
        item["overlap_ucGT"] = item.get("uc_gt_overlap", "")
        item["overlap_usGT"] = item.get("us_gt_overlap", "")
        item["overlap_nonGT"] = item.get("non_gt_overlap", "")
        item["main_category"] = item.get("primary_category", "")
        cut_by_valley = False
        blocked_by_valley = False
        for event in cut_events:
            if event.get("dataset") != item.get("dataset") or event.get("video_id") != item.get("video_id"):
                continue
            pi_start = int(float(event.get("pi_start", -1)))
            pi_end = int(float(event.get("pi_end", -1)))
            start = int(float(item.get("start", -1)))
            end = int(float(item.get("end", -1)))
            if pi_start <= start and end <= pi_end:
                cut_by_valley = True
                anchor_start = int(float(event.get("start", pi_start)))
                anchor_end = int(float(event.get("end", pi_end)))
                blocked_by_valley = pi_start < anchor_start and anchor_end < pi_end
                break
        item["cut_by_valley"] = cut_by_valley
        item["blocked_by_valley"] = blocked_by_valley
        out.append(item)
    return out


def plot_example_for_key(path: Path, title: str, key, before_rows, after_rows, ni_rows, gt_rows, decomp) -> None:
    if key not in decomp:
        return
    data = decomp[key]
    frames = data["frames"]
    raw = data["curves"]["raw_score"]
    gt_by_video = group_gt(gt_rows)
    before = group_predictions(before_rows).get(key, [])
    after = group_predictions(after_rows).get(key, [])
    eval_gt = ranges_for_eval(gt_by_video.get(key, []))
    us_gt = ranges_for_group(gt_by_video.get(key, []), "unsupportable")
    ni = [(int(row["start"]), int(row["end"])) for row in ni_rows.get(key, [])]
    tracks = [("PI before", before, "#E15759"), ("PI after", after, "#59A14F"), ("NI", ni, "#4C78A8"), ("s/uc-GT", eval_gt, "#2CA02C"), ("us-GT", us_gt, "#D62728")]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(tracks) + 1, 1, figsize=(15, 8.8), sharex=True, gridspec_kw={"height_ratios": [2.3] + [0.62] * len(tracks)})
    axes[0].plot(frames, raw, color="#333333", linewidth=1)
    peak_mask = raw >= 0.6
    if np.any(peak_mask):
        axes[0].scatter(frames[peak_mask], raw[peak_mask], s=12, color="#E15759", label="peaks >= 0.6")
        axes[0].legend(loc="upper right", fontsize=8, frameon=False)
    axes[0].set_title(f"{title} | {key[0]} | {key[1]}")
    axes[0].set_ylabel("score")
    for ax, (label, ranges, color) in zip(axes[1:], tracks):
        for start, end in ranges:
            ax.axvspan(start, end, ymin=0.15, ymax=0.85, color=color, alpha=0.75)
        ax.set_ylabel(label, rotation=0, ha="right", va="center")
        ax.set_yticks([])
        ax.grid(axis="x", alpha=0.1)
    axes[-1].set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def write_examples(reports: Path, before_rows, after_rows, events, ni_by_video, gt_rows, decomp, overmerge_rows=None) -> None:
    out = reports / "low_fp_examples"
    event_keys = [(row["dataset"], row["video_id"]) for row in events]
    keys = []
    if event_keys:
        keys.append(("01_valley_cut_fp", event_keys[0]))
    protected = [row for row in events if row.get("action") == "protected_eval_gt_overlap"]
    if protected:
        keys.append(("03_protected_sgt_ucgt", (protected[0]["dataset"], protected[0]["video_id"])))
    diag = pi_diagnostics("low_fp_with_valley_cut_final", after_rows, gt_rows, {}, decomp)
    major_fp = [row for row in diag if str(row.get("is_major_fp")).lower() == "true" or row.get("primary_category") == "non-GT"]
    if major_fp:
        keys.append(("05_far_from_gt_fp", (major_fp[0]["dataset"], major_fp[0]["video_id"])))
    success = [row for row in diag if row.get("primary_category") in {"s-GT", "uc-GT"} and int(row.get("non_gt_overlap", 0)) < int(row.get("duration", 1)) * 0.5]
    if success:
        keys.append(("04_success_interval", (success[0]["dataset"], success[0]["video_id"])))
    grouped_before = group_predictions(before_rows)
    grouped_after = group_predictions(after_rows)
    for key in grouped_before:
        if key not in grouped_after or len(grouped_before[key]) > len(grouped_after[key]):
            keys.append(("02_overmerge_reduced", key))
            break
    seen = set()
    for name, key in keys:
        if (name, key) in seen:
            continue
        seen.add((name, key))
        plot_example_for_key(out / f"{name}.png", name, key, before_rows, after_rows, ni_by_video, gt_rows, decomp)
    if overmerge_rows:
        grouped_overmerge = group_predictions(overmerge_rows)
        for key, intervals in grouped_overmerge.items():
            final_intervals = grouped_after.get(key, [])
            overmerge_duration = max((end - start for start, end in intervals), default=0)
            final_duration = max((end - start for start, end in final_intervals), default=0)
            if overmerge_duration > final_duration or len(intervals) < len(final_intervals):
                plot_example_for_key(out / "02_large_merge_gap_overmerge.png", "02_large_merge_gap_overmerge", key, overmerge_rows, after_rows, ni_by_video, gt_rows, decomp)
                break


def write_report(reports: Path, final_row: dict, ablation_rows: list[dict], scan_rows: dict[str, list[dict]]) -> None:
    by_variant = {row["Variant"]: row for row in ablation_rows}
    no_valley = by_variant["no_valley_cut"]
    lines = [
        "# Low-FP Ablation And Scan Report",
        "",
        "## 1. Final Configuration",
        "",
        "The fixed final precision-first configuration is `low_fp_with_valley_cut_final`. It uses the selected Low-FP base configuration plus negative-evidence / valley post-processing.",
        "",
        "```json",
        json.dumps(FINAL_CONFIG, indent=2, ensure_ascii=False),
        "```",
        "",
        "## 2. Main Result",
        "",
        f"- Eval Recall: {fmt(final_row['Eval Recall'])}.",
        f"- GT Precision: {fmt(final_row['GT Precision'])}.",
        f"- FP Duration: {int(float(final_row['FP Duration']))}.",
        f"- FP Ratio in PI: {fmt(final_row['FP Ratio in PI'])}.",
        "- This is a precision-first operating point, not a recall-optimal configuration.",
        "",
        "## 3. Ablation Study",
        "",
        f"- Valley cut incremental gain versus `no_valley_cut`: FP Duration changes by {int(float(final_row['FP Duration']) - float(no_valley['FP Duration']))} frames, GT Precision changes by {fmt(float(final_row['GT Precision']) - float(no_valley['GT Precision']))}.",
        "- The valley gain is a light refinement; most FP reduction comes from the precision-first base filters and penalties.",
        "",
        markdown_table(ablation_rows, ["Variant", "PI Ratio", "Eval Recall", "GT Precision", "FP Duration", "FP Ratio in PI", "Delta Eval Recall vs final_full", "Delta GT Precision vs final_full", "Delta FP Duration vs final_full"]),
        "",
        "## 4. Parameter Scan",
        "",
        "- The local scans vary one parameter around the final setting while keeping the rest fixed.",
        "- Higher fusion thresholds and stronger post-filters tend to reduce PI/FP but can reduce Eval Recall.",
        "- Larger merge gaps can increase over-merge risk; smaller gaps support precision-first behavior but should be checked for recall loss.",
        "- Valley thresholds control the strength of normal-anchor cuts; the final setting is conservative because s-GT contact is nonzero.",
        "",
    ]
    for name, rows in scan_rows.items():
        lines.extend([f"### {name}", "", markdown_table(rows, ["Parameter", "Value", "PI Ratio", "Eval Recall", "GT Precision", "FP Duration", "FP Ratio in PI", "FP Removed by NI", "TP Lost by NI", "NI-over-sGT Ratio"]), ""])
    lines.extend(
        [
            "## 5. Limitations",
            "",
            "- FP Ratio remains nontrivial even after valley cut.",
            "- GT/VAD evidence is layered and boundary-uncertain.",
            "- Valley detector can touch s-GT; `protect_sgt_ucgt=True` is required.",
            "- Low-FP sacrifices Eval Recall, so it is suited for low-false-positive reporting rather than maximum recall.",
        ]
    )
    (reports / "low_fp_ablation_and_scan_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gt_stats_csv", type=Path, default=Path("outputs/26-07-07-14-43-gt-score-window-curves/outputs/gt_interval_score_stats.csv"))
    parser.add_argument("--gt_support_csv", type=Path, default=Path("outputs/26-07-07-15-14-gt-score-alignment-analysis/outputs/gt_support_classification.csv"))
    parser.add_argument("--video_inventory_csv", type=Path, default=Path("outputs/26-07-07-14-43-gt-score-window-curves/outputs/video_score_curve_inventory.csv"))
    parser.add_argument("--existing_interval_root", type=Path, default=Path("outputs"))
    parser.add_argument("--reuse_cached_curves", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    reports = args.output_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    gt_rows, inventory, decomp, existing, warnings = setup_inputs(args)
    ablation_rows, preds_by_name, events_by_name, ni_by_name = run_variant_set(build_ablation_variants(), existing, decomp, gt_rows, inventory, warnings)
    write_ablation(reports, ablation_rows)
    final_row = next(row for row in ablation_rows if row["Variant"] == "final_full")
    scan_outputs = {}
    for scan_name, param, values in scan_specs():
        scan_outputs[scan_name] = write_scan(reports, scan_name, param, values, existing, decomp, gt_rows, inventory, warnings)
    final_ni = ni_by_name["final_full"]
    final_events = events_by_name["final_full"]
    write_csv(
        reports / "pi_interval_diagnostics_low_fp_final.csv",
        standardized_pi_diagnostics(BASE_METHOD, preds_by_name["no_valley_cut"], gt_rows, inventory, decomp),
        PI_DIAGNOSTIC_FIELDS,
    )
    write_csv(
        reports / "pi_interval_diagnostics_low_fp_with_valley_cut_final.csv",
        standardized_pi_diagnostics(FINAL_METHOD, preds_by_name["final_full"], gt_rows, inventory, decomp, final_events),
        PI_DIAGNOSTIC_FIELDS,
    )
    write_csv(reports / "negative_evidence_diagnostics.csv", [
        {field: final_row.get(field, "") for field in ["Variant", "FP Removed by NI", "TP Lost by NI", "NI-over-sGT Ratio", "NI-over-ucGT Ratio", "NI-over-usGT Ratio", "PI Cut Count", "Merge Blocked Count"]}
    ])
    write_csv(reports / "negative_evidence_intervals_final.csv", negative_diagnostics_rows(FINAL_METHOD, final_ni, gt_rows, inventory))
    write_csv(reports / "negative_evidence_events_final.csv", final_events)
    write_examples(reports, preds_by_name["no_valley_cut"], preds_by_name["final_full"], final_events, final_ni, gt_rows, decomp, preds_by_name["large_merge_gap"])
    write_report(reports, final_row, ablation_rows, scan_outputs)
    shutil.copy2(reports / "low_fp_ablation_and_scan_report.md", args.output_dir / "low-fp-ablation-scan_report.md")

    program = args.output_dir / "programs" / "scripts" / Path(__file__).name
    program.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), program)
    summary = {
        "output_dir": str(args.output_dir).replace("\\", "/"),
        "final_config_name": "low_fp_with_valley_cut_final",
        "report": str(reports / "low_fp_ablation_and_scan_report.md").replace("\\", "/"),
        "ablation_csv": str(reports / "low_fp_ablation_summary.csv").replace("\\", "/"),
        "final_metrics": {key: final_row.get(key, "") for key in CORE_METRICS},
        "warning_count": len(warnings),
    }
    write_json(args.output_dir / "low_fp_ablation_and_scan_summary.json", summary)
    (args.output_dir / "MANIFEST.md").write_text(
        "\n".join(
            [
                "# MANIFEST",
                "",
                "- `low-fp-ablation-scan_report.md`: root archive report copy.",
                "- `reports/low_fp_ablation_and_scan_report.md`: total report.",
                "- `reports/low_fp_ablation_summary.md/csv`: ablation summary.",
                "- `reports/scan_*.md/csv`: local parameter scans.",
                "- `reports/pi_interval_diagnostics_low_fp_final.csv`: PI diagnostics before valley cut.",
                "- `reports/pi_interval_diagnostics_low_fp_with_valley_cut_final.csv`: PI diagnostics after valley cut.",
                "- `reports/negative_evidence_diagnostics.csv`: final NI aggregate diagnostics.",
                "- `reports/low_fp_examples/*.png`: example plots.",
                "- `programs/scripts/run_low_fp_ablation_and_scan.py`: copied generator script.",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
