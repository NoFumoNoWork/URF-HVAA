import argparse
import csv
import json
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_interval_methods import group_gt, load_gt_rows, load_inventory, merge_ranges, write_csv  # noqa: E402
from scripts.plot_low_fp_visualization import (  # noqa: E402
    DEFAULT_GT_STATS,
    DEFAULT_GT_SUPPORT,
    DEFAULT_INVENTORY,
    DEFAULT_SOURCE_ARCHIVE,
    HIGH_UNMARKED_THRESHOLD,
    LOW_MARKED_THRESHOLD,
    NORMAL_ANCHOR_MARGIN,
    group_intervals,
    in_any_interval,
    ranges_for_gt,
    read_csv,
)
from scripts.run_spectral_final_materials import DEFAULT_CACHE_SOURCE  # noqa: E402
from scripts.run_spectral_param_scan import precompute_curves  # noqa: E402
from scripts.run_spectral_score_decomposition import group_positive_runs, local_find_peaks, safe_name  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("outputs/26-07-09-16-18-low-fp-h4-visualization")
DEFAULT_H4_DIR = Path("outputs/26-07-09-15-25-h4-resource-prep")
DEFAULT_VIS_INDEX = Path("outputs/26-07-08-11-15-low-fp-visualization/reports/low_fp_visualization_index.csv")
CONFIGS_PATH = Path("configs/selected_low_fp_config.yaml")


def boolish(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def clean_rows(rows: list[dict]) -> list[dict]:
    cleaned = []
    for row in rows:
        cleaned.append({str(key).strip().lstrip("\ufeff"): value for key, value in row.items()})
    return cleaned


def as_int(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def as_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def write_text(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def yaml_scalar(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if value is None:
        return "null"
    text = str(value)
    if text == "" or any(ch in text for ch in [":", "#", "{", "}", "[", "]", ","]):
        return json.dumps(text, ensure_ascii=False)
    return text


def write_yaml(path: Path, payload: dict) -> None:
    lines = []
    for key, value in payload.items():
        if isinstance(value, dict):
            lines.append(f"{key}:")
            for sub_key in sorted(value):
                lines.append(f"  {sub_key}: {yaml_scalar(value[sub_key])}")
        else:
            lines.append(f"{key}: {yaml_scalar(value)}")
    write_text(path, lines)


def select_final_config(scan_csv: Path) -> tuple[dict, dict]:
    rows = clean_rows(read_csv(scan_csv))
    preferred = None
    for row in rows:
        if row.get("Variant") == "scan_fusion_threshold_0p38":
            preferred = row
            break
    if preferred is None:
        for row in rows:
            if row.get("Parameter") == "fusion_threshold" and row.get("Value") in {"0.380", "0.38"}:
                preferred = row
                break
    if preferred is None:
        raise FileNotFoundError(f"Could not locate final fusion_threshold=0.38 config in {scan_csv}")
    config = json.loads(preferred["config_json"])
    metrics = {k: preferred[k] for k in preferred if k not in {"config_json"}}
    return config, metrics


def load_h4_by_video(h4_dir: Path) -> tuple[dict[tuple[str, str], list[dict]], dict[tuple[str, str], list[dict]]]:
    gap_rows = clean_rows(read_csv(h4_dir / "prediction_gaps.csv"))
    h4_rows = clean_rows(read_csv(h4_dir / "h4_diagnostic_table.csv"))
    gaps_by_video = defaultdict(list)
    h4_by_video = defaultdict(list)
    for row in gap_rows:
        if boolish(row.get("has_h4_in_gap")) or boolish(row.get("has_h4_near_gap")):
            gaps_by_video[(row["dataset"], row["video_id"])].append(row)
    for row in h4_rows:
        if boolish(row.get("inside_prediction_gap")) or boolish(row.get("near_prediction_gap")):
            h4_by_video[(row["dataset"], row["video_id"])].append(row)
    return gaps_by_video, h4_by_video


def choose_case(vis_index: Path, gaps_by_video: dict, h4_by_video: dict) -> tuple[tuple[str, str], dict]:
    rows = clean_rows(read_csv(vis_index))
    best_key = None
    best_score = -1
    best_row = {}
    for row in rows:
        key = (row["dataset"], row["video_id"])
        h4_gap_count = len(gaps_by_video.get(key, []))
        h4_candidate_count = len(h4_by_video.get(key, []))
        final_count = as_int(row.get("final_pi_count"))
        valley_count = as_int(row.get("valley_cut_count"))
        high_count = as_int(row.get("high_unmarked_peak_count"))
        low_count = as_int(row.get("low_marked_segment_count"))
        gt_count = as_int(row.get("gt_supportable_count")) + as_int(row.get("gt_uncertain_count")) + as_int(row.get("gt_unsupportable_count"))
        if final_count <= 0 or h4_gap_count <= 0:
            continue
        score = (
            h4_gap_count * 4
            + h4_candidate_count
            + valley_count * 6
            + final_count * 2
            + low_count * 2
            + high_count * 3
            + gt_count * 2
        )
        if valley_count <= 0:
            score -= 20
        if score > best_score:
            best_key = key
            best_score = score
            best_row = dict(row)
            best_row["h4_gap_count"] = h4_gap_count
            best_row["h4_candidate_count"] = h4_candidate_count
            best_row["selection_score"] = score
    if best_key is None:
        raise RuntimeError("No case found with both final low-FP intervals and H4 near-gap candidates.")
    return best_key, best_row


def event_ranges(rows: list[dict]) -> list[tuple[int, int]]:
    ranges = []
    for row in rows:
        start = max(as_int(row.get("pi_start", row.get("start"))), as_int(row.get("start")) - NORMAL_ANCHOR_MARGIN)
        end = min(as_int(row.get("pi_end", row.get("end"))), as_int(row.get("end")) + NORMAL_ANCHOR_MARGIN)
        if end > start:
            ranges.append((start, end))
    return merge_ranges(ranges)


def gap_ranges(rows: list[dict]) -> list[tuple[int, int]]:
    return merge_ranges([(as_int(row["gap_start"]), as_int(row["gap_end"])) for row in rows if as_int(row["gap_end"]) > as_int(row["gap_start"])])


def build_case_tables(
    output_dir: Path,
    key: tuple[str, str],
    data: dict,
    gt_rows: list[dict],
    pi_ranges: list[tuple[int, int]],
    cut_ranges: list[tuple[int, int]],
    h4_gap_rows: list[dict],
    h4_rows: list[dict],
) -> dict:
    frames = data["frames"]
    stride = int(data.get("stride", 16) or 16)
    curves = data["curves"]
    raw = curves["raw_score"]
    smooth = curves.get("rolling_mean_100", raw)
    h4_ranges = gap_ranges(h4_gap_rows)

    high_peak_idx = local_find_peaks(smooth, HIGH_UNMARKED_THRESHOLD, 1e-6)
    high_unmarked = {
        int(frames[idx])
        for idx in high_peak_idx
        if smooth[idx] > HIGH_UNMARKED_THRESHOLD and not in_any_interval(int(frames[idx]), pi_ranges)
    }
    low_marked_mask = np.asarray([in_any_interval(int(frame), pi_ranges) and value < LOW_MARKED_THRESHOLD for frame, value in zip(frames, smooth)], dtype=bool)
    low_marked_ranges = group_positive_runs(frames, low_marked_mask, stride, max_gap_frames=stride * 2)

    curve_rows = []
    for frame, raw_value, smooth_value in zip(frames, raw, smooth):
        frame_i = int(frame)
        curve_rows.append(
            {
                "dataset": key[0],
                "video_id": key[1],
                "frame": frame_i,
                "raw_score": f"{float(raw_value):.6f}",
                "smooth_100": f"{float(smooth_value):.6f}",
                "inside_final_low_fp_interval": int(in_any_interval(frame_i, pi_ranges)),
                "inside_h4_bridge_gap": int(in_any_interval(frame_i, h4_ranges)),
                "high_unmarked_peak": int(frame_i in high_unmarked),
                "low_marked_point": int(bool(low_marked_mask[len(curve_rows)])),
            }
        )
    write_csv(
        output_dir / "case_curve_data.csv",
        curve_rows,
        [
            "dataset",
            "video_id",
            "frame",
            "raw_score",
            "smooth_100",
            "inside_final_low_fp_interval",
            "inside_h4_bridge_gap",
            "high_unmarked_peak",
            "low_marked_point",
        ],
    )

    gt_out = []
    for idx, row in enumerate(gt_rows):
        gt_out.append(
            {
                "interval_id": f"gt_{idx}",
                "dataset": key[0],
                "video_id": key[1],
                "source_stage": "gt_support_classification",
                "interval_role": row.get("support_group", ""),
                "start": row["start"],
                "end": row["end"],
                "length": as_int(row["end"]) - as_int(row["start"]),
                "label": row.get("label", ""),
                "anomaly_id": row.get("anomaly_id", ""),
                "score_summary": "",
                "removed_by_valley_cut": 0,
                "involved_in_h4_merge": 0,
                "final_selected": "",
                "notes": "GT interval used for evaluation/visualization, not a model input.",
            }
        )
    write_csv(
        output_dir / "case_gt_intervals.csv",
        gt_out,
        [
            "interval_id",
            "dataset",
            "video_id",
            "source_stage",
            "interval_role",
            "start",
            "end",
            "length",
            "label",
            "anomaly_id",
            "score_summary",
            "removed_by_valley_cut",
            "involved_in_h4_merge",
            "final_selected",
            "notes",
        ],
    )

    final_out = [
        {
            "interval_id": f"final_{idx}",
            "dataset": key[0],
            "video_id": key[1],
            "source_stage": "final_low_fp",
            "interval_role": "predicted_abnormal",
            "start": s,
            "end": e,
            "length": e - s,
            "score_summary": "",
            "removed_by_valley_cut": 0,
            "involved_in_h4_merge": int(any(max(s, as_int(row["gap_start"])) < min(e, as_int(row["gap_end"])) for row in h4_gap_rows)),
            "final_selected": 1,
            "notes": "Final interval retained by low-FP with valley cut.",
        }
        for idx, (s, e) in enumerate(pi_ranges)
    ]
    cut_out = [
        {
            "interval_id": f"valley_cut_{idx}",
            "dataset": key[0],
            "video_id": key[1],
            "source_stage": "negative_evidence_postprocess",
            "interval_role": "valley_cut_removed",
            "start": s,
            "end": e,
            "length": e - s,
            "score_summary": "",
            "removed_by_valley_cut": 1,
            "involved_in_h4_merge": int(any(max(s, as_int(row["gap_start"])) < min(e, as_int(row["gap_end"])) for row in h4_gap_rows)),
            "final_selected": 0,
            "notes": "Removed by normal-anchor/valley-cut negative evidence.",
        }
        for idx, (s, e) in enumerate(cut_ranges)
    ]
    h4_gap_out = []
    for idx, row in enumerate(h4_gap_rows):
        h4_gap_out.append(
            {
                "interval_id": f"h4_gap_{idx}",
                "dataset": key[0],
                "video_id": key[1],
                "source_stage": "h4_gap_enrichment",
                "interval_role": "h4_bridge_or_recheck_candidate_gap",
                "gap_id": row.get("gap_id", ""),
                "start": row.get("gap_start", ""),
                "end": row.get("gap_end", ""),
                "length": row.get("gap_len", ""),
                "score_summary": f"left_mean={row.get('left_mean_score', '')}; gap_mean={row.get('gap_mean_score', '')}; right_mean={row.get('right_mean_score', '')}",
                "removed_by_valley_cut": 0,
                "involved_in_h4_merge": 1,
                "final_selected": "",
                "notes": "Caption/gap-level H4 bridge or recheck candidate; not enabled as frame-level fusion.",
                "has_h4_in_gap": row.get("has_h4_in_gap", ""),
                "has_h4_near_gap": row.get("has_h4_near_gap", ""),
                "h4_count_in_gap": row.get("h4_count_in_gap", ""),
                "h4_count_near_gap": row.get("h4_count_near_gap", ""),
                "h4_types_near_gap": row.get("h4_types_near_gap", ""),
                "strongest_h4_score": row.get("strongest_h4_score", ""),
                "merge_oracle_label": row.get("merge_oracle_label", ""),
            }
        )
    h4_candidate_out = []
    for row in h4_rows:
        h4_candidate_out.append(
            {
                "dataset": key[0],
                "video_id": key[1],
                "h4_id": row.get("h4_id", ""),
                "h4_position": row.get("h4_position", ""),
                "h4_type": row.get("h4_type", ""),
                "h4_score": row.get("h4_score", ""),
                "inside_gt": row.get("inside_gt", ""),
                "inside_prediction_gap": row.get("inside_prediction_gap", ""),
                "near_prediction_gap": row.get("near_prediction_gap", ""),
                "nearest_gap_id": row.get("nearest_gap_id", ""),
                "gap_oracle_label": row.get("gap_oracle_label", ""),
                "source_dataset_or_proxy": row.get("source_dataset_or_proxy", ""),
            }
        )

    write_csv(output_dir / "case_final_intervals.csv", final_out, ["interval_id", "dataset", "video_id", "source_stage", "interval_role", "start", "end", "length", "score_summary", "removed_by_valley_cut", "involved_in_h4_merge", "final_selected", "notes"])
    write_csv(output_dir / "case_valley_cut_intervals.csv", cut_out, ["interval_id", "dataset", "video_id", "source_stage", "interval_role", "start", "end", "length", "score_summary", "removed_by_valley_cut", "involved_in_h4_merge", "final_selected", "notes"])
    write_csv(
        output_dir / "case_h4_gaps.csv",
        h4_gap_out,
        [
            "dataset",
            "video_id",
            "source_stage",
            "interval_role",
            "interval_id",
            "gap_id",
            "start",
            "end",
            "length",
            "score_summary",
            "removed_by_valley_cut",
            "involved_in_h4_merge",
            "final_selected",
            "notes",
            "has_h4_in_gap",
            "has_h4_near_gap",
            "h4_count_in_gap",
            "h4_count_near_gap",
            "h4_types_near_gap",
            "strongest_h4_score",
            "merge_oracle_label",
        ],
    )
    write_csv(
        output_dir / "case_h4_candidates.csv",
        h4_candidate_out,
        [
            "dataset",
            "video_id",
            "h4_id",
            "h4_position",
            "h4_type",
            "h4_score",
            "inside_gt",
            "inside_prediction_gap",
            "near_prediction_gap",
            "nearest_gap_id",
            "gap_oracle_label",
            "source_dataset_or_proxy",
        ],
    )
    breakdown = final_out + cut_out + h4_gap_out + gt_out
    write_csv(output_dir / "case_interval_breakdown.csv", breakdown, sorted({field for row in breakdown for field in row}))

    return {
        "high_unmarked_peak_count": len(high_unmarked),
        "low_marked_segment_count": len(low_marked_ranges),
        "h4_gap_count": len(h4_gap_rows),
        "h4_candidate_count": len(h4_rows),
        "final_pi_count": len(pi_ranges),
        "valley_cut_count": len(cut_ranges),
        "gt_support_counts": dict(Counter(row.get("support_group", "unknown") for row in gt_rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source_archive", type=Path, default=DEFAULT_SOURCE_ARCHIVE)
    parser.add_argument("--h4_dir", type=Path, default=DEFAULT_H4_DIR)
    parser.add_argument("--visualization_index_csv", type=Path, default=DEFAULT_VIS_INDEX)
    parser.add_argument("--gt_stats_csv", type=Path, default=DEFAULT_GT_STATS)
    parser.add_argument("--gt_support_csv", type=Path, default=DEFAULT_GT_SUPPORT)
    parser.add_argument("--video_inventory_csv", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--video_id", default="")
    parser.add_argument("--reuse_cached_curves", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "programs" / "scripts").mkdir(parents=True, exist_ok=True)

    required = [
        args.source_archive / "reports" / "low_fp_parameter_scan_summary.csv",
        args.source_archive / "reports" / "pi_interval_diagnostics_low_fp_with_valley_cut_final.csv",
        args.source_archive / "reports" / "negative_evidence_events_final.csv",
        args.h4_dir / "prediction_gaps.csv",
        args.h4_dir / "h4_diagnostic_table.csv",
        args.gt_stats_csv,
        args.gt_support_csv,
        args.video_inventory_csv,
    ]
    checks = []
    missing = []
    for path in required:
        exists = path.exists()
        checks.append(f"- `{path.as_posix()}`: {'found' if exists else 'missing'}")
        if not exists:
            missing.append(path.as_posix())
    if missing:
        raise FileNotFoundError("Missing required inputs: " + ", ".join(missing))

    config, metrics = select_final_config(args.source_archive / "reports" / "low_fp_parameter_scan_summary.csv")
    config_payload = {
        "config_name": "low_fp_with_valley_cut_final",
        "selected_from": (args.source_archive / "reports" / "low_fp_parameter_scan_summary.csv").as_posix(),
        "selection_reason": "precision-first low-FP operating point with valley-cut negative evidence postprocess; selected from prior ablation archive, not rerun in this stage",
        "metrics": metrics,
        "parameters": config,
    }
    write_yaml(args.output_dir / "selected_low_fp_config.yaml", config_payload)
    CONFIGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_yaml(CONFIGS_PATH, config_payload)

    gaps_by_video, h4_by_video = load_h4_by_video(args.h4_dir)
    if args.dataset and args.video_id:
        key = (args.dataset, args.video_id)
        selected_row = {"manual_selection": True, "h4_gap_count": len(gaps_by_video.get(key, [])), "h4_candidate_count": len(h4_by_video.get(key, []))}
    else:
        key, selected_row = choose_case(args.visualization_index_csv, gaps_by_video, h4_by_video)

    gt_rows = load_gt_rows(args.gt_stats_csv, args.gt_support_csv)
    gt_by_video = group_gt(gt_rows)
    inventory = load_inventory(args.video_inventory_csv, gt_rows)
    warnings = []
    pre_args = SimpleNamespace(output_dir=DEFAULT_CACHE_SOURCE.parent.parent if DEFAULT_CACHE_SOURCE.exists() else args.output_dir / "outputs", reuse_cached_curves=args.reuse_cached_curves)
    decomp = precompute_curves(pre_args, inventory, warnings)
    if key not in decomp:
        raise RuntimeError(f"Selected case {key} was not found in cached/precomputed score curves")

    source_reports = args.source_archive / "reports"
    pi_rows = clean_rows(read_csv(source_reports / "pi_interval_diagnostics_low_fp_with_valley_cut_final.csv"))
    event_rows = clean_rows(read_csv(source_reports / "negative_evidence_events_final.csv"))
    pi_by_video = group_intervals(pi_rows)
    cut_by_video = defaultdict(list)
    for row in event_rows:
        if row.get("action") == "cut_pi":
            cut_by_video[(row["dataset"], row["video_id"])].append(row)

    pi_ranges = pi_by_video.get(key, [])
    cut_ranges = event_ranges(cut_by_video.get(key, []))
    h4_gap_rows = gaps_by_video.get(key, [])
    h4_rows = h4_by_video.get(key, [])
    case_stats = build_case_tables(args.output_dir, key, decomp[key], gt_by_video.get(key, []), pi_ranges, cut_ranges, h4_gap_rows, h4_rows)

    write_text(args.output_dir / "input_check.md", ["# Input Check", "", *checks])
    write_text(
        args.output_dir / "missing_or_uncertain_inputs.md",
        [
            "# Missing Or Uncertain Inputs",
            "",
            "- No missing required CSV inputs were detected for this stage.",
            "- Original videos and frame-accurate shot-boundary annotations were not available in these resources.",
            "- H4 remains a caption/gap-level proxy signal here. It is not treated as visual shot-boundary evidence or as an enabled frame-level fusion component.",
            "- The supportable/uncertain/unsupportable labels come from existing GT-score alignment resources and should be audited against raw video before being used as visual evidence.",
        ],
    )
    write_text(
        args.output_dir / "selected_case_info.md",
        [
            "# Selected Case Info",
            "",
            f"- Dataset: `{key[0]}`",
            f"- Video ID: `{key[1]}`",
            "- Selection rule: choose a case with final low-FP intervals, H4 near/in-gap candidates, and preferably valley-cut plus low/high diagnostic markers.",
            f"- Selection metadata: `{json.dumps(selected_row, ensure_ascii=False)}`",
            f"- Final low-FP abnormal intervals: {case_stats['final_pi_count']}",
            f"- Valley-cut removed intervals: {case_stats['valley_cut_count']}",
            f"- H4 bridge/recheck candidate gaps: {case_stats['h4_gap_count']}",
            f"- H4 candidates near/in prediction gaps: {case_stats['h4_candidate_count']}",
            f"- High unmarked peaks above {HIGH_UNMARKED_THRESHOLD}: {case_stats['high_unmarked_peak_count']}",
            f"- Low marked segments below {LOW_MARKED_THRESHOLD}: {case_stats['low_marked_segment_count']}",
            f"- GT support counts: `{json.dumps(case_stats['gt_support_counts'], ensure_ascii=False)}`",
        ],
    )
    write_text(
        args.output_dir / "low_fp_config_notes.md",
        [
            "# Low-FP Config Notes",
            "",
            "- Selected config: `low_fp_with_valley_cut_final` / `final-precision-first` from the prior ablation archive.",
            "- It is a low-FP operating point because it uses a stricter fusion threshold (`0.38`), low-score and residual penalties, minimum-duration filtering, normal-anchor blocking/shrinking, and valley-cut negative-evidence postprocessing.",
            f"- Prior metrics: Eval Recall `{metrics.get('Eval Recall')}`, GT Precision `{metrics.get('GT Precision')}`, FP Duration `{metrics.get('FP Duration')}`, FP Ratio in PI `{metrics.get('FP Ratio in PI')}`.",
            "- Compared with looser ablations, the trade-off is lower recall in exchange for fewer false-positive frames and cleaner interval boundaries.",
            "- H4 is not enabled as a frame-level fusion input in this selected config. This stage overlays H4 only as a gap/interval-level merge or recheck proxy.",
        ],
    )
    write_text(
        args.output_dir / "current_system_workflow.md",
        [
            "# Current low-FP system workflow",
            "",
            "## 1. Inputs",
            "",
            "- VLM captions: used upstream by the H4 resource-prep stage to derive caption-level boundary candidates.",
            "- Anomaly score / refined score: cached per-video score curves loaded through the existing spectral decomposition cache.",
            "- GT intervals: used for evaluation and visualization only; they are not an inference-time model input.",
            "- Supportability / risk-support labels: loaded from `gt-score-alignment-analysis` resources as supportable / uncertain / unsupportable GT groups.",
            "- H4 candidates: loaded from `h4_diagnostic_table.csv` and `prediction_gaps.csv` as caption/gap-level boundary or bridge evidence.",
            "- Prior low-FP outputs: final interval diagnostics and negative-evidence events from the low-FP ablation archive.",
            "",
            "## 2. Main pipeline",
            "",
            "1. Load caption-derived resources, score curves, GT support labels, final interval diagnostics, and valley-cut event tables.",
            "2. Reuse the prior `final-precision-first` low-FP configuration with `fusion_threshold=0.38` and the existing cached score curves.",
            "3. Generate/restore interval proposals from the prior low-FP archive rather than rerunning a new parameter scan.",
            "4. Apply the archived final low-FP interval output, which already reflects thresholding, penalties, minimum-duration filtering, normal-anchor logic, and valley-cut negative evidence.",
            "5. Overlay H4 prediction-gap candidates as possible merge/recheck regions. In this stage they are diagnostic overlays, not frame-level score components.",
            "6. Export per-frame curve data plus interval tables for GT support, final low-FP intervals, valley-cut removals, and H4 bridge/recheck gaps.",
            "7. Draw the four-row visualization and write audit notes.",
            "",
            "## 3. Role of H4",
            "",
            "- H4 is not used as a frame-level anomaly score term in the selected low-FP configuration.",
            "- H4 is shown as a prediction-gap-level merge or recheck signal: it can indicate that two adjacent abnormal fragments may need review or reconstruction.",
            "- H4 candidates are not treated as true camera transitions, true scene changes, or guaranteed event boundaries.",
            "- H4 gaps can correspond to same event continuation, new viewpoint, related consequence, narrative jump, caption phrasing change, or true scene change.",
            "",
            "## 4. Rationale",
            "",
            "- The selected low-FP configuration suppresses false positives through a stricter fusion threshold, low-score/residual penalties, minimum-duration filtering, and normal-anchor/valley-cut negative evidence.",
            "- This can reduce spurious intervals, but it can also fragment events or miss weak abnormal portions.",
            "- Valley cut and H4 are useful to inspect together because valley cut removes low-evidence regions while H4 may flag nearby gaps that deserve interval-level merge or recheck.",
            "",
            "## 5. Known limitations",
            "",
            "- H4 is caption-level proxy evidence, not visual shot-boundary evidence.",
            "- H4-based merge could still connect different events without semantic or visual validation.",
            "- High-score missed peaks and low-score selected regions show that score-only thresholds are imperfect and need interval-level diagnostics.",
            "- Supportable / uncertain / unsupportable labels come from existing GT-score alignment analysis and should be audited against raw video before being used as visual evidence.",
        ],
    )
    write_text(
        args.output_dir / "current_system_parameters.md",
        [
            "# Current low-FP system parameters",
            "",
            "| Parameter | Current value | Source | Role |",
            "| --- | ---: | --- | --- |",
            f"| selected_config | `low_fp_with_valley_cut_final` | prior ablation archive | final low-FP operating point |",
            f"| fusion_threshold | `{config.get('fusion_threshold')}` | selected config | stricter interval proposal threshold |",
            f"| raw_weight | `{config.get('raw_weight')}` | selected config | score fusion component |",
            f"| residual_weight | `{config.get('residual_weight')}` | selected config | residual evidence component |",
            f"| trend_weight | `{config.get('trend_weight')}` | selected config | trend evidence component |",
            f"| sg_weight | `{config.get('sg_weight')}` | selected config | SG component weight, effectively disabled in score weight |",
            f"| length_penalty_weight | `{config.get('length_penalty_weight')}` | selected config | penalizes interval shapes likely to inflate FP |",
            f"| low_residual_penalty_weight | `{config.get('low_residual_penalty_weight')}` | selected config | suppresses weak residual intervals |",
            f"| low_score_threshold | `{config.get('low_score_threshold')}` | selected config | negative evidence threshold |",
            f"| raw_low_threshold | `{config.get('raw_low_threshold')}` | selected config | raw-score low evidence threshold |",
            f"| post_min_duration | `{config.get('post_min_duration')}` frames | selected config | removes very short final intervals |",
            f"| merge_gap_frames | `{config.get('merge_gap_frames')}` frames | selected config | non-H4 small-gap merge setting in archived pipeline |",
            f"| min_normal_duration | `{config.get('min_normal_duration')}` frames | selected config | normal-anchor minimum support |",
            f"| normal_anchor_margin | `{config.get('normal_anchor_margin')}` frames | selected config | margin used around normal anchor / valley cut |",
            f"| max_normal_std | `{config.get('max_normal_std')}` | selected config | constrains normal-anchor stability |",
            f"| trend_window | `{config.get('trend_window')}` frames | selected config | trend extraction window |",
            f"| sg_window_length | `{config.get('sg_window_length')}` | selected config | Savitzky-Golay window length |",
            f"| sg_polyorder | `{config.get('sg_polyorder')}` | selected config | Savitzky-Golay polynomial order |",
            f"| airpls_lambda | `{config.get('airpls_lambda')}` | selected config | baseline/residual extraction |",
            f"| smoothing_window_for_figure | `100` frames | cached `rolling_mean_100` | fourth-row visualization curve |",
            f"| high_unmarked_threshold | `{HIGH_UNMARKED_THRESHOLD}` | visualization rule | marks unreported smoothed peaks |",
            f"| low_marked_threshold | `{LOW_MARKED_THRESHOLD}` | visualization rule | marks selected low-score portions |",
            "| H4 window | `unknown` | inferred from `prediction_gaps.csv` near/in-gap flags | exact construction belongs to prior H4 resource-prep stage |",
            "| H4 type restrictions | `none applied in this stage` | visualization rule | all near/in-gap H4 types are displayed |",
            "| H4 score constraint | `none applied in this stage` | visualization rule | scores are exported but not filtered for the figure |",
            "| enable_h4_merge | `false for final low-FP output; overlay only here` | current stage policy | H4 is a merge/recheck proxy, not final fusion |",
            "| enable_semantic_continuity | `unknown / not applied here` | current resources | would require a separate continuity check before strict H4 merge |",
            "",
            "Parameters marked `unknown` are not reconstructed by this stage; they should be recovered from the H4 resource-prep code before a formal H4 merge experiment.",
        ],
    )
    write_text(
        args.output_dir / "README_visualization.md",
        [
            "# Low-FP H4 Visualization Package",
            "",
            "## Inputs",
            "",
            "- Low-FP archive: `outputs/26-07-07-22-50-low-fp-ablation-scan/`.",
            "- Prior visualization index: `outputs/26-07-08-11-15-low-fp-visualization/reports/low_fp_visualization_index.csv`.",
            "- H4 resources: `outputs/26-07-09-15-25-h4-resource-prep/prediction_gaps.csv` and `h4_diagnostic_table.csv`.",
            "- GT support labels: `outputs/26-07-07-15-14-gt-score-alignment-analysis/outputs/gt_support_classification.csv`.",
            "",
            "## Run Commands",
            "",
            "```bash",
            "python scripts/extract_low_fp_case_data.py",
            "python scripts/plot_low_fp_h4_visualization.py",
            "```",
            "",
            "To switch case:",
            "",
            "```bash",
            "python scripts/extract_low_fp_case_data.py --dataset UCF-Crime --video_id Explosion010_x264",
            "python scripts/plot_low_fp_h4_visualization.py",
            "```",
            "",
            "To switch the low-FP source archive, pass `--source_archive` to the extraction script. The archive must contain compatible final interval diagnostics, negative-evidence events, and parameter summary files.",
            "",
            "## Outputs",
            "",
            "- `fig_low_fp_case_visualization.png/.pdf/.svg`: report figure generated by `scripts/plot_low_fp_h4_visualization.py`.",
            "- `case_curve_data.csv`: per-frame raw score, 100-frame smooth score, final interval membership, and H4-gap membership.",
            "- `case_interval_breakdown.csv`: unified interval table for GT support groups, final low-FP intervals, valley-cut removals, and H4 bridge/recheck gaps.",
            "- `case_h4_gaps.csv` and `case_h4_candidates.csv`: H4 proxy evidence for the selected case.",
            "- `selected_case_info.md`: why this case was selected and its diagnostic counts.",
            "- `low_fp_config_notes.md`, `current_system_workflow.md`, and `current_system_parameters.md`: audit notes for the current system state.",
            "- `missing_or_uncertain_inputs.md`: explicit limits of the current evidence package.",
        ],
    )
    write_text(
        args.output_dir / "visualization_notes.md",
        [
            "# Visualization Notes",
            "",
            "- Supportable / uncertain / unsupportable GT labels were found and loaded from the existing GT-score alignment analysis.",
            "- The selected case contains supportable GT intervals; uncertain and unsupportable intervals are absent for this case, so their legend entries are retained for consistency but no spans appear.",
            "- Row 3 uses H4 bridge/recheck candidate gaps from `prediction_gaps.csv`. These are not confirmed H4 merge outputs from the final low-FP system.",
            "- H4 candidate positions are caption-level boundary proxies and should not be read as visual camera transitions.",
        ],
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "output_dir": args.output_dir.as_posix(),
                "selected_case": {"dataset": key[0], "video_id": key[1]},
                "case_stats": case_stats,
                "selected_config": "low_fp_with_valley_cut_final",
                "source_archive": args.source_archive.as_posix(),
                "h4_dir": args.h4_dir.as_posix(),
                "warnings": len(warnings),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    shutil.copy2(Path(__file__), args.output_dir / "programs" / "scripts" / Path(__file__).name)
    print(json.dumps({"output_dir": args.output_dir.as_posix(), "selected_case": {"dataset": key[0], "video_id": key[1]}, "case_stats": case_stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
