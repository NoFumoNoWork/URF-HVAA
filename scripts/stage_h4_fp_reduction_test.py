import argparse
import csv
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_interval_methods import (  # noqa: E402
    as_int,
    duration,
    group_gt,
    intersect_duration,
    interval_overlap_duration,
    load_gt_rows,
    load_inventory,
    merge_ranges,
    read_csv,
    write_csv,
)
from scripts.run_spectral_final_materials import DEFAULT_CACHE_SOURCE  # noqa: E402
from scripts.run_spectral_param_scan import precompute_curves  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("outputs/26-07-09-16-49-h4-fp-reduction-test")
DEFAULT_BASELINE_ARCHIVE = Path("outputs/26-07-07-22-50-low-fp-ablation-scan")
DEFAULT_H4_DIR = Path("outputs/26-07-09-15-25-h4-resource-prep")
DEFAULT_GT_STATS = Path("outputs/26-07-07-14-43-gt-score-window-curves/outputs/gt_interval_score_stats.csv")
DEFAULT_GT_SUPPORT = Path("outputs/26-07-07-15-14-gt-score-alignment-analysis/outputs/gt_support_classification.csv")
DEFAULT_INVENTORY = Path("outputs/26-07-07-14-43-gt-score-window-curves/outputs/video_score_curve_inventory.csv")

H4_FILTERS = {
    "all_h4": None,
    "possible_context_forgetting": {"possible_context_forgetting"},
    "lexical_topic_boundary": {"lexical_topic_boundary"},
    "explicit_transition_boundary": {"explicit_transition_boundary"},
    "multi_scene_compression_boundary": {"multi_scene_compression_boundary"},
    "possible_context_forgetting+lexical_topic_boundary": {"possible_context_forgetting", "lexical_topic_boundary"},
    "explicit_transition_boundary+multi_scene_compression_boundary": {"explicit_transition_boundary", "multi_scene_compression_boundary"},
    "exclude_event_onset_not_h4": {"__exclude_event_onset_not_h4__"},
}


def clean_rows(rows: list[dict]) -> list[dict]:
    return [{str(k).strip().lstrip("\ufeff"): v for k, v in row.items()} for row in rows]


def as_float(value, default=0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def boolish(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def split_types(value: str) -> set[str]:
    return {item.strip() for item in str(value or "").replace(",", ";").split(";") if item.strip()}


def h4_type_match(types: set[str], filter_name: str) -> bool:
    spec = H4_FILTERS[filter_name]
    if spec is None:
        return bool(types)
    if "__exclude_event_onset_not_h4__" in spec:
        return bool(types) and "event_onset_not_h4" not in types
    return bool(types & spec)


def load_baseline_intervals(path: Path) -> dict[tuple[str, str], list[tuple[int, int]]]:
    by_video = defaultdict(list)
    for row in clean_rows(read_csv(path)):
        key = (row["dataset"], row["video_id"])
        start = as_int(row.get("start"))
        end = as_int(row.get("end"))
        if end > start:
            by_video[key].append((start, end))
    return {key: merge_ranges(value) for key, value in by_video.items()}


def load_valley_events(path: Path) -> dict[tuple[str, str], list[tuple[int, int]]]:
    by_video = defaultdict(list)
    for row in clean_rows(read_csv(path)):
        if row.get("action") != "cut_pi":
            continue
        key = (row["dataset"], row["video_id"])
        start = as_int(row.get("start"))
        end = as_int(row.get("end"))
        if end > start:
            by_video[key].append((start, end))
    return {key: merge_ranges(value) for key, value in by_video.items()}


def load_h4(h4_dir: Path) -> tuple[dict, dict]:
    candidates = defaultdict(list)
    for row in clean_rows(read_csv(h4_dir / "h4_diagnostic_table.csv")):
        key = (row["dataset"], row["video_id"])
        candidates[key].append(
            {
                "position": as_int(row.get("h4_position")),
                "types": split_types(row.get("h4_type")),
                "score": as_float(row.get("h4_score")),
                "inside_prediction_gap": boolish(row.get("inside_prediction_gap")),
                "near_prediction_gap": boolish(row.get("near_prediction_gap")),
                "nearest_gap_id": row.get("nearest_gap_id", ""),
                "gap_oracle_label": row.get("gap_oracle_label", ""),
            }
        )
    gaps = defaultdict(list)
    for row in clean_rows(read_csv(h4_dir / "prediction_gaps.csv")):
        key = (row["dataset"], row["video_id"])
        types = split_types(row.get("h4_types_near_gap"))
        if not (boolish(row.get("has_h4_in_gap")) or boolish(row.get("has_h4_near_gap"))):
            continue
        gaps[key].append(
            {
                "start": as_int(row.get("gap_start")),
                "end": as_int(row.get("gap_end")),
                "types": types,
                "score": as_float(row.get("strongest_h4_score")),
                "oracle": row.get("merge_oracle_label", ""),
                "gap_id": row.get("gap_id", ""),
            }
        )
    return dict(candidates), dict(gaps)


def frames_and_curve(curve_data: dict, curve_name: str = "rolling_mean_100") -> tuple[np.ndarray, np.ndarray]:
    frames = np.asarray(curve_data["frames"], dtype=int)
    curves = curve_data["curves"]
    values = np.asarray(curves.get(curve_name, curves.get("raw_score")), dtype=float)
    return frames, values


def interval_mean(curve_data: dict, start: int, end: int, curve_name: str = "rolling_mean_100") -> float:
    frames, values = frames_and_curve(curve_data, curve_name)
    mask = (frames >= start) & (frames < end)
    if not np.any(mask):
        return math.nan
    return float(np.nanmean(values[mask]))


def point_score(curve_data: dict, position: int, curve_name: str = "rolling_mean_100") -> float:
    frames, values = frames_and_curve(curve_data, curve_name)
    if len(frames) == 0:
        return math.nan
    idx = int(np.argmin(np.abs(frames - position)))
    return float(values[idx])


def low_score_ranges(curve_data: dict, start: int, end: int, threshold: float, min_len: int) -> list[tuple[int, int]]:
    frames, values = frames_and_curve(curve_data, "rolling_mean_100")
    stride = int(curve_data.get("stride", 16) or 16)
    mask = (frames >= start) & (frames < end) & (values < threshold)
    ranges = []
    run_start = None
    last = None
    for frame, active in zip(frames, mask):
        frame = int(frame)
        if active and run_start is None:
            run_start = frame
        if not active and run_start is not None:
            stop = int(last) + stride
            if stop - run_start >= min_len:
                ranges.append((run_start, stop))
            run_start = None
        last = frame
    if run_start is not None and last is not None:
        stop = int(last) + stride
        if stop - run_start >= min_len:
            ranges.append((run_start, stop))
    return merge_ranges(ranges)


def subtract_ranges(intervals: list[tuple[int, int]], cuts: list[tuple[int, int]]) -> list[tuple[int, int]]:
    pieces = []
    cuts = merge_ranges(cuts)
    for start, end in intervals:
        cursor = start
        for cut_start, cut_end in cuts:
            if cut_end <= cursor or cut_start >= end:
                continue
            if cut_start > cursor:
                pieces.append((cursor, min(cut_start, end)))
            cursor = max(cursor, cut_end)
            if cursor >= end:
                break
        if cursor < end:
            pieces.append((cursor, end))
    return merge_ranges(pieces)


def filter_segments(intervals: list[tuple[int, int]], curve_data: dict | None, min_duration: int, min_mean_score: float | None) -> list[tuple[int, int]]:
    kept = []
    for start, end in intervals:
        if end - start < min_duration:
            continue
        if curve_data is not None and min_mean_score is not None:
            score = interval_mean(curve_data, start, end)
            if not math.isnan(score) and score < min_mean_score:
                continue
        kept.append((start, end))
    return merge_ranges(kept)


def score_only_filter(baseline: dict, decomp: dict, min_mean: float, min_duration: int) -> dict:
    out = {}
    for key, intervals in baseline.items():
        data = decomp.get(key)
        out[key] = filter_segments(intervals, data, min_duration, min_mean)
    return out


def valley_only_stronger(baseline: dict, decomp: dict, low_threshold: float, min_low_len: int, min_duration: int) -> dict:
    out = {}
    for key, intervals in baseline.items():
        data = decomp.get(key)
        if not data:
            out[key] = intervals
            continue
        cuts = []
        for start, end in intervals:
            cuts.extend(low_score_ranges(data, start, end, low_threshold, min_low_len))
        out[key] = filter_segments(subtract_ranges(intervals, cuts), data, min_duration, None)
    return out


def h4_suppression(baseline: dict, decomp: dict, h4_candidates: dict, filter_name: str, local_score_max: float, cut_margin: int, min_duration: int, min_mean: float) -> dict:
    out = {}
    for key, intervals in baseline.items():
        data = decomp.get(key)
        cuts = []
        for start, end in intervals:
            for cand in h4_candidates.get(key, []):
                pos = cand["position"]
                if not (start + cut_margin < pos < end - cut_margin):
                    continue
                if not h4_type_match(cand["types"], filter_name):
                    continue
                score = point_score(data, pos) if data else math.nan
                if math.isnan(score) or score <= local_score_max:
                    cuts.append((pos - cut_margin, pos + cut_margin))
        out[key] = filter_segments(subtract_ranges(intervals, cuts), data, min_duration, min_mean)
    return out


def h4_gated_valley(baseline: dict, decomp: dict, valley_events: dict, h4_candidates: dict, filter_name: str, h4_window: int, expand_margin: int, min_duration: int, min_mean: float) -> dict:
    out = {}
    for key, intervals in baseline.items():
        data = decomp.get(key)
        cuts = []
        for start, end in valley_events.get(key, []):
            near = False
            for cand in h4_candidates.get(key, []):
                if h4_type_match(cand["types"], filter_name) and start - h4_window <= cand["position"] <= end + h4_window:
                    near = True
                    break
            if near:
                cuts.append((start - expand_margin, end + expand_margin))
        out[key] = filter_segments(subtract_ranges(intervals, cuts), data, min_duration, min_mean)
    return out


def simple_gap_merge(baseline: dict, decomp: dict, max_gap: int, min_gap_score: float, h4_gaps: dict | None = None, filter_name: str | None = None) -> dict:
    out = {}
    for key, intervals in baseline.items():
        data = decomp.get(key)
        items = sorted(intervals)
        if not items:
            out[key] = []
            continue
        merged = [items[0]]
        for start, end in items[1:]:
            last_start, last_end = merged[-1]
            gap_len = start - last_end
            should_merge = gap_len <= max_gap
            if should_merge and data is not None and gap_len > 0:
                gap_score = interval_mean(data, last_end, start)
                should_merge = math.isnan(gap_score) or gap_score >= min_gap_score
            if should_merge and h4_gaps is not None and filter_name is not None:
                for gap in h4_gaps.get(key, []):
                    overlaps_gap = max(last_end, gap["start"]) < min(start, gap["end"])
                    near_gap = gap["start"] <= start + 32 and gap["end"] >= last_end - 32
                    if (overlaps_gap or near_gap) and h4_type_match(gap["types"], filter_name):
                        should_merge = False
                        break
            if should_merge:
                merged[-1] = (last_start, end)
            else:
                merged.append((start, end))
        out[key] = merge_ranges(merged)
    return out


def flatten_intervals(intervals_by_video: dict) -> list[dict]:
    rows = []
    for key in sorted(intervals_by_video):
        for idx, (start, end) in enumerate(intervals_by_video[key]):
            rows.append({"dataset": key[0], "video_id": key[1], "interval_id": f"{key[1]}_{idx}", "start": start, "end": end, "length": end - start})
    return rows


def evaluate(intervals_by_video: dict, gt_by_video: dict, inventory: dict) -> dict:
    keys = set(inventory) | set(gt_by_video) | set(intervals_by_video)
    tp = fp = fn = tn = total_duration = 0
    pred_count = fp_interval_count = tp_interval_count = 0
    lengths = []
    gt_hit_count = 0
    gt_count = 0
    pred_over_gt_count = 0
    for key in keys:
        pred = merge_ranges(intervals_by_video.get(key, []))
        gt = merge_ranges([(int(row["start"]), int(row["end"])) for row in gt_by_video.get(key, [])])
        video_len = as_int(inventory.get(key, {}).get("video_length"))
        if not video_len:
            candidates = [0]
            candidates.extend([end for _, end in pred])
            candidates.extend([end for _, end in gt])
            video_len = max(candidates)
        pred_dur = duration(pred)
        gt_dur = duration(gt)
        overlap = intersect_duration(pred, gt)
        tp += overlap
        fp += pred_dur - overlap
        fn += gt_dur - overlap
        total_duration += max(video_len, duration(merge_ranges(pred + gt)))
        pred_count += len(pred)
        lengths.extend([end - start for start, end in pred])
        for interval in pred:
            if interval_overlap_duration(interval, gt) > 0:
                tp_interval_count += 1
                pred_over_gt_count += 1
            else:
                fp_interval_count += 1
        for interval in gt:
            gt_count += 1
            if interval_overlap_duration(interval, pred) > 0:
                gt_hit_count += 1
    tn = max(0, total_duration - tp - fp - fn)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    seg_precision = tp_interval_count / pred_count if pred_count else 0.0
    seg_recall = gt_hit_count / gt_count if gt_count else 0.0
    seg_f1 = 2 * seg_precision * seg_recall / (seg_precision + seg_recall) if seg_precision + seg_recall else 0.0
    frag_ratio = pred_over_gt_count / gt_hit_count if gt_hit_count else 0.0
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "num_final_predicted_intervals": pred_count,
        "num_FP_intervals": fp_interval_count,
        "num_TP_overlapping_intervals": tp_interval_count,
        "mean_interval_length": mean(lengths) if lengths else 0.0,
        "median_interval_length": median(lengths) if lengths else 0.0,
        "GT_fragmentation_ratio": frag_ratio,
        "segment_precision": seg_precision,
        "segment_recall": seg_recall,
        "segment_F1": seg_f1,
        "total_duration": total_duration,
    }


def build_method_rows(methods: list[dict], baseline_metrics: dict) -> list[dict]:
    rows = []
    tp0 = baseline_metrics["TP"]
    fp0 = baseline_metrics["FP"]
    recall0 = baseline_metrics["recall"]
    precision0 = baseline_metrics["precision"]
    for item in methods:
        metrics = item["metrics"]
        delta_tp = metrics["TP"] - tp0
        delta_fp = metrics["FP"] - fp0
        fp_reduction = fp0 - metrics["FP"]
        tp_lost = max(0, tp0 - metrics["TP"])
        recall_loss = max(0.0, recall0 - metrics["recall"])
        precision_gain = metrics["precision"] - precision0
        row = {
            "method_name": item["method_name"],
            "method_family": item["method_family"],
            "h4_type_filter": item.get("h4_type_filter", "none"),
            "threshold_params": item.get("threshold_params", ""),
            "TP": metrics["TP"],
            "FP": metrics["FP"],
            "FN": metrics["FN"],
            "TN": metrics["TN"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "F1": metrics["F1"],
            "delta_TP": delta_tp,
            "delta_FP": delta_fp,
            "delta_recall": metrics["recall"] - recall0,
            "delta_precision": precision_gain,
            "FP_reduction": fp_reduction,
            "TP_retention": metrics["TP"] / tp0 if tp0 else 0.0,
            "recall_retention": metrics["recall"] / recall0 if recall0 else 0.0,
            "FP_reduction_per_TP_lost": fp_reduction / tp_lost if tp_lost else math.inf if fp_reduction > 0 else 0.0,
            "precision_gain_per_recall_loss": precision_gain / recall_loss if recall_loss else math.inf if precision_gain > 0 else 0.0,
            "net_utility_lambda_1": fp_reduction - 1 * tp_lost,
            "net_utility_lambda_2": fp_reduction - 2 * tp_lost,
            "net_utility_lambda_5": fp_reduction - 5 * tp_lost,
            "net_utility_lambda_10": fp_reduction - 10 * tp_lost,
            "pass_tp_retention_99": metrics["TP"] / tp0 >= 0.99 if tp0 else False,
            "pass_tp_retention_98": metrics["TP"] / tp0 >= 0.98 if tp0 else False,
            "pass_tp_retention_95": metrics["TP"] / tp0 >= 0.95 if tp0 else False,
            "on_pareto_front": False,
            "num_final_predicted_intervals": metrics["num_final_predicted_intervals"],
            "num_FP_intervals": metrics["num_FP_intervals"],
            "num_TP_overlapping_intervals": metrics["num_TP_overlapping_intervals"],
            "mean_interval_length": metrics["mean_interval_length"],
            "median_interval_length": metrics["median_interval_length"],
            "GT_fragmentation_ratio": metrics["GT_fragmentation_ratio"],
            "segment_precision": metrics["segment_precision"],
            "segment_recall": metrics["segment_recall"],
            "segment_F1": metrics["segment_F1"],
        }
        rows.append(row)
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            better_or_equal = other["FP"] <= row["FP"] and other["TP_retention"] >= row["TP_retention"] and other["precision"] >= row["precision"]
            strictly_better = other["FP"] < row["FP"] or other["TP_retention"] > row["TP_retention"] or other["precision"] > row["precision"]
            if better_or_equal and strictly_better:
                dominated = True
                break
        row["on_pareto_front"] = not dominated
    return rows


def best_by_constraints(rows: list[dict], thresholds: list[float]) -> list[dict]:
    out = []
    for threshold in thresholds:
        eligible = [row for row in rows if row["TP_retention"] >= threshold]
        if not eligible:
            out.append({"constraint": f"TP_retention>={threshold}", "best_method": "none", "FP": "", "TP_retention": "", "precision": "", "recall": "", "reason": "No method met the constraint."})
            continue
        best = min(eligible, key=lambda row: (row["FP"], -row["precision"], -row["recall"]))
        out.append(
            {
                "constraint": f"TP_retention>={threshold}",
                "best_method": best["method_name"],
                "FP": best["FP"],
                "TP_retention": best["TP_retention"],
                "precision": best["precision"],
                "recall": best["recall"],
                "reason": "Lowest FP among methods satisfying the TP retention constraint.",
            }
        )
    return out


def is_h4(row: dict) -> bool:
    return row["method_family"] in {"M4_h4_suppression_cut", "M5_h4_gated_valley_cut", "M6_h4_veto_merge"}


def decide_added_value(rows: list[dict], baseline: dict) -> dict:
    h4_rows = [row for row in rows if is_h4(row)]
    non_h4_rows = [row for row in rows if not is_h4(row)]
    useful_h4 = [row for row in h4_rows if row["FP"] < baseline["FP"] and row["TP_retention"] >= 0.98 and row["precision"] > baseline["precision"]]
    best_h4 = min(h4_rows, key=lambda row: (0 if row["TP_retention"] >= 0.98 else 1, row["FP"], -row["precision"]), default=None)
    best_non_h4 = min([row for row in non_h4_rows if row["TP_retention"] >= 0.98], key=lambda row: (row["FP"], -row["precision"]), default=None)
    if useful_h4 and best_non_h4:
        best_useful_h4 = min(useful_h4, key=lambda row: (row["FP"], -row["precision"]))
        if best_useful_h4["FP"] < best_non_h4["FP"]:
            case = "Case 1"
            has_added = True
            reason = "An H4 variant reduced FP at TP retention >= 0.98 and beat score-only/valley-only controls."
            chosen_h4 = best_useful_h4
        else:
            case = "Case 2"
            has_added = False
            reason = "H4 reduced FP, but non-H4 score/valley controls reached equal or lower FP under the same TP-retention constraint."
            chosen_h4 = best_useful_h4
    elif any(row["FP"] < baseline["FP"] for row in h4_rows):
        case = "Case 3"
        has_added = False
        reason = "Some H4 variants reduced FP, but they did not preserve TP at the required level or did not improve precision."
        chosen_h4 = best_h4
    else:
        case = "Case 4"
        has_added = False
        reason = "H4 variants did not reduce FP relative to the low-FP baseline."
        chosen_h4 = best_h4
    return {
        "has_added_value": has_added,
        "best_h4_method": chosen_h4["method_name"] if chosen_h4 else "none",
        "best_non_h4_method": best_non_h4["method_name"] if best_non_h4 else "none",
        "conclusion_case": case,
        "short_reason": reason,
        "recommended_next_step": "Run type-specific semantic-continuity validation before enabling H4 rules." if has_added else "Treat current H4 as diagnostic overlay unless future VLM outputs add explicit event-continuity labels.",
    }


def write_plots(output_dir: Path, rows: list[dict], baseline_name: str) -> None:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    colors = ["#4E79A7" if not is_h4(row) else "#E15759" for row in rows]
    sizes = [80 if row["method_name"] == baseline_name else 36 for row in rows]

    plt.figure(figsize=(9, 6))
    plt.scatter([row["TP_retention"] for row in rows], [row["FP"] for row in rows], c=colors, s=sizes, alpha=0.75)
    for row in rows:
        if row["method_name"] == baseline_name or row["on_pareto_front"]:
            plt.annotate(row["method_name"][:36], (row["TP_retention"], row["FP"]), fontsize=7, alpha=0.8)
    plt.xlabel("TP retention vs baseline")
    plt.ylabel("FP duration")
    plt.title("Pareto view: FP duration vs TP retention")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(fig_dir / "pareto_fp_vs_tp_retention.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.scatter([row["recall"] for row in rows], [row["precision"] for row in rows], c=colors, s=sizes, alpha=0.75)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision/recall trade-off: H4 variants vs controls")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(fig_dir / "precision_recall_tradeoff_h4_vs_baselines.png", dpi=160)
    plt.close()

    eligible = sorted([row for row in rows if row["TP_retention"] >= 0.98], key=lambda row: row["FP_reduction"], reverse=True)[:24]
    plt.figure(figsize=(11, 6))
    plt.bar(range(len(eligible)), [row["FP_reduction"] for row in eligible], color=["#E15759" if is_h4(row) else "#4E79A7" for row in eligible])
    plt.xticks(range(len(eligible)), [row["method_name"][:30] for row in eligible], rotation=75, ha="right", fontsize=7)
    plt.ylabel("FP reduction vs baseline")
    plt.title("FP reduction under TP retention >= 0.98")
    plt.tight_layout()
    plt.savefig(fig_dir / "fp_reduction_by_method.png", dpi=160)
    plt.close()

    h4_best = []
    for filter_name in H4_FILTERS:
        candidates = [row for row in rows if is_h4(row) and row["h4_type_filter"] == filter_name]
        if candidates:
            h4_best.append(max(candidates, key=lambda row: row["FP_reduction"]))
    plt.figure(figsize=(10, 5.8))
    plt.bar(range(len(h4_best)), [row["FP_reduction"] for row in h4_best], color="#E15759")
    plt.xticks(range(len(h4_best)), [row["h4_type_filter"] for row in h4_best], rotation=70, ha="right", fontsize=8)
    plt.ylabel("Best FP reduction vs baseline")
    plt.title("H4 type restrictions: best FP reduction")
    plt.tight_layout()
    plt.savefig(fig_dir / "h4_type_fp_reduction.png", dpi=160)
    plt.close()


def write_case_plot(output_dir: Path, baseline: dict, variant: dict, gt_by_video: dict, decomp: dict) -> None:
    fig_dir = output_dir / "figures"
    best_key = None
    best_gain = -1
    for key in set(baseline) | set(variant):
        gt = merge_ranges([(int(row["start"]), int(row["end"])) for row in gt_by_video.get(key, [])])
        base_fp = duration(baseline.get(key, [])) - intersect_duration(baseline.get(key, []), gt)
        var_fp = duration(variant.get(key, [])) - intersect_duration(variant.get(key, []), gt)
        base_tp = intersect_duration(baseline.get(key, []), gt)
        var_tp = intersect_duration(variant.get(key, []), gt)
        gain = (base_fp - var_fp) - 2 * max(0, base_tp - var_tp)
        if gain > best_gain and key in decomp:
            best_key = key
            best_gain = gain
    if best_key is None:
        return
    frames, raw = frames_and_curve(decomp[best_key], "raw_score")
    _, smooth = frames_and_curve(decomp[best_key], "rolling_mean_100")
    gt = merge_ranges([(int(row["start"]), int(row["end"])) for row in gt_by_video.get(best_key, [])])
    plt.figure(figsize=(14, 7))
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(frames, raw, color="#333333", linewidth=1.0)
    for start, end in gt:
        ax1.axvspan(start, end, color="#2CA02C", alpha=0.22)
    ax1.set_ylabel("raw score + GT")
    ax1.set_title(f"Case visualization: {best_key[0]} / {best_key[1]}")
    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    for start, end in baseline.get(best_key, []):
        ax2.axvspan(start, end, ymin=0.18, ymax=0.82, color="#4E79A7", alpha=0.75)
    ax2.set_yticks([])
    ax2.set_ylabel("baseline")
    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    ax3.plot(frames, smooth, color="#666666", linewidth=1.0)
    for start, end in variant.get(best_key, []):
        ax3.axvspan(start, end, ymin=0.0, ymax=0.25, color="#E15759", alpha=0.45)
    ax3.set_ylabel("variant + smooth")
    ax3.set_xlabel("frame")
    for ax in [ax1, ax2, ax3]:
        ax.grid(axis="x", alpha=0.15)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(fig_dir / "best_variant_case_visualization.png", dpi=160)
    plt.close()


def fmt(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isinf(value):
            return "inf"
        return f"{value:.4f}"
    return str(value)


def report_table(rows: list[dict], fields: list[str], limit: int = 12) -> list[str]:
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] + ["---:" for _ in fields[1:]]) + " |"]
    for row in rows[:limit]:
        lines.append("| " + " | ".join(fmt(row.get(field, "")) for field in fields) + " |")
    return lines


def write_report(output_dir: Path, rows: list[dict], best_rows: list[dict], decision: dict, baseline_metrics: dict, skipped: list[str], input_checks: list[dict]) -> None:
    baseline_row = next(row for row in rows if row["method_family"] == "M0_baseline")
    best_h4 = next((row for row in rows if row["method_name"] == decision["best_h4_method"]), None)
    best_non = next((row for row in rows if row["method_name"] == decision["best_non_h4_method"]), None)
    top98 = sorted([row for row in rows if row["TP_retention"] >= 0.98], key=lambda row: (row["FP"], -row["precision"]))[:10]
    h4_type_summary = []
    for filter_name in H4_FILTERS:
        candidates = [row for row in rows if is_h4(row) and row["h4_type_filter"] == filter_name]
        if candidates:
            best = max(candidates, key=lambda row: row["FP_reduction"])
            h4_type_summary.append(best)
    pareto = [row for row in rows if row["on_pareto_front"]]
    lines = [
        "# Stage: H4 Added-value Test for FP Reduction",
        "",
        "## 1. Question",
        "",
        "This stage asks whether H4, as a caption-derived structural feature, can reduce false-positive duration while preserving the TP/recall behavior of the current low-FP system. H4 is not evaluated as a frame-level score booster here; it is only tested as a cut, suppression, valley-gating, or merge-veto signal.",
        "",
        "## Input Check",
        "",
        "| input | status | role |",
        "| --- | --- | --- |",
    ]
    for item in input_checks:
        lines.append(f"| `{item['path']}` | {item['status']} | {item['role']} |")
    lines.extend(
        [
            "",
        "## 2. Baseline",
        "",
        "- Baseline: `low_fp_with_valley_cut_final` / `final-precision-first`.",
        "- Source: `outputs/26-07-07-22-50-low-fp-ablation-scan/reports/pi_interval_diagnostics_low_fp_with_valley_cut_final.csv`.",
        "- H4 is disabled in the baseline.",
        "- Config description: fusion threshold `0.38`, final low-FP intervals restored from the prior archive, valley-cut negative evidence already applied in the baseline outputs.",
        "",
        *report_table([baseline_row], ["method_name", "TP", "FP", "FN", "precision", "recall", "F1", "num_final_predicted_intervals", "num_FP_intervals"], 1),
        "",
        "## 3. Experimental Design",
        "",
        "- M0: baseline low-FP final intervals.",
        "- M1: score-only stricter filters over the same final intervals.",
        "- M2: valley-cut-only stronger setting using low-smoothed-score valleys, without H4.",
        "- M3: simple gap / score-shape merge, without H4.",
        "- M4: H4-risky boundary suppression, cutting final intervals around H4 candidates when local score is weak.",
        "- M5: H4-gated valley cut, expanding valley removals only when selected H4 types are nearby.",
        "- M6: H4 veto over naive gap merge, rejecting otherwise allowed merges when selected H4 types mark the gap.",
        "",
        "The score-only and valley-only controls are required because H4 has added value only if it beats ordinary stricter post-processing under the same TP-retention constraint.",
        "",
        "## 4. Metrics and Success Criteria",
        "",
        "- `TP`, `FP`, and `FN` are frame-duration counts against all GT intervals.",
        "- `TP_retention = TP_variant / TP_baseline` and `recall_retention = recall_variant / recall_baseline`.",
        "- `FP_reduction = FP_baseline - FP_variant`.",
        "- A useful variant should reduce FP, keep `TP_retention >= 0.98` or `recall_retention >= 0.98`, improve precision, and avoid substantially worse F1.",
        "- Pareto front uses lower FP, higher TP retention, and higher precision.",
        "- Net utility is `FP_reduction - lambda * TP_loss`, with lambda in `[1, 2, 5, 10]`.",
        "",
        "## 5. Results",
        "",
        "- CSV: `method_metrics.csv`.",
        "- CSV: `best_methods_by_constraint.csv`.",
        "- Decision JSON: `h4_added_value_decision.json`.",
        "- Figure: `figures/pareto_fp_vs_tp_retention.png`.",
        "- Figure: `figures/precision_recall_tradeoff_h4_vs_baselines.png`.",
        "- Figure: `figures/fp_reduction_by_method.png`.",
        "",
        *report_table(top98, ["method_name", "method_family", "h4_type_filter", "TP_retention", "FP", "FP_reduction", "precision", "recall", "F1"], 10),
        "",
        "## 6. H4 Type Analysis",
        "",
        "Best FP-reduction result by H4 type filter:",
        "",
        *report_table(sorted(h4_type_summary, key=lambda row: row["FP_reduction"], reverse=True), ["h4_type_filter", "method_name", "TP_retention", "FP_reduction", "precision", "recall"], 12),
        "",
        "## 7. Pareto Analysis",
        "",
        f"- Pareto-front methods: {len(pareto)}.",
        f"- H4 on Pareto front: {'yes' if any(is_h4(row) for row in pareto) else 'no'}.",
        "",
        *report_table(sorted(pareto, key=lambda row: (row["FP"], -row["TP_retention"]))[:12], ["method_name", "method_family", "h4_type_filter", "TP_retention", "FP", "precision", "recall"], 12),
        "",
        "Best methods by TP-retention constraint:",
        "",
        *report_table(best_rows, ["constraint", "best_method", "FP", "TP_retention", "precision", "recall", "reason"], 10),
        "",
        "## 8. Case Study",
        "",
        "- Figure: `figures/best_variant_case_visualization.png`.",
        "- If the best H4 method satisfies the success standard, the figure shows a success case; otherwise it shows the strongest/failure case where H4 altered intervals but did not establish independent added value.",
        "",
        "## 9. Interpretation",
        "",
        f"- Decision: `{decision['conclusion_case']}`.",
        f"- Has added value: `{decision['has_added_value']}`.",
        f"- Best H4 method: `{decision['best_h4_method']}`.",
        f"- Best non-H4 method: `{decision['best_non_h4_method']}`.",
        f"- Reason: {decision['short_reason']}",
        "",
    ]
    )
    if best_h4:
        lines.extend(["Best H4 summary:", "", *report_table([best_h4], ["method_name", "TP_retention", "FP", "FP_reduction", "precision", "recall", "F1"], 1), ""])
    if best_non:
        lines.extend(["Best non-H4 summary:", "", *report_table([best_non], ["method_name", "TP_retention", "FP", "FP_reduction", "precision", "recall", "F1"], 1), ""])
    lines.extend(
        [
            "## 10. Limitations",
            "",
            "- H4 is a caption-level feature, not a verified camera transition or visual shot boundary.",
            "- Original video was not available to judge whether two sides of a boundary are the same event.",
            "- This experiment only tests added value over the current low-FP pipeline and its available score/valley resources.",
            "- If H4 does not beat score-only or valley-only controls, it cannot be claimed as an independent contributor to FP reduction.",
            "- Some H4 construction details, such as the exact near-gap window, are inherited from prior resource-prep outputs rather than recomputed here.",
            "",
            "## 11. Next Step",
            "",
            f"{decision['recommended_next_step']}",
        ]
    )
    if skipped:
        lines.extend(["", "## Input or Method Warnings", "", *[f"- {item}" for item in skipped]])
    (output_dir / "stage_h4_fp_reduction_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_archive", type=Path, default=DEFAULT_BASELINE_ARCHIVE)
    parser.add_argument("--h4_resource_dir", type=Path, default=DEFAULT_H4_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tp_retention_thresholds", nargs="+", type=float, default=[0.99, 0.98, 0.95])
    parser.add_argument("--seed", type=int, default=20260709)
    parser.add_argument("--gt_stats_csv", type=Path, default=DEFAULT_GT_STATS)
    parser.add_argument("--gt_support_csv", type=Path, default=DEFAULT_GT_SUPPORT)
    parser.add_argument("--video_inventory_csv", type=Path, default=DEFAULT_INVENTORY)
    args = parser.parse_args()

    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "figures").mkdir(parents=True, exist_ok=True)
    skipped = []
    required = [
        (args.baseline_archive / "reports" / "pi_interval_diagnostics_low_fp_with_valley_cut_final.csv", "baseline final intervals"),
        (args.baseline_archive / "reports" / "negative_evidence_events_final.csv", "baseline valley-cut event archive"),
        (args.h4_resource_dir / "prediction_gaps.csv", "H4 gap-level bridge/recheck candidates"),
        (args.h4_resource_dir / "h4_diagnostic_table.csv", "H4 candidate positions and types"),
        (args.gt_stats_csv, "GT interval score statistics"),
        (args.gt_support_csv, "supportable/uncertain/unsupportable GT labels"),
        (args.video_inventory_csv, "video score-curve inventory and duration estimates"),
    ]
    input_checks = []
    for path, role in required:
        input_checks.append({"path": path.as_posix(), "status": "found" if path.exists() else "missing", "role": role})
        if not path.exists():
            skipped.append(f"Missing required input: `{path.as_posix()}`")
    if skipped:
        (args.output_dir / "stage_h4_fp_reduction_report.md").write_text("# Stage: H4 Added-value Test for FP Reduction\n\n" + "\n".join(f"- {item}" for item in skipped) + "\n", encoding="utf-8")
        raise FileNotFoundError("; ".join(skipped))

    gt_rows = load_gt_rows(args.gt_stats_csv, args.gt_support_csv)
    gt_by_video = group_gt(gt_rows)
    inventory = load_inventory(args.video_inventory_csv, gt_rows)
    warnings = []
    pre_args = SimpleNamespace(output_dir=DEFAULT_CACHE_SOURCE.parent.parent if DEFAULT_CACHE_SOURCE.exists() else args.output_dir / "outputs", reuse_cached_curves=True)
    decomp = precompute_curves(pre_args, inventory, warnings)
    baseline = load_baseline_intervals(args.baseline_archive / "reports" / "pi_interval_diagnostics_low_fp_with_valley_cut_final.csv")
    valley_events = load_valley_events(args.baseline_archive / "reports" / "negative_evidence_events_final.csv")
    h4_candidates, h4_gaps = load_h4(args.h4_resource_dir)

    methods = []

    def add_method(name: str, family: str, intervals: dict, params: str, h4_filter: str = "none") -> None:
        methods.append(
            {
                "method_name": name,
                "method_family": family,
                "h4_type_filter": h4_filter,
                "threshold_params": params,
                "intervals": intervals,
                "metrics": evaluate(intervals, gt_by_video, inventory),
            }
        )

    add_method("M0_baseline_low_fp", "M0_baseline", baseline, "fusion_threshold=0.38; H4=false")
    for min_mean in [0.28, 0.32, 0.36, 0.40, 0.44]:
        add_method(f"M1_score_only_mean_ge_{min_mean:.2f}", "M1_score_only_stricter_threshold", score_only_filter(baseline, decomp, min_mean, 32), f"min_mean_score={min_mean}; min_duration=32")
    for threshold in [0.18, 0.22, 0.26, 0.30]:
        for min_low_len in [16, 32, 48]:
            add_method(f"M2_valley_only_low_{threshold:.2f}_len_{min_low_len}", "M2_valley_cut_only_stronger", valley_only_stronger(baseline, decomp, threshold, min_low_len, 32), f"low_smooth_threshold={threshold}; min_low_len={min_low_len}; min_duration=32")
    for max_gap in [32, 64, 96, 128]:
        for min_gap_score in [0.00, 0.20, 0.30, 0.40]:
            add_method(f"M3_simple_merge_gap_{max_gap}_score_{min_gap_score:.2f}", "M3_simple_gap_score_shape_merge", simple_gap_merge(baseline, decomp, max_gap, min_gap_score), f"max_gap={max_gap}; min_gap_score={min_gap_score}")
    for filter_name in H4_FILTERS:
        for local_score in [0.25, 0.35, 0.45]:
            add_method(
                f"M4_h4_cut_{filter_name}_score_{local_score:.2f}",
                "M4_h4_suppression_cut",
                h4_suppression(baseline, decomp, h4_candidates, filter_name, local_score, 8, 32, 0.20),
                f"local_score_max={local_score}; cut_margin=8; min_duration=32; min_mean=0.20",
                filter_name,
            )
        for expand in [16, 32, 64]:
            add_method(
                f"M5_h4_valley_{filter_name}_expand_{expand}",
                "M5_h4_gated_valley_cut",
                h4_gated_valley(baseline, decomp, valley_events, h4_candidates, filter_name, 64, expand, 32, 0.20),
                f"h4_window=64; expand_margin={expand}; min_duration=32; min_mean=0.20",
                filter_name,
            )
        for max_gap in [64, 96, 128]:
            add_method(
                f"M6_h4_veto_{filter_name}_gap_{max_gap}",
                "M6_h4_veto_merge",
                simple_gap_merge(baseline, decomp, max_gap, 0.20, h4_gaps, filter_name),
                f"max_gap={max_gap}; min_gap_score=0.20; veto_filter={filter_name}",
                filter_name,
            )

    baseline_metrics = methods[0]["metrics"]
    rows = build_method_rows(methods, baseline_metrics)
    fields = list(rows[0].keys())
    write_csv(args.output_dir / "method_metrics.csv", rows, fields)
    best_rows = best_by_constraints(rows, args.tp_retention_thresholds)
    write_csv(args.output_dir / "best_methods_by_constraint.csv", best_rows, ["constraint", "best_method", "FP", "TP_retention", "precision", "recall", "reason"])
    decision = decide_added_value(rows, rows[0])
    (args.output_dir / "h4_added_value_decision.json").write_text(json.dumps(decision, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_plots(args.output_dir, rows, "M0_baseline_low_fp")
    selected_variant_name = decision["best_h4_method"] if decision["best_h4_method"] != "none" else "M0_baseline_low_fp"
    selected_method = next((m for m in methods if m["method_name"] == selected_variant_name), methods[0])
    write_case_plot(args.output_dir, baseline, selected_method["intervals"], gt_by_video, decomp)
    write_report(args.output_dir, rows, best_rows, decision, baseline_metrics, skipped, input_checks)
    write_csv(args.output_dir / "baseline_intervals.csv", flatten_intervals(baseline), ["dataset", "video_id", "interval_id", "start", "end", "length"])
    write_csv(args.output_dir / "selected_h4_variant_intervals.csv", flatten_intervals(selected_method["intervals"]), ["dataset", "video_id", "interval_id", "start", "end", "length"])
    program_dir = args.output_dir / "programs" / "scripts"
    program_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), program_dir / Path(__file__).name)
    print(
        json.dumps(
            {
                "output_dir": args.output_dir.as_posix(),
                "baseline": {key: baseline_metrics[key] for key in ["TP", "FP", "FN", "precision", "recall", "F1"]},
                "best_h4_method": decision["best_h4_method"],
                "best_non_h4_method": decision["best_non_h4_method"],
                "has_added_value": decision["has_added_value"],
                "conclusion_case": decision["conclusion_case"],
                "recommended_next_step": decision["recommended_next_step"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
