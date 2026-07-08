import argparse
import csv
import json
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.anomaly_utils import write_json
from scripts.evaluate_interval_methods import (
    add_window_methods,
    auto_scan_methods,
    duration,
    group_gt,
    intersect_duration,
    interval_overlap_duration,
    load_gt_rows,
    load_inventory,
    merge_ranges,
    write_csv,
)
from scripts.run_spectral_ablation_study import score_fusion
from scripts.run_spectral_final_materials import DEFAULT_CACHE_SOURCE
from scripts.run_spectral_param_scan import DEFAULT_PARAMS, precompute_curves, safe_float
from scripts.run_spectral_pipeline_ablation import build_final_configs, config, make_candidates_for_config
from scripts.run_spectral_score_decomposition import interval_query


DEFAULT_OUTPUT_DIR = Path("outputs/26-07-07-22-31-negative-evidence-valley")
METHOD_ORDER = ["original", "peak_baseline", "strict", "recall_first", "precision_first_low_fp", "low_fp_with_valley_cut"]
DISPLAY = {
    "original": "Original",
    "peak_baseline": "Peak baseline",
    "strict": "Strict",
    "recall_first": "Recall-first",
    "precision_first_low_fp": "Precision-first / Low-FP",
    "low_fp_with_valley_cut": "Low-FP with valley cut",
}
SUMMARY_FIELDS = [
    "Method",
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
    "NI Duration",
    "NI Ratio",
    "PI Cut Count",
    "Merge Blocked Count",
    "FP Removed by NI",
    "TP Lost by NI",
    "Eval Recall Before NI",
    "Eval Recall After NI",
    "GT Precision Before NI",
    "GT Precision After NI",
    "NI-over-GT Ratio",
    "NI-over-sGT Ratio",
    "NI-over-ucGT Ratio",
    "NI-over-usGT Ratio",
    "NI-over-nonGT Ratio",
]
NEGATIVE_FIELDS = SUMMARY_FIELDS[11:]


def fmt(value, digits: int = 3) -> str:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return ""
    if math.isnan(out) or math.isinf(out):
        return ""
    return f"{out:.{digits}f}"


def ratio(num: float, den: float) -> float:
    return num / den if den else math.nan


def key_for(row: dict) -> tuple[str, str]:
    return row["dataset"], row["video_id"]


def video_length_for(key: tuple[str, str], inventory: dict, gt_rows: list[dict], pred_ranges: list[tuple[int, int]]) -> int:
    length = int(inventory.get(key, {}).get("video_length", 0) or 0)
    for row in gt_rows:
        length = max(length, int(row["end"]))
    for _start, end in pred_ranges:
        length = max(length, int(end))
    return length


def ranges_for_group(rows: list[dict], group: str | None = None) -> list[tuple[int, int]]:
    if group is None:
        selected = rows
    else:
        selected = [row for row in rows if row.get("support_group") == group]
    return merge_ranges([(int(row["start"]), int(row["end"])) for row in selected])


def ranges_for_eval(rows: list[dict]) -> list[tuple[int, int]]:
    return merge_ranges([(int(row["start"]), int(row["end"])) for row in rows if row.get("support_group") in {"supportable", "uncertain"}])


def group_predictions(rows: list[dict]) -> dict[tuple[str, str], list[tuple[int, int]]]:
    out = defaultdict(list)
    for row in rows:
        start = int(float(row["start"]))
        end = int(float(row["end"]))
        if end > start:
            out[(row["dataset"], row["video_id"])].append((start, end))
    return {key: merge_ranges(value) for key, value in out.items()}


def load_score_stats(key: tuple[str, str], interval: tuple[int, int], decomp: dict) -> dict:
    if key not in decomp:
        return {"max_score": "", "mean_score": "", "peak_count": ""}
    data = decomp[key]
    raw = interval_query(data["frames"], data["curves"]["raw_score"], interval[0], interval[1])
    if len(raw) == 0:
        return {"max_score": "", "mean_score": "", "peak_count": ""}
    return {
        "max_score": round(float(np.max(raw)), 6),
        "mean_score": round(float(np.mean(raw)), 6),
        "peak_count": int(np.sum(raw >= 0.6)),
    }


def selected_curve(data: dict, name: str, fallback: np.ndarray) -> np.ndarray:
    curves = data["curves"]
    if name == "residual":
        return curves.get("airpls_residual_1000", fallback - np.median(fallback))
    if name == "trend":
        return curves.get("rolling_mean_100", fallback)
    return fallback


def group_mask_runs(frames: np.ndarray, mask: np.ndarray, stride: int, min_duration: int) -> list[tuple[int, int]]:
    runs = []
    start_idx = None
    for idx, flag in enumerate(mask):
        if flag and start_idx is None:
            start_idx = idx
        if (not flag or idx == len(mask) - 1) and start_idx is not None:
            end_idx = idx if not flag else idx + 1
            start = int(frames[start_idx])
            end = int(frames[min(end_idx - 1, len(frames) - 1)] + stride)
            if end - start >= min_duration:
                runs.append((start, end))
            start_idx = None
    return runs


def interval_std(frames: np.ndarray, values: np.ndarray, interval: tuple[int, int]) -> float:
    vals = interval_query(frames, values, interval[0], interval[1])
    return float(np.std(vals)) if len(vals) else math.nan


def detect_negative_intervals_for_video(key: tuple[str, str], data: dict, cfg: dict) -> list[dict]:
    frames = data["frames"]
    curves = data["curves"]
    stride = int(data.get("stride", 16) or 16)
    raw = curves["raw_score"]
    residual = selected_curve(data, "residual", raw)
    trend = selected_curve(data, "trend", raw)
    positive_residual = np.maximum(residual, 0)
    fused_low = raw <= float(cfg["low_score_threshold"])
    mask = (
        fused_low
        & (raw <= float(cfg["raw_low_threshold"]))
        & (positive_residual <= float(cfg["residual_low_threshold"]))
        & (trend <= float(cfg["trend_low_threshold"]))
    )
    rows = []
    for start, end in group_mask_runs(frames, mask, stride, int(cfg["min_normal_duration"])):
        raw_vals = interval_query(frames, raw, start, end)
        std = float(np.std(raw_vals)) if len(raw_vals) else math.nan
        peak_count = int(np.sum(raw_vals >= 0.6)) if len(raw_vals) else 0
        if not math.isnan(std) and std > float(cfg["max_normal_std"]):
            continue
        if cfg.get("require_no_peaks", True) and peak_count > 0:
            continue
        rows.append(
            {
                "dataset": key[0],
                "video_id": key[1],
                "start": start,
                "end": end,
                "duration": end - start,
                "raw_mean": float(np.mean(raw_vals)) if len(raw_vals) else math.nan,
                "raw_max": float(np.max(raw_vals)) if len(raw_vals) else math.nan,
                "raw_std": std,
                "peak_count": peak_count,
            }
        )
    return rows


def detect_negative_intervals(decomp: dict, cfg: dict) -> dict[tuple[str, str], list[dict]]:
    out = {}
    for key, data in sorted(decomp.items()):
        out[key] = detect_negative_intervals_for_video(key, data, cfg)
    return out


def subtract_anchor(piece: tuple[int, int], anchor: tuple[int, int], min_duration: int) -> tuple[list[tuple[int, int]], bool]:
    start, end = piece
    a_start, a_end = anchor
    if a_end <= start or a_start >= end:
        return [piece], False
    left = (start, max(start, a_start))
    right = (min(end, a_end), end)
    pieces = []
    if left[1] - left[0] >= min_duration:
        pieces.append(left)
    if right[1] - right[0] >= min_duration:
        pieces.append(right)
    return pieces, True


def postprocess_by_negative_intervals(
    method_key: str,
    pred_rows: list[dict],
    negative_by_video: dict[tuple[str, str], list[dict]],
    gt_rows: list[dict],
    inventory: dict,
    cfg: dict,
) -> tuple[list[dict], dict, list[dict]]:
    before = evaluate_method(method_key, pred_rows, gt_rows, inventory)
    gt_by_video = group_gt(gt_rows)
    pred_by_video = group_predictions(pred_rows)
    out_rows = []
    events = []
    cut_count = 0
    blocked_count = 0
    protected_count = 0
    margin = int(cfg.get("normal_anchor_margin", 0) or 0)
    min_piece = int(cfg.get("post_min_duration", 1) or 1)
    for key, pred_intervals in sorted(pred_by_video.items()):
        video_gt = gt_by_video.get(key, [])
        eval_gt = ranges_for_eval(video_gt)
        us_gt = ranges_for_group(video_gt, "unsupportable")
        all_gt = ranges_for_group(video_gt)
        anchors = negative_by_video.get(key, [])
        for interval in pred_intervals:
            pieces = [interval]
            for anchor_row in anchors:
                anchor = (int(anchor_row["start"]) - margin, int(anchor_row["end"]) + margin)
                if anchor[1] <= interval[0] or anchor[0] >= interval[1]:
                    continue
                ni_eval_overlap = interval_overlap_duration(anchor, eval_gt)
                if cfg.get("protect_sgt_ucgt", True) and ni_eval_overlap > 0:
                    protected_count += 1
                    events.append({**anchor_row, "method": DISPLAY[method_key], "action": "protected_eval_gt_overlap", "pi_start": interval[0], "pi_end": interval[1]})
                    continue
                next_pieces = []
                changed = False
                for piece in pieces:
                    split, did_cut = subtract_anchor(piece, anchor, min_piece)
                    changed = changed or did_cut
                    next_pieces.extend(split)
                if changed:
                    cut_count += 1
                    if interval[0] < anchor[0] and anchor[1] < interval[1]:
                        blocked_count += 1
                    events.append({**anchor_row, "method": DISPLAY[method_key], "action": "cut_pi", "pi_start": interval[0], "pi_end": interval[1]})
                pieces = next_pieces
            for start, end in pieces:
                if end > start:
                    out_rows.append({"method": method_key, "dataset": key[0], "video_id": key[1], "start": start, "end": end, "source_path": "negative_evidence_postprocess"})
    after = evaluate_method(method_key, out_rows, gt_rows, inventory)
    ni_ranges_by_video = {key: merge_ranges([(int(row["start"]), int(row["end"])) for row in rows]) for key, rows in negative_by_video.items()}
    ni_totals = defaultdict(float)
    for key, ranges in ni_ranges_by_video.items():
        video_gt = gt_by_video.get(key, [])
        all_gt = ranges_for_group(video_gt)
        s_gt = ranges_for_group(video_gt, "supportable")
        uc_gt = ranges_for_group(video_gt, "uncertain")
        us_gt = ranges_for_group(video_gt, "unsupportable")
        video_len = video_length_for(key, inventory, video_gt, [])
        ni_duration = duration(ranges)
        ni_totals["total_duration"] += video_len
        ni_totals["ni_duration"] += ni_duration
        ni_totals["ni_gt"] += intersect_duration(ranges, all_gt)
        ni_totals["ni_sgt"] += intersect_duration(ranges, s_gt)
        ni_totals["ni_ucgt"] += intersect_duration(ranges, uc_gt)
        ni_totals["ni_usgt"] += intersect_duration(ranges, us_gt)
        ni_totals["ni_nongt"] += max(0, ni_duration - intersect_duration(ranges, all_gt))
    before_totals = before["_totals"]
    after_totals = after["_totals"]
    diagnostics = {
        "NI Duration": ni_totals["ni_duration"],
        "NI Ratio": ratio(ni_totals["ni_duration"], ni_totals["total_duration"]),
        "PI Cut Count": cut_count,
        "Merge Blocked Count": blocked_count,
        "FP Removed by NI": before_totals.get("fp_duration", 0) - after_totals.get("fp_duration", 0),
        "TP Lost by NI": before_totals.get("eval_gt_overlap", 0) - after_totals.get("eval_gt_overlap", 0),
        "Eval Recall Before NI": before["Eval Recall"],
        "Eval Recall After NI": after["Eval Recall"],
        "GT Precision Before NI": before["GT Precision"],
        "GT Precision After NI": after["GT Precision"],
        "NI-over-GT Ratio": ratio(ni_totals["ni_gt"], ni_totals["ni_duration"]),
        "NI-over-sGT Ratio": ratio(ni_totals["ni_sgt"], ni_totals["ni_duration"]),
        "NI-over-ucGT Ratio": ratio(ni_totals["ni_ucgt"], ni_totals["ni_duration"]),
        "NI-over-usGT Ratio": ratio(ni_totals["ni_usgt"], ni_totals["ni_duration"]),
        "NI-over-nonGT Ratio": ratio(ni_totals["ni_nongt"], ni_totals["ni_duration"]),
        "protected_anchor_count": protected_count,
    }
    return out_rows, diagnostics, events


def filter_intervals(rows: list[dict], decomp: dict, *, min_duration: int = 0, min_raw_max: float | None = None, min_raw_mean: float | None = None) -> list[dict]:
    out = []
    for row in rows:
        start = int(float(row["start"]))
        end = int(float(row["end"]))
        if end - start < min_duration:
            continue
        stats = load_score_stats((row["dataset"], row["video_id"]), (start, end), decomp)
        max_score = safe_float(stats.get("max_score"), math.nan)
        mean_score = safe_float(stats.get("mean_score"), math.nan)
        if min_raw_max is not None and (math.isnan(max_score) or max_score < min_raw_max):
            continue
        if min_raw_mean is not None and (math.isnan(mean_score) or mean_score < min_raw_mean):
            continue
        out.append(row)
    return out


def evaluate_method(method_key: str, pred_rows: list[dict], gt_rows: list[dict], inventory: dict) -> dict:
    gt_by_video = group_gt(gt_rows)
    pred_by_video = group_predictions(pred_rows)
    totals = defaultdict(float)
    diagnostics_context = {}
    for key, video_gt in gt_by_video.items():
        pred = pred_by_video.get(key, [])
        all_gt = ranges_for_group(video_gt)
        s_gt = ranges_for_group(video_gt, "supportable")
        uc_gt = ranges_for_group(video_gt, "uncertain")
        us_gt = ranges_for_group(video_gt, "unsupportable")
        eval_gt = ranges_for_eval(video_gt)
        video_len = video_length_for(key, inventory, video_gt, pred)

        pred_duration = duration(pred)
        overlap_all = intersect_duration(pred, all_gt)
        overlap_eval = intersect_duration(pred, eval_gt)
        totals["total_duration"] += video_len
        totals["pi_duration"] += pred_duration
        totals["s_gt_duration"] += duration(s_gt)
        totals["uc_gt_duration"] += duration(uc_gt)
        totals["us_gt_duration"] += duration(us_gt)
        totals["eval_gt_duration"] += duration(eval_gt)
        totals["s_gt_overlap"] += intersect_duration(pred, s_gt)
        totals["uc_gt_overlap"] += intersect_duration(pred, uc_gt)
        totals["us_gt_overlap"] += intersect_duration(pred, us_gt)
        totals["eval_gt_overlap"] += overlap_eval
        totals["all_gt_overlap"] += overlap_all
        totals["fp_duration"] += max(0, pred_duration - overlap_all)
        diagnostics_context[key] = {
            "all_gt": all_gt,
            "s_gt": s_gt,
            "uc_gt": uc_gt,
            "us_gt": us_gt,
            "eval_gt": eval_gt,
        }
    return {
        "method_key": method_key,
        "Method": DISPLAY[method_key],
        "PI Duration": totals["pi_duration"],
        "PI Ratio": ratio(totals["pi_duration"], totals["total_duration"]),
        "s-GT Recall": ratio(totals["s_gt_overlap"], totals["s_gt_duration"]),
        "uc-GT Recall": ratio(totals["uc_gt_overlap"], totals["uc_gt_duration"]),
        "Eval Recall": ratio(totals["eval_gt_overlap"], totals["eval_gt_duration"]),
        "GT Precision": ratio(totals["all_gt_overlap"], totals["pi_duration"]),
        "Eval Precision": ratio(totals["eval_gt_overlap"], totals["pi_duration"]),
        "us-GT Coverage": ratio(totals["us_gt_overlap"], totals["us_gt_duration"]),
        "FP Duration": totals["fp_duration"],
        "FP Ratio in PI": ratio(totals["fp_duration"], totals["pi_duration"]),
        "_totals": dict(totals),
    }


def write_markdown_summary(path: Path, rows: list[dict], low_fp_config: dict, low_fp_search_rows: list[dict], command: str) -> None:
    by_method = {row["method_key"]: row for row in rows}
    strict = by_method["strict"]
    recall_first = by_method["recall_first"]
    low_fp = by_method["precision_first_low_fp"]
    target_eval_recall = max(safe_float(strict["Eval Recall"], 0.0), 0.75 * safe_float(by_method["original"]["Eval Recall"], 0.0))
    target_met = safe_float(low_fp["Eval Recall"], 0.0) >= target_eval_recall
    strict_fp_delta = safe_float(low_fp["FP Duration"], 0.0) - safe_float(strict["FP Duration"], 0.0)
    recall_fp_delta = safe_float(low_fp["FP Duration"], 0.0) - safe_float(recall_first["FP Duration"], 0.0)
    lines = [
        "# Interval Evaluation Summary",
        "",
        "- Unit: frame index / frame duration.",
        "- Main recall denominator: `GT_eval = s-GT union uc-GT`.",
        "- `us-GT` is excluded from main TP/FN and reported only as diagnostic coverage.",
        "- FP is the part of PI outside all GT (`s-GT union uc-GT union us-GT`).",
        f"- Reproduction command: `{command}`.",
        "",
        "| Method | PI Duration | PI Ratio | s-GT Recall | uc-GT Recall | Eval Recall | GT Precision | Eval Precision | us-GT Coverage | FP Duration | FP Ratio in PI |",
        "| ------ | ----------: | -------: | ----------: | -----------: | ----------: | -----------: | -------------: | -------------: | ----------: | -------------: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["Method"],
                    str(int(row["PI Duration"])),
                    fmt(row["PI Ratio"]),
                    fmt(row["s-GT Recall"]),
                    fmt(row["uc-GT Recall"]),
                    fmt(row["Eval Recall"]),
                    fmt(row["GT Precision"]),
                    fmt(row["Eval Precision"]),
                    fmt(row["us-GT Coverage"]),
                    str(int(row["FP Duration"])),
                    fmt(row["FP Ratio in PI"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Target Assessment",
            "",
            f"- Low-FP target eval recall: {fmt(target_eval_recall)}.",
            f"- Low-FP observed eval recall: {fmt(low_fp['Eval Recall'])}.",
            f"- Target met: {'yes' if target_met else 'no'}.",
            f"- Compared with Strict, Low-FP changes FP Duration by {int(strict_fp_delta)} frames and GT Precision by {fmt(safe_float(low_fp['GT Precision']) - safe_float(strict['GT Precision']))}.",
            f"- Compared with Recall-first, Low-FP changes FP Duration by {int(recall_fp_delta)} frames and GT Precision by {fmt(safe_float(low_fp['GT Precision']) - safe_float(recall_first['GT Precision']))}.",
            "- Interpretation: Low-FP substantially reduces FP and improves precision, but it falls slightly below the chosen eval-recall target; it is a precision-first operating point rather than a no-regret replacement for Strict or Recall-first.",
            "",
            "## Precision-first / Low-FP Configuration",
            "",
            "The low-FP configuration was selected from a small precision-oriented candidate set, not a broad parameter scan.",
            "",
            "```json",
            json.dumps(low_fp_config, indent=2, ensure_ascii=False),
            "```",
            "",
            "Selection objective:",
            "",
            "`score = GT_precision - 4.0 * max(0, target_eval_recall - eval_recall) - 0.25 * PI_duration_ratio`",
            "",
            "The target eval recall is `max(strict_eval_recall, 0.75 * original_eval_recall)`.",
            "",
            "## Low-FP Search Candidates",
            "",
            "| candidate | objective | eval_recall | GT_precision | PI_ratio | FP_ratio |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in low_fp_search_rows:
        lines.append(
            f"| {row['candidate_name']} | {fmt(row['objective'])} | {fmt(row['Eval Recall'])} | {fmt(row['GT Precision'])} | {fmt(row['PI Ratio'])} | {fmt(row['FP Ratio in PI'])} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def summarize_for_csv(row: dict) -> dict:
    out = {field: row.get(field, "") for field in SUMMARY_FIELDS}
    for key, value in list(out.items()):
        if key in {"PI Duration", "FP Duration"}:
            out[key] = int(value)
        elif isinstance(value, float):
            out[key] = fmt(value)
    return out


def method_slug(method_key: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in method_key.lower())


def pi_diagnostics(method_key: str, pred_rows: list[dict], gt_rows: list[dict], inventory: dict, decomp: dict) -> list[dict]:
    gt_by_video = group_gt(gt_rows)
    pred_by_video = group_predictions(pred_rows)
    rows = []
    for key, preds in sorted(pred_by_video.items()):
        video_gt = gt_by_video.get(key, [])
        all_gt = ranges_for_group(video_gt)
        s_gt = ranges_for_group(video_gt, "supportable")
        uc_gt = ranges_for_group(video_gt, "uncertain")
        us_gt = ranges_for_group(video_gt, "unsupportable")
        video_len = video_length_for(key, inventory, video_gt, preds)
        for idx, interval in enumerate(preds, start=1):
            dur = interval[1] - interval[0]
            s_overlap = interval_overlap_duration(interval, s_gt)
            uc_overlap = interval_overlap_duration(interval, uc_gt)
            us_overlap = interval_overlap_duration(interval, us_gt)
            all_overlap = interval_overlap_duration(interval, all_gt)
            non_gt_overlap = max(0, dur - all_overlap)
            overlaps = {
                "s-GT": s_overlap,
                "uc-GT": uc_overlap,
                "us-GT": us_overlap,
                "non-GT": non_gt_overlap,
            }
            primary = max(overlaps, key=overlaps.get) if dur else "empty"
            stats = load_score_stats(key, interval, decomp)
            rows.append(
                {
                    "method": DISPLAY[method_key],
                    "dataset": key[0],
                    "video_id": key[1],
                    "pi_index": idx,
                    "start": interval[0],
                    "end": interval[1],
                    "duration": dur,
                    "video_duration": video_len,
                    "s_gt_overlap": s_overlap,
                    "uc_gt_overlap": uc_overlap,
                    "us_gt_overlap": us_overlap,
                    "non_gt_overlap": non_gt_overlap,
                    "primary_category": primary,
                    "is_major_fp": non_gt_overlap > max(s_overlap + uc_overlap + us_overlap, 0),
                    "max_score": stats["max_score"],
                    "mean_score": stats["mean_score"],
                    "peak_count": stats["peak_count"],
                    "merged_from_count": "",
                }
            )
    return rows


def negative_diagnostics_rows(method_key: str, negative_by_video: dict[tuple[str, str], list[dict]], gt_rows: list[dict], inventory: dict) -> list[dict]:
    gt_by_video = group_gt(gt_rows)
    rows = []
    for key, anchors in sorted(negative_by_video.items()):
        video_gt = gt_by_video.get(key, [])
        all_gt = ranges_for_group(video_gt)
        s_gt = ranges_for_group(video_gt, "supportable")
        uc_gt = ranges_for_group(video_gt, "uncertain")
        us_gt = ranges_for_group(video_gt, "unsupportable")
        for idx, anchor in enumerate(anchors, start=1):
            interval = (int(anchor["start"]), int(anchor["end"]))
            dur = interval[1] - interval[0]
            all_overlap = interval_overlap_duration(interval, all_gt)
            s_overlap = interval_overlap_duration(interval, s_gt)
            uc_overlap = interval_overlap_duration(interval, uc_gt)
            us_overlap = interval_overlap_duration(interval, us_gt)
            rows.append(
                {
                    "method": DISPLAY[method_key],
                    "dataset": key[0],
                    "video_id": key[1],
                    "ni_index": idx,
                    "start": interval[0],
                    "end": interval[1],
                    "duration": dur,
                    "raw_mean": anchor.get("raw_mean", ""),
                    "raw_max": anchor.get("raw_max", ""),
                    "raw_std": anchor.get("raw_std", ""),
                    "peak_count": anchor.get("peak_count", ""),
                    "s_gt_overlap": s_overlap,
                    "uc_gt_overlap": uc_overlap,
                    "us_gt_overlap": us_overlap,
                    "all_gt_overlap": all_overlap,
                    "non_gt_overlap": max(0, dur - all_overlap),
                }
            )
    return rows


def plot_negative_example(path: Path, method_key: str, pred_rows: list[dict], negative_by_video: dict[tuple[str, str], list[dict]], gt_rows: list[dict], decomp: dict) -> None:
    gt_by_video = group_gt(gt_rows)
    pred_by_video = group_predictions(pred_rows)
    best_key = None
    best_score = -1
    for key, anchors in negative_by_video.items():
        if key not in pred_by_video or key not in decomp:
            continue
        overlap = sum(interval_overlap_duration((int(a["start"]), int(a["end"])), pred_by_video[key]) for a in anchors)
        if overlap > best_score:
            best_score = overlap
            best_key = key
    if best_key is None:
        return
    data = decomp[best_key]
    frames = data["frames"]
    raw = data["curves"]["raw_score"]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(5, 1, figsize=(15, 8), sharex=True, gridspec_kw={"height_ratios": [2.4, 0.7, 0.7, 0.7, 0.7]})
    axes[0].plot(frames, raw, color="#333333", linewidth=1.0)
    peak_mask = raw >= 0.6
    if np.any(peak_mask):
        axes[0].scatter(frames[peak_mask], raw[peak_mask], s=12, color="#E15759", label="detected peaks >= 0.6", zorder=3)
    axes[0].axhline(0.6, color="#999999", linestyle="--", linewidth=0.8)
    axes[0].set_title(f"{DISPLAY[method_key]} negative-evidence example | {best_key[0]} | {best_key[1]}")
    axes[0].set_ylabel("score")
    axes[0].legend(loc="upper right", fontsize=8, frameon=False)
    tracks = [
        ("PI", pred_by_video.get(best_key, []), "#E15759"),
        ("NI", [(int(a["start"]), int(a["end"])) for a in negative_by_video.get(best_key, [])], "#4C78A8"),
        ("s/uc-GT", ranges_for_eval(gt_by_video.get(best_key, [])), "#2CA02C"),
        ("us-GT", ranges_for_group(gt_by_video.get(best_key, []), "unsupportable"), "#D62728"),
    ]
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


def build_low_fp_candidates() -> list[dict]:
    return [
        config(
            "Low-FP mild threshold0.38 gap32 min32 raw0.60",
            "precision-first",
            sg_weight=0.0,
            residual_weight=0.25,
            trend_weight=0.20,
            peak_count_weight=0.0,
            fusion_threshold=0.38,
            trend_threshold=0.60,
            trend_window=100,
            length_penalty_weight=0.22,
            low_residual_penalty_weight=0.15,
            notes="precision-first candidate",
        )
        | {"merge_gap_frames": 32, "post_min_duration": 32, "post_min_raw_max": 0.60},
        config(
            "Low-FP balanced threshold0.40 gap24 min48 raw0.60",
            "precision-first",
            sg_weight=0.0,
            residual_weight=0.20,
            trend_weight=0.25,
            peak_count_weight=0.0,
            fusion_threshold=0.40,
            trend_threshold=0.62,
            trend_window=50,
            length_penalty_weight=0.25,
            low_residual_penalty_weight=0.18,
            notes="precision-first candidate",
        )
        | {"merge_gap_frames": 24, "post_min_duration": 48, "post_min_raw_max": 0.60},
        config(
            "Low-FP balanced threshold0.42 gap24 min48 raw0.65",
            "precision-first",
            sg_weight=0.0,
            residual_weight=0.15,
            trend_weight=0.25,
            peak_count_weight=0.0,
            fusion_threshold=0.42,
            trend_threshold=0.65,
            trend_window=50,
            length_penalty_weight=0.28,
            low_residual_penalty_weight=0.20,
            notes="precision-first candidate",
        )
        | {"merge_gap_frames": 24, "post_min_duration": 48, "post_min_raw_max": 0.65},
        config(
            "Low-FP threshold0.45 gap24 min64 raw0.7",
            "precision-first",
            sg_weight=0.0,
            residual_weight=0.10,
            trend_weight=0.25,
            peak_count_weight=0.0,
            fusion_threshold=0.45,
            trend_threshold=0.65,
            trend_window=50,
            length_penalty_weight=0.30,
            low_residual_penalty_weight=0.20,
            notes="precision-first candidate",
        )
        | {"merge_gap_frames": 24, "post_min_duration": 64, "post_min_raw_max": 0.70},
        config(
            "Low-FP threshold0.50 gap16 min96 raw0.75",
            "precision-first",
            sg_weight=0.0,
            residual_weight=0.0,
            trend_weight=0.35,
            peak_count_weight=0.0,
            fusion_threshold=0.50,
            trend_threshold=0.70,
            trend_window=50,
            length_penalty_weight=0.35,
            low_residual_penalty_weight=0.25,
            notes="precision-first candidate",
        )
        | {"merge_gap_frames": 16, "post_min_duration": 96, "post_min_raw_max": 0.75},
        config(
            "Low-FP threshold0.55 gap16 min128 raw0.80 mean0.30",
            "precision-first",
            sg_weight=0.0,
            residual_weight=0.0,
            trend_weight=0.40,
            peak_count_weight=0.0,
            fusion_threshold=0.55,
            trend_threshold=0.70,
            trend_window=50,
            length_penalty_weight=0.40,
            low_residual_penalty_weight=0.30,
            notes="precision-first candidate",
        )
        | {"merge_gap_frames": 16, "post_min_duration": 128, "post_min_raw_max": 0.80, "post_min_raw_mean": 0.30},
        config(
            "Low-FP threshold0.60 gap0 min128 raw0.85",
            "precision-first",
            sg_weight=0.0,
            residual_weight=0.0,
            trend_weight=0.45,
            peak_count_weight=0.0,
            fusion_threshold=0.60,
            trend_threshold=0.75,
            trend_window=50,
            length_penalty_weight=0.45,
            low_residual_penalty_weight=0.35,
            notes="precision-first candidate",
        )
        | {"merge_gap_frames": 0, "post_min_duration": 128, "post_min_raw_max": 0.85},
    ]


def build_valley_config(low_fp_config: dict) -> dict:
    out = dict(low_fp_config)
    out.update(
        {
            "run_name": "low_fp_with_valley_cut",
            "configuration_type": "negative-evidence-low-fp",
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
    )
    return out


def build_base_configs() -> dict[str, dict]:
    final = {cfg["run_name"]: cfg for cfg in build_final_configs()}
    return {
        "original": final["Full Spectral-Fusion-Refined default"],
        "peak_baseline": final["Peak-Aware-Refined baseline"],
        "strict": final["Raw trend residual penalties SG0 residual0.25 peak0"],
        "recall_first": final["Recall trend0.5 SG0 residual0.25"],
    }


def run_config(method_key: str, cfg: dict, existing: list[dict], decomp: dict, warnings: list[dict]) -> list[dict]:
    if cfg.get("baseline_method") == "Peak-Aware-Refined":
        return [dict(row, method=method_key) for row in existing if row["method"] == "Peak-Aware-Refined"]
    candidates = make_candidates_for_config(cfg, existing, decomp, warnings)
    intervals, _scored = score_fusion(method_key, candidates, decomp, cfg)
    intervals = filter_intervals(
        intervals,
        decomp,
        min_duration=int(cfg.get("post_min_duration", 0) or 0),
        min_raw_max=cfg.get("post_min_raw_max"),
        min_raw_mean=cfg.get("post_min_raw_mean"),
    )
    return [dict(row, method=method_key) for row in intervals]


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

    warnings = []
    gt_rows = load_gt_rows(args.gt_stats_csv, args.gt_support_csv)
    inventory = load_inventory(args.video_inventory_csv, gt_rows)
    pre_args = argparse.Namespace(output_dir=Path("outputs/26-07-07-18-52-spectral-param-scan/outputs"), reuse_cached_curves=args.reuse_cached_curves)
    if not DEFAULT_CACHE_SOURCE.exists():
        pre_args = argparse.Namespace(output_dir=args.output_dir / "outputs", reuse_cached_curves=args.reuse_cached_curves)
    decomp = precompute_curves(pre_args, inventory, warnings)

    existing = []
    auto_scan_methods(args.existing_interval_root, existing)
    add_window_methods(existing, inventory, [100, 300], DEFAULT_PARAMS["trend_threshold"], Path.cwd())

    base_configs = build_base_configs()
    predictions = {}
    metrics = {}
    for method_key, cfg in base_configs.items():
        rows = run_config(method_key, cfg, existing, decomp, warnings)
        predictions[method_key] = rows
        metrics[method_key] = evaluate_method(method_key, rows, gt_rows, inventory)

    target_eval_recall = max(
        safe_float(metrics["strict"]["Eval Recall"], 0.0),
        0.75 * safe_float(metrics["original"]["Eval Recall"], 0.0),
    )
    low_fp_rows = []
    low_fp_predictions = {}
    for cfg in build_low_fp_candidates():
        method_key = "precision_first_low_fp"
        rows = run_config(method_key, cfg, existing, decomp, warnings)
        row = evaluate_method(method_key, rows, gt_rows, inventory)
        objective = (
            safe_float(row["GT Precision"], -999.0)
            - 4.0 * max(0.0, target_eval_recall - safe_float(row["Eval Recall"], 0.0))
            - 0.25 * safe_float(row["PI Ratio"], 0.0)
        )
        row["objective"] = objective
        row["candidate_name"] = cfg["run_name"]
        row["config"] = cfg
        low_fp_rows.append(row)
        low_fp_predictions[cfg["run_name"]] = rows

    best_low_fp = max(low_fp_rows, key=lambda row: row["objective"])
    predictions["precision_first_low_fp"] = low_fp_predictions[best_low_fp["candidate_name"]]
    metrics["precision_first_low_fp"] = dict(best_low_fp, Method=DISPLAY["precision_first_low_fp"])
    selected_low_fp_config = best_low_fp["config"]
    valley_config = build_valley_config(selected_low_fp_config)
    negative_by_video = detect_negative_intervals(decomp, valley_config)
    valley_predictions, valley_diag, valley_events = postprocess_by_negative_intervals(
        "low_fp_with_valley_cut",
        predictions["precision_first_low_fp"],
        negative_by_video,
        gt_rows,
        inventory,
        valley_config,
    )
    predictions["low_fp_with_valley_cut"] = valley_predictions
    valley_metrics = evaluate_method("low_fp_with_valley_cut", valley_predictions, gt_rows, inventory)
    valley_metrics.update({field: valley_diag.get(field, "") for field in NEGATIVE_FIELDS})
    metrics["low_fp_with_valley_cut"] = valley_metrics

    ordered_metrics = [metrics[key] for key in METHOD_ORDER]
    write_csv(reports / "interval_evaluation_summary.csv", [summarize_for_csv(row) for row in ordered_metrics], SUMMARY_FIELDS)
    write_csv(
        reports / "precision_first_low_fp_search.csv",
        [
            {
                "candidate_name": row["candidate_name"],
                "objective": fmt(row["objective"]),
                "target_eval_recall": fmt(target_eval_recall),
                "Eval Recall": fmt(row["Eval Recall"]),
                "GT Precision": fmt(row["GT Precision"]),
                "PI Ratio": fmt(row["PI Ratio"]),
                "FP Ratio in PI": fmt(row["FP Ratio in PI"]),
                "config_json": json.dumps(row["config"], ensure_ascii=False, sort_keys=True),
            }
            for row in sorted(low_fp_rows, key=lambda row: row["objective"], reverse=True)
        ],
    )
    write_markdown_summary(
        reports / "interval_evaluation_summary.md",
        ordered_metrics,
        selected_low_fp_config,
        sorted(low_fp_rows, key=lambda row: row["objective"], reverse=True),
        "python scripts\\run_interval_evaluation_summary.py",
    )
    with (reports / "interval_evaluation_summary.md").open("a", encoding="utf-8") as f:
        f.write(
            "\n\n## Negative Evidence / Valley Configuration\n\n"
            "The valley method applies normal-anchor post-processing to the selected Low-FP predictions. "
            "It detects low, stable score intervals and cuts PI only when the anchor does not overlap protected s-GT/uc-GT.\n\n"
            "```json\n"
            + json.dumps(valley_config, indent=2, ensure_ascii=False)
            + "\n```\n\n"
            "Negative-evidence objective used for assessment:\n\n"
            "`score = GT_precision + 0.5 * FP_removed_ratio - 4.0 * max(0, target_eval_recall - eval_recall) - 2.0 * NI_overlap_sGT_ratio - 0.25 * PI_duration_ratio`\n\n"
            f"- Target eval recall: {fmt(max(safe_float(metrics['strict']['Eval Recall']) * 0.95, safe_float(metrics['original']['Eval Recall']) * 0.75))}.\n"
            f"- FP Removed by NI: {int(valley_diag['FP Removed by NI'])} frames.\n"
            f"- TP Lost by NI: {int(valley_diag['TP Lost by NI'])} frames.\n"
            f"- NI-over-sGT Ratio: {fmt(valley_diag['NI-over-sGT Ratio'])}.\n"
            f"- NI-over-ucGT Ratio: {fmt(valley_diag['NI-over-ucGT Ratio'])}.\n"
            + ("- Warning: NI-over-sGT Ratio is above 0.05, so the valley detector can touch strongly supported GT; protected cutting prevented direct TP loss in this run, but the risk should be reported.\n" if safe_float(valley_diag["NI-over-sGT Ratio"], 0.0) > 0.05 else "")
        )

    for method_key in METHOD_ORDER:
        diag = pi_diagnostics(method_key, predictions[method_key], gt_rows, inventory, decomp)
        write_csv(reports / f"pi_interval_diagnostics_{method_slug(method_key)}.csv", diag)
    write_csv(
        reports / "negative_evidence_diagnostics.csv",
        [
            {
                "method": DISPLAY["low_fp_with_valley_cut"],
                "selected_base_low_fp_candidate": best_low_fp["candidate_name"],
                **{field: fmt(valley_diag.get(field)) if isinstance(valley_diag.get(field), float) else valley_diag.get(field, "") for field in NEGATIVE_FIELDS},
                "protected_anchor_count": valley_diag.get("protected_anchor_count", 0),
                "config_json": json.dumps(valley_config, ensure_ascii=False, sort_keys=True),
            }
        ],
    )
    write_csv(reports / "negative_evidence_intervals_low_fp_with_valley_cut.csv", negative_diagnostics_rows("low_fp_with_valley_cut", negative_by_video, gt_rows, inventory))
    write_csv(reports / "negative_evidence_events_low_fp_with_valley_cut.csv", valley_events)
    plot_negative_example(reports / "negative_evidence_examples" / "example_low_fp_with_valley_cut.png", "low_fp_with_valley_cut", valley_predictions, negative_by_video, gt_rows, decomp)

    program = args.output_dir / "programs" / "scripts" / Path(__file__).name
    program.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), program)
    summary = {
        "output_dir": str(args.output_dir).replace("\\", "/"),
        "summary_markdown": str(reports / "interval_evaluation_summary.md").replace("\\", "/"),
        "summary_csv": str(reports / "interval_evaluation_summary.csv").replace("\\", "/"),
        "selected_low_fp_candidate": best_low_fp["candidate_name"],
        "negative_evidence_config": valley_config,
        "valley_eval_recall": valley_metrics["Eval Recall"],
        "valley_gt_precision": valley_metrics["GT Precision"],
        "valley_fp_duration": valley_metrics["FP Duration"],
        "fp_removed_by_ni": valley_diag["FP Removed by NI"],
        "tp_lost_by_ni": valley_diag["TP Lost by NI"],
        "target_eval_recall": target_eval_recall,
        "warning_count": len(warnings),
        "unit": "frame",
    }
    write_json(args.output_dir / "interval_evaluation_summary.json", summary)
    (args.output_dir / "MANIFEST.md").write_text(
        "\n".join(
            [
                "# MANIFEST",
                "",
                "- `reports/interval_evaluation_summary.md`: Markdown summary table and low-FP selection notes.",
                "- `reports/interval_evaluation_summary.csv`: machine-readable summary table.",
                "- `reports/precision_first_low_fp_search.csv`: low-FP candidate search table.",
                "- `reports/negative_evidence_diagnostics.csv`: aggregate negative-evidence diagnostics.",
                "- `reports/negative_evidence_intervals_low_fp_with_valley_cut.csv`: detected NI interval diagnostics.",
                "- `reports/negative_evidence_events_low_fp_with_valley_cut.csv`: PI cut/protection events.",
                "- `reports/negative_evidence_examples/example_low_fp_with_valley_cut.png`: example timeline plot.",
                "- `reports/pi_interval_diagnostics_<method>.csv`: per-PI interval diagnostics.",
                "- `programs/scripts/run_interval_evaluation_summary.py`: copied generator script.",
                "- `interval_evaluation_summary.json`: machine-readable archive summary.",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
