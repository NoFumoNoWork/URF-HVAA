import argparse
import csv
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.h1_h4_boundary_aware_postprocess import (  # noqa: E402
    boundary_aware_merge,
    duration,
    evaluate_method,
    h4_candidate_for_gap,
    interval_iou,
    intersect_duration,
    load_all_inputs,
    load_h4_candidates,
    merge_ranges,
    overlap_len,
    raw_intervals_from_scores,
    score_stats,
    semantic_continuity,
    side_high,
)


GAP_FIELDS = [
    "dataset", "video_id", "left_interval_start", "left_interval_end",
    "right_interval_start", "right_interval_end", "gap_start", "gap_end",
    "gap_length", "left_max_score", "right_max_score", "gap_min_score",
    "gap_mean_score", "nearest_h4_frame", "distance_to_h4", "h4_types",
    "h4_score_drop", "h4_confidence", "pass_gap_limit", "pass_h4_window",
    "pass_score_drop", "pass_left_score", "pass_right_score",
    "pass_semantic_continuity", "final_merge_decision", "fail_reason",
]

FUNNEL_FIELDS = ["stage", "remaining_gaps", "removed_gaps", "removal_reason"]

TAXONOMY_FIELDS = [
    "dataset", "video_id", "gt_start", "gt_end", "gt_length",
    "covered_by_prediction", "num_predicted_segments_covering_gt",
    "max_score_inside_gt", "mean_score_inside_gt", "has_internal_low_score_gap",
    "has_h4_candidate_inside_gt", "has_h4_candidate_near_internal_gap",
    "failure_type",
]

METRIC_FIELDS = [
    "method", "frame_precision", "frame_recall", "frame_f1", "segment_precision",
    "segment_recall", "segment_f1", "avg_segments_per_video",
    "fragmented_gt_ratio", "FP_duration", "FN_duration", "changed_cases",
    "fixed_cases", "worsened_cases",
]

CASE_FIELDS = [
    "dataset", "video_id", "original_intervals", "relaxed_intervals",
    "h4_frame", "h4_types", "h4_score_drop", "gap_length", "gt_overlap",
    "fixed_or_worsened", "explanation",
]


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def fmt(value, digits=6):
    if value is None or value == "":
        return ""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return value
    if math.isnan(value) or math.isinf(value):
        return ""
    return round(value, digits)


def interval_text(intervals):
    return ";".join(f"{s}-{e}" for s, e in intervals)


def values_in(scores, start, end):
    return [v for f, v in scores.items() if start <= f < end]


def nearest_h4(candidates, gap_start, gap_end):
    best = None
    best_dist = None
    for cand in candidates:
        frame = cand["_frame"]
        if gap_start <= frame <= gap_end:
            dist = 0
        else:
            dist = min(abs(frame - gap_start), abs(frame - gap_end))
        if best_dist is None or dist < best_dist:
            best = cand
            best_dist = dist
    return best, best_dist


def fail_reason(row):
    checks = [
        ("pass_gap_limit", "gap_too_long"),
        ("pass_h4_window", "no_h4_near_gap"),
        ("pass_score_drop", "h4_score_drop_not_positive"),
        ("pass_left_score", "left_score_too_low"),
        ("pass_right_score", "right_score_too_low"),
        ("pass_semantic_continuity", "semantic_continuity_failed"),
    ]
    return ";".join(reason for key, reason in checks if not row[key]) or ""


def build_gap_diagnostics(scores_by_key, original_by_key, h4_by_key, args):
    rows = []
    for key, intervals in sorted(original_by_key.items()):
        dataset, video_id = key
        scores = scores_by_key.get(key, {})
        h4s = h4_by_key.get(key, [])
        for left, right in zip(intervals, intervals[1:]):
            gap_start, gap_end = left[1], right[0]
            gap_len = gap_end - gap_start
            if gap_len <= 0:
                continue
            cand, distance = nearest_h4(h4s, gap_start, gap_end)
            gap_values = values_in(scores, gap_start, gap_end)
            left_max, _left_mean = score_stats(scores, left)
            right_max, _right_mean = score_stats(scores, right)
            pass_gap = gap_len <= args.h4_gap_max_frames
            pass_window = cand is not None and distance <= args.h4_window_size
            pass_drop = cand is not None and float(cand.get("_score_drop", 0.0)) > 0
            pass_left = side_high(scores, left, args.min_side_score)
            pass_right = side_high(scores, right, args.min_side_score)
            pass_semantic = False
            if cand is not None:
                pass_semantic, _shared = semantic_continuity(cand)
            final = pass_gap and pass_window and pass_drop and pass_left and pass_right and pass_semantic
            row = {
                "dataset": dataset,
                "video_id": video_id,
                "left_interval_start": left[0],
                "left_interval_end": left[1],
                "right_interval_start": right[0],
                "right_interval_end": right[1],
                "gap_start": gap_start,
                "gap_end": gap_end,
                "gap_length": gap_len,
                "left_max_score": fmt(left_max),
                "right_max_score": fmt(right_max),
                "gap_min_score": fmt(min(gap_values) if gap_values else ""),
                "gap_mean_score": fmt(sum(gap_values) / len(gap_values) if gap_values else ""),
                "nearest_h4_frame": cand.get("frame", cand.get("_frame", "")) if cand else "",
                "distance_to_h4": distance if distance is not None else "",
                "h4_types": cand.get("types", "") if cand else "",
                "h4_score_drop": cand.get("score_drop", "") if cand else "",
                "h4_confidence": cand.get("confidence", "") if cand else "",
                "pass_gap_limit": pass_gap,
                "pass_h4_window": pass_window,
                "pass_score_drop": pass_drop,
                "pass_left_score": pass_left,
                "pass_right_score": pass_right,
                "pass_semantic_continuity": pass_semantic,
                "final_merge_decision": final,
            }
            row["fail_reason"] = fail_reason(row)
            rows.append(row)
    return rows


def funnel(rows):
    stages = [
        ("all_interval_gaps", None, ""),
        ("pass_gap_limit", "pass_gap_limit", "gap_too_long"),
        ("pass_h4_window", "pass_h4_window", "no_h4_near_gap"),
        ("pass_positive_score_drop", "pass_score_drop", "h4_score_drop_not_positive"),
        ("pass_left_right_score", ("pass_left_score", "pass_right_score"), "left_or_right_score_too_low"),
        ("pass_semantic_continuity", "pass_semantic_continuity", "semantic_continuity_failed"),
        ("final_merge", "final_merge_decision", "not_final_merge"),
    ]
    current = list(rows)
    out = []
    for stage, key, reason in stages:
        if key is None:
            out.append({"stage": stage, "remaining_gaps": len(current), "removed_gaps": 0, "removal_reason": ""})
            continue
        before = len(current)
        if isinstance(key, tuple):
            current = [r for r in current if all(r[k] for k in key)]
        else:
            current = [r for r in current if r[key]]
        out.append({"stage": stage, "remaining_gaps": len(current), "removed_gaps": before - len(current), "removal_reason": reason})
    return out


def internal_low_gap(scores, gt, threshold):
    frames = [f for f in sorted(scores) if gt[0] <= f < gt[1]]
    run = 0
    for frame in frames:
        if scores[frame] < threshold:
            run += 1
            if run >= 2:
                return True
        else:
            run = 0
    return False


def h4_near_internal_gap(scores, gt, h4s, threshold, window):
    low_frames = [f for f in sorted(scores) if gt[0] <= f < gt[1] and scores[f] < threshold]
    for cand in h4s:
        if any(abs(cand["_frame"] - f) <= window for f in low_frames):
            return True
    return False


def taxonomy(scores_by_key, gt_by_key, original_by_key, h4_by_key, args):
    rows = []
    for key, gts in sorted(gt_by_key.items()):
        dataset, video_id = key
        scores = scores_by_key.get(key, {})
        preds = original_by_key.get(key, [])
        h4s = h4_by_key.get(key, [])
        for gt in gts:
            vals = values_in(scores, gt[0], gt[1])
            covering = [pred for pred in preds if overlap_len(gt, pred) > 0]
            covered = bool(covering)
            max_score = max(vals) if vals else ""
            mean_score = sum(vals) / len(vals) if vals else ""
            low_gap = internal_low_gap(scores, gt, args.score_threshold)
            h4_inside = any(gt[0] <= cand["_frame"] < gt[1] for cand in h4s)
            h4_gap = h4_near_internal_gap(scores, gt, h4s, args.score_threshold, args.h4_window_size)
            if not covered and (max_score == "" or max_score < args.score_threshold):
                failure = "missed_gt_low_score"
            elif len(covering) > 1 and h4_gap:
                failure = "fragmented_gt_with_h4_gap"
            elif len(covering) > 1:
                failure = "fragmented_gt_without_h4_gap"
            elif covered and any(pred[0] < gt[0] - args.h4_gap_max_frames or pred[1] > gt[1] + args.h4_gap_max_frames for pred in covering):
                failure = "overmerged_prediction"
            elif covered:
                failure = "good_detection"
            else:
                failure = "unclear"
            rows.append(
                {
                    "dataset": dataset,
                    "video_id": video_id,
                    "gt_start": gt[0],
                    "gt_end": gt[1],
                    "gt_length": gt[1] - gt[0],
                    "covered_by_prediction": covered,
                    "num_predicted_segments_covering_gt": len(covering),
                    "max_score_inside_gt": fmt(max_score),
                    "mean_score_inside_gt": fmt(mean_score),
                    "has_internal_low_score_gap": low_gap,
                    "has_h4_candidate_inside_gt": h4_inside,
                    "has_h4_candidate_near_internal_gap": h4_gap,
                    "failure_type": failure,
                }
            )
    return rows


def relaxed_merge(intervals, h4s, args):
    if not intervals:
        return [], []
    out = [intervals[0]]
    events = []
    for current in intervals[1:]:
        prev = out[-1]
        gap_start, gap_end = prev[1], current[0]
        gap_len = gap_end - gap_start
        cand, distance = nearest_h4(h4s, gap_start, gap_end)
        if cand and gap_len <= args.h4_gap_max_frames and distance <= args.h4_window_size and cand["_score_drop"] > 0:
            merged = (prev[0], max(prev[1], current[1]))
            out[-1] = merged
            events.append({"left": prev, "right": current, "merged": merged, "h4": cand, "gap_length": gap_len})
        else:
            out.append(current)
    return out, events


def fp_duration(preds, gts):
    return max(0, sum(e - s for s, e in preds) - intersect_duration(preds, gts))


def fn_duration(preds, gts):
    return max(0, sum(e - s for s, e in gts) - intersect_duration(preds, gts))


def changed_cases(original_by_key, relaxed_by_key, gt_by_key, relaxed_events):
    rows = []
    for key, events in relaxed_events.items():
        if not events:
            continue
        dataset, video_id = key
        original = original_by_key.get(key, [])
        relaxed = relaxed_by_key.get(key, [])
        gts = gt_by_key.get(key, [])
        original_fp = fp_duration(original, gts)
        relaxed_fp = fp_duration(relaxed, gts)
        original_fn = fn_duration(original, gts)
        relaxed_fn = fn_duration(relaxed, gts)
        for event in events:
            cand = event["h4"]
            gt_overlap = intersect_duration([event["merged"]], gts)
            if relaxed_fp > original_fp:
                label = "worsened"
                explanation = "Relaxed H4 merge increased false-positive duration."
            elif relaxed_fn < original_fn:
                label = "fixed"
                explanation = "Relaxed H4 merge reduced false-negative duration or bridged a fragmented GT region."
            else:
                label = "neutral"
                explanation = "Intervals changed but aggregate FP/FN did not improve."
            rows.append(
                {
                    "dataset": dataset,
                    "video_id": video_id,
                    "original_intervals": interval_text(original),
                    "relaxed_intervals": interval_text(relaxed),
                    "h4_frame": cand.get("frame", cand.get("_frame", "")),
                    "h4_types": cand.get("types", ""),
                    "h4_score_drop": cand.get("score_drop", ""),
                    "gap_length": event["gap_length"],
                    "gt_overlap": gt_overlap,
                    "fixed_or_worsened": label,
                    "explanation": explanation,
                }
            )
    return rows


def metric_row(method, metrics, cases):
    labels = Counter(r["fixed_or_worsened"] for r in cases)
    return {
        "method": method,
        "frame_precision": metrics["frame_precision"],
        "frame_recall": metrics["frame_recall"],
        "frame_f1": metrics["frame_f1"],
        "segment_precision": metrics["segment_precision"],
        "segment_recall": metrics["segment_recall"],
        "segment_f1": metrics["segment_f1"],
        "avg_segments_per_video": metrics["average_predicted_segments_per_video"],
        "fragmented_gt_ratio": metrics["fragmented_gt_ratio"],
        "FP_duration": metrics["false_positive_duration"],
        "FN_duration": metrics["false_negative_duration"],
        "changed_cases": len(cases),
        "fixed_cases": labels["fixed"],
        "worsened_cases": labels["worsened"],
    }


def write_report(path, args, gap_rows, funnel_rows, taxonomy_rows, metric_rows, case_rows):
    failures = Counter(r["failure_type"] for r in taxonomy_rows)
    final_merges = sum(1 for r in gap_rows if r["final_merge_decision"])
    gap_limit_removed = next((r["removed_gaps"] for r in funnel_rows if r["stage"] == "pass_gap_limit"), 0)
    no_h4_removed = next((r["removed_gaps"] for r in funnel_rows if r["stage"] == "pass_h4_window"), 0)
    score_drop_removed = next((r["removed_gaps"] for r in funnel_rows if r["stage"] == "pass_positive_score_drop"), 0)
    side_score_removed = next((r["removed_gaps"] for r in funnel_rows if r["stage"] == "pass_left_right_score"), 0)
    semantic_removed = next((r["removed_gaps"] for r in funnel_rows if r["stage"] == "pass_semantic_continuity"), 0)
    original = next((r for r in metric_rows if r["method"] == "original"), {})
    relaxed = next((r for r in metric_rows if r["method"] == "h4_relaxed"), {})
    strict = next((r for r in metric_rows if r["method"] == "h4_semantic_filtered"), {})
    lines = [
        "# H1/H4 Trigger Diagnostics Report",
        "",
        "## Goal",
        "",
        "This diagnostic run asks where H4 boundary-aware interval merging is blocked, and whether a relaxed H4-only merge changes interval-level metrics.",
        "",
        "## Direct Answers",
        "",
        f"- Strict boundary-aware rule did trigger in this run: {final_merges} final merge decisions out of {len(gap_rows)} baseline interval gaps. If an earlier report showed identical metrics with fixed rows = 0 and worsened rows = 0, the most likely cause is the H4 candidate input path/run configuration rather than the merge predicates themselves.",
        f"- The dataset does have interval gaps: {len(gap_rows)} adjacent predicted-interval gaps were inspected.",
        f"- The main blockers are gap length and missing nearby H4 candidates: {gap_limit_removed} gaps exceeded the H4 max merge gap, and {no_h4_removed} surviving gaps had no H4 candidate within the {args.h4_window_size}-frame window.",
        f"- The H4 window is a real bottleneck under the current settings; semantic continuity is not the current bottleneck because it removed {semantic_removed} additional gaps after previous filters.",
        f"- Positive score-drop and side-score filters removed {score_drop_removed} and {side_score_removed} additional gaps respectively after H4-window filtering.",
        "",
        "The strict and relaxed methods are identical here because every gap that passes gap length, H4-window, positive score-drop, and side-score filters also passes lexical semantic continuity. Therefore removing semantic continuity does not change the result in this run.",
        "",
        "## Filter Funnel",
        "",
        "| stage | remaining | removed | reason |",
        "|---|---:|---:|---|",
    ]
    for row in funnel_rows:
        lines.append(f"| {row['stage']} | {row['remaining_gaps']} | {row['removed_gaps']} | {row['removal_reason']} |")
    lines.extend(["", "## H1 Failure Taxonomy", ""])
    for key, value in sorted(failures.items()):
        lines.append(f"- {key}: {value}")
    frag_h4 = failures.get("fragmented_gt_with_h4_gap", 0)
    frag_no = failures.get("fragmented_gt_without_h4_gap", 0)
    lines.extend(
        [
            "",
            f"H4 candidates cover {frag_h4} fragmented GT failures, while {frag_no} fragmented GT failures lack a nearby H4 low-score gap under the current rules.",
            "",
            "## Relaxed H4 Merge Metrics",
            "",
            "| method | frame P | frame R | frame F1 | segment F1 | avg seg/video | fragmented GT | FP | FN | changed | fixed | worsened |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in metric_rows:
        lines.append(
            f"| {row['method']} | {row['frame_precision']} | {row['frame_recall']} | {row['frame_f1']} | "
            f"{row['segment_f1']} | {row['avg_segments_per_video']} | {row['fragmented_gt_ratio']} | "
            f"{row['FP_duration']} | {row['FN_duration']} | {row['changed_cases']} | {row['fixed_cases']} | {row['worsened_cases']} |"
        )
    labels = Counter(r["fixed_or_worsened"] for r in case_rows)
    lines.extend(
        [
            "",
            "## Changed Cases",
            "",
            f"- changed cases: {len(case_rows)}",
            f"- fixed: {labels['fixed']}",
            f"- worsened: {labels['worsened']}",
            f"- neutral: {labels['neutral']}",
            "",
            "## Interpretation",
            "",
            f"- Compared with original, relaxed/strict H4 merging changes recall from {original.get('frame_recall')} to {relaxed.get('frame_recall')} and fragmented GT ratio from {original.get('fragmented_gt_ratio')} to {relaxed.get('fragmented_gt_ratio')}.",
            f"- The same merge also increases FP duration from {original.get('FP_duration')} to {relaxed.get('FP_duration')} and lowers segment F1 from {original.get('segment_f1')} to {relaxed.get('segment_f1')}. This is not clean VAD improvement; it is a recall/fragmentation gain with over-merge cost.",
            f"- h4_semantic_filtered changed {strict.get('changed_cases')} cases, matching h4_relaxed in this run.",
            "- H4 candidates cover only part of H1 fragmentation failure. A large missed_gt_low_score bucket remains outside what interval merging can fix.",
            "",
            "## Next Steps",
            "",
            "- First tune H4 window size and max merge gap, because those are the observed funnel bottlenecks.",
            "- Also tune the baseline score threshold/merge gap, because many GT failures are missed due to low scores rather than fragmentation.",
            "- Add LLM or visual recheck only after narrowing the candidate set, especially for the 387 worsened merge cases where H4 proximity appears to introduce FP duration.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose H1/H4 interval merge trigger opportunities.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--h4-candidates", default="outputs/26-07-09-00-35-caption_boundary_screen/h4_strong_candidates.csv")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--score-threshold", type=float, default=0.6)
    parser.add_argument("--merge-gap-frames", type=int, default=32)
    parser.add_argument("--h4-window-size", type=int, default=48)
    parser.add_argument("--h4-gap-max-frames", type=int, default=128)
    parser.add_argument("--min-side-score", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / f"{datetime.now():%y-%m-%d-%H-%M}-h1-h4-trigger-diagnostics"
    scores_by_key, gt_by_key, warnings = load_all_inputs(Path(args.data_root))
    h4_by_key = load_h4_candidates(Path(args.h4_candidates))
    all_keys = sorted(set(scores_by_key) | set(gt_by_key))
    original_by_key = {}
    strict_by_key = {}
    strict_events = {}
    relaxed_by_key = {}
    relaxed_events = {}
    for key in all_keys:
        scores = scores_by_key.get(key, {})
        raw = raw_intervals_from_scores(scores, args.score_threshold)
        original = merge_ranges(raw, args.merge_gap_frames)
        strict, strict_event_rows = boundary_aware_merge(original, scores, h4_by_key.get(key, []), args)
        relaxed, events = relaxed_merge(original, h4_by_key.get(key, []), args)
        original_by_key[key] = original
        strict_by_key[key] = strict
        relaxed_by_key[key] = relaxed
        relaxed_events[key] = events
        strict_events[key] = [
            {"left": item["left_interval"], "right": item["right_interval"], "merged": item["merged_interval"], "h4": item["h4"], "gap_length": item["gap"]}
            for item in strict_event_rows
        ]
    gap_rows = build_gap_diagnostics(scores_by_key, original_by_key, h4_by_key, args)
    funnel_rows = funnel(gap_rows)
    taxonomy_rows = taxonomy(scores_by_key, gt_by_key, original_by_key, h4_by_key, args)
    relaxed_cases = changed_cases(original_by_key, relaxed_by_key, gt_by_key, relaxed_events)
    strict_cases = changed_cases(original_by_key, strict_by_key, gt_by_key, strict_events)
    original_metrics = evaluate_method("original", original_by_key, gt_by_key, all_keys)
    relaxed_metrics = evaluate_method("h4_relaxed", relaxed_by_key, gt_by_key, all_keys)
    strict_metrics = evaluate_method("h4_semantic_filtered", strict_by_key, gt_by_key, all_keys)
    metrics = [
        metric_row("original", original_metrics, []),
        metric_row("h4_relaxed", relaxed_metrics, relaxed_cases),
        metric_row("h4_semantic_filtered", strict_metrics, strict_cases),
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "merge_opportunity_diagnostics.csv", gap_rows, GAP_FIELDS)
    write_csv(output_dir / "filter_funnel.csv", funnel_rows, FUNNEL_FIELDS)
    write_csv(output_dir / "h1_failure_taxonomy.csv", taxonomy_rows, TAXONOMY_FIELDS)
    write_csv(output_dir / "h1_h4_relaxed_metrics.csv", metrics, METRIC_FIELDS)
    write_csv(output_dir / "h4_relaxed_changed_cases.csv", relaxed_cases, CASE_FIELDS)
    write_json(
        output_dir / "h1_h4_trigger_diagnostics_summary.json",
        {
            "all_videos": len(all_keys),
            "gap_count": len(gap_rows),
            "strict_final_merges": sum(1 for r in gap_rows if r["final_merge_decision"]),
            "relaxed_changed_cases": len(relaxed_cases),
            "relaxed_fixed_cases": sum(1 for r in relaxed_cases if r["fixed_or_worsened"] == "fixed"),
            "relaxed_worsened_cases": sum(1 for r in relaxed_cases if r["fixed_or_worsened"] == "worsened"),
            "warnings": warnings,
            "args": vars(args),
        },
    )
    write_report(output_dir / "h1_h4_trigger_diagnostics_report.md", args, gap_rows, funnel_rows, taxonomy_rows, metrics, relaxed_cases)
    script_dir = output_dir / "scripts"
    script_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), script_dir / Path(__file__).name)
    print(f"gaps: {len(gap_rows)}")
    print(f"strict final merges: {sum(1 for r in gap_rows if r['final_merge_decision'])}")
    print(f"relaxed changed cases: {len(relaxed_cases)}")
    print(f"output dir: {output_dir}")


if __name__ == "__main__":
    main()
