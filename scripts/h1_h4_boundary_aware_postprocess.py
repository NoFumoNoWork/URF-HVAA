import argparse
import csv
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


TASK_NAME = "h1_h4"

DATASET_CONFIGS = [
    {
        "dataset": "MSAD",
        "score_dir": Path("MSAD/refined_scores/videollama3"),
        "gt_file": Path("MSAD/annotations/test.txt"),
    },
    {
        "dataset": "UBNormal",
        "score_dir": Path("UBNormal/refined_scores/videollama3"),
        "gt_file": Path("UBNormal/annotations/temporal.txt"),
    },
    {
        "dataset": "UCF-Crime",
        "score_dir": Path("ucf_crime/refined_scores/videollama3"),
        "gt_file": Path("ucf_crime/annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt"),
    },
    {
        "dataset": "XD-Violence",
        "score_dir": Path("xd_violence/refined_scores/videollama3"),
        "gt_file": Path("xd_violence/annotations/temporal_anomaly_annotation_for_testing_videos.txt"),
    },
]

RISK_SEMANTIC_TERMS = {
    "fight", "fighting", "attack", "attacking", "chase", "chasing", "running",
    "gun", "shooting", "explosion", "crash", "accident", "fire", "smoke",
    "injured", "blood", "distress", "screaming", "yelling", "panic", "threat",
    "weapon", "stab", "choke", "fall", "falling", "car", "bus", "van",
    "helicopter", "knife", "flames", "burning", "hit", "punch", "kicking",
}

INTERVAL_FIELDS = [
    "method", "dataset", "video_id", "start_frame", "end_frame", "max_score",
    "mean_score", "length", "matched_gt", "iou_with_gt", "error_type",
    "merge_reason", "source_intervals", "h4_frame",
]

COMPARE_FIELDS = [
    "dataset", "video_id", "num_original_intervals", "num_boundary_aware_intervals",
    "delta_intervals", "original_gt_fragment_count", "boundary_aware_gt_fragment_count",
    "delta_gt_fragment_count", "original_fp_duration", "boundary_aware_fp_duration",
    "delta_fp_duration", "num_h4_merges",
]

METRIC_FIELDS = [
    "method", "frame_precision", "frame_recall", "frame_f1", "segment_precision",
    "segment_recall", "segment_f1", "average_predicted_segments_per_video",
    "fragmented_gt_ratio", "false_positive_duration", "false_negative_duration",
    "predicted_duration", "gt_duration", "tp_duration", "num_predicted_segments",
    "num_gt_segments", "num_videos",
]

CASE_FIELDS = [
    "dataset", "video_id", "original_intervals", "boundary_aware_intervals",
    "h4_frame", "h4_types", "score_drop", "caption_before", "caption_current",
    "caption_after", "gt_interval", "original_error_type", "fixed_or_worsened",
    "explanation",
]


def normalize_stem(stem):
    return re.sub(r"\(\d+\)$", "", Path(stem).stem).strip()


def read_csv(path):
    if not path or not Path(path).exists():
        return []
    with Path(path).open("r", newline="", encoding="utf-8-sig", errors="replace") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def as_float(value, default=None):
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value, default=0):
    try:
        if value in ("", None):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def fmt(value, digits=6):
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ""
        return round(value, digits)
    return value


def load_scores(path):
    if not path or not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    out = {}
    for key, value in raw.items():
        try:
            out[int(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return dict(sorted(out.items()))


def score_json_index(score_dir):
    result = {}
    if not score_dir.exists():
        return result
    excluded = {"context_prompt.txt", "format_prompt.txt", "highest_lowest_intervals.json", "suspicious_part_phrases.json"}
    for path in sorted(score_dir.glob("*.json")):
        if path.name in excluded:
            continue
        result.setdefault(path.stem, path)
        result.setdefault(normalize_stem(path.stem), path)
    return result


def parse_gt_file(path):
    rows = defaultdict(list)
    if not path.exists():
        return rows
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = re.split(r"[\s,]+", line)
        if len(parts) < 3:
            continue
        video_id = normalize_stem(parts[0])
        nums = []
        for item in parts[1:]:
            if re.fullmatch(r"-?\d+(?:\.\d+)?", item):
                nums.append(int(float(item)))
        for start, end in zip(nums[0::2], nums[1::2]):
            if start == -1 or end == -1:
                break
            if end > start:
                rows[video_id].append((start, end))
    return {key: merge_ranges(value) for key, value in rows.items()}


def load_all_inputs(data_root):
    scores_by_key = {}
    gt_by_key = defaultdict(list)
    warnings = []
    for cfg in DATASET_CONFIGS:
        dataset = cfg["dataset"]
        score_dir = data_root / cfg["score_dir"]
        gt_file = data_root / cfg["gt_file"]
        if not score_dir.exists():
            warnings.append(f"{dataset}: missing score_dir {score_dir}")
            continue
        if not gt_file.exists():
            warnings.append(f"{dataset}: missing gt_file {gt_file}")
        gt_rows = parse_gt_file(gt_file)
        for video_id, intervals in gt_rows.items():
            gt_by_key[(dataset, video_id)] = intervals
        for video_id, path in score_json_index(score_dir).items():
            if (dataset, video_id) in scores_by_key:
                continue
            scores = load_scores(path)
            if scores:
                scores_by_key[(dataset, video_id)] = scores
    return scores_by_key, dict(gt_by_key), warnings


def merge_ranges(intervals, gap=0):
    valid = sorted((int(s), int(e)) for s, e in intervals if e > s)
    if not valid:
        return []
    merged = [valid[0]]
    for s, e in valid[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e + gap:
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))
    return merged


def overlap_len(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def duration(intervals):
    return sum(e - s for s, e in intervals)


def intersect_duration(a, b):
    a = merge_ranges(a)
    b = merge_ranges(b)
    i = j = 0
    total = 0
    while i < len(a) and j < len(b):
        s = max(a[i][0], b[j][0])
        e = min(a[i][1], b[j][1])
        if e > s:
            total += e - s
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return total


def interval_iou(a, b):
    inter = overlap_len(a, b)
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union else 0.0


def score_stride(scores):
    frames = sorted(scores)
    diffs = [b - a for a, b in zip(frames, frames[1:]) if b > a]
    if not diffs:
        return 16
    return Counter(diffs).most_common(1)[0][0]


def raw_intervals_from_scores(scores, threshold):
    if not scores:
        return []
    frames = sorted(scores)
    stride = score_stride(scores)
    intervals = []
    start = None
    last = None
    for frame in frames:
        if scores[frame] >= threshold:
            if start is None:
                start = frame
            last = frame
        elif start is not None:
            intervals.append((start, last + stride))
            start = None
            last = None
    if start is not None:
        intervals.append((start, last + stride))
    return intervals


def score_stats(scores, interval):
    values = [value for frame, value in scores.items() if interval[0] <= frame < interval[1]]
    if not values:
        return "", ""
    return max(values), sum(values) / len(values)


def best_gt(interval, gt_intervals):
    best = None
    best_iou = 0.0
    for gt in gt_intervals:
        iou = interval_iou(interval, gt)
        if iou > best_iou:
            best = gt
            best_iou = iou
    return best, best_iou


def interval_error_type(interval, gt_intervals):
    if any(overlap_len(interval, gt) > 0 for gt in gt_intervals):
        return "tp_overlap"
    return "false_positive"


def interval_rows(method, dataset, video_id, intervals, scores, gt_intervals, merge_reasons=None):
    rows = []
    merge_reasons = merge_reasons or {}
    for interval in intervals:
        reason_meta = merge_reasons.get(interval, {}) if isinstance(merge_reasons.get(interval, {}), dict) else {"merge_reason": merge_reasons.get(interval, "")}
        max_score, mean_score = score_stats(scores, interval)
        match, iou = best_gt(interval, gt_intervals)
        rows.append(
            {
                "method": method,
                "dataset": dataset,
                "video_id": video_id,
                "start_frame": interval[0],
                "end_frame": interval[1],
                "max_score": fmt(max_score),
                "mean_score": fmt(mean_score),
                "length": interval[1] - interval[0],
                "matched_gt": f"{match[0]}-{match[1]}" if match else "",
                "iou_with_gt": fmt(iou),
                "error_type": interval_error_type(interval, gt_intervals),
                "merge_reason": reason_meta.get("merge_reason", ""),
                "source_intervals": reason_meta.get("source_intervals", ""),
                "h4_frame": reason_meta.get("h4_frame", ""),
            }
        )
    return rows


def token_set(*texts):
    words = set()
    for text in texts:
        for item in re.findall(r"[a-z0-9]+", (text or "").lower()):
            if len(item) > 2:
                words.add(item)
    return words


def semantic_continuity(candidate):
    before = token_set(candidate.get("caption_before"), candidate.get("caption_current"))
    after = token_set(candidate.get("caption_current"), candidate.get("caption_after"))
    shared = (before & after) & RISK_SEMANTIC_TERMS
    if shared:
        return True, ";".join(sorted(shared))
    generic_shared = (before & after) - {"video", "shows", "scene", "camera", "person", "people", "man", "woman"}
    return len(generic_shared) >= 2, ";".join(sorted(list(generic_shared))[:8])


def load_h4_candidates(path):
    if not path.exists():
        raise FileNotFoundError(f"H4 candidates file not found: {path}")
    rows = read_csv(path)
    by_key = defaultdict(list)
    seen = set()
    for row in rows:
        dataset = row.get("dataset", "")
        video_id = normalize_stem(row.get("video_id", ""))
        frame = as_int(row.get("frame"), None)
        drop = as_float(row.get("score_drop"), 0.0)
        types = row.get("types", "")
        if frame is None:
            continue
        if drop is None or drop <= 0:
            continue
        if "possible_context_forgetting" not in types and "multi_scene_compression_boundary" not in types:
            continue
        if "event_onset_not_h4" in types:
            continue
        dedupe_key = (dataset, video_id, frame, types, row.get("caption_current", ""))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        row = dict(row)
        row["_frame"] = frame
        row["_score_drop"] = drop
        by_key[(dataset, video_id)].append(row)
    return {key: sorted(value, key=lambda r: r["_frame"]) for key, value in by_key.items()}


def h4_candidate_for_gap(candidates, left, right, h4_window_size):
    gap_start, gap_end = left[1], right[0]
    for cand in candidates:
        frame = cand["_frame"]
        if gap_start - h4_window_size <= frame <= gap_end + h4_window_size:
            ok, shared = semantic_continuity(cand)
            if ok:
                return cand, shared
    return None, ""


def side_high(scores, interval, min_side_score):
    max_score, mean_score = score_stats(scores, interval)
    max_score = as_float(max_score, 0.0)
    mean_score = as_float(mean_score, 0.0)
    return max(max_score, mean_score) >= min_side_score


def boundary_aware_merge(baseline_intervals, scores, h4_candidates, args):
    if not baseline_intervals:
        return [], []
    result = [baseline_intervals[0]]
    events = []
    for current in baseline_intervals[1:]:
        prev = result[-1]
        gap = current[0] - prev[1]
        if gap <= args.merge_gap_frames:
            result[-1] = (prev[0], max(prev[1], current[1]))
            continue
        cand, shared = h4_candidate_for_gap(h4_candidates, prev, current, args.h4_window_size)
        if (
            cand
            and gap <= args.h4_gap_max_frames
            and side_high(scores, prev, args.min_side_score)
            and side_high(scores, current, args.min_side_score)
        ):
            merged = (prev[0], max(prev[1], current[1]))
            result[-1] = merged
            events.append(
                {
                    "merged_interval": merged,
                    "left_interval": prev,
                    "right_interval": current,
                    "h4": cand,
                    "shared_terms": shared,
                    "gap": gap,
                }
            )
        else:
            result.append(current)
    return result, events


def intervals_to_text(intervals):
    return ";".join(f"{s}-{e}" for s, e in intervals)


def gt_fragments(gt_intervals, pred_intervals):
    count = 0
    fragmented = 0
    for gt in gt_intervals:
        overlaps = [pred for pred in pred_intervals if overlap_len(gt, pred) > 0]
        if overlaps:
            count += len(overlaps)
        if len(overlaps) > 1:
            fragmented += 1
    return count, fragmented


def evaluate_method(method, pred_by_key, gt_by_key, all_keys):
    tp = fp = fn = pred_dur = gt_dur = 0
    pred_segments = 0
    gt_segments = 0
    segment_tp = 0
    fragmented_gt = 0
    for key in all_keys:
        preds = merge_ranges(pred_by_key.get(key, []))
        gts = merge_ranges(gt_by_key.get(key, []))
        pred_segments += len(preds)
        gt_segments += len(gts)
        pred_dur += duration(preds)
        gt_dur += duration(gts)
        local_tp = intersect_duration(preds, gts)
        tp += local_tp
        fp += max(0, duration(preds) - local_tp)
        fn += max(0, duration(gts) - local_tp)
        segment_tp += sum(1 for pred in preds if any(overlap_len(pred, gt) > 0 for gt in gts))
        for gt in gts:
            if sum(1 for pred in preds if overlap_len(pred, gt) > 0) > 1:
                fragmented_gt += 1
    frame_p = tp / pred_dur if pred_dur else 0.0
    frame_r = tp / gt_dur if gt_dur else 0.0
    seg_p = segment_tp / pred_segments if pred_segments else 0.0
    seg_r = sum(1 for key in all_keys for gt in gt_by_key.get(key, []) if any(overlap_len(gt, pred) > 0 for pred in pred_by_key.get(key, []))) / gt_segments if gt_segments else 0.0
    return {
        "method": method,
        "frame_precision": fmt(frame_p),
        "frame_recall": fmt(frame_r),
        "frame_f1": fmt(2 * frame_p * frame_r / (frame_p + frame_r) if frame_p + frame_r else 0.0),
        "segment_precision": fmt(seg_p),
        "segment_recall": fmt(seg_r),
        "segment_f1": fmt(2 * seg_p * seg_r / (seg_p + seg_r) if seg_p + seg_r else 0.0),
        "average_predicted_segments_per_video": fmt(pred_segments / len(all_keys) if all_keys else 0.0),
        "fragmented_gt_ratio": fmt(fragmented_gt / gt_segments if gt_segments else 0.0),
        "false_positive_duration": fp,
        "false_negative_duration": fn,
        "predicted_duration": pred_dur,
        "gt_duration": gt_dur,
        "tp_duration": tp,
        "num_predicted_segments": pred_segments,
        "num_gt_segments": gt_segments,
        "num_videos": len(all_keys),
    }


def video_fp_duration(preds, gts):
    local_tp = intersect_duration(preds, gts)
    return max(0, duration(preds) - local_tp)


def video_fn_duration(preds, gts):
    local_tp = intersect_duration(preds, gts)
    return max(0, duration(gts) - local_tp)


def build_case_rows(key, original, aware, gt_intervals, events):
    dataset, video_id = key
    original_fp = video_fp_duration(original, gt_intervals)
    aware_fp = video_fp_duration(aware, gt_intervals)
    original_fn = video_fn_duration(original, gt_intervals)
    aware_fn = video_fn_duration(aware, gt_intervals)
    original_frag = sum(1 for gt in gt_intervals if sum(1 for pred in original if overlap_len(gt, pred) > 0) > 1)
    aware_frag = sum(1 for gt in gt_intervals if sum(1 for pred in aware if overlap_len(gt, pred) > 0) > 1)
    rows = []
    for event in events:
        cand = event["h4"]
        merged = event["merged_interval"]
        match, _iou = best_gt(merged, gt_intervals)
        if aware_fp > original_fp:
            label = "worsened"
            explanation = "Boundary-aware merge increased false-positive duration; likely over-merge risk."
        elif aware_frag < original_frag or aware_fn < original_fn:
            label = "fixed"
            explanation = "Boundary-aware merge reduced GT fragmentation or false-negative duration near an H4 candidate."
        elif aware_fp == original_fp and aware_fn == original_fn:
            label = "neutral"
            explanation = "Intervals changed by H4 merge but aggregate FP/FN did not change."
        else:
            label = "unclear"
            explanation = "Mixed interval metric change; needs visual inspection."
        rows.append(
            {
                "dataset": dataset,
                "video_id": video_id,
                "original_intervals": intervals_to_text(original),
                "boundary_aware_intervals": intervals_to_text(aware),
                "h4_frame": cand.get("frame", cand.get("_frame", "")),
                "h4_types": cand.get("types", ""),
                "score_drop": cand.get("score_drop", ""),
                "caption_before": cand.get("caption_before", ""),
                "caption_current": cand.get("caption_current", ""),
                "caption_after": cand.get("caption_after", ""),
                "gt_interval": f"{match[0]}-{match[1]}" if match else "",
                "original_error_type": "fragmented_gt" if original_frag else "score_discontinuity_gap",
                "fixed_or_worsened": label,
                "explanation": explanation,
            }
        )
    return rows


def comparison_row(key, original, aware, gt_intervals, events):
    original_frag_count, _ = gt_fragments(gt_intervals, original)
    aware_frag_count, _ = gt_fragments(gt_intervals, aware)
    return {
        "dataset": key[0],
        "video_id": key[1],
        "num_original_intervals": len(original),
        "num_boundary_aware_intervals": len(aware),
        "delta_intervals": len(aware) - len(original),
        "original_gt_fragment_count": original_frag_count,
        "boundary_aware_gt_fragment_count": aware_frag_count,
        "delta_gt_fragment_count": aware_frag_count - original_frag_count,
        "original_fp_duration": video_fp_duration(original, gt_intervals),
        "boundary_aware_fp_duration": video_fp_duration(aware, gt_intervals),
        "delta_fp_duration": video_fp_duration(aware, gt_intervals) - video_fp_duration(original, gt_intervals),
        "num_h4_merges": len(events),
    }


def write_report(path, args, metrics, fixed_rows, failed_rows, warnings):
    original = next(row for row in metrics if row["method"] == "original")
    aware = next(row for row in metrics if row["method"] == "boundary_aware")
    lines = [
        "# H1/H4 Boundary-Aware Interval Report",
        "",
        "## Goal",
        "",
        "This experiment tests whether H4 caption-boundary candidates can selectively repair anomaly-score discontinuities at the interval level. It compares baseline threshold/merge intervals with a boundary-aware merge rule.",
        "",
        "## Baseline H1 Failure Pattern",
        "",
        "Baseline intervals are generated from raw anomaly scores using a score threshold and a standard temporal merge gap. Fragmentation is measured by GT intervals covered by multiple predicted segments.",
        "",
        "## H4 Candidate Alignment With H1 Failures",
        "",
        f"- H4 candidate file: `{args.h4_candidates}`",
        f"- score_threshold: {args.score_threshold}",
        f"- baseline merge_gap_frames: {args.merge_gap_frames}",
        f"- H4 window size: {args.h4_window_size}",
        f"- H4 max merge gap: {args.h4_gap_max_frames}",
        "",
        "## Boundary-Aware Post-Processing Rule",
        "",
        "The boundary-aware rule only merges adjacent predicted intervals when both sides retain high anomaly evidence, the gap is near a positive-score-drop H4 candidate, the gap is below the H4 limit, and before/current/after captions share risk/object/action semantics. Such merges are marked `boundary_induced_score_drop`.",
        "",
        "## Quantitative Comparison",
        "",
        "| method | frame P | frame R | frame F1 | segment P | segment R | segment F1 | avg seg/video | fragmented GT | FP duration | FN duration |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics:
        lines.append(
            f"| {row['method']} | {row['frame_precision']} | {row['frame_recall']} | {row['frame_f1']} | "
            f"{row['segment_precision']} | {row['segment_recall']} | {row['segment_f1']} | "
            f"{row['average_predicted_segments_per_video']} | {row['fragmented_gt_ratio']} | "
            f"{row['false_positive_duration']} | {row['false_negative_duration']} |"
        )
    lines.extend(
        [
            "",
            "## Improved Cases",
            "",
            f"- fixed rows: {len(fixed_rows)}",
        ]
    )
    for row in fixed_rows[:10]:
        lines.append(f"- {row['dataset']} / {row['video_id']} @ H4 {row['h4_frame']}: {row['explanation']}")
    lines.extend(["", "## Worsened Cases", "", f"- worsened rows: {len(failed_rows)}"])
    for row in failed_rows[:10]:
        lines.append(f"- {row['dataset']} / {row['video_id']} @ H4 {row['h4_frame']}: {row['explanation']}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- H4 candidates can explain a subset of score discontinuities where interval fragments are separated by caption-level context boundaries.",
            "- Boundary-aware merging should be interpreted as a selective recheck signal, not a universal improvement, unless metrics improve without unacceptable FP growth.",
            "- The comparison above should be read with both fixed and worsened cases, because over-merging can introduce false positives.",
            "",
            "## Limitations",
            "",
            "- The semantic continuity test is lexical and caption-based; it is not direct visual shot-boundary verification.",
            "- GT annotation granularity differs across datasets, and MSAD compact labels may not encode the same interval semantics as movie datasets.",
            "- The rule uses fixed thresholds; sensitivity analysis is still needed.",
            "",
            "## Next steps",
            "",
            "- Manually inspect fixed and worsened cases with raw frames/video.",
            "- Tune score threshold, H4 gap size, and semantic continuity rules on a validation split.",
            "- Compare against existing low-FP/valley-cut interval post-processing before making final claims.",
        ]
    )
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {item}" for item in warnings[:100])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="H1/H4 boundary-aware interval post-processing experiment.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--h4-candidates", default="outputs/caption_boundary_screen/h4_strong_candidates.csv")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--score-threshold", type=float, default=0.6)
    parser.add_argument("--merge-gap-frames", type=int, default=32)
    parser.add_argument("--h4-window-size", type=int, default=48)
    parser.add_argument("--h4-gap-max-frames", type=int, default=128)
    parser.add_argument("--min-side-score", type=float, default=0.5)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / f"{datetime.now():%y-%m-%d-%H-%M}-{TASK_NAME}"
    data_root = Path(args.data_root)
    scores_by_key, gt_by_key, warnings = load_all_inputs(data_root)
    h4_by_key = load_h4_candidates(Path(args.h4_candidates))
    all_keys = sorted(set(scores_by_key) | set(gt_by_key))
    original_by_key = {}
    aware_by_key = {}
    original_rows = []
    aware_rows = []
    compare_rows = []
    case_rows = []
    merge_event_count = 0
    for key in all_keys:
        dataset, video_id = key
        scores = scores_by_key.get(key, {})
        gt_intervals = gt_by_key.get(key, [])
        raw = raw_intervals_from_scores(scores, args.score_threshold)
        original = merge_ranges(raw, args.merge_gap_frames)
        aware, events = boundary_aware_merge(original, scores, h4_by_key.get(key, []), args)
        aware_merge_meta = {}
        for event in events:
            cand = event["h4"]
            aware_merge_meta[event["merged_interval"]] = {
                "merge_reason": "boundary_induced_score_drop",
                "source_intervals": f"{event['left_interval'][0]}-{event['left_interval'][1]};{event['right_interval'][0]}-{event['right_interval'][1]}",
                "h4_frame": cand.get("frame", cand.get("_frame", "")),
            }
        original_by_key[key] = original
        aware_by_key[key] = aware
        merge_event_count += len(events)
        original_rows.extend(interval_rows("original", dataset, video_id, original, scores, gt_intervals))
        aware_rows.extend(interval_rows("boundary_aware", dataset, video_id, aware, scores, gt_intervals, aware_merge_meta))
        compare_rows.append(comparison_row(key, original, aware, gt_intervals, events))
        if events:
            case_rows.extend(build_case_rows(key, original, aware, gt_intervals, events))
    metrics = [
        evaluate_method("original", original_by_key, gt_by_key, all_keys),
        evaluate_method("boundary_aware", aware_by_key, gt_by_key, all_keys),
    ]
    fixed_rows = [row for row in case_rows if row["fixed_or_worsened"] == "fixed"]
    failed_rows = [row for row in case_rows if row["fixed_or_worsened"] == "worsened"]
    neutral_rows = [row for row in case_rows if row["fixed_or_worsened"] not in {"fixed", "worsened"}]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "original_intervals.csv", original_rows, INTERVAL_FIELDS)
    write_csv(output_dir / "boundary_aware_intervals.csv", aware_rows, INTERVAL_FIELDS)
    write_csv(output_dir / "interval_level_comparison.csv", compare_rows, COMPARE_FIELDS)
    write_csv(output_dir / "h1_h4_interval_metrics.csv", metrics, METRIC_FIELDS)
    write_csv(output_dir / "h4_fixed_cases.csv", fixed_rows + neutral_rows, CASE_FIELDS)
    write_csv(output_dir / "h4_failed_cases.csv", failed_rows, CASE_FIELDS)
    write_json(
        output_dir / "h1_h4_boundary_aware_summary.json",
        {
            "scores_videos": len(scores_by_key),
            "gt_videos": len(gt_by_key),
            "all_videos": len(all_keys),
            "h4_candidate_videos": len(h4_by_key),
            "merge_event_count": merge_event_count,
            "fixed_case_count": len(fixed_rows),
            "failed_case_count": len(failed_rows),
            "neutral_case_count": len(neutral_rows),
            "warnings": warnings,
            "args": vars(args),
        },
    )
    write_report(output_dir / "h1_h4_boundary_aware_report.md", args, metrics, fixed_rows, failed_rows, warnings)
    script_dir = output_dir / "scripts"
    script_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), script_dir / Path(__file__).name)
    print(f"videos: {len(all_keys)}")
    print(f"merge events: {merge_event_count}")
    print(f"fixed cases: {len(fixed_rows)}")
    print(f"worsened cases: {len(failed_rows)}")
    print(f"output dir: {output_dir}")


if __name__ == "__main__":
    main()
