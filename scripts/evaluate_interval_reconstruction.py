import argparse
import csv
from collections import defaultdict
from pathlib import Path


def as_float(value, default=0.0):
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig", errors="replace") as f:
        return list(csv.DictReader(f))


def load_intervals(path):
    """Load prediction intervals from a CSV.

    Expected columns are flexible: pred_start/pred_end or start_frame/end_frame.
    Returns a dict keyed by (dataset, video_id). Frame-level recall and segment
    F1 can move in different directions after merging, so downstream code should
    inspect both rather than optimizing a single number.
    """
    grouped = defaultdict(list)
    for row in read_csv(path):
        dataset = row.get("dataset", "")
        video_id = row.get("video_id", "")
        start = as_float(row.get("pred_start", row.get("start_frame", row.get("start", 0))))
        end = as_float(row.get("pred_end", row.get("end_frame", row.get("end", 0))))
        if end > start:
            grouped[(dataset, video_id)].append((start, end))
    return {key: sorted(value) for key, value in grouped.items()}


def load_gt(path):
    """Load GT intervals from prediction-style CSV rows.

    For raw annotation text files, use the resource prep script's parser instead.
    This skeleton intentionally keeps GT loading simple for experiment outputs.
    """
    grouped = defaultdict(list)
    for row in read_csv(path):
        dataset = row.get("dataset", "")
        video_id = row.get("video_id", "")
        start = as_float(row.get("gt_start", row.get("start", row.get("start_frame", 0))))
        end = as_float(row.get("gt_end", row.get("end", row.get("end_frame", 0))))
        if end > start:
            grouped[(dataset, video_id)].append((start, end))
    return {key: sorted(value) for key, value in grouped.items()}


def interval_overlap(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def duration(intervals):
    return sum(max(0.0, end - start) for start, end in intervals)


def union_duration(intervals):
    if not intervals:
        return 0.0
    merged = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return duration(merged)


def intersection_duration(a_intervals, b_intervals):
    total = 0.0
    for a in a_intervals:
        for b in b_intervals:
            total += interval_overlap(a, b)
    return total


def frame_level_metrics(pred_intervals, gt_intervals):
    """Compute frame-level precision/recall/F1.

    Merging adjacent predictions often increases recall by filling gaps, but it
    can also expand false positives. This project cares about that TP-FP tradeoff.
    """
    tp = intersection_duration(pred_intervals, gt_intervals)
    pred_dur = union_duration(pred_intervals)
    gt_dur = union_duration(gt_intervals)
    precision = tp / pred_dur if pred_dur else 0.0
    recall = tp / gt_dur if gt_dur else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "frame_precision": precision,
        "frame_recall": recall,
        "frame_f1": f1,
        "tp_duration": tp,
        "pred_duration": pred_dur,
        "gt_duration": gt_dur,
    }


def segment_level_metrics(pred_intervals, gt_intervals, iou_threshold=0.1):
    """Compute segment-level precision/recall/F1 with greedy matching.

    Segment F1 may drop after a merge even when frame recall rises, because
    oversized intervals can become poor segment matches or introduce FP segments.
    """
    matched_gt = set()
    tp = 0
    for pred in pred_intervals:
        best_idx = None
        best_iou = 0.0
        for idx, gt in enumerate(gt_intervals):
            if idx in matched_gt:
                continue
            overlap = interval_overlap(pred, gt)
            union = max(pred[1], gt[1]) - min(pred[0], gt[0])
            iou = overlap / union if union else 0.0
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx is not None and best_iou >= iou_threshold:
            matched_gt.add(best_idx)
            tp += 1
    precision = tp / len(pred_intervals) if pred_intervals else 0.0
    recall = tp / len(gt_intervals) if gt_intervals else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "segment_precision": precision,
        "segment_recall": recall,
        "segment_f1": f1,
        "segment_tp": tp,
    }


def gt_fragmentation_ratio(pred_intervals, gt_intervals):
    fragmented = 0
    for gt in gt_intervals:
        overlaps = [pred for pred in pred_intervals if interval_overlap(pred, gt) > 0]
        if len(overlaps) > 1:
            fragmented += 1
    return fragmented / len(gt_intervals) if gt_intervals else 0.0


def false_positive_duration(pred_intervals, gt_intervals):
    tp = intersection_duration(pred_intervals, gt_intervals)
    return max(0.0, union_duration(pred_intervals) - tp)


def compare_before_after(original_pred, merged_pred, gt):
    """Compare interval sets before and after a merge strategy."""
    before = frame_level_metrics(original_pred, gt)
    after = frame_level_metrics(merged_pred, gt)
    delta_tp = after["tp_duration"] - before["tp_duration"]
    delta_fp = false_positive_duration(merged_pred, gt) - false_positive_duration(original_pred, gt)
    return {
        "delta_tp": delta_tp,
        "delta_fp": delta_fp,
        "before_frame_f1": before["frame_f1"],
        "after_frame_f1": after["frame_f1"],
    }


def net_merge_utility(delta_tp, delta_fp, lambda_fp=1.0):
    """Utility for TP-FP tradeoff: positive TP gain minus FP penalty."""
    return delta_tp - lambda_fp * delta_fp


def main():
    parser = argparse.ArgumentParser(description="Basic interval reconstruction evaluation skeleton.")
    parser.add_argument("--pred", help="Prediction interval CSV")
    parser.add_argument("--gt", help="GT interval CSV in normalized interval format")
    args = parser.parse_args()
    if not args.pred or not args.gt:
        print("Skeleton ready. Provide --pred and --gt to run basic loading.")
        return
    pred = load_intervals(args.pred)
    gt = load_gt(args.gt)
    keys = sorted(set(pred) | set(gt))
    for key in keys[:10]:
        print(key, frame_level_metrics(pred.get(key, []), gt.get(key, [])))


if __name__ == "__main__":
    main()
