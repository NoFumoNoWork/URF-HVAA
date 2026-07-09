import argparse
import csv
from collections import defaultdict
from pathlib import Path


INTERVAL_FIELDS = [
    "dataset", "video_id", "pred_interval_id", "pred_start", "pred_end",
    "pred_len", "mean_score", "max_score", "overlaps_gt",
    "matched_gt_interval_ids",
]

GAP_FIELDS = [
    "dataset", "video_id", "gap_id", "left_pred_interval_id",
    "right_pred_interval_id", "gap_start", "gap_end", "gap_len",
    "left_mean_score", "right_mean_score", "gap_mean_score", "score_dip",
    "has_h4_in_gap", "has_h4_near_gap", "nearest_h4_distance",
    "h4_count_in_gap", "h4_count_near_gap", "h4_types_near_gap",
    "strongest_h4_score", "caption_before", "caption_inside",
    "caption_after", "left_gt_ids", "right_gt_ids", "gap_gt_overlap_ratio",
    "same_gt_on_both_sides", "merge_oracle_label",
]


def read_csv(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig", errors="replace") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_float(value, default=None):
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value, default=None):
    value = as_float(value, default)
    if value is None:
        return default
    return int(value)


def fmt(value, digits=6):
    if value in ("", None):
        return ""
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return value


def split_ids(value):
    return {item for item in str(value or "").split(";") if item}


def group_timeline(rows):
    grouped = defaultdict(list)
    for row in rows:
        key = (row.get("dataset", ""), row.get("video_id", ""))
        grouped[key].append(row)
    for key in grouped:
        grouped[key].sort(key=lambda row: as_float(row.get("center"), 0))
    return grouped


def build_intervals_for_video(dataset, video_id, rows, threshold, min_interval_len):
    intervals = []
    current = []
    for row in rows:
        score = as_float(row.get("anomaly_score"))
        if score is not None and score >= threshold:
            current.append(row)
        elif current:
            maybe_add_interval(dataset, video_id, current, intervals, min_interval_len)
            current = []
    if current:
        maybe_add_interval(dataset, video_id, current, intervals, min_interval_len)
    return intervals


def maybe_add_interval(dataset, video_id, items, intervals, min_interval_len):
    start = as_int(items[0].get("start"), 0)
    end = as_int(items[-1].get("end"), start)
    pred_len = max(0, end - start)
    if pred_len < min_interval_len:
        return
    scores = [as_float(row.get("anomaly_score")) for row in items if as_float(row.get("anomaly_score")) is not None]
    gt_ids = set()
    for row in items:
        gt_ids.update(split_ids(row.get("gt_interval_id")))
    pred_id = f"{video_id}_pred_{len(intervals)}"
    intervals.append({
        "dataset": dataset,
        "video_id": video_id,
        "pred_interval_id": pred_id,
        "pred_start": start,
        "pred_end": end,
        "pred_len": pred_len,
        "mean_score": fmt(sum(scores) / len(scores) if scores else ""),
        "max_score": fmt(max(scores) if scores else ""),
        "overlaps_gt": bool(gt_ids),
        "matched_gt_interval_ids": ";".join(sorted(gt_ids)),
    })


def items_between(rows, start, end):
    return [
        row for row in rows
        if as_float(row.get("center"), -1) is not None and start <= as_float(row.get("center"), -1) <= end
    ]


def item_caption(rows, default=""):
    for row in rows:
        caption = row.get("caption", "")
        if caption:
            return caption
    return default


def h4_features(rows, gap_start, gap_end, h4_window):
    in_gap = []
    near_gap = []
    nearest_distance = None
    types = set()
    scores = []
    for row in rows:
        dist = as_float(row.get("nearest_h4_distance"))
        if dist is None:
            continue
        center = as_float(row.get("center"), 0)
        h4_count = as_int(row.get("h4_count_nearby"), 0) or 0
        h4_type = row.get("nearest_h4_type", "")
        h4_score = as_float(row.get("nearest_h4_score"))
        inside = gap_start <= center <= gap_end
        near = (gap_start - h4_window) <= center <= (gap_end + h4_window)
        if h4_count > 0 and inside:
            in_gap.append(row)
        if h4_count > 0 and near:
            near_gap.append(row)
            if h4_type:
                types.update(item for item in h4_type.split(";") if item)
            if h4_score is not None:
                scores.append(h4_score)
        if nearest_distance is None or dist < nearest_distance:
            nearest_distance = dist
    return {
        "has_h4_in_gap": bool(in_gap),
        "has_h4_near_gap": bool(near_gap),
        "nearest_h4_distance": fmt(nearest_distance),
        "h4_count_in_gap": sum(as_int(row.get("h4_count_nearby"), 0) or 0 for row in in_gap),
        "h4_count_near_gap": sum(as_int(row.get("h4_count_nearby"), 0) or 0 for row in near_gap),
        "h4_types_near_gap": ";".join(sorted(types)),
        "strongest_h4_score": fmt(max(scores) if scores else ""),
    }


def oracle_label(left_gt, right_gt, gap_rows):
    gap_gt_items = [row for row in gap_rows if str(row.get("is_gt")).lower() == "true"]
    ratio = len(gap_gt_items) / len(gap_rows) if gap_rows else 0.0
    same = bool(left_gt and right_gt and left_gt.intersection(right_gt))
    if same and ratio >= 0.5:
        return "positive_merge", ratio, True
    if not left_gt or not right_gt:
        return "risky_merge" if ratio > 0 else "unknown", ratio, False
    if not same:
        return "negative_merge", ratio, False
    return "risky_merge", ratio, same


def build_gaps_for_video(dataset, video_id, rows, intervals, h4_window):
    gaps = []
    by_span = sorted(intervals, key=lambda row: as_int(row["pred_start"], 0))
    for idx, (left, right) in enumerate(zip(by_span, by_span[1:])):
        gap_start = as_int(left["pred_end"], 0)
        gap_end = as_int(right["pred_start"], gap_start)
        if gap_end < gap_start:
            continue
        gap_rows = items_between(rows, gap_start, gap_end)
        gap_scores = [as_float(row.get("anomaly_score")) for row in gap_rows if as_float(row.get("anomaly_score")) is not None]
        left_mean = as_float(left.get("mean_score"), 0)
        right_mean = as_float(right.get("mean_score"), 0)
        gap_mean = sum(gap_scores) / len(gap_scores) if gap_scores else ""
        score_dip = min(left_mean, right_mean) - gap_mean if gap_mean != "" else ""
        left_gt = split_ids(left.get("matched_gt_interval_ids"))
        right_gt = split_ids(right.get("matched_gt_interval_ids"))
        label, gap_gt_ratio, same_gt = oracle_label(left_gt, right_gt, gap_rows)
        h4 = h4_features(rows, gap_start, gap_end, h4_window)
        before_rows = [row for row in rows if as_int(row.get("end"), 0) <= gap_start]
        after_rows = [row for row in rows if as_int(row.get("start"), 0) >= gap_end]
        gaps.append({
            "dataset": dataset,
            "video_id": video_id,
            "gap_id": f"{video_id}_gap_{idx}",
            "left_pred_interval_id": left["pred_interval_id"],
            "right_pred_interval_id": right["pred_interval_id"],
            "gap_start": gap_start,
            "gap_end": gap_end,
            "gap_len": max(0, gap_end - gap_start),
            "left_mean_score": left.get("mean_score", ""),
            "right_mean_score": right.get("mean_score", ""),
            "gap_mean_score": fmt(gap_mean),
            "score_dip": fmt(score_dip),
            "caption_before": item_caption(reversed(before_rows)),
            "caption_inside": item_caption(gap_rows),
            "caption_after": item_caption(after_rows),
            "left_gt_ids": ";".join(sorted(left_gt)),
            "right_gt_ids": ";".join(sorted(right_gt)),
            "gap_gt_overlap_ratio": fmt(gap_gt_ratio),
            "same_gt_on_both_sides": same_gt,
            "merge_oracle_label": label,
            **h4,
        })
    return gaps


def write_notes(path, threshold, min_interval_len, h4_window, num_intervals, num_gaps):
    lines = [
        "# Prediction gap notes",
        "",
        f"- score_threshold: {threshold}",
        f"- min_interval_len: {min_interval_len}",
        f"- h4_window: {h4_window}",
        f"- prediction intervals: {num_intervals}",
        f"- prediction gaps: {num_gaps}",
        "",
        "## Interpretation limits",
        "",
        "- Prediction intervals are generated from canonical timeline anomaly scores when no external interval file is supplied.",
        "- `merge_oracle_label` uses GT information and is only for upper-bound diagnostics, not a deployable merge rule.",
        "- H4 features in gap rows are caption-level boundary-candidate features. They do not imply a true camera transition or a new scene.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(canonical_timeline, output_dir, threshold, min_interval_len, h4_window):
    rows = read_csv(canonical_timeline)
    grouped = group_timeline(rows)
    all_intervals = []
    all_gaps = []
    for (dataset, video_id), items in grouped.items():
        intervals = build_intervals_for_video(dataset, video_id, items, threshold, min_interval_len)
        all_intervals.extend(intervals)
        all_gaps.extend(build_gaps_for_video(dataset, video_id, items, intervals, h4_window))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "prediction_intervals.csv", all_intervals, INTERVAL_FIELDS)
    write_csv(output_dir / "prediction_gaps.csv", all_gaps, GAP_FIELDS)
    write_notes(output_dir / "prediction_gap_notes.md", threshold, min_interval_len, h4_window, len(all_intervals), len(all_gaps))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-timeline", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--score-threshold", type=float, default=0.6)
    parser.add_argument("--min-interval-len", type=int, default=1)
    parser.add_argument("--h4-window", type=int, default=60)
    args = parser.parse_args()
    run(Path(args.canonical_timeline), Path(args.output_dir), args.score_threshold, args.min_interval_len, args.h4_window)


if __name__ == "__main__":
    main()
