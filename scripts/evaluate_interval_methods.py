import argparse
import csv
import json
import math
import shutil
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.anomaly_utils import write_json  # noqa: E402


SUPPORTABLE_TYPES = {"strongly_score_supported", "weakly_score_supported"}
UNSUPPORTABLE_TYPES = {"score_unsupported", "unobserved_or_missing_score"}
UNCERTAIN_TYPES = {"sparsely_sampled", "barely_sampled", "ambiguous_mid_score"}


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        if not fields:
            return
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(value, default=0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value, default=0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def ratio(num: float, den: float) -> float:
    return num / den if den else math.nan


def clean_interval(start, end) -> tuple[int, int] | None:
    s = as_int(start)
    e = as_int(end)
    if e <= s:
        return None
    return s, e


def merge_ranges(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    valid = sorted((s, e) for s, e in intervals if e > s)
    if not valid:
        return []
    merged = [valid[0]]
    for s, e in valid[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e:
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))
    return merged


def duration(intervals: list[tuple[int, int]]) -> int:
    return sum(e - s for s, e in intervals)


def intersect_duration(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
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


def overlaps(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def interval_iou(a: tuple[int, int], b: tuple[int, int]) -> float:
    inter = overlaps(a, b)
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union else 0.0


def interval_overlap_duration(interval: tuple[int, int], intervals: list[tuple[int, int]]) -> int:
    pieces = []
    for other in intervals:
        s = max(interval[0], other[0])
        e = min(interval[1], other[1])
        if e > s:
            pieces.append((s, e))
    return duration(merge_ranges(pieces))


def support_group(row: dict) -> str:
    recoverable = str(row.get("recoverable_by_postprocessing", "")).strip().lower()
    support_type = str(row.get("support_type", "")).strip()
    if recoverable == "true" or support_type in SUPPORTABLE_TYPES:
        return "supportable"
    if recoverable == "false" or support_type in UNSUPPORTABLE_TYPES:
        return "unsupportable"
    return "uncertain"


def load_gt_rows(gt_stats_csv: Path, gt_support_csv: Path) -> list[dict]:
    rows = read_csv(gt_support_csv)
    if not rows:
        rows = read_csv(gt_stats_csv)
    normalized = []
    for row in rows:
        interval = clean_interval(row.get("gt_start"), row.get("gt_end"))
        if not interval:
            continue
        out = dict(row)
        out["dataset"] = row.get("dataset", "")
        out["video_id"] = row.get("video_id", "")
        out["label"] = row.get("label", "")
        out["start"] = interval[0]
        out["end"] = interval[1]
        out["duration"] = interval[1] - interval[0]
        out["support_group"] = support_group(row)
        normalized.append(out)
    return normalized


def load_inventory(video_inventory_csv: Path, gt_rows: list[dict]) -> dict[tuple[str, str], dict]:
    inventory = {}
    for row in read_csv(video_inventory_csv):
        key = (row.get("dataset", ""), row.get("video_id", ""))
        inventory[key] = {
            "dataset": key[0],
            "video_id": key[1],
            "label": row.get("label", ""),
            "score_json_path": row.get("score_json_path", ""),
            "score_point_count": as_int(row.get("score_point_count")),
            "score_stride_est": as_int(row.get("score_stride_est"), 16),
            "video_length": as_int(row.get("video_length_frame_est")),
        }
    for row in gt_rows:
        key = (row["dataset"], row["video_id"])
        inventory.setdefault(
            key,
            {
                "dataset": key[0],
                "video_id": key[1],
                "label": row.get("label", ""),
                "score_json_path": row.get("score_json_path", ""),
                "score_point_count": 0,
                "score_stride_est": 16,
                "video_length": 0,
            },
        )
        inventory[key]["video_length"] = max(inventory[key]["video_length"], row["end"])
    return inventory


def add_interval(rows: list[dict], method: str, dataset: str, video_id: str, start, end, source: Path | str) -> None:
    interval = clean_interval(start, end)
    if not interval:
        return
    rows.append(
        {
            "method": method,
            "dataset": dataset,
            "video_id": video_id,
            "start": interval[0],
            "end": interval[1],
            "source_path": str(source),
        }
    )


def parse_dataset_video_list_json(path: Path, method: str, rows: list[dict]) -> bool:
    data = read_json(path)
    if not isinstance(data, dict):
        return False
    found = 0
    for dataset, videos in data.items():
        if not isinstance(videos, dict):
            continue
        for video_id, intervals in videos.items():
            if isinstance(intervals, list):
                for item in intervals:
                    if isinstance(item, dict):
                        before = len(rows)
                        add_interval(rows, method, dataset, video_id, item.get("start"), item.get("end"), path)
                        found += len(rows) - before
            elif isinstance(intervals, dict) and isinstance(intervals.get("intervals"), list):
                for item in intervals.get("intervals", []):
                    if isinstance(item, dict):
                        before = len(rows)
                        add_interval(rows, method, dataset, video_id, item.get("start"), item.get("end"), path)
                        found += len(rows) - before
    return found > 0


def parse_hierarchical(path: Path, rows: list[dict]) -> bool:
    data = read_json(path)
    found = 0
    for video_id, meta in data.items():
        if not isinstance(meta, dict):
            continue
        dataset = meta.get("dataset", "")
        vid = meta.get("video_id", video_id)
        for event in meta.get("events", []):
            before = len(rows)
            add_interval(rows, "Hierarchical-Merged", dataset, vid, event.get("merged_start"), event.get("merged_end"), path)
            found += len(rows) - before
            for micro in event.get("micro_intervals", []):
                before = len(rows)
                scale = micro.get("scale")
                method = f"Hierarchical-Micro-{scale}F" if scale else "Hierarchical-Micro"
                add_interval(rows, method, dataset, vid, micro.get("start"), micro.get("end"), path)
                found += len(rows) - before
    return found > 0


def parse_refined(path: Path, rows: list[dict]) -> bool:
    data = read_json(path)
    found = 0
    for video_id, meta in data.items():
        if not isinstance(meta, dict):
            continue
        dataset = meta.get("dataset", "")
        vid = meta.get("video_id", video_id)
        for item in meta.get("refined_intervals", []):
            before = len(rows)
            add_interval(rows, "Peak-Aware-Refined", dataset, vid, item.get("start"), item.get("end"), path)
            found += len(rows) - before
    return found > 0


def parse_wmax_baselines(path: Path, rows: list[dict]) -> bool:
    data = read_json(path)
    found = 0
    for method_key, by_dataset in data.items():
        if not isinstance(by_dataset, dict):
            continue
        method = {
            "wmax": "Wmax-Baseline",
            "random": "Random-Same-Length",
            "random_same_length": "Random-Same-Length",
            "peak_expanded": "Peak-Expanded-Baseline",
            "peak_expand": "Peak-Expanded-Baseline",
        }.get(method_key, f"Wmax-{method_key}")
        for dataset, videos in by_dataset.items():
            if not isinstance(videos, dict):
                continue
            for video_id, item in videos.items():
                items = item if isinstance(item, list) else [item]
                for interval in items:
                    if isinstance(interval, dict):
                        before = len(rows)
                        add_interval(rows, method, dataset, video_id, interval.get("start"), interval.get("end"), path)
                        found += len(rows) - before
    return found > 0


def parse_csv_intervals(path: Path, rows: list[dict]) -> bool:
    csv_rows = read_csv(path)
    if not csv_rows:
        return False
    fields = set(csv_rows[0])
    if not {"dataset", "video_id", "start", "end"}.issubset(fields):
        return False
    found = 0
    for row in csv_rows:
        method = row.get("method") or path.stem
        before = len(rows)
        add_interval(rows, method, row.get("dataset", ""), row.get("video_id", ""), row.get("start"), row.get("end"), path)
        found += len(rows) - before
    return found > 0


def parse_known_interval_file(path: Path, rows: list[dict]) -> tuple[bool, str]:
    try:
        name = path.name
        if name == "hierarchical_intervals.json":
            return parse_hierarchical(path, rows), "hierarchical"
        if name == "refined_intervals.json":
            return parse_refined(path, rows), "peak_refined"
        if name == "intervals.json" and "wmax_replacement_baselines" in str(path):
            return parse_wmax_baselines(path, rows), "wmax_baselines"
        if name.startswith("topk_k") and name.endswith(".json"):
            k = name.replace("topk_k", "").replace(".json", "")
            return parse_dataset_video_list_json(path, f"TopK-{k}", rows), "topk"
        if name == "multiscale_intervals.json":
            return parse_dataset_video_list_json(path, "Multiscale", rows), "multiscale"
        if name.startswith("adaptive_intervals") and name.endswith(".json"):
            return parse_dataset_video_list_json(path, "Adaptive", rows), "adaptive"
        if path.suffix.lower() == ".csv":
            return parse_csv_intervals(path, rows), "csv"
        if path.suffix.lower() == ".json":
            before = len(rows)
            ok = parse_dataset_video_list_json(path, path.stem, rows)
            return ok, "generic_json" if len(rows) > before else "generic_json_empty"
    except Exception as exc:  # noqa: BLE001
        return False, f"error: {exc}"
    return False, "unsupported"


def auto_scan_methods(root_output_dir: Path, rows: list[dict]) -> list[dict]:
    candidates = []
    patterns = [
        "**/hierarchical_intervals.json",
        "**/refined_intervals.json",
        "**/topk_k*.json",
        "**/multiscale_intervals.json",
        "**/adaptive_intervals*.json",
        "**/wmax_replacement_baselines/intervals.json",
    ]
    seen = set()
    for pattern in patterns:
        for path in root_output_dir.glob(pattern):
            if path in seen:
                continue
            seen.add(path)
            candidates.append(path)

    skipped = []
    for path in sorted(candidates):
        before = len(rows)
        ok, reason = parse_known_interval_file(path, rows)
        added = len(rows) - before
        skipped.append(
            {
                "source_path": str(path),
                "parsed": ok,
                "reason": reason,
                "intervals_added": added,
            }
        )
    return skipped


def load_methods_config(path: Path | None, rows: list[dict]) -> list[dict]:
    if not path:
        return []
    data = read_json(path)
    skipped = []
    for item in data.get("methods", []):
        source = Path(item["path"])
        method = item.get("method", source.stem)
        if not source.is_absolute():
            source = Path.cwd() / source
        before = len(rows)
        if source.suffix.lower() == ".csv":
            ok = parse_csv_intervals(source, rows)
        else:
            ok = parse_dataset_video_list_json(source, method, rows)
        skipped.append({"source_path": str(source), "parsed": ok, "reason": "methods_config", "intervals_added": len(rows) - before})
    return skipped


def load_score_series(path: Path) -> dict[int, float]:
    if not path.exists():
        return {}
    raw = read_json(path)
    if isinstance(raw, dict):
        out = {}
        for k, v in raw.items():
            try:
                out[int(k)] = float(v)
            except (TypeError, ValueError):
                continue
        return dict(sorted(out.items()))
    if isinstance(raw, list):
        return {i: as_float(v) for i, v in enumerate(raw)}
    return {}


def add_window_methods(
    rows: list[dict],
    inventory: dict[tuple[str, str], dict],
    window_sizes: list[int],
    threshold: float,
    root: Path,
) -> list[dict]:
    skipped = []
    for key, meta in inventory.items():
        score_path_text = meta.get("score_json_path", "")
        if not score_path_text:
            continue
        score_path = Path(score_path_text)
        if not score_path.is_absolute():
            score_path = root / score_path
        scores = load_score_series(score_path)
        if not scores:
            skipped.append({"source_path": str(score_path), "parsed": False, "reason": "missing_or_empty_score_json", "intervals_added": 0})
            continue
        frames = sorted(scores)
        stride = meta.get("score_stride_est") or (frames[1] - frames[0] if len(frames) > 1 else 16)
        video_len = max(meta.get("video_length", 0), frames[-1] + stride)
        for win in window_sizes:
            positive = []
            step = max(1, win)
            for start in range(0, max(video_len, 1), step):
                end = min(video_len, start + win)
                vals = [scores[f] for f in frames if start <= f < end]
                if vals and sum(vals) / len(vals) >= threshold:
                    positive.append((start, end))
            for start, end in merge_ranges(positive):
                add_interval(rows, f"Window-{win}F", meta["dataset"], meta["video_id"], start, end, score_path)
    return skipped


def group_gt(gt_rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    out = defaultdict(list)
    for row in gt_rows:
        out[(row["dataset"], row["video_id"])].append(row)
    return out


def group_pred(pred_rows: list[dict]) -> dict[tuple[str, str, str], list[dict]]:
    out = defaultdict(list)
    for row in pred_rows:
        out[(row["method"], row["dataset"], row["video_id"])].append(row)
    return out


def video_length_for(key: tuple[str, str], inventory: dict[tuple[str, str], dict], gt_video_rows: list[dict], pred_ranges: list[tuple[int, int]]) -> int:
    length = inventory.get(key, {}).get("video_length", 0)
    for row in gt_video_rows:
        length = max(length, row["end"])
    for _, e in pred_ranges:
        length = max(length, e)
    return length


def evaluate_one_video(
    method: str,
    dataset: str,
    video_id: str,
    gt_rows: list[dict],
    pred_rows: list[dict],
    inventory: dict[tuple[str, str], dict],
    iou_thresholds: list[float],
) -> dict:
    pred_raw = [(as_int(r["start"]), as_int(r["end"])) for r in pred_rows if as_int(r["end"]) > as_int(r["start"])]
    pred = merge_ranges(pred_raw)
    gt_raw = [(r["start"], r["end"]) for r in gt_rows]
    gt_merged = merge_ranges(gt_raw)
    inter = intersect_duration(pred, gt_merged)
    total_pred = duration(pred)
    total_gt = duration(gt_merged)
    video_len = video_length_for((dataset, video_id), inventory, gt_rows, pred)

    gt_hit = sum(1 for gt in gt_raw if any(overlaps(gt, p) > 0 for p in pred))
    pred_hit = sum(1 for p in pred if any(overlaps(p, gt) > 0 for gt in gt_merged))
    lengths = [e - s for s, e in pred]

    row = {
        "method": method,
        "dataset": dataset,
        "video_id": video_id,
        "total_video_duration": video_len,
        "total_gt_duration": total_gt,
        "total_predicted_duration": total_pred,
        "intersection_duration_pred_gt": inter,
        "predicted_GT_fraction": ratio(inter, total_pred),
        "GT_coverage": ratio(inter, total_gt),
        "GT_uncovered_ratio": 1 - ratio(inter, total_gt) if total_gt else math.nan,
        "over_coverage_duration": max(0, total_pred - inter),
        "over_coverage_ratio": ratio(max(0, total_pred - inter), total_pred),
        "predicted_duration_ratio": ratio(total_pred, video_len),
        "gt_event_count": len(gt_raw),
        "predicted_event_count": len(pred),
        "gt_event_hit_count": gt_hit,
        "gt_event_missed_count": len(gt_raw) - gt_hit,
        "gt_event_hit_ratio": ratio(gt_hit, len(gt_raw)),
        "gt_event_missed_ratio": ratio(len(gt_raw) - gt_hit, len(gt_raw)),
        "pred_interval_count": len(pred),
        "pred_interval_with_gt_overlap_count": pred_hit,
        "pred_interval_no_gt_overlap_count": len(pred) - pred_hit,
        "pred_interval_gt_overlap_ratio": ratio(pred_hit, len(pred)),
        "mean_predicted_interval_length": statistics.mean(lengths) if lengths else math.nan,
        "median_predicted_interval_length": statistics.median(lengths) if lengths else math.nan,
    }
    for threshold in iou_thresholds:
        count = sum(1 for gt in gt_raw if any(interval_iou(gt, p) >= threshold for p in pred))
        row[f"gt_event_hit_count_iou_{threshold}"] = count
        row[f"gt_event_hit_ratio_iou_{threshold}"] = ratio(count, len(gt_raw))
    for group in ("supportable", "unsupportable", "uncertain"):
        group_rows = [r for r in gt_rows if r["support_group"] == group]
        group_ranges = merge_ranges([(r["start"], r["end"]) for r in group_rows])
        group_duration = duration(group_ranges)
        covered_duration = intersect_duration(pred, group_ranges)
        event_hit = sum(1 for gt in group_rows if any(overlaps((gt["start"], gt["end"]), p) > 0 for p in pred))
        row[f"{group}_gt_duration"] = group_duration
        row[f"{group}_gt_covered_duration"] = covered_duration
        row[f"{group}_gt_coverage"] = ratio(covered_duration, group_duration)
        row[f"{group}_gt_uncovered_ratio"] = 1 - ratio(covered_duration, group_duration) if group_duration else math.nan
        row[f"{group}_gt_event_count"] = len(group_rows)
        row[f"{group}_gt_event_hit_count"] = event_hit
        row[f"{group}_gt_event_hit_ratio"] = ratio(event_hit, len(group_rows))
    return row


def aggregate_rows(rows: list[dict], keys: tuple[str, str], iou_thresholds: list[float]) -> dict:
    out = {key: rows[0][key] for key in keys}
    sum_fields = [
        "total_video_duration",
        "total_gt_duration",
        "total_predicted_duration",
        "intersection_duration_pred_gt",
        "over_coverage_duration",
        "gt_event_count",
        "predicted_event_count",
        "gt_event_hit_count",
        "gt_event_missed_count",
        "pred_interval_count",
        "pred_interval_with_gt_overlap_count",
        "pred_interval_no_gt_overlap_count",
        "supportable_gt_duration",
        "supportable_gt_covered_duration",
        "supportable_gt_event_count",
        "supportable_gt_event_hit_count",
        "unsupportable_gt_duration",
        "unsupportable_gt_covered_duration",
        "unsupportable_gt_event_count",
        "unsupportable_gt_event_hit_count",
        "uncertain_gt_duration",
        "uncertain_gt_covered_duration",
        "uncertain_gt_event_count",
        "uncertain_gt_event_hit_count",
    ]
    for field in sum_fields:
        out[field] = sum(as_float(r.get(field)) for r in rows)
    for threshold in iou_thresholds:
        out[f"gt_event_hit_count_iou_{threshold}"] = sum(as_float(r.get(f"gt_event_hit_count_iou_{threshold}")) for r in rows)
    pred_lengths = []
    for row in rows:
        n = as_int(row.get("pred_interval_count"))
        med = as_float(row.get("median_predicted_interval_length"), math.nan)
        if not math.isnan(med):
            pred_lengths.extend([med] * max(1, n))

    out["predicted_duration_ratio"] = ratio(out["total_predicted_duration"], out["total_video_duration"])
    out["predicted_GT_fraction"] = ratio(out["intersection_duration_pred_gt"], out["total_predicted_duration"])
    out["GT_coverage"] = ratio(out["intersection_duration_pred_gt"], out["total_gt_duration"])
    out["GT_uncovered_ratio"] = 1 - out["GT_coverage"] if not math.isnan(out["GT_coverage"]) else math.nan
    out["over_coverage_ratio"] = ratio(out["over_coverage_duration"], out["total_predicted_duration"])
    out["gt_event_hit_ratio"] = ratio(out["gt_event_hit_count"], out["gt_event_count"])
    out["gt_event_missed_ratio"] = ratio(out["gt_event_missed_count"], out["gt_event_count"])
    out["pred_interval_gt_overlap_ratio"] = ratio(out["pred_interval_with_gt_overlap_count"], out["pred_interval_count"])
    for group in ("supportable", "unsupportable", "uncertain"):
        out[f"{group}_gt_coverage"] = ratio(out[f"{group}_gt_covered_duration"], out[f"{group}_gt_duration"])
        out[f"{group}_gt_uncovered_ratio"] = 1 - out[f"{group}_gt_coverage"] if not math.isnan(out[f"{group}_gt_coverage"]) else math.nan
        out[f"{group}_gt_event_hit_ratio"] = ratio(out[f"{group}_gt_event_hit_count"], out[f"{group}_gt_event_count"])
    for threshold in iou_thresholds:
        out[f"gt_event_hit_ratio_iou_{threshold}"] = ratio(out[f"gt_event_hit_count_iou_{threshold}"], out["gt_event_count"])
    out["mean_predicted_interval_length"] = ratio(out["total_predicted_duration"], out["pred_interval_count"])
    out["median_predicted_interval_length"] = statistics.median(pred_lengths) if pred_lengths else math.nan
    out["num_predicted_intervals"] = out["pred_interval_count"]
    out["balanced_score"] = (
        0.4 * (0 if math.isnan(out["GT_coverage"]) else out["GT_coverage"])
        + 0.3 * (0 if math.isnan(out["predicted_GT_fraction"]) else out["predicted_GT_fraction"])
        + 0.2 * (0 if math.isnan(out["supportable_gt_coverage"]) else out["supportable_gt_coverage"])
        - 0.1 * (0 if math.isnan(out["predicted_duration_ratio"]) else out["predicted_duration_ratio"])
    )
    return out


def evaluate_methods(
    pred_rows: list[dict],
    gt_rows: list[dict],
    inventory: dict[tuple[str, str], dict],
    iou_thresholds: list[float],
) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict]]:
    gt_by_video = group_gt(gt_rows)
    pred_by_video = group_pred(pred_rows)
    methods = sorted({row["method"] for row in pred_rows})
    all_video_keys = sorted(gt_by_video)
    per_video = []
    for method in methods:
        for dataset, video_id in all_video_keys:
            preds = pred_by_video.get((method, dataset, video_id), [])
            per_video.append(evaluate_one_video(method, dataset, video_id, gt_by_video[(dataset, video_id)], preds, inventory, iou_thresholds))

    overall = []
    for method in methods:
        method_rows = [r for r in per_video if r["method"] == method]
        for dataset in sorted({r["dataset"] for r in method_rows}):
            overall.append(aggregate_rows([r for r in method_rows if r["dataset"] == dataset], ("method", "dataset"), iou_thresholds))
        all_row = aggregate_rows(method_rows, ("method",), iou_thresholds)
        all_row["dataset"] = "ALL"
        overall.append(all_row)

    support_rows = []
    for row in overall:
        for group in ("supportable", "unsupportable", "uncertain"):
            support_rows.append(
                {
                    "method": row["method"],
                    "dataset": row["dataset"],
                    "support_group": group,
                    "gt_duration": row[f"{group}_gt_duration"],
                    "covered_duration": row[f"{group}_gt_covered_duration"],
                    "coverage": row[f"{group}_gt_coverage"],
                    "uncovered_ratio": row[f"{group}_gt_uncovered_ratio"],
                    "event_count": row[f"{group}_gt_event_count"],
                    "event_hit_count": row[f"{group}_gt_event_hit_count"],
                    "event_hit_ratio": row[f"{group}_gt_event_hit_ratio"],
                }
            )

    iou_rows = []
    for row in overall:
        for threshold in iou_thresholds:
            iou_rows.append(
                {
                    "method": row["method"],
                    "dataset": row["dataset"],
                    "iou_threshold": threshold,
                    "gt_event_count": row["gt_event_count"],
                    "gt_event_hit_count": row[f"gt_event_hit_count_iou_{threshold}"],
                    "gt_event_hit_ratio": row[f"gt_event_hit_ratio_iou_{threshold}"],
                }
            )

    ranking = []
    all_overall = [r for r in overall if r["dataset"] == "ALL"]
    rank_specs = [
        ("best_gt_coverage", "GT_coverage"),
        ("best_predicted_GT_fraction", "predicted_GT_fraction"),
        ("best_supportable_gt_coverage", "supportable_gt_coverage"),
        ("best_balanced_score", "balanced_score"),
    ]
    for rank_name, field in rank_specs:
        ordered = sorted(all_overall, key=lambda r: (-math.inf if math.isnan(as_float(r.get(field), math.nan)) else as_float(r.get(field))), reverse=True)
        for idx, row in enumerate(ordered, start=1):
            ranking.append(
                {
                    "ranking_dimension": rank_name,
                    "rank": idx,
                    "method": row["method"],
                    "dataset": "ALL",
                    "score": row.get(field),
                    "GT_coverage": row["GT_coverage"],
                    "predicted_GT_fraction": row["predicted_GT_fraction"],
                    "supportable_gt_coverage": row["supportable_gt_coverage"],
                    "unsupportable_gt_coverage": row["unsupportable_gt_coverage"],
                    "predicted_duration_ratio": row["predicted_duration_ratio"],
                    "balanced_score": row["balanced_score"],
                }
            )
    return per_video, overall, support_rows, iou_rows, ranking


def fmt(value) -> str:
    try:
        value = float(value)
        if math.isnan(value):
            return "NA"
        return f"{value:.3f}"
    except (TypeError, ValueError):
        return str(value)


def top_methods(overall: list[dict], limit: int = 16) -> list[dict]:
    all_rows = [r for r in overall if r["dataset"] == "ALL"]
    return sorted(all_rows, key=lambda r: as_float(r.get("balanced_score"), -999), reverse=True)[:limit]


def plot_coverage_purity(overall: list[dict], output: Path) -> None:
    rows = top_methods(overall, 18)
    fig, ax = plt.subplots(figsize=(11, 7))
    for dataset, marker in [("ALL", "o")]:
        xs = [as_float(r["predicted_GT_fraction"], math.nan) for r in rows if r["dataset"] == dataset]
        ys = [as_float(r["GT_coverage"], math.nan) for r in rows if r["dataset"] == dataset]
        sizes = [80 + 1000 * min(0.5, max(0.0, as_float(r["predicted_duration_ratio"], 0))) for r in rows if r["dataset"] == dataset]
        ax.scatter(xs, ys, s=sizes, alpha=0.68, marker=marker, label=dataset)
        for row in rows:
            if row["dataset"] == dataset:
                ax.annotate(row["method"], (as_float(row["predicted_GT_fraction"]), as_float(row["GT_coverage"])), fontsize=8)
    ax.set_xlabel("predicted_GT_fraction (purity)")
    ax.set_ylabel("GT_coverage (recall)")
    ax.set_title("Method Coverage-Purity Trade-off")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_supportable(overall: list[dict], output: Path) -> None:
    rows = top_methods(overall, 18)
    fig, ax = plt.subplots(figsize=(10, 7))
    xs = [as_float(r["supportable_gt_coverage"], math.nan) for r in rows]
    ys = [as_float(r["unsupportable_gt_coverage"], math.nan) for r in rows]
    ax.scatter(xs, ys, alpha=0.72, s=100)
    for row in rows:
        ax.annotate(row["method"], (as_float(row["supportable_gt_coverage"]), as_float(row["unsupportable_gt_coverage"])), fontsize=8)
    ax.set_xlabel("supportable_gt_coverage")
    ax.set_ylabel("unsupportable_gt_coverage")
    ax.set_title("Supportable vs Unsupportable GT Coverage")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_bar(rows: list[dict], field: str, title: str, ylabel: str, output: Path) -> None:
    rows = top_methods(rows, 16)
    labels = [r["method"] for r in rows]
    vals = [as_float(r.get(field), 0.0) for r in rows]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(labels)), vals, color="#4C78A8")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_event_hit(overall: list[dict], output: Path) -> None:
    rows = top_methods(overall, 16)
    labels = [r["method"] for r in rows]
    hit = [as_float(r.get("gt_event_hit_ratio"), 0.0) for r in rows]
    miss = [as_float(r.get("gt_event_missed_ratio"), 0.0) for r in rows]
    fig, ax = plt.subplots(figsize=(12, 6))
    x = list(range(len(labels)))
    ax.bar(x, hit, label="hit", color="#59A14F")
    ax.bar(x, miss, bottom=hit, label="missed", color="#E15759")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("event ratio")
    ax.set_title("GT Event Hit and Missed Ratio")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_plots(overall: list[dict], output_dir: Path) -> None:
    plot_coverage_purity(overall, output_dir / "fig_method_gt_coverage_vs_purity.png")
    plot_supportable(overall, output_dir / "fig_method_supportable_vs_unsupportable_coverage.png")
    plot_bar(overall, "predicted_duration_ratio", "Predicted Duration Ratio", "predicted_duration_ratio", output_dir / "fig_method_predicted_duration_ratio.png")
    plot_event_hit(overall, output_dir / "fig_method_event_hit_ratio.png")
    plot_bar(overall, "predicted_GT_fraction", "Prediction Purity", "predicted_GT_fraction", output_dir / "fig_method_purity_bar.png")


def describe_method(method: str) -> str:
    if method.startswith("Hierarchical-Micro"):
        return "Micro intervals: fine-grained abnormal fragments selected from score-thresholded windows at a specific scale."
    if method == "Hierarchical-Merged":
        return "Merged intervals: adjacent or overlapping micro fragments are merged into coarser event-level intervals."
    if method == "Peak-Aware-Refined":
        return "Peak-aware refined: uses local peak evidence to rescue, split, and refine hierarchical intervals."
    if method.startswith("Window-"):
        return "Window method: fixed-size windows are marked abnormal when their mean anomaly score passes the score threshold."
    if method.startswith("TopK-"):
        return "Top-K: selects the highest-scoring candidate windows per video."
    if method == "Multiscale":
        return "Multiscale: collects top candidate windows across several temporal scales."
    if method == "Adaptive":
        return "Adaptive: selects intervals using adaptive score/spacing rules tuned for long multi-anomaly videos."
    if method == "Wmax-Baseline":
        return "Wmax baseline: the original best-scoring window-like interval per video."
    if method == "Random-Same-Length":
        return "Random same-length baseline: random intervals with matched length, useful as a sanity baseline."
    if method == "Peak-Expanded-Baseline":
        return "Peak-expanded baseline: baseline interval expanded around score peaks."
    return "Other detected method: parsed from existing interval files."


def write_report(output_dir: Path, overall: list[dict], ranking: list[dict], methods: list[str], skipped: list[dict]) -> Path:
    all_rows = [r for r in overall if r["dataset"] == "ALL"]
    best_balanced = next((r for r in ranking if r["ranking_dimension"] == "best_balanced_score" and r["rank"] == 1), None)
    best_recall = next((r for r in ranking if r["ranking_dimension"] == "best_gt_coverage" and r["rank"] == 1), None)
    best_purity = next((r for r in ranking if r["ranking_dimension"] == "best_predicted_GT_fraction" and r["rank"] == 1), None)
    best_support = next((r for r in ranking if r["ranking_dimension"] == "best_supportable_gt_coverage" and r["rank"] == 1), None)
    best_row = next((r for r in all_rows if best_balanced and r["method"] == best_balanced["method"]), None)
    rejected_methods = {"Random-Same-Length"}
    operational_rows = [
        r
        for r in all_rows
        if r["method"] not in rejected_methods and as_float(r.get("predicted_duration_ratio"), 1.0) <= 0.65
    ]
    operational_best = max(operational_rows, key=lambda r: as_float(r.get("balanced_score"), -999), default=best_row)
    recall_rows = [r for r in all_rows if r["method"] not in rejected_methods]
    recall_best = max(recall_rows, key=lambda r: as_float(r.get("GT_coverage"), -999), default=best_row)

    lines = [
        "# Interval Method Evaluation Report",
        "",
        "## Executive summary",
        "",
    ]
    if best_row:
        duration_warning = "yes" if operational_best and as_float(operational_best["predicted_duration_ratio"], 0) > 0.35 and as_float(operational_best["predicted_GT_fraction"], 0) < 0.5 else "limited/not primary"
        lines.extend(
            [
                f"- Raw balanced_score leader: `{best_row['method']}`; it is treated as a sanity/broad-coverage baseline, not as the recommended detector.",
                f"- Recommended balanced current method: `{operational_best['method'] if operational_best else 'NA'}`.",
                f"- GT coverage: {fmt(operational_best['GT_coverage'] if operational_best else math.nan)}.",
                f"- predicted_GT_fraction: {fmt(operational_best['predicted_GT_fraction'] if operational_best else math.nan)}.",
                f"- supportable_gt_coverage: {fmt(operational_best['supportable_gt_coverage'] if operational_best else math.nan)}.",
                f"- unsupportable_gt_coverage: {fmt(operational_best['unsupportable_gt_coverage'] if operational_best else math.nan)}.",
                f"- Evidence of gaining coverage by broadening intervals: {duration_warning}; predicted_duration_ratio={fmt(operational_best['predicted_duration_ratio'] if operational_best else math.nan)}.",
                "",
                "This report does not use ordinary frame-level accuracy as the main metric because normal frames dominate anomaly videos and can make accuracy look artificially high.",
                "",
            ]
        )
    lines.extend(
        [
            "## Method descriptions",
            "",
        ]
    )
    for method in methods:
        lines.append(f"- `{method}`: {describe_method(method)}")
    lines.extend(
        [
            "",
            "## Metrics definition",
            "",
            "- `predicted_GT_fraction`: fraction of predicted abnormal duration that overlaps human GT; this is interval purity.",
            "- `GT_coverage`: fraction of GT abnormal duration covered by predictions; this is duration-level recall.",
            "- `GT_uncovered_ratio`: one minus GT coverage.",
            "- `supportable_gt_coverage`: coverage on GT intervals whose scores are strongly or weakly supported.",
            "- `unsupportable_gt_coverage`: coverage on GT intervals without score evidence; high values can come from broad intervals and are not automatically good.",
            "- `predicted_duration_ratio`: predicted abnormal duration divided by total video duration.",
            "- `event hit ratio`: fraction of GT events with any prediction overlap.",
            "- `IoU-threshold hit ratio`: stricter event hit ratio requiring interval IoU above the threshold.",
            "",
            "## Overall comparison",
            "",
            "- CSV: `method_overall_metrics.csv`",
            "- Figure: `fig_method_gt_coverage_vs_purity.png`",
            "- Figure: `fig_method_predicted_duration_ratio.png`",
            "",
            "| method | GT_coverage | predicted_GT_fraction | supportable_gt_coverage | unsupportable_gt_coverage | predicted_duration_ratio | balanced_score |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(all_rows, key=lambda r: as_float(r.get("balanced_score"), -999), reverse=True)[:12]:
        lines.append(
            f"| `{row['method']}` | {fmt(row['GT_coverage'])} | {fmt(row['predicted_GT_fraction'])} | "
            f"{fmt(row['supportable_gt_coverage'])} | {fmt(row['unsupportable_gt_coverage'])} | "
            f"{fmt(row['predicted_duration_ratio'])} | {fmt(row['balanced_score'])} |"
        )
    lines.extend(
        [
            "",
            "## Supportability-aware comparison",
            "",
            f"- Best supportable GT coverage: `{best_support['method'] if best_support else 'NA'}`.",
            "- Unsupportable coverage is reported separately because it may reflect over-wide predictions rather than true score-supported detection.",
            "",
            "## Coverage-purity trade-off",
            "",
            "- High GT coverage with low predicted_GT_fraction indicates broad intervals that cover GT plus much normal/non-GT time.",
            "- High predicted_GT_fraction with high GT_uncovered_ratio indicates a conservative method with good purity but many missed GT events.",
            "- The current operating point should be selected from the coverage-purity frontier rather than by maximizing a single score.",
            "",
            "## Best method recommendation",
            "",
            f"- Best recall-oriented method excluding random sanity baseline: `{recall_best['method'] if recall_best else (best_recall['method'] if best_recall else 'NA')}`.",
            f"- Best purity-oriented method: `{best_purity['method'] if best_purity else 'NA'}`.",
            f"- Best raw balanced_score method: `{best_balanced['method'] if best_balanced else 'NA'}`; this is diagnostic and may select over-broad baselines.",
            f"- Recommended current method: `{operational_best['method'] if operational_best else 'NA'}`.",
            "",
            "## Limitations",
            "",
            "- This is an offline evaluation and does not modify VDA, VLM, LLM scoring, interval extraction, or peak refinement.",
            "- Supportable and unsupportable groups depend on the existing score-support classification.",
            "- Human GT and VDA score may differ in definition and temporal granularity.",
            "- Window methods can increase coverage by widening intervals, often reducing purity.",
            "- IoU is sensitive to interval length and can penalize short/long interval mismatch.",
            "",
            "## Next steps",
            "",
            "- Do not directly choose 300F/1000F windows as the final method only because they improve coverage.",
            "- Choose an operating point on the coverage-purity frontier.",
            "- To exceed the post-processing upper bound, revisit VDA: VLM descriptions, LLM scoring, and annotation consistency.",
            "",
            "## Parsed and skipped inputs",
            "",
            f"- Parsed methods: {len(methods)}.",
            f"- Candidate/score files with warnings: {sum(1 for row in skipped if not row.get('parsed'))}.",
        ]
    )
    report = output_dir / "interval_method_evaluation_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


def copy_archive_files(root: Path, output_dir: Path, report: Path, script_path: Path) -> None:
    root = root.resolve()
    archive_root = output_dir.parent.resolve()
    archive_rel = archive_root.relative_to(root)
    shutil.copy2(report, archive_root / "interval-method-evaluation_report.md")
    program_dst = archive_root / "programs" / "scripts" / script_path.name
    program_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(script_path, program_dst)
    manifest = [
        "# interval-method-evaluation",
        "",
        f"- archive_folder: `{archive_rel.as_posix()}`",
        "- primary_report: `interval-method-evaluation_report.md`",
        "",
        "## Contents",
        "",
        "- `programs/`: copied script used for the evaluation.",
        "- `outputs/`: generated CSV metrics, normalized intervals, figures, JSON summary, and detailed report.",
    ]
    (archive_root / "MANIFEST.md").write_text("\n".join(manifest), encoding="utf-8")

    layout = root / "reports" / "ARTIFACT_LAYOUT.md"
    line = f"- `{archive_rel.as_posix()}/`"
    if layout.exists():
        text = layout.read_text(encoding="utf-8")
        if line not in text:
            layout.write_text(text.rstrip() + "\n" + line + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_output_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--gt_stats_csv", type=Path, default=Path("outputs/26-07-07-14-43-gt-score-window-curves/outputs/gt_interval_score_stats.csv"))
    parser.add_argument("--gt_support_csv", type=Path, default=Path("outputs/26-07-07-15-14-gt-score-alignment-analysis/outputs/gt_support_classification.csv"))
    parser.add_argument("--video_inventory_csv", type=Path, default=Path("outputs/26-07-07-14-43-gt-score-window-curves/outputs/video_score_curve_inventory.csv"))
    parser.add_argument("--methods_config", type=Path)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/26-07-07-15-59-interval-method-evaluation/outputs"))
    parser.add_argument("--iou_thresholds", default="0.1,0.3,0.5")
    parser.add_argument("--window_sizes", default="30,100,300,1000")
    parser.add_argument("--score_positive_threshold", type=float, default=0.6)
    parser.add_argument("--make_plots", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    root = Path.cwd()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_rows = load_gt_rows(args.gt_stats_csv, args.gt_support_csv)
    inventory = load_inventory(args.video_inventory_csv, gt_rows)

    interval_rows = []
    skipped = []
    skipped.extend(load_methods_config(args.methods_config, interval_rows))
    if not args.methods_config:
        skipped.extend(auto_scan_methods(args.root_output_dir, interval_rows))
    window_sizes = [as_int(x) for x in args.window_sizes.split(",") if x.strip()]
    skipped.extend(add_window_methods(interval_rows, inventory, window_sizes, args.score_positive_threshold, root))

    iou_thresholds = [float(x) for x in args.iou_thresholds.split(",") if x.strip()]
    per_video, overall, support_rows, iou_rows, ranking = evaluate_methods(interval_rows, gt_rows, inventory, iou_thresholds)

    interval_fields = ["method", "dataset", "video_id", "start", "end", "source_path"]
    write_csv(output_dir / "all_method_intervals_normalized.csv", interval_rows, interval_fields)
    write_csv(output_dir / "skipped_interval_files.csv", skipped)
    write_csv(output_dir / "method_per_video_metrics.csv", per_video)
    write_csv(output_dir / "method_overall_metrics.csv", overall)
    write_csv(output_dir / "method_supportability_metrics.csv", support_rows)
    write_csv(output_dir / "method_iou_event_metrics.csv", iou_rows)
    write_csv(output_dir / "method_ranking.csv", ranking)

    if args.make_plots:
        write_plots(overall, output_dir)

    methods = sorted({row["method"] for row in interval_rows})
    report = write_report(output_dir, overall, ranking, methods, skipped)
    summary = {
        "gt_interval_count": len(gt_rows),
        "video_count": len(inventory),
        "method_count": len(methods),
        "methods": methods,
        "normalized_interval_count": len(interval_rows),
        "warning_count": sum(1 for row in skipped if not row.get("parsed")),
        "report": str(report),
    }
    write_json(output_dir / "interval_method_evaluation_summary.json", summary)
    copy_archive_files(root, output_dir, report, Path(__file__).resolve())
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
