import argparse
import csv
import json
import math
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.anomaly_utils import merge_intervals, repo_root, write_csv, write_json  # noqa: E402


DEFAULT_GT_STATS = Path("outputs/26-07-07-14-43-gt-score-window-curves/outputs/gt_interval_score_stats.csv")
DEFAULT_VIDEO_INVENTORY = Path("outputs/26-07-07-14-43-gt-score-window-curves/outputs/video_score_curve_inventory.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/26-07-07-15-14-gt-score-alignment-analysis/outputs")
SUPPORT_ORDER = [
    "strongly_score_supported",
    "weakly_score_supported",
    "ambiguous_mid_score",
    "score_unsupported",
    "sparsely_sampled",
    "barely_sampled",
    "unobserved_or_missing_score",
]
DURATION_BINS = [
    (0, 30, "0-30"),
    (30, 100, "30-100"),
    (100, 300, "100-300"),
    (300, 1000, "300-1000"),
    (1000, float("inf"), "1000+"),
]


def parse_float(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) or math.isinf(out) else out


def parse_int(value, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def ratio(num: float, den: float) -> float:
    return round(num / den, 6) if den else 0.0


def mean(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(float(np.mean(vals)), 6) if vals else None


def median(values: list[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return round(float(np.median(vals)), 6) if vals else None


def split_labels(label: str) -> list[str]:
    labels = [item.strip() for item in re.split(r"[,;/|]+", label or "") if item.strip()]
    return labels or ["UNKNOWN"]


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def classify_support(row: dict, inventory: dict, args: argparse.Namespace) -> dict:
    key = (row["dataset"], row["video_id"])
    inv = inventory.get(key, {})
    has_scores = str(inv.get("has_scores", "True")).lower() == "true"
    count = parse_int(row.get("score_point_count"))
    mean_score = parse_float(row.get("mean_score"))
    max_score = parse_float(row.get("max_score"))
    duration = max(1, parse_int(row.get("gt_duration"), 1))

    if count == 0 or mean_score is None or max_score is None or not has_scores:
        support = "unobserved_or_missing_score"
    elif count < args.min_sparse_points:
        support = "barely_sampled"
    elif count < args.min_well_sampled_points:
        support = "sparsely_sampled"
    elif max_score >= args.strong_max_threshold or mean_score >= args.strong_mean_threshold:
        support = "strongly_score_supported"
    elif max_score >= args.weak_max_threshold:
        support = "weakly_score_supported"
    elif max_score < args.unsupported_max_threshold:
        support = "score_unsupported"
    else:
        support = "ambiguous_mid_score"

    if count < args.min_sparse_points or mean_score is None or max_score is None:
        shape = "sparse_or_unknown"
    elif mean_score >= args.strong_mean_threshold and max_score >= args.strong_max_threshold:
        shape = "sustained_response"
    elif max_score >= args.strong_max_threshold and mean_score < args.strong_mean_threshold:
        shape = "localized_response"
    elif max_score < args.weak_max_threshold:
        shape = "weak_or_no_response"
    else:
        shape = "ambiguous_response"

    if support in {"strongly_score_supported", "weakly_score_supported"}:
        recoverable = "True"
    elif support in {"score_unsupported", "unobserved_or_missing_score", "barely_sampled"}:
        recoverable = "False"
    else:
        recoverable = "uncertain"

    if count == 0:
        sampling = "unobserved"
    elif count < args.min_sparse_points:
        sampling = "barely_sampled"
    elif count < args.min_well_sampled_points:
        sampling = "sparse"
    else:
        sampling = "well_sampled"

    out = dict(row)
    out.update(
        {
            "has_scores": str(has_scores),
            "support_type": support,
            "response_shape": shape,
            "score_density": round(count / duration, 6),
            "score_sampling_level": sampling,
            "recoverable_by_postprocessing": recoverable,
        }
    )
    return out


def load_score_curve(path_text: str, stride_est: int | None, warnings: list[dict], context: dict) -> list[tuple[int, float]]:
    if not path_text:
        warnings.append({**context, "reason": "missing_score_json_path"})
        return []
    path = repo_root() / path_text
    if not path.exists():
        warnings.append({**context, "score_json_path": path_text, "reason": "score_json_path_not_found"})
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        warnings.append({**context, "score_json_path": path_text, "reason": f"json_parse_error:{exc}"})
        return []

    curve = []
    if isinstance(raw, dict):
        for key, value in raw.items():
            try:
                curve.append((int(float(key)), float(value)))
            except (TypeError, ValueError):
                continue
    elif isinstance(raw, list):
        stride = stride_est or 1
        for idx, item in enumerate(raw):
            try:
                if isinstance(item, dict):
                    frame = parse_int(item.get("frame", item.get("frame_index", idx * stride)), idx * stride)
                    score = float(item.get("score", item.get("value")))
                else:
                    frame = idx * stride
                    score = float(item)
                curve.append((frame, score))
            except (TypeError, ValueError):
                continue
    else:
        warnings.append({**context, "score_json_path": path_text, "reason": "unsupported_score_json_format"})
        return []
    curve = sorted(curve)
    if not curve:
        warnings.append({**context, "score_json_path": path_text, "reason": "no_numeric_scores"})
    return curve


def in_any_interval(frame: int, intervals: list[dict]) -> bool:
    return any(int(item["gt_start"]) <= frame < int(item["gt_end"]) for item in intervals)


def overlap_len(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start))


def nearest_gt_distance(start: int, end: int, intervals: list[dict]) -> int:
    if not intervals:
        return -1
    distances = []
    for item in intervals:
        gs, ge = int(item["gt_start"]), int(item["gt_end"])
        if overlap_len(start, end, gs, ge) > 0:
            distances.append(0)
        elif end <= gs:
            distances.append(gs - end)
        else:
            distances.append(start - ge)
    return min(distances)


def group_high_points(points: list[tuple[int, float]], stride: int, intervals: list[dict], dataset: str, video_id: str, label: str) -> list[dict]:
    if not points:
        return []
    groups = []
    current = [points[0]]
    max_gap = max(1, int(stride * 1.5)) if stride else 1
    for point in points[1:]:
        if point[0] - current[-1][0] <= max_gap:
            current.append(point)
        else:
            groups.append(current)
            current = [point]
    groups.append(current)
    rows = []
    for group in groups:
        frames = [p[0] for p in group]
        scores = [p[1] for p in group]
        start = min(frames)
        end = max(frames) + (stride or 1)
        rows.append(
            {
                "dataset": dataset,
                "video_id": video_id,
                "label": label,
                "start_frame": start,
                "end_frame": end,
                "duration": end - start,
                "mean_score": round(float(np.mean(scores)), 6),
                "max_score": round(float(np.max(scores)), 6),
                "score_point_count": len(scores),
                "nearest_gt_distance": nearest_gt_distance(start, end, intervals),
                "overlaps_gt": any(overlap_len(start, end, int(gt["gt_start"]), int(gt["gt_end"])) > 0 for gt in intervals),
                "reason": "score_positive_gt_negative",
            }
        )
    return rows


def build_video_alignment(classified: list[dict], inventory_rows: list[dict], args: argparse.Namespace) -> list[dict]:
    by_video = defaultdict(list)
    for row in classified:
        by_video[(row["dataset"], row["video_id"])].append(row)
    inventory = {(r["dataset"], r["video_id"]): r for r in inventory_rows}
    rows = []
    for key, items in sorted(by_video.items()):
        inv = inventory.get(key, {})
        n = len(items)
        counts = Counter(row["support_type"] for row in items)
        missing = counts["unobserved_or_missing_score"]
        sparse = counts["sparsely_sampled"] + counts["barely_sampled"]
        recoverable_count = sum(1 for row in items if row["recoverable_by_postprocessing"] == "True")
        unsupported = counts["score_unsupported"]
        unsupported_ratio = ratio(unsupported, n)
        recoverable_ratio = ratio(recoverable_count, n)
        sparse_missing_ratio = ratio(sparse + missing, n)
        if sparse_missing_ratio >= 0.5 or str(inv.get("has_scores", "True")).lower() != "true":
            align_type = "sparse_or_unreliable"
        elif recoverable_ratio >= 0.7 and unsupported_ratio <= 0.2:
            align_type = "good_alignment"
        elif n >= 2 and unsupported_ratio >= 0.4:
            align_type = "under_sensitive"
        else:
            align_type = "mixed_alignment"
        rows.append(
            {
                "dataset": key[0],
                "video_id": key[1],
                "label": inv.get("label", items[0].get("label", "")),
                "gt_interval_count": n,
                "score_point_count": inv.get("score_point_count", ""),
                "score_stride_est": inv.get("score_stride_est", ""),
                "video_length_frame_est": inv.get("video_length_frame_est", ""),
                "num_gt_strong": counts["strongly_score_supported"],
                "num_gt_weak": counts["weakly_score_supported"],
                "num_gt_unsupported": unsupported,
                "num_gt_sparse": sparse,
                "num_gt_missing": missing,
                "num_gt_ambiguous": counts["ambiguous_mid_score"],
                "ratio_gt_strong": ratio(counts["strongly_score_supported"], n),
                "ratio_gt_weak": ratio(counts["weakly_score_supported"], n),
                "ratio_gt_unsupported": unsupported_ratio,
                "ratio_gt_sparse_or_missing": sparse_missing_ratio,
                "mean_gt_mean_score": mean([parse_float(row.get("mean_score")) for row in items]),
                "mean_gt_max_score": mean([parse_float(row.get("max_score")) for row in items]),
                "max_gt_max_score": mean([max([parse_float(row.get("max_score")) for row in items if parse_float(row.get("max_score")) is not None], default=None)]),
                "postprocessing_recoverable_gt_count": recoverable_count,
                "postprocessing_recoverable_gt_ratio": recoverable_ratio,
                "video_alignment_type": align_type,
            }
        )
    return rows


def label_summary(classified: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in classified:
        for label in split_labels(row.get("label", "")):
            groups[(row["dataset"], label)].append(row)
    out = []
    for (dataset, label), rows in sorted(groups.items()):
        n = len(rows)
        counts = Counter(r["support_type"] for r in rows)
        shapes = Counter(r["response_shape"] for r in rows)
        recoverable = sum(1 for r in rows if r["recoverable_by_postprocessing"] == "True")
        sparse_missing = counts["sparsely_sampled"] + counts["barely_sampled"] + counts["unobserved_or_missing_score"]
        out.append(
            {
                "dataset": dataset,
                "label": label,
                "gt_interval_count": n,
                "mean_of_mean_score": mean([parse_float(r.get("mean_score")) for r in rows]),
                "mean_of_max_score": mean([parse_float(r.get("max_score")) for r in rows]),
                "median_max_score": median([parse_float(r.get("max_score")) for r in rows]),
                "unsupported_ratio": ratio(counts["score_unsupported"], n),
                "strong_supported_ratio": ratio(counts["strongly_score_supported"], n),
                "weak_supported_ratio": ratio(counts["weakly_score_supported"], n),
                "sparse_or_missing_ratio": ratio(sparse_missing, n),
                "localized_response_ratio": ratio(shapes["localized_response"], n),
                "sustained_response_ratio": ratio(shapes["sustained_response"], n),
                "recoverable_ratio": ratio(recoverable, n),
            }
        )
    return out


def duration_bin(duration: int) -> str:
    for lo, hi, label in DURATION_BINS:
        if lo <= duration < hi:
            return label
    return "unknown"


def duration_summary(classified: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in classified:
        groups[duration_bin(parse_int(row.get("gt_duration")))].append(row)
    out = []
    for _, _, label in DURATION_BINS:
        rows = groups.get(label, [])
        if not rows:
            continue
        n = len(rows)
        counts = Counter(r["support_type"] for r in rows)
        shapes = Counter(r["response_shape"] for r in rows)
        out.append(
            {
                "duration_bin": label,
                "gt_count": n,
                "mean_score_mean": mean([parse_float(r.get("mean_score")) for r in rows]),
                "mean_score_max": mean([parse_float(r.get("max_score")) for r in rows]),
                "median_max_score": median([parse_float(r.get("max_score")) for r in rows]),
                "unsupported_ratio": ratio(counts["score_unsupported"], n),
                "strong_supported_ratio": ratio(counts["strongly_score_supported"], n),
                "localized_response_ratio": ratio(shapes["localized_response"], n),
                "sustained_response_ratio": ratio(shapes["sustained_response"], n),
                "mean_score_point_count": mean([parse_float(r.get("score_point_count")) for r in rows]),
            }
        )
    return out


def outside_gt_analysis(classified: list[dict], inventory_rows: list[dict], args: argparse.Namespace, warnings: list[dict]) -> tuple[list[dict], list[dict]]:
    gt_by_video = defaultdict(list)
    for row in classified:
        gt_by_video[(row["dataset"], row["video_id"])].append(row)
    out_rows = []
    intervals = []
    for inv in inventory_rows:
        key = (inv["dataset"], inv["video_id"])
        gts = gt_by_video.get(key, [])
        if not gts:
            continue
        stride = parse_int(inv.get("score_stride_est"), 1) or 1
        curve = load_score_curve(inv.get("score_json_path", ""), stride, warnings, {"dataset": inv["dataset"], "video_id": inv["video_id"]})
        if not curve:
            out_rows.append(
                {
                    "dataset": inv["dataset"],
                    "video_id": inv["video_id"],
                    "label": inv.get("label", ""),
                    "inside_gt_score_points": 0,
                    "outside_gt_score_points": 0,
                    "inside_gt_high_score_points": 0,
                    "outside_gt_high_score_points": 0,
                    "inside_gt_high_score_ratio": 0,
                    "outside_gt_high_score_ratio": 0,
                    "outside_gt_max_score": None,
                    "outside_gt_mean_score": None,
                    "outside_gt_high_score_interval_count": 0,
                }
            )
            continue
        inside = [(f, s) for f, s in curve if in_any_interval(f, gts)]
        outside = [(f, s) for f, s in curve if not in_any_interval(f, gts)]
        inside_high = [(f, s) for f, s in inside if s >= args.score_positive_threshold]
        outside_high = [(f, s) for f, s in outside if s >= args.score_positive_threshold]
        outside_intervals = group_high_points(outside_high, stride, gts, inv["dataset"], inv["video_id"], inv.get("label", ""))
        intervals.extend(outside_intervals)
        out_rows.append(
            {
                "dataset": inv["dataset"],
                "video_id": inv["video_id"],
                "label": inv.get("label", ""),
                "inside_gt_score_points": len(inside),
                "outside_gt_score_points": len(outside),
                "inside_gt_high_score_points": len(inside_high),
                "outside_gt_high_score_points": len(outside_high),
                "inside_gt_high_score_ratio": ratio(len(inside_high), len(inside)),
                "outside_gt_high_score_ratio": ratio(len(outside_high), len(outside)),
                "outside_gt_max_score": round(max([s for _, s in outside], default=0), 6) if outside else None,
                "outside_gt_mean_score": round(float(np.mean([s for _, s in outside])), 6) if outside else None,
                "outside_gt_high_score_interval_count": len(outside_intervals),
            }
        )
    return out_rows, intervals


def window_confusion_for_threshold(inventory_rows: list[dict], classified: list[dict], args: argparse.Namespace, threshold: float, warnings: list[dict]) -> tuple[list[dict], list[dict]]:
    gt_by_video = defaultdict(list)
    for row in classified:
        gt_by_video[(row["dataset"], row["video_id"])].append(row)
    summary = defaultdict(lambda: Counter())
    label_summary_counts = defaultdict(lambda: Counter())
    for inv in inventory_rows:
        key = (inv["dataset"], inv["video_id"])
        gts = gt_by_video.get(key, [])
        if not gts:
            continue
        stride = parse_int(inv.get("score_stride_est"), 1) or 1
        curve = load_score_curve(inv.get("score_json_path", ""), stride, warnings, {"dataset": inv["dataset"], "video_id": inv["video_id"]})
        if not curve:
            continue
        frame_scores = dict(curve)
        max_frame = max(max(frame_scores), parse_int(inv.get("video_length_frame_est"), max(frame_scores) + stride))
        labels = split_labels(inv.get("label", ""))
        for window_size in args.window_sizes:
            for start in range(0, max_frame + 1, window_size):
                end = start + window_size
                vals = [s for f, s in curve if start <= f < end]
                if not vals:
                    continue
                gt_pos = any(overlap_len(start, end, int(gt["gt_start"]), int(gt["gt_end"])) > 0 for gt in gts)
                score_pos = max(vals) >= threshold or float(np.mean(vals)) >= args.strong_mean_threshold
                bucket = ("gt_pos_" if gt_pos else "gt_neg_") + ("score_pos" if score_pos else "score_neg")
                summary[(inv["dataset"], window_size)][bucket] += 1
                for label in labels:
                    label_summary_counts[(inv["dataset"], label, window_size)][bucket] += 1
    rows = []
    for (dataset, window_size), c in sorted(summary.items()):
        gp = c["gt_pos_score_pos"] + c["gt_pos_score_neg"]
        gn = c["gt_neg_score_pos"] + c["gt_neg_score_neg"]
        sp = c["gt_pos_score_pos"] + c["gt_neg_score_pos"]
        sn = c["gt_pos_score_neg"] + c["gt_neg_score_neg"]
        rows.append(
            {
                "dataset": dataset,
                "window_size": window_size,
                "gt_pos_score_pos": c["gt_pos_score_pos"],
                "gt_pos_score_neg": c["gt_pos_score_neg"],
                "gt_neg_score_pos": c["gt_neg_score_pos"],
                "gt_neg_score_neg": c["gt_neg_score_neg"],
                "gt_pos_count": gp,
                "gt_neg_count": gn,
                "score_pos_count": sp,
                "score_neg_count": sn,
                "gt_pos_score_pos_ratio": ratio(c["gt_pos_score_pos"], gp),
                "gt_pos_score_neg_ratio": ratio(c["gt_pos_score_neg"], gp),
                "gt_neg_score_pos_ratio": ratio(c["gt_neg_score_pos"], gn),
                "gt_neg_score_neg_ratio": ratio(c["gt_neg_score_neg"], gn),
            }
        )
    label_rows = []
    for (dataset, label, window_size), c in sorted(label_summary_counts.items()):
        gp = c["gt_pos_score_pos"] + c["gt_pos_score_neg"]
        gn = c["gt_neg_score_pos"] + c["gt_neg_score_neg"]
        label_rows.append(
            {
                "dataset": dataset,
                "label": label,
                "window_size": window_size,
                "gt_pos_score_pos": c["gt_pos_score_pos"],
                "gt_pos_score_neg": c["gt_pos_score_neg"],
                "gt_neg_score_pos": c["gt_neg_score_pos"],
                "gt_neg_score_neg": c["gt_neg_score_neg"],
                "gt_pos_count": gp,
                "gt_neg_count": gn,
                "gt_pos_score_pos_ratio": ratio(c["gt_pos_score_pos"], gp),
                "gt_pos_score_neg_ratio": ratio(c["gt_pos_score_neg"], gp),
                "gt_neg_score_pos_ratio": ratio(c["gt_neg_score_pos"], gn),
                "gt_neg_score_neg_ratio": ratio(c["gt_neg_score_neg"], gn),
            }
        )
    return rows, label_rows


def upper_bound_summary(classified: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in classified:
        groups[(row["dataset"], "ALL")].append(row)
        for label in split_labels(row.get("label", "")):
            groups[(row["dataset"], label)].append(row)
    rows = []
    for (dataset, label), items in sorted(groups.items()):
        recoverable = 0
        unrecoverable = 0
        uncertain = 0
        for row in items:
            support = row["support_type"]
            shape = row["response_shape"]
            if support in {"strongly_score_supported", "weakly_score_supported"} or shape in {"localized_response", "sustained_response"}:
                recoverable += 1
            elif support in {"score_unsupported", "unobserved_or_missing_score", "barely_sampled"}:
                unrecoverable += 1
            else:
                uncertain += 1
        n = len(items)
        rows.append(
            {
                "dataset": dataset,
                "label": label,
                "gt_count": n,
                "recoverable_count": recoverable,
                "unrecoverable_count": unrecoverable,
                "uncertain_count": uncertain,
                "recoverable_ratio": ratio(recoverable, n),
                "unrecoverable_ratio": ratio(unrecoverable, n),
                "uncertain_ratio": ratio(uncertain, n),
            }
        )
    return rows


def threshold_sensitivity(thresholds: list[float], classified: list[dict], inventory_rows: list[dict], args: argparse.Namespace, warnings: list[dict]) -> list[dict]:
    rows = []
    ub = upper_bound_summary(classified)
    overall_rec = mean([parse_float(r["recoverable_ratio"]) for r in ub if r["label"] == "ALL"])
    for threshold in thresholds:
        local_args = argparse.Namespace(**vars(args))
        local_args.score_positive_threshold = threshold
        outside_rows, _ = outside_gt_analysis(classified, inventory_rows, local_args, warnings)
        window_rows, _ = window_confusion_for_threshold(inventory_rows, classified, local_args, threshold, warnings)
        outside_ratio = mean([parse_float(r["outside_gt_high_score_ratio"]) for r in outside_rows])
        gt_pos_score_pos = mean([parse_float(r["gt_pos_score_pos_ratio"]) for r in window_rows])
        gt_neg_score_pos = mean([parse_float(r["gt_neg_score_pos_ratio"]) for r in window_rows])
        rows.append(
            {
                "score_positive_threshold": threshold,
                "recoverable_ratio": overall_rec,
                "outside_gt_high_score_ratio": outside_ratio,
                "gt_pos_score_pos_ratio": gt_pos_score_pos,
                "gt_neg_score_pos_ratio": gt_neg_score_pos,
            }
        )
    return rows


def plot_support_by_dataset(classified: list[dict], out: Path) -> None:
    datasets = sorted({r["dataset"] for r in classified})
    x = np.arange(len(SUPPORT_ORDER))
    width = 0.8 / max(1, len(datasets))
    fig, ax = plt.subplots(figsize=(12, 5))
    for idx, dataset in enumerate(datasets):
        counts = Counter(r["support_type"] for r in classified if r["dataset"] == dataset)
        ax.bar(x + idx * width, [counts[s] for s in SUPPORT_ORDER], width=width, label=dataset)
    ax.set_xticks(x + width * (len(datasets) - 1) / 2)
    ax.set_xticklabels(SUPPORT_ORDER, rotation=35, ha="right")
    ax.set_ylabel("GT interval count")
    ax.set_title("GT support classification by dataset")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_label_topk(label_rows: list[dict], out: Path) -> None:
    totals = defaultdict(int)
    for row in label_rows:
        totals[row["label"]] += parse_int(row["gt_interval_count"])
    top = [label for label, _ in sorted(totals.items(), key=lambda x: x[1], reverse=True)[:10]]
    ratios = defaultdict(lambda: defaultdict(float))
    for row in label_rows:
        if row["label"] in top:
            ratios[row["label"]]["strong"] += parse_float(row["strong_supported_ratio"]) or 0
            ratios[row["label"]]["weak"] += parse_float(row["weak_supported_ratio"]) or 0
            ratios[row["label"]]["unsupported"] += parse_float(row["unsupported_ratio"]) or 0
            ratios[row["label"]]["sparse"] += parse_float(row["sparse_or_missing_ratio"]) or 0
    x = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(12, 5))
    bottom = np.zeros(len(top))
    for key in ["strong", "weak", "unsupported", "sparse"]:
        vals = np.asarray([ratios[label][key] for label in top])
        ax.bar(x, vals, bottom=bottom, label=key)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(top, rotation=35, ha="right")
    ax.set_ylabel("ratio sum across datasets")
    ax.set_title("Top labels: support ratios")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_duration_scatter(classified: list[dict], y_key: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    datasets = sorted({r["dataset"] for r in classified})
    for dataset in datasets:
        rows = [r for r in classified if r["dataset"] == dataset and parse_float(r.get(y_key)) is not None]
        ax.scatter([parse_int(r["gt_duration"]) for r in rows], [parse_float(r[y_key]) for r in rows], alpha=0.35, s=14, label=dataset)
    ax.set_xscale("log")
    ax.set_xlabel("GT duration (frames, log scale)")
    ax.set_ylabel(y_key)
    ax.set_title(f"GT duration vs {y_key}")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_window_confusion(window_rows: list[dict], out: Path) -> None:
    labels = [f"{r['dataset']}-{r['window_size']}" for r in window_rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, 5))
    bottom = np.zeros(len(labels))
    keys = ["gt_pos_score_pos_ratio", "gt_pos_score_neg_ratio", "gt_neg_score_pos_ratio", "gt_neg_score_neg_ratio"]
    for key in keys:
        vals = np.asarray([parse_float(r[key]) or 0 for r in window_rows])
        ax.bar(x, vals, bottom=bottom, label=key)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("ratio")
    ax.set_title("Window-level GT/score consistency")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_recoverable(upper_rows: list[dict], out: Path) -> None:
    rows = [r for r in upper_rows if r["label"] == "ALL"]
    labels = [r["dataset"] for r in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7, 4))
    bottom = np.zeros(len(labels))
    for key in ["recoverable_ratio", "uncertain_ratio", "unrecoverable_ratio"]:
        vals = np.asarray([parse_float(r[key]) or 0 for r in rows])
        ax.bar(x, vals, bottom=bottom, label=key)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title("Post-processing recoverable upper bound")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_outside(outside_rows: list[dict], outside_intervals: list[dict], out: Path) -> None:
    datasets = sorted({r["dataset"] for r in outside_rows})
    x = np.arange(len(datasets))
    ratios = [mean([parse_float(r["outside_gt_high_score_ratio"]) for r in outside_rows if r["dataset"] == d]) or 0 for d in datasets]
    counts = [sum(1 for r in outside_intervals if r["dataset"] == d) for d in datasets]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.bar(x - 0.18, ratios, width=0.36, label="mean outside high-score ratio")
    ax2 = ax1.twinx()
    ax2.bar(x + 0.18, counts, width=0.36, alpha=0.5, label="outside high-score intervals")
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets)
    ax1.set_ylabel("ratio")
    ax2.set_ylabel("interval count")
    ax1.set_title("Outside-GT high-score evidence")
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_threshold_sensitivity(rows: list[dict], out: Path) -> None:
    if not rows:
        return
    x = [parse_float(r["score_positive_threshold"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8, 5))
    for key in ["recoverable_ratio", "outside_gt_high_score_ratio", "gt_pos_score_pos_ratio", "gt_neg_score_pos_ratio"]:
        ax.plot(x, [parse_float(r[key]) for r in rows], marker="o", label=key)
    ax.set_xlabel("score_positive_threshold")
    ax.set_ylabel("ratio")
    ax.set_ylim(0, 1)
    ax.set_title("Threshold sensitivity")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def write_report(path: Path, args: argparse.Namespace, classified: list[dict], video_rows: list[dict], label_rows: list[dict], duration_rows: list[dict], outside_rows: list[dict], outside_intervals: list[dict], window_rows: list[dict], upper_rows: list[dict], warnings: list[dict]) -> None:
    counts = Counter(r["support_type"] for r in classified)
    total = len(classified)
    recover_all = [r for r in upper_rows if r["label"] == "ALL"]
    top_unsupported = sorted(label_rows, key=lambda r: (parse_float(r["unsupported_ratio"]) or 0, parse_int(r["gt_interval_count"])), reverse=True)[:5]
    best_window = max(window_rows, key=lambda r: parse_float(r["gt_pos_score_pos_ratio"]) or 0) if window_rows else None
    lines = [
        "# GT-Score Alignment Analysis Report",
        "",
        "## Method",
        "",
        "This analysis diagnoses consistency between human event-level GT and score-level anomaly evidence. It does not assume either GT or score is absolutely correct. The goal is to separate post-processing error, score-supported misses, score-unsupported GT intervals, and score-positive GT-negative regions.",
        "",
        "## Inputs",
        "",
        f"- gt_stats_csv: `{args.gt_stats_csv}`",
        f"- video_inventory_csv: `{args.video_inventory_csv}`",
        f"- GT intervals: {total}",
        f"- videos: {len(video_rows)}",
        f"- score JSON paths: {len({row.get('score_json_path', '') for row in video_rows if row.get('score_json_path', '')})}",
        f"- warnings/skipped score files: {len(warnings)}",
        "",
        "## Thresholds And Label Definitions",
        "",
        "The following thresholds were used in this run:",
        "",
        f"- `score_positive_threshold`: {args.score_positive_threshold}. A score point or fixed window is treated as score-positive at this threshold.",
        f"- `strong_mean_threshold`: {args.strong_mean_threshold}. A GT interval is strongly supported if its mean score reaches this value, provided it has enough score samples.",
        f"- `strong_max_threshold`: {args.strong_max_threshold}. A GT interval is strongly supported if its max score reaches this value, provided it has enough score samples.",
        f"- `weak_max_threshold`: {args.weak_max_threshold}. A GT interval is weakly supported if its max score reaches this value but it is not strongly supported.",
        f"- `unsupported_max_threshold`: {args.unsupported_max_threshold}. A sufficiently sampled GT interval with max score below this value is score-unsupported.",
        f"- `min_sparse_points`: {args.min_sparse_points}. Intervals with fewer than this many score points are `barely_sampled`.",
        f"- `min_well_sampled_points`: {args.min_well_sampled_points}. Intervals with at least `min_sparse_points` but fewer than this many score points are `sparsely_sampled`.",
        f"- `window_sizes`: {', '.join(str(x) for x in args.window_sizes)} frames.",
        f"- `threshold_sweep`: {args.threshold_sweep}. Used only for sensitivity analysis.",
        "",
        "`support_type` is assigned in this priority order:",
        "",
        "- `unobserved_or_missing_score`: `score_point_count == 0`, missing mean/max score, or missing score file.",
        "- `barely_sampled`: `0 < score_point_count < min_sparse_points`.",
        "- `sparsely_sampled`: `min_sparse_points <= score_point_count < min_well_sampled_points`. This rule is evaluated before strong/weak support, so short intervals with high scores can still be marked sparse.",
        "- `strongly_score_supported`: enough samples and (`max_score >= strong_max_threshold` or `mean_score >= strong_mean_threshold`).",
        "- `weakly_score_supported`: enough samples and `max_score >= weak_max_threshold`, but not strongly supported.",
        "- `score_unsupported`: enough samples and `max_score < unsupported_max_threshold`.",
        "- `ambiguous_mid_score`: all remaining middle-score cases.",
        "",
        "`response_shape` is assigned as:",
        "",
        "- `sustained_response`: `mean_score >= strong_mean_threshold` and `max_score >= strong_max_threshold`.",
        "- `localized_response`: `max_score >= strong_max_threshold` and `mean_score < strong_mean_threshold`.",
        "- `weak_or_no_response`: `max_score < weak_max_threshold`.",
        "- `sparse_or_unknown`: too few score points or missing score values.",
        "- `moderate_response`: all remaining response-shape cases.",
        "",
        "`recoverable_by_postprocessing` is `True` for `strongly_score_supported` and `weakly_score_supported`, `False` for `score_unsupported`, `unobserved_or_missing_score`, and `barely_sampled`, and `uncertain` for `sparsely_sampled` and `ambiguous_mid_score`.",
        "",
        "Window-level score-positive logic uses `max(window_scores) >= score_positive_threshold` or `mean(window_scores) >= strong_mean_threshold`.",
        "",
        "Post-processing upper bound treats an interval as recoverable if it is strongly/weakly score-supported or has a `localized_response`/`sustained_response`; sparse intervals remain uncertain unless this response-shape evidence is present.",
        "",
        "## GT Support Summary",
        "",
        "- `gt_support_classification.csv` contains original GT rows plus support classification fields.",
        "- `fig_gt_support_by_dataset.png` visualizes support counts by dataset.",
    ]
    for key in SUPPORT_ORDER:
        lines.append(f"- {key}: {counts[key]} ({ratio(counts[key], total):.2%})")
    lines.extend(["", "## Label-wise Summary", ""])
    lines.append("Labels with high unsupported ratios among the current top rows:")
    for row in top_unsupported:
        lines.append(f"- {row['dataset']} / {row['label']}: unsupported={float(row['unsupported_ratio']):.2%}, n={row['gt_interval_count']}")
    lines.extend(["", "## Duration-Score Relationship", ""])
    lines.append("Duration bins are summarized in `duration_score_summary.csv`; scatter plots are `fig_duration_vs_max_score.png` and `fig_duration_vs_mean_score.png`.")
    lines.extend(["", "## Window-level Consistency", ""])
    if best_window:
        lines.append(f"Highest GT+Score+ ratio: {best_window['dataset']} window={best_window['window_size']} with {float(best_window['gt_pos_score_pos_ratio']):.2%}.")
    lines.append("Full results are in `window_confusion_summary.csv` and `label_window_confusion_summary.csv`.")
    lines.extend(["", "## Outside-GT High Score Analysis", ""])
    lines.append(f"Detected outside-GT high-score intervals: {len(outside_intervals)}. Per-video ratios are in `video_outside_gt_score_summary.csv`; intervals are in `outside_gt_high_score_intervals.csv`.")
    lines.extend(["", "## Post-processing Upper Bound", ""])
    for row in recover_all:
        lines.append(f"- {row['dataset']}: recoverable={float(row['recoverable_ratio']):.2%}, uncertain={float(row['uncertain_ratio']):.2%}, unrecoverable={float(row['unrecoverable_ratio']):.2%}")
    lines.append("")
    lines.append("Peak-aware refinement can only recover score-supported local anomalies; it cannot recover GT intervals where the upstream scorer provides no abnormal signal.")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Score stride may undersample short GT intervals.",
            "- Multi-label videos do not uniquely attribute each interval to one label.",
            "- Score thresholds are diagnostic and need sensitivity analysis.",
            "- GT and VLM score may differ in definition and temporal granularity.",
            "- This analysis does not prove whether human annotation or VLM score is wrong.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_archive_manifest(output_dir: Path) -> None:
    root = repo_root()
    archive_root = output_dir.parent if output_dir.name == "outputs" else output_dir
    programs_dir = archive_root / "programs/scripts"
    programs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / "scripts/analyze_gt_score_alignment.py", programs_dir / "analyze_gt_score_alignment.py")
    report_src = output_dir / "gt_score_alignment_report.md"
    if report_src.exists():
        shutil.copy2(report_src, archive_root / "gt-score-alignment-analysis_report.md")
    manifest = [
        "# gt-score-alignment-analysis",
        "",
        f"- archive_folder: `{archive_root.relative_to(root)}`",
        "- primary_report: `gt-score-alignment-analysis_report.md`",
        "",
        "## Contents",
        "",
        "- `programs/`: copied script needed to reproduce or inspect this experiment.",
        "- `outputs/`: CSV, JSON, figures, skipped-file log, and detailed Markdown report.",
        "- `gt-score-alignment-analysis_report.md`: copied top-level report for quick access.",
    ]
    (archive_root / "MANIFEST.md").write_text("\n".join(manifest), encoding="utf-8")


def parse_thresholds(text: str | None) -> list[float]:
    if not text:
        return []
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_stats_csv", type=Path, default=DEFAULT_GT_STATS)
    parser.add_argument("--video_inventory_csv", type=Path, default=DEFAULT_VIDEO_INVENTORY)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--score_positive_threshold", type=float, default=0.6)
    parser.add_argument("--strong_mean_threshold", type=float, default=0.6)
    parser.add_argument("--strong_max_threshold", type=float, default=0.8)
    parser.add_argument("--weak_max_threshold", type=float, default=0.5)
    parser.add_argument("--unsupported_max_threshold", type=float, default=0.4)
    parser.add_argument("--min_well_sampled_points", type=int, default=5)
    parser.add_argument("--min_sparse_points", type=int, default=2)
    parser.add_argument("--window_sizes", default="30,100,300")
    parser.add_argument("--make_plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--threshold_sweep", default="0.4,0.5,0.6,0.7,0.8")
    args = parser.parse_args()
    args.window_sizes = [int(item.strip()) for item in str(args.window_sizes).split(",") if item.strip()]

    root = repo_root()
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    gt_rows = read_csv_rows(root / args.gt_stats_csv)
    inventory_rows = read_csv_rows(root / args.video_inventory_csv)
    inventory = {(row["dataset"], row["video_id"]): row for row in inventory_rows}
    warnings = []

    classified = [classify_support(row, inventory, args) for row in gt_rows]
    write_csv(output_dir / "gt_support_classification.csv", classified)
    video_alignment = build_video_alignment(classified, inventory_rows, args)
    write_csv(output_dir / "video_alignment_summary.csv", video_alignment)
    labels = label_summary(classified)
    write_csv(output_dir / "label_alignment_summary.csv", labels)
    durations = duration_summary(classified)
    write_csv(output_dir / "duration_score_summary.csv", durations)
    outside_rows, outside_intervals = outside_gt_analysis(classified, inventory_rows, args, warnings)
    write_csv(output_dir / "video_outside_gt_score_summary.csv", outside_rows)
    write_csv(output_dir / "outside_gt_high_score_intervals.csv", outside_intervals)
    window_rows, label_window_rows = window_confusion_for_threshold(inventory_rows, classified, args, args.score_positive_threshold, warnings)
    write_csv(output_dir / "window_confusion_summary.csv", window_rows)
    write_csv(output_dir / "label_window_confusion_summary.csv", label_window_rows)
    upper = upper_bound_summary(classified)
    write_csv(output_dir / "postprocessing_upper_bound_summary.csv", upper)
    sweep_rows = threshold_sensitivity(parse_thresholds(args.threshold_sweep), classified, inventory_rows, args, warnings)
    write_csv(output_dir / "threshold_sensitivity_summary.csv", sweep_rows)
    write_csv(output_dir / "skipped_score_files.csv", warnings)

    if args.make_plots:
        plot_support_by_dataset(classified, output_dir / "fig_gt_support_by_dataset.png")
        plot_label_topk(labels, output_dir / "fig_gt_support_by_label_topk.png")
        plot_duration_scatter(classified, "max_score", output_dir / "fig_duration_vs_max_score.png")
        plot_duration_scatter(classified, "mean_score", output_dir / "fig_duration_vs_mean_score.png")
        plot_window_confusion(window_rows, output_dir / "fig_window_confusion_30_100_300.png")
        plot_recoverable(upper, output_dir / "fig_recoverable_upper_bound.png")
        plot_outside(outside_rows, outside_intervals, output_dir / "fig_outside_gt_high_score_by_dataset.png")
        plot_threshold_sensitivity(sweep_rows, output_dir / "fig_threshold_sensitivity.png")

    write_report(output_dir / "gt_score_alignment_report.md", args, classified, inventory_rows, labels, durations, outside_rows, outside_intervals, window_rows, upper, warnings)
    write_archive_manifest(output_dir)
    write_json(
        output_dir / "gt_score_alignment_summary.json",
        {
            "gt_count": len(classified),
            "video_count": len(inventory_rows),
            "support_counts": dict(Counter(row["support_type"] for row in classified)),
            "warning_count": len(warnings),
            "outside_gt_high_score_interval_count": len(outside_intervals),
            "output_dir": str(args.output_dir),
        },
    )
    print(
        json.dumps(
            {
                "gt_count": len(classified),
                "video_count": len(inventory_rows),
                "warning_count": len(warnings),
                "outside_gt_high_score_intervals": len(outside_intervals),
                "report": str((output_dir / "gt_score_alignment_report.md").relative_to(root)),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
