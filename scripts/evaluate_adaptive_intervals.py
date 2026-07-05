import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.anomaly_utils import (  # noqa: E402
    DATASET_CONFIGS,
    bin_value,
    coverage_by_intervals,
    estimate_video_length,
    load_scores,
    parse_temporal_annotations,
    repo_root,
    safe_name,
    score_metadata,
    score_path_for_video,
    write_json,
)
from src.score_filter import find_extreme_intervals  # noqa: E402
from src.topk_score_filter import find_topk_intervals  # noqa: E402


COUNT_BINS = [(0, 2, "1"), (2, 4, "2-3"), (4, 8, "4-7"), (8, 10**9, "8+")]
LENGTH_BINS = [(0, 1000, "<1k"), (1000, 3000, "1k-3k"), (3000, 6000, "3k-6k"), (6000, 10**9, "6k+")]
DEFAULT_CASES = [
    ("XD-Violence", "v=38GQ9L2meyE__#1_label_B6-0-0"),
    ("XD-Violence", "v=uQY15O3LKI0__#1_label_B6-0-0"),
    ("UCF-Crime", "Assault010_x264"),
]


def load_adaptive(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def adaptive_intervals_for(preds: dict, dataset: str, video_id: str) -> list[dict]:
    value = preds.get(dataset, {}).get(video_id, {})
    if isinstance(value, list):
        return value
    return value.get("intervals", [])


def overlap_count_for_gt(gt: dict, intervals: list[dict]) -> int:
    return sum(1 for item in intervals if max(0, min(gt["end"], item["end"]) - max(gt["start"], item["start"])) > 0)


def covered_gt_count_for_pred(pred: dict, gts: list[dict]) -> int:
    return sum(1 for gt in gts if max(0, min(gt["end"], pred["end"]) - max(gt["start"], pred["start"])) > 0)


def summarize_rows(rows: list[dict]) -> dict:
    total = len(rows)
    missed = sum(row["missed"] for row in rows)
    videos = defaultdict(list)
    for row in rows:
        videos[(row["dataset"], row["video_id"])].append(row)

    def group_stats(key_fn):
        groups = defaultdict(list)
        for row in rows:
            groups[key_fn(row)].append(row)
        return {
            key: {
                "segments": len(items),
                "segment_miss_rate": round(sum(i["missed"] for i in items) / len(items), 6),
                "mean_coverage": round(sum(i["coverage"] for i in items) / len(items), 6),
            }
            for key, items in sorted(groups.items())
        }

    return {
        "segments": total,
        "covered_segments": total - missed,
        "missed_segments": missed,
        "segment_miss_rate": round(missed / total, 6) if total else None,
        "video_count": len(videos),
        "videos_with_any_miss": sum(any(item["missed"] for item in items) for items in videos.values()),
        "video_any_miss_rate": round(sum(any(item["missed"] for item in items) for items in videos.values()) / len(videos), 6) if videos else None,
        "mean_coverage": round(sum(row["coverage"] for row in rows) / total, 6) if total else None,
        "coverage_by_dataset": group_stats(lambda row: row["dataset"]),
        "coverage_by_anomaly_count_bin": group_stats(lambda row: bin_value(row["anomaly_count"], COUNT_BINS)),
        "coverage_by_video_length_bin": group_stats(lambda row: bin_value(row["video_length"], LENGTH_BINS)),
    }


def evaluate_adaptive(preds: dict) -> tuple[dict, list[dict], dict]:
    root = repo_root()
    rows = []
    pred_counts = []
    pred_durations = []
    gt_pred_overlap_counts = []
    pred_gt_overlap_counts = []
    for dataset, cfg in DATASET_CONFIGS.items():
        annotations = parse_temporal_annotations(root / cfg["annotation"], dataset)
        for video_id, meta in annotations.items():
            if not meta["intervals"]:
                continue
            score_path = score_path_for_video(video_id, [root / p for p in cfg["score_dirs"]])
            scores = load_scores(score_path)
            video_len = estimate_video_length(score_metadata(scores), meta["intervals"])
            intervals = adaptive_intervals_for(preds, dataset, video_id)
            pred_counts.append(len(intervals))
            pred_durations.extend([item["duration"] for item in intervals])
            pred_gt_overlap_counts.extend([covered_gt_count_for_pred(pred, meta["intervals"]) for pred in intervals])
            for gt_idx, gt in enumerate(meta["intervals"], start=1):
                cov = coverage_by_intervals(gt, intervals)
                gt_pred_overlap_counts.append(overlap_count_for_gt(gt, intervals))
                rows.append(
                    {
                        "dataset": dataset,
                        "video_id": video_id,
                        "anomaly_id": gt_idx,
                        "coverage": cov,
                        "missed": cov < 0.1,
                        "anomaly_count": len(meta["intervals"]),
                        "video_length": video_len,
                    }
                )
    summary = summarize_rows(rows)
    summary.update(
        {
            "mean_pred_intervals_per_video": round(statistics.mean(pred_counts), 6) if pred_counts else 0,
            "median_pred_intervals_per_video": round(statistics.median(pred_counts), 6) if pred_counts else 0,
            "max_pred_intervals_per_video": max(pred_counts) if pred_counts else 0,
            "mean_pred_interval_duration": round(statistics.mean(pred_durations), 6) if pred_durations else 0,
            "redundancy": {
                "mean_pred_intervals_covering_each_gt": round(statistics.mean(gt_pred_overlap_counts), 6) if gt_pred_overlap_counts else 0,
                "max_pred_intervals_covering_each_gt": max(gt_pred_overlap_counts) if gt_pred_overlap_counts else 0,
                "mean_gt_intervals_covered_by_each_pred": round(statistics.mean(pred_gt_overlap_counts), 6) if pred_gt_overlap_counts else 0,
                "max_gt_intervals_covered_by_each_pred": max(pred_gt_overlap_counts) if pred_gt_overlap_counts else 0,
            },
        }
    )
    return summary, rows, {"pred_counts": pred_counts, "pred_durations": pred_durations}


def baseline_wmax_summary() -> dict:
    root = repo_root()
    rows = []
    counts = []
    durations = []
    for dataset, cfg in DATASET_CONFIGS.items():
        annotations = parse_temporal_annotations(root / cfg["annotation"], dataset)
        for video_id, meta in annotations.items():
            if not meta["intervals"]:
                continue
            score_path = score_path_for_video(video_id, [root / p for p in cfg["score_dirs"]])
            scores = load_scores(score_path)
            if not scores:
                continue
            best_s, best_e, *_ = find_extreme_intervals({str(k): v for k, v in scores.items()})
            interval = {"start": best_s, "end": best_e, "duration": best_e - best_s}
            counts.append(1)
            durations.append(best_e - best_s)
            video_len = estimate_video_length(score_metadata(scores), meta["intervals"])
            for gt in meta["intervals"]:
                cov = coverage_by_intervals(gt, [interval])
                rows.append({"dataset": dataset, "video_id": video_id, "coverage": cov, "missed": cov < 0.1, "anomaly_count": len(meta["intervals"]), "video_length": video_len})
    summary = summarize_rows(rows)
    summary["mean_pred_intervals_per_video"] = 1.0
    summary["mean_pred_interval_duration"] = round(statistics.mean(durations), 6) if durations else 0
    return summary


def baseline_topk_summary(k: int) -> dict:
    root = repo_root()
    path = root / "outputs/topk_coverage_results.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        item = next((row for row in existing if row["k"] == k), None)
        if item:
            item = dict(item)
            item["mean_pred_intervals_per_video"] = float(k)
            return item
    rows = []
    for dataset, cfg in DATASET_CONFIGS.items():
        annotations = parse_temporal_annotations(root / cfg["annotation"], dataset)
        for video_id, meta in annotations.items():
            if not meta["intervals"]:
                continue
            scores = load_scores(score_path_for_video(video_id, [root / p for p in cfg["score_dirs"]]))
            if not scores:
                continue
            intervals = find_topk_intervals(scores, k=k)
            video_len = estimate_video_length(score_metadata(scores), meta["intervals"])
            for gt in meta["intervals"]:
                cov = coverage_by_intervals(gt, intervals)
                rows.append({"dataset": dataset, "video_id": video_id, "coverage": cov, "missed": cov < 0.1, "anomaly_count": len(meta["intervals"]), "video_length": video_len})
    summary = summarize_rows(rows)
    summary["mean_pred_intervals_per_video"] = float(k)
    return summary


def baseline_multiscale_summary() -> dict | None:
    root = repo_root()
    path = root / "reports/multiscale_coverage_report.md"
    interval_path = root / "outputs/multiscale_intervals.json"
    if not interval_path.exists():
        return None
    preds = json.loads(interval_path.read_text(encoding="utf-8"))
    rows = []
    counts = []
    durations = []
    for dataset, cfg in DATASET_CONFIGS.items():
        annotations = parse_temporal_annotations(root / cfg["annotation"], dataset)
        for video_id, meta in annotations.items():
            if not meta["intervals"]:
                continue
            intervals = preds.get(dataset, {}).get(video_id, [])
            counts.append(len(intervals))
            durations.extend([item.get("duration", item["end"] - item["start"]) for item in intervals])
            scores = load_scores(score_path_for_video(video_id, [root / p for p in cfg["score_dirs"]]))
            video_len = estimate_video_length(score_metadata(scores), meta["intervals"])
            for gt in meta["intervals"]:
                cov = coverage_by_intervals(gt, intervals)
                rows.append({"dataset": dataset, "video_id": video_id, "coverage": cov, "missed": cov < 0.1, "anomaly_count": len(meta["intervals"]), "video_length": video_len})
    summary = summarize_rows(rows)
    summary["mean_pred_intervals_per_video"] = round(statistics.mean(counts), 6) if counts else 0
    summary["mean_pred_interval_duration"] = round(statistics.mean(durations), 6) if durations else 0
    summary["source_report"] = str(path.relative_to(root)) if path.exists() else ""
    return summary


def draw_tradeoff(path: Path, comparison: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for item in comparison:
        ax.scatter(item["mean_pred_intervals_per_video"], item["segment_miss_rate"], s=70)
        ax.annotate(item["method"], (item["mean_pred_intervals_per_video"], item["segment_miss_rate"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("mean predicted intervals per video")
    ax.set_ylabel("segment miss rate")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def draw_timeline(dataset: str, video_id: str, preds: dict, output_dir: Path) -> dict:
    root = repo_root()
    cfg = DATASET_CONFIGS[dataset]
    annotations = parse_temporal_annotations(root / cfg["annotation"], dataset)
    meta = annotations.get(video_id)
    if not meta:
        return {"dataset": dataset, "video_id": video_id, "status": "missing annotation"}
    scores = load_scores(score_path_for_video(video_id, [root / p for p in cfg["score_dirs"]]))
    if not scores:
        return {"dataset": dataset, "video_id": video_id, "status": "missing score"}
    best_s, best_e, *_ = find_extreme_intervals({str(k): v for k, v in scores.items()})
    topk = find_topk_intervals(scores, k=5)
    adaptive = adaptive_intervals_for(preds, dataset, video_id)
    video_len = estimate_video_length(score_metadata(scores), meta["intervals"])

    tracks = [
        ("GT", meta["intervals"], "#d95f02"),
        ("Wmax", [{"start": best_s, "end": best_e}], "#1b9e77"),
        ("TopK5", topk, "#7570b3"),
        ("Adaptive", adaptive, "#e7298a"),
    ]
    fig, axes = plt.subplots(len(tracks) + 1, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1, 1, 2]})
    for ax, (label, intervals, color) in zip(axes[:-1], tracks):
        ax.set_ylabel(label)
        ax.set_yticks([])
        for item in intervals:
            ax.broken_barh([(item["start"], item["end"] - item["start"])], (0, 1), facecolors=color, alpha=0.85)
    axes[-1].plot(list(scores.keys()), list(scores.values()), color="#386cb0", linewidth=1.2)
    axes[-1].set_ylabel("score")
    axes[-1].set_xlabel("frame")
    axes[-1].grid(alpha=0.25)
    axes[-1].set_xlim(0, max(video_len, max(scores)))
    fig.suptitle(f"{dataset} | {video_id} | events={len(meta['intervals'])} | adaptive={len(adaptive)}", fontsize=10)
    fig.tight_layout()
    out = output_dir / dataset / f"{safe_name(video_id)}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return {"dataset": dataset, "video_id": video_id, "status": "ok", "output": str(out.relative_to(root))}


def write_report(path: Path, adaptive_summary: dict, comparison: list[dict], timeline_results: list[dict]) -> None:
    lines = [
        "# Adaptive Interval Selection Report",
        "",
        "## Method",
        "",
        "The adaptive method slides over fixed windows `300, 600, 1200` and adaptive windows `max_frame//20, max_frame//10`, then scores each candidate with:",
        "",
        "`proposal_score = 0.6 * mean_score + 0.3 * top20_mean_score + 0.1 * max_score`",
        "",
        "Candidates above the within-video 85th percentile are retained, then overlapping or near-adjacent intervals are merged with IoU >= 0.3 or gap <= 150 frames. Final intervals use the highest-proposal candidate in each cluster as their score carrier.",
        "",
        "## Parameters",
        "",
        "- threshold_percentile: 85",
        "- merge_iou: 0.3",
        "- merge_gap: 150 frames",
        "- min_duration: 60 frames",
        "- post_filter_percentile: 75",
        "",
        "## Coverage And Output Trade-Off",
        "",
        "| Method | Segment miss rate | Video any miss rate | Mean coverage | Mean intervals/video |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in comparison:
        lines.append(
            f"| {item['method']} | {item['segment_miss_rate']:.2%} | {item['video_any_miss_rate']:.2%} | "
            f"{item['mean_coverage']:.3f} | {item['mean_pred_intervals_per_video']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Adaptive Output Statistics",
            "",
            f"- mean_pred_intervals_per_video: {adaptive_summary['mean_pred_intervals_per_video']}",
            f"- median_pred_intervals_per_video: {adaptive_summary['median_pred_intervals_per_video']}",
            f"- max_pred_intervals_per_video: {adaptive_summary['max_pred_intervals_per_video']}",
            f"- mean_pred_interval_duration: {adaptive_summary['mean_pred_interval_duration']}",
            f"- redundancy: `{adaptive_summary['redundancy']}`",
            "",
            "## Fragmentation",
            "",
            "The mean interval count and redundancy statistics are the main fragmentation indicators. High interval count with low miss rate indicates recall-oriented behavior; high redundancy means multiple predicted intervals overlap the same GT event.",
            "",
            "## Typical Cases",
            "",
        ]
    )
    for item in timeline_results:
        lines.append(f"- {item['dataset']} `{item['video_id']}`: {item['status']} {item.get('output', '')}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Adaptive intervals are not fixed-K; they trade more flexible event count for coverage.",
            "- Compare adaptive against Wmax and Top-K to decide whether adaptive thresholding gives better recall per emitted interval.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("outputs/adaptive_intervals/adaptive_intervals_p85_gap150_iou03.json"))
    args = parser.parse_args()

    root = repo_root()
    preds = load_adaptive(root / args.input)
    adaptive_summary, rows, _ = evaluate_adaptive(preds)
    comparison = [
        {"method": "original_wmax", **baseline_wmax_summary()},
        {"method": "topk_k5", **baseline_topk_summary(5)},
        {"method": "topk_k10", **baseline_topk_summary(10)},
    ]
    multiscale = baseline_multiscale_summary()
    if multiscale:
        comparison.append({"method": "multiscale_k10", **multiscale})
    comparison.append({"method": "adaptive", **adaptive_summary})

    write_json(root / "outputs/adaptive_intervals/adaptive_coverage_results.json", {"adaptive": adaptive_summary, "comparison": comparison, "rows": rows})
    draw_tradeoff(root / "outputs/adaptive_interval_tradeoff.png", comparison)
    timeline_results = [draw_timeline(dataset, video_id, preds, root / "outputs/adaptive_timeline_plots") for dataset, video_id in DEFAULT_CASES]
    write_report(root / "reports/adaptive_interval_selection_report.md", adaptive_summary, comparison, timeline_results)
    print(json.dumps({"adaptive": adaptive_summary, "report": "reports/adaptive_interval_selection_report.md"}, indent=2))


if __name__ == "__main__":
    main()
