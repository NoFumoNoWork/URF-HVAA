import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.anomaly_utils import (  # noqa: E402
    DATASET_CONFIGS,
    load_scores,
    parse_temporal_annotations,
    repo_root,
    score_path_for_video,
    score_values_in_interval,
    write_json,
)
from src.adaptive_interval_selection import interval_gap, interval_iou, percentile, top_fraction_mean  # noqa: E402


DEFAULT_MICRO_WINDOWS = [30, 100, 300]
DEFAULT_CONTEXT_WINDOWS = [600, 1200]
DEFAULT_SCALE_THRESHOLDS = {30: 90.0, 100: 85.0, 300: 75.0}


def parse_int_list(text: str) -> list[int]:
    if not text:
        return []
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_scale_thresholds(text: str) -> dict[int, float]:
    if not text:
        return {}
    thresholds = {}
    for item in text.split(","):
        if not item.strip():
            continue
        scale, value = item.split(":", 1)
        thresholds[int(scale.strip())] = float(value.strip())
    return thresholds


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def rounded(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None


def proposal_score(values: list[float]) -> tuple[float, float, float, float]:
    mean_score = sum(values) / len(values) if values else 0.0
    max_score = max(values) if values else 0.0
    top20_mean_score = top_fraction_mean(values, 0.2)
    score = 0.6 * mean_score + 0.3 * top20_mean_score + 0.1 * max_score
    return mean_score, max_score, top20_mean_score, score


def generate_scale_candidates(scores: dict[int, float], window_sizes: list[int]) -> list[dict]:
    frames = sorted(scores)
    candidates = []
    for scale in window_sizes:
        for start in frames:
            end = start + scale
            values = score_values_in_interval(scores, start, end)
            if not values:
                continue
            mean_score, max_score, top20_mean_score, score = proposal_score(values)
            candidates.append(
                {
                    "start": int(start),
                    "end": int(end),
                    "duration": int(scale),
                    "mean_score": rounded(mean_score),
                    "max_score": rounded(max_score),
                    "top20_mean_score": rounded(top20_mean_score),
                    "proposal_score": rounded(score),
                    "scale": int(scale),
                    "num_score_points": len(values),
                }
            )
    return candidates


def select_micro_proposals(
    scores: dict[int, float],
    window_sizes: list[int],
    threshold_percentile: float,
    scale_thresholds: dict[int, float],
) -> tuple[list[dict], dict]:
    all_candidates = generate_scale_candidates(scores, window_sizes)
    retained = []
    threshold_by_scale = {}
    count_by_scale = {}
    retained_by_scale = {}
    for scale in window_sizes:
        scale_candidates = [item for item in all_candidates if item["scale"] == scale]
        scores_for_scale = [item["proposal_score"] for item in scale_candidates]
        pct = scale_thresholds.get(scale, threshold_percentile)
        threshold = percentile(scores_for_scale, pct)
        threshold_by_scale[str(scale)] = rounded(threshold)
        count_by_scale[str(scale)] = len(scale_candidates)
        selected = [item for item in scale_candidates if threshold is not None and item["proposal_score"] >= threshold]
        retained_by_scale[str(scale)] = len(selected)
        retained.extend(selected)
    retained.sort(key=lambda item: (item["start"], item["end"], item["scale"], -item["proposal_score"]))
    metadata = {
        "window_sizes": window_sizes,
        "threshold_percentile": threshold_percentile,
        "scale_threshold_percentiles": {str(k): v for k, v in sorted(scale_thresholds.items()) if k in window_sizes},
        "threshold_by_scale": threshold_by_scale,
        "candidate_count_by_scale": count_by_scale,
        "retained_count_by_scale": retained_by_scale,
        "candidate_count": len(all_candidates),
        "retained_candidate_count": len(retained),
    }
    return retained, metadata


def cluster_micro_proposals(candidates: list[dict], merge_iou: float, merge_gap: int) -> list[list[dict]]:
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda item: (item["start"], item["end"], item["scale"], -item["proposal_score"]))
    clusters = [[ordered[0]]]
    span = {"start": ordered[0]["start"], "end": ordered[0]["end"]}
    for cand in ordered[1:]:
        if interval_iou(span, cand) >= merge_iou or interval_gap(span, cand) <= merge_gap:
            clusters[-1].append(cand)
            span["start"] = min(span["start"], cand["start"])
            span["end"] = max(span["end"], cand["end"])
        else:
            clusters.append([cand])
            span = {"start": cand["start"], "end": cand["end"]}
    return clusters


def gap_stats(scores: dict[int, float], prev: dict, current: dict, prev_sub_id: int, current_sub_id: int) -> dict | None:
    gap_start = int(prev["end"])
    gap_end = int(current["start"])
    if gap_end <= gap_start:
        return None
    values = score_values_in_interval(scores, gap_start, gap_end)
    return {
        "between_sub_ids": [prev_sub_id, current_sub_id],
        "gap_start": gap_start,
        "gap_end": gap_end,
        "gap_duration": gap_end - gap_start,
        "gap_mean_score": rounded(sum(values) / len(values)) if values else None,
        "gap_min_score": rounded(min(values)) if values else None,
        "gap_max_score": rounded(max(values)) if values else None,
    }


def cluster_to_event(cluster: list[dict], scores: dict[int, float], event_id: int) -> dict:
    ordered = sorted(cluster, key=lambda item: (item["start"], item["end"], item["scale"], -item["proposal_score"]))
    micro_intervals = []
    for sub_id, item in enumerate(ordered, start=1):
        micro = {
            "sub_id": sub_id,
            "start": item["start"],
            "end": item["end"],
            "duration": item["duration"],
            "scale": item["scale"],
            "mean_score": item["mean_score"],
            "max_score": item["max_score"],
            "top20_mean_score": item["top20_mean_score"],
            "proposal_score": item["proposal_score"],
            "num_score_points": item["num_score_points"],
        }
        micro_intervals.append(micro)
    gaps = []
    if micro_intervals:
        rightmost = micro_intervals[0]
        for micro in micro_intervals[1:]:
            if micro["start"] > rightmost["end"]:
                gap = gap_stats(scores, rightmost, micro, rightmost["sub_id"], micro["sub_id"])
                if gap:
                    gaps.append(gap)
            if micro["end"] > rightmost["end"]:
                rightmost = micro
    merged_start = min(item["start"] for item in ordered)
    merged_end = max(item["end"] for item in ordered)
    event_values = score_values_in_interval(scores, merged_start, merged_end)
    return {
        "event_id": event_id,
        "merged_start": int(merged_start),
        "merged_end": int(merged_end),
        "merged_duration": int(merged_end - merged_start),
        "event_score": rounded(max(item["proposal_score"] for item in ordered)),
        "event_mean_score": rounded(sum(event_values) / len(event_values)) if event_values else None,
        "event_max_score": rounded(max(event_values)) if event_values else None,
        "scale_sources": sorted({item["scale"] for item in ordered}),
        "micro_interval_count": len(micro_intervals),
        "micro_intervals": micro_intervals,
        "gaps": gaps,
    }


def context_summary(scores: dict[int, float], window_sizes: list[int], threshold_percentile: float) -> list[dict]:
    if not window_sizes:
        return []
    candidates = generate_scale_candidates(scores, window_sizes)
    summaries = []
    for scale in window_sizes:
        scale_candidates = [item for item in candidates if item["scale"] == scale]
        threshold = percentile([item["proposal_score"] for item in scale_candidates], threshold_percentile)
        top = [item for item in scale_candidates if threshold is not None and item["proposal_score"] >= threshold]
        top.sort(key=lambda item: item["proposal_score"], reverse=True)
        summaries.extend(top[:10])
    summaries.sort(key=lambda item: (item["start"], item["end"], item["scale"]))
    return summaries


def generate_video(
    dataset: str,
    video_id: str,
    label: str,
    score_path: Path | None,
    scores: dict[int, float],
    args: argparse.Namespace,
) -> dict:
    scale_thresholds = dict(DEFAULT_SCALE_THRESHOLDS)
    scale_thresholds.update(parse_scale_thresholds(args.scale_thresholds))
    micro_windows = parse_int_list(args.micro_windows)
    micro_proposals, metadata = select_micro_proposals(scores, micro_windows, args.threshold_percentile, scale_thresholds)
    clusters = cluster_micro_proposals(micro_proposals, args.merge_iou, args.merge_gap)
    events = [cluster_to_event(cluster, scores, event_id) for event_id, cluster in enumerate(clusters, start=1)]
    root = repo_root()
    metadata.update(
        {
            "merge_iou": args.merge_iou,
            "merge_gap": args.merge_gap,
            "event_count": len(events),
            "context_window_sizes": parse_int_list(args.context_windows) if args.include_context else [],
        }
    )
    result = {
        "dataset": dataset,
        "video_id": video_id,
        "label": label,
        "score_json_path": str(score_path.relative_to(root)) if score_path else "",
        "events": events,
        "metadata": metadata,
    }
    if args.include_context:
        result["context_intervals"] = context_summary(scores, parse_int_list(args.context_windows), args.context_threshold_percentile)
    return result


def generate(args: argparse.Namespace) -> tuple[dict, list[dict]]:
    root = repo_root()
    output = {}
    rows = []
    for dataset, cfg in DATASET_CONFIGS.items():
        annotations = parse_temporal_annotations(root / cfg["annotation"], dataset)
        for video_id in sorted(annotations):
            meta = annotations[video_id]
            score_path = score_path_for_video(video_id, [root / p for p in cfg["score_dirs"]])
            scores = load_scores(score_path)
            if not scores:
                continue
            key = video_id if video_id not in output else f"{dataset}::{video_id}"
            video_result = generate_video(dataset, video_id, meta["label"], score_path, scores, args)
            output[key] = video_result
            event_counts = [event["micro_interval_count"] for event in video_result["events"]]
            gap_durations = [gap["gap_duration"] for event in video_result["events"] for gap in event["gaps"]]
            rows.append(
                {
                    "dataset": dataset,
                    "video_id": video_id,
                    "output_key": key,
                    "score_json_path": video_result["score_json_path"],
                    "event_count": len(video_result["events"]),
                    "micro_proposal_count": video_result["metadata"]["retained_candidate_count"],
                    "mean_micro_intervals_per_event": round(sum(event_counts) / len(event_counts), 3) if event_counts else 0,
                    "mean_gap_duration": round(sum(gap_durations) / len(gap_durations), 3) if gap_durations else 0,
                    "thresholds": json.dumps(video_result["metadata"]["threshold_by_scale"], sort_keys=True),
                }
            )
    return output, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold_percentile", type=float, default=75.0)
    parser.add_argument("--scale_thresholds", default="30:90,100:85,300:75")
    parser.add_argument("--micro_windows", default="30,100,300")
    parser.add_argument("--context_windows", default="600,1200")
    parser.add_argument("--include_context", action="store_true")
    parser.add_argument("--context_threshold_percentile", type=float, default=85.0)
    parser.add_argument("--merge_iou", type=float, default=0.3)
    parser.add_argument("--merge_gap", type=int, default=150)
    parser.add_argument("--output", type=Path, default=Path("outputs/hierarchical_intervals/hierarchical_intervals.json"))
    args = parser.parse_args()

    root = repo_root()
    results, rows = generate(args)
    write_json(root / args.output, results)
    write_csv(root / "outputs/hierarchical_intervals/hierarchical_intervals_summary.csv", rows)
    print(
        json.dumps(
            {
                "videos": len(results),
                "events": sum(len(video["events"]) for video in results.values()),
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
