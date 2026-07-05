from __future__ import annotations

import math


def score_frames(scores: dict) -> list[int]:
    return sorted(int(k) for k in scores.keys())


def score_value(scores: dict, frame: int) -> float:
    if frame in scores:
        return float(scores[frame])
    return float(scores[str(frame)])


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[int(rank)]
    weight = rank - lo
    return ordered[lo] * (1 - weight) + ordered[hi] * weight


def adaptive_window_sizes(scores: dict, min_window_size: int = 60) -> list[int]:
    frames = score_frames(scores)
    if not frames:
        return []
    max_frame = frames[-1]
    sizes = [300, 600, 1200, max_frame // 20, max_frame // 10]
    return sorted({int(size) for size in sizes if int(size) >= min_window_size})


def top_fraction_mean(values: list[float], fraction: float = 0.2) -> float:
    if not values:
        return 0.0
    keep = max(1, math.ceil(len(values) * fraction))
    return sum(sorted(values, reverse=True)[:keep]) / keep


def generate_candidates(scores: dict, window_sizes: list[int]) -> list[dict]:
    frames = score_frames(scores)
    candidates = []
    for window_size in window_sizes:
        for i, start in enumerate(frames):
            end = start + window_size
            values = [score_value(scores, frame) for frame in frames[i:] if frame < end]
            if not values:
                continue
            mean_score = sum(values) / len(values)
            max_score = max(values)
            top20_mean_score = top_fraction_mean(values, 0.2)
            proposal_score = 0.6 * mean_score + 0.3 * top20_mean_score + 0.1 * max_score
            candidates.append(
                {
                    "start": int(start),
                    "end": int(end),
                    "duration": int(window_size),
                    "mean_score": mean_score,
                    "max_score": max_score,
                    "top20_mean_score": top20_mean_score,
                    "proposal_score": proposal_score,
                    "num_score_points": len(values),
                    "window_size": int(window_size),
                }
            )
    return candidates


def interval_iou(a: dict, b: dict) -> float:
    inter = max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
    union = max(a["end"], b["end"]) - min(a["start"], b["start"])
    return inter / union if union > 0 else 0.0


def interval_gap(a: dict, b: dict) -> int:
    if b["start"] >= a["end"]:
        return b["start"] - a["end"]
    if a["start"] >= b["end"]:
        return a["start"] - b["end"]
    return 0


def _cluster_to_interval(cluster: list[dict]) -> dict:
    best = max(cluster, key=lambda item: item["proposal_score"])
    scale_sources = sorted({int(item["window_size"]) for item in cluster})
    start = min(item["start"] for item in cluster)
    end = max(item["end"] for item in cluster)
    return {
        "start": int(start),
        "end": int(end),
        "duration": int(end - start),
        "mean_score": round(best["mean_score"], 6),
        "max_score": round(best["max_score"], 6),
        "top20_mean_score": round(best["top20_mean_score"], 6),
        "proposal_score": round(best["proposal_score"], 6),
        "scale_sources": scale_sources,
        "num_merged_windows": len(cluster),
    }


def merge_candidates(candidates: list[dict], merge_iou: float = 0.3, merge_gap: int = 150) -> list[dict]:
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda item: (item["start"], item["end"]))
    clusters: list[list[dict]] = [[ordered[0]]]
    cluster_span = {"start": ordered[0]["start"], "end": ordered[0]["end"]}
    for cand in ordered[1:]:
        should_merge = interval_iou(cluster_span, cand) >= merge_iou or interval_gap(cluster_span, cand) <= merge_gap
        if should_merge:
            clusters[-1].append(cand)
            cluster_span["start"] = min(cluster_span["start"], cand["start"])
            cluster_span["end"] = max(cluster_span["end"], cand["end"])
        else:
            clusters.append([cand])
            cluster_span = {"start": cand["start"], "end": cand["end"]}
    return [_cluster_to_interval(cluster) for cluster in clusters]


def select_adaptive_intervals(
    scores: dict,
    threshold_percentile: float = 85.0,
    merge_iou: float = 0.3,
    merge_gap: int = 150,
    min_duration: int = 60,
    post_filter_percentile: float | None = 75.0,
) -> tuple[list[dict], dict]:
    sizes = adaptive_window_sizes(scores)
    candidates = generate_candidates(scores, sizes)
    proposal_scores = [item["proposal_score"] for item in candidates]
    threshold = percentile(proposal_scores, threshold_percentile)
    retained = [item for item in candidates if threshold is not None and item["proposal_score"] >= threshold]
    merged = merge_candidates(retained, merge_iou=merge_iou, merge_gap=merge_gap)
    post_threshold = percentile(proposal_scores, post_filter_percentile) if post_filter_percentile is not None else None
    filtered = []
    for item in merged:
        if item["duration"] < min_duration:
            continue
        if post_threshold is not None and item["proposal_score"] < post_threshold:
            continue
        filtered.append(item)
    filtered.sort(key=lambda item: (item["start"], item["end"]))
    for event_id, item in enumerate(filtered, start=1):
        item["event_id"] = event_id
    metadata = {
        "window_sizes": sizes,
        "candidate_count": len(candidates),
        "threshold_percentile": threshold_percentile,
        "percentile_threshold": round(threshold, 6) if threshold is not None else None,
        "retained_candidate_count": len(retained),
        "merged_interval_count": len(merged),
        "final_interval_count": len(filtered),
        "post_filter_percentile": post_filter_percentile,
        "post_filter_threshold": round(post_threshold, 6) if post_threshold is not None else None,
    }
    return filtered, metadata
