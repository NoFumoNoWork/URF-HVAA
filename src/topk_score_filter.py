from __future__ import annotations


def _score_frames(scores: dict) -> list[int]:
    return sorted(int(k) for k in scores.keys())


def _value(scores: dict, frame: int) -> float:
    if frame in scores:
        return float(scores[frame])
    return float(scores[str(frame)])


def _overlap(a: dict, b: dict) -> int:
    return max(0, min(a["end"], b["end"]) - max(a["start"], b["start"]))


def _iou(a: dict, b: dict) -> float:
    inter = _overlap(a, b)
    union = max(a["end"], b["end"]) - min(a["start"], b["start"])
    return inter / union if union > 0 else 0.0


def _overlap_ratio(a: dict, b: dict) -> float:
    inter = _overlap(a, b)
    denom = max(1, min(a["end"] - a["start"], b["end"] - b["start"]))
    return inter / denom


def candidate_intervals(scores: dict, window_size: int | None = None) -> list[dict]:
    frames = _score_frames(scores)
    if not frames:
        return []
    max_frame = frames[-1]
    if window_size is None:
        window_size = max(max_frame // 10, 300)

    candidates = []
    n = len(frames)
    for i, start in enumerate(frames):
        end = start + window_size
        values = [_value(scores, frame) for frame in frames[i:n] if frame < end]
        if not values:
            continue
        candidates.append(
            {
                "start": start,
                "end": end,
                "mean_score": sum(values) / len(values),
                "max_score": max(values),
                "num_score_points": len(values),
                "window_size": window_size,
            }
        )
    return candidates


def find_topk_intervals(
    scores: dict,
    k: int = 5,
    window_size: int | None = None,
    nms_iou: float = 0.5,
    min_mean_score: float | None = None,
) -> list[dict]:
    candidates = candidate_intervals(scores, window_size=window_size)
    if min_mean_score is not None:
        candidates = [c for c in candidates if c["mean_score"] >= min_mean_score]
    candidates.sort(key=lambda c: (c["mean_score"], c["max_score"], c["num_score_points"]), reverse=True)

    selected = []
    for cand in candidates:
        if all(_iou(cand, kept) <= nms_iou and _overlap_ratio(cand, kept) <= nms_iou for kept in selected):
            selected.append(cand)
        if len(selected) >= k:
            break

    for rank, item in enumerate(selected, start=1):
        item["rank"] = rank
        item["mean_score"] = round(item["mean_score"], 6)
        item["max_score"] = round(item["max_score"], 6)
    return selected


def find_multiscale_intervals(
    scores: dict,
    window_sizes: list[int] | None = None,
    adaptive_scales: bool = True,
    k_per_scale: int = 5,
    final_k: int = 10,
    nms_iou: float = 0.5,
    min_mean_score: float | None = None,
) -> list[dict]:
    frames = _score_frames(scores)
    if not frames:
        return []
    max_frame = frames[-1]
    scales = list(window_sizes or [300, 600, 1200])
    if adaptive_scales:
        scales.extend([max(max_frame // 20, 1), max(max_frame // 10, 1)])
    scales = sorted({int(s) for s in scales if int(s) > 0})

    all_candidates = []
    for scale in scales:
        top = find_topk_intervals(
            scores,
            k=k_per_scale,
            window_size=scale,
            nms_iou=nms_iou,
            min_mean_score=min_mean_score,
        )
        for item in top:
            item = dict(item)
            item["scale"] = scale
            all_candidates.append(item)

    all_candidates.sort(key=lambda c: (c["mean_score"], c["max_score"], c["num_score_points"]), reverse=True)
    selected = []
    for cand in all_candidates:
        if all(_iou(cand, kept) <= nms_iou and _overlap_ratio(cand, kept) <= nms_iou for kept in selected):
            selected.append(cand)
        if len(selected) >= final_k:
            break

    for rank, item in enumerate(selected, start=1):
        item["rank"] = rank
    return selected
