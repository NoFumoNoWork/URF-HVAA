from __future__ import annotations

import random


def score_frames(scores: dict) -> list[int]:
    return sorted(int(k) for k in scores.keys())


def score_value(scores: dict, frame: int) -> float:
    if frame in scores:
        return float(scores[frame])
    return float(scores[str(frame)])


def default_window_size(scores: dict) -> int:
    frames = score_frames(scores)
    if not frames:
        return 0
    return max(frames[-1] // 10, 300)


def score_values_in_window(scores: dict, start: int, end: int) -> list[float]:
    return [score_value(scores, frame) for frame in score_frames(scores) if start <= frame < end]


def interval_stats(scores: dict, start: int, end: int, window_size: int) -> dict:
    values = score_values_in_window(scores, start, end)
    return {
        "start": int(start),
        "end": int(end),
        "mean_score": round(sum(values) / len(values), 6) if values else None,
        "max_score": round(max(values), 6) if values else None,
        "num_score_points": len(values),
        "window_size": int(window_size),
    }


def clamp_interval_around(center: int, window_size: int, video_length: int) -> tuple[int, int]:
    if window_size <= 0:
        return 0, 0
    if video_length <= 0:
        start = max(0, center - window_size // 2)
        return start, start + window_size
    if window_size >= video_length:
        return 0, window_size
    start = center - window_size // 2
    start = max(0, min(start, video_length - window_size))
    return int(start), int(start + window_size)


def random_same_length_interval(
    scores: dict,
    video_length: int,
    rng: random.Random,
    window_size: int | None = None,
) -> dict:
    if window_size is None:
        window_size = default_window_size(scores)
    if window_size <= 0:
        return interval_stats(scores, 0, 0, 0)
    max_start = max(0, video_length - window_size)
    start = rng.randint(0, max_start) if max_start > 0 else 0
    return interval_stats(scores, start, start + window_size, window_size)


def peak_expanded_interval(scores: dict, video_length: int, window_size: int | None = None) -> dict:
    frames = score_frames(scores)
    if not frames:
        return interval_stats(scores, 0, 0, 0)
    if window_size is None:
        window_size = default_window_size(scores)
    max_score = max(score_value(scores, frame) for frame in frames)
    peak_frames = [frame for frame in frames if score_value(scores, frame) == max_score]
    candidates = []
    for peak in peak_frames:
        start, end = clamp_interval_around(peak, window_size, video_length)
        item = interval_stats(scores, start, end, window_size)
        item["peak_frame"] = int(peak)
        item["peak_score"] = round(max_score, 6)
        candidates.append(item)
    candidates.sort(
        key=lambda item: (
            item["mean_score"] if item["mean_score"] is not None else float("-inf"),
            item["num_score_points"],
            -abs(item["peak_frame"] - (item["start"] + item["end"]) / 2),
        ),
        reverse=True,
    )
    best = candidates[0]
    best["num_max_peaks"] = len(peak_frames)
    return best
