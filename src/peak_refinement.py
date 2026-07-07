from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable

import numpy as np
from scipy.signal import find_peaks, peak_prominences, peak_widths


def _as_float_array(scores: Iterable[float]) -> np.ndarray:
    return np.asarray(list(scores), dtype=float)


def _round(value: float | int | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return round(float(value), 6)


def estimate_baseline(scores, window: int = 101, method: str = "median", quantile: float = 0.3) -> np.ndarray:
    values = _as_float_array(scores)
    if len(values) == 0:
        return np.asarray([], dtype=float)
    if window < 1:
        raise ValueError("baseline window must be >= 1")
    if method not in {"median", "quantile"}:
        raise ValueError(f"unsupported baseline method: {method}")
    if not 0 <= quantile <= 1:
        raise ValueError("baseline quantile must be in [0, 1]")

    radius = max(0, window // 2)
    baseline = np.zeros_like(values, dtype=float)
    for idx in range(len(values)):
        lo = max(0, idx - radius)
        hi = min(len(values), idx + radius + 1)
        chunk = values[lo:hi]
        if method == "median":
            baseline[idx] = float(np.median(chunk))
        else:
            baseline[idx] = float(np.quantile(chunk, quantile))
    return baseline


def _adaptive_threshold(residual: np.ndarray, mad_k: float) -> float:
    if len(residual) == 0:
        return 0.0
    residual_median = float(np.median(residual))
    mad = float(np.median(np.abs(residual - residual_median)))
    return residual_median + mad_k * mad


def _peak_area(residual: np.ndarray, left: int, right: int) -> float:
    if len(residual) == 0 or right <= left:
        return 0.0
    clipped = np.maximum(residual[left:right], 0)
    return float(np.sum(clipped))


def detect_peaks(
    scores,
    baseline=None,
    min_prominence=None,
    min_height=None,
    min_width: int = 3,
    min_distance: int = 10,
    mad_k: float = 2.5,
) -> list[dict]:
    values = _as_float_array(scores)
    if len(values) == 0:
        return []
    base = estimate_baseline(values) if baseline is None else _as_float_array(baseline)
    if len(base) != len(values):
        raise ValueError("baseline length must match scores length")

    residual = values - base
    adaptive = _adaptive_threshold(residual, mad_k)
    height_threshold = adaptive if min_height is None else float(min_height)
    distance = max(1, int(min_distance))
    raw_peaks, props = find_peaks(residual, height=height_threshold, distance=distance)
    if len(raw_peaks) == 0:
        return []

    prominences, left_bases, right_bases = peak_prominences(residual, raw_peaks)
    widths, _, left_ips, right_ips = peak_widths(residual, raw_peaks, rel_height=0.5)
    min_prom = adaptive if min_prominence is None else float(min_prominence)

    peaks = []
    for idx, prom, width, left_base, right_base, left_ip, right_ip in zip(raw_peaks, prominences, widths, left_bases, right_bases, left_ips, right_ips):
        if prom < min_prom or width < min_width:
            continue
        left_boundary = max(0, int(math.floor(left_ip)))
        right_boundary = min(len(values), int(math.ceil(right_ip)) + 1)
        peaks.append(
            {
                "peak_index": int(idx),
                "height_raw": _round(values[idx]),
                "height_residual": _round(residual[idx]),
                "prominence": _round(prom),
                "width": _round(width),
                "left_base": int(left_base),
                "right_base": int(right_base),
                "left_boundary": left_boundary,
                "right_boundary": right_boundary,
                "area_residual": _round(_peak_area(residual, left_boundary, right_boundary)),
                "local_baseline": _round(base[idx]),
            }
        )
    return peaks


def expand_peak_intervals(scores, baseline, peaks: list[dict], stop_ratio: float = 0.2, min_len: int = 3, max_len: int | None = None) -> list[dict]:
    values = _as_float_array(scores)
    base = _as_float_array(baseline)
    if len(values) != len(base):
        raise ValueError("baseline length must match scores length")
    residual = values - base
    intervals = []
    if len(values) == 0:
        return intervals

    stronger = sorted(peaks, key=lambda item: item["peak_index"])
    for peak in stronger:
        idx = int(peak["peak_index"])
        prom = float(peak.get("prominence") or 0)
        stop_value = max(0.0, stop_ratio * prom)
        start = idx
        end = idx + 1

        left_limit = 0
        right_limit = len(values)
        for other in stronger:
            other_idx = int(other["peak_index"])
            other_prom = float(other.get("prominence") or 0)
            if other_prom > prom and other_idx < idx:
                left_limit = max(left_limit, other_idx + 1)
            elif other_prom > prom and other_idx > idx:
                right_limit = min(right_limit, other_idx)

        while start > left_limit and residual[start - 1] > stop_value:
            start -= 1
            if max_len is not None and end - start >= max_len:
                break
        while end < right_limit and residual[end] > stop_value:
            end += 1
            if max_len is not None and end - start >= max_len:
                break
        if end - start < min_len:
            pad = min_len - (end - start)
            start = max(left_limit, start - (pad // 2))
            end = min(right_limit, max(end, start + min_len))
        intervals.append(
            {
                "start": int(start),
                "end": int(end),
                "duration": int(end - start),
                "peak_index": idx,
                "peak_prominence": _round(prom),
                "peak_area": _round(_peak_area(residual, start, end)),
                "source": "peak_expanded",
            }
        )
    return intervals


def _overlap_len(a: dict, b: dict) -> int:
    return max(0, min(int(a["end"]), int(b["end"])) - max(int(a["start"]), int(b["start"])))


def _covered(candidate: dict, intervals: list[dict], overlap_tolerance: int = 0) -> bool:
    return any(_overlap_len(candidate, item) > overlap_tolerance for item in intervals)


def rescue_peak_intervals(
    peak_intervals: list[dict],
    micro_intervals: list[dict],
    merged_intervals: list[dict],
    min_prominence: float,
    min_area: float,
    overlap_tolerance: int = 0,
) -> list[dict]:
    rescued = []
    originals = list(micro_intervals) + list(merged_intervals)
    for item in peak_intervals:
        prom = float(item.get("peak_prominence") or 0)
        area = float(item.get("peak_area") or 0)
        if _covered(item, originals, overlap_tolerance):
            continue
        if prom < min_prominence or area < min_area:
            continue
        duration = int(item["end"]) - int(item["start"])
        isolated = duration <= 3
        rescued.append(
            {
                "start": int(item["start"]),
                "end": int(item["end"]),
                "duration": duration,
                "source": "peak_rescue",
                "peak_index": int(item["peak_index"]),
                "peak_prominence": _round(prom),
                "peak_area": _round(area),
                "confidence_hint": "low_isolated_sharp_spike" if isolated else "high_prominence_rescue",
                "reason": "high-prominence peak not covered by original intervals",
            }
        )
    return rescued


def _interval_peaks(interval: dict, peaks: list[dict]) -> list[dict]:
    return [peak for peak in peaks if int(interval["start"]) <= int(peak["peak_index"]) < int(interval["end"])]


def _micro_gaps_in_merged(merged: dict, micro_intervals: list[dict]) -> list[tuple[int, int, dict, dict]]:
    inside = [item for item in micro_intervals if _overlap_len(merged, item) > 0]
    inside.sort(key=lambda item: (int(item["start"]), int(item["end"])))
    gaps = []
    if len(inside) < 2:
        return gaps
    rightmost = dict(inside[0])
    for current in inside[1:]:
        if int(current["start"]) > int(rightmost["end"]):
            gaps.append((int(rightmost["end"]), int(current["start"]), rightmost, current))
        if int(current["end"]) > int(rightmost["end"]):
            rightmost = dict(current)
    return gaps


def split_merged_intervals_by_peak_gaps(
    merged_intervals: list[dict],
    micro_intervals: list[dict],
    peaks: list[dict],
    scores,
    baseline,
    min_gap_len: int = 50,
    low_score_quantile: float = 0.4,
) -> tuple[list[dict], list[dict]]:
    values = _as_float_array(scores)
    base = _as_float_array(baseline)
    if len(values) != len(base):
        raise ValueError("baseline length must match scores length")
    residual = values - base
    low_threshold = float(np.quantile(residual, low_score_quantile)) if len(residual) else 0.0
    refined = []
    diagnostics = []

    for merged in merged_intervals:
        split_points = []
        for gap_start, gap_end, left_micro, right_micro in _micro_gaps_in_merged(merged, micro_intervals):
            gap_len = gap_end - gap_start
            gap_residual = residual[gap_start:gap_end] if gap_end > gap_start else np.asarray([])
            gap_mean = float(np.mean(gap_residual)) if len(gap_residual) else 0.0
            peaks_in_gap = _interval_peaks({"start": gap_start, "end": gap_end}, peaks)
            left_support = bool(_interval_peaks(left_micro, peaks))
            right_support = bool(_interval_peaks(right_micro, peaks))
            should_split = gap_len >= min_gap_len and gap_mean <= low_threshold and not peaks_in_gap and (left_support or right_support)
            if should_split:
                split_points.append((gap_start, gap_end))
                diagnostics.append(
                    {
                        "original_merged": [int(merged["start"]), int(merged["end"])],
                        "split_gap": [gap_start, gap_end],
                        "reason": "long low-residual gap without peak support",
                        "gap_length": gap_len,
                        "gap_mean_residual": _round(gap_mean),
                        "num_peaks_in_gap": 0,
                    }
                )

        if not split_points:
            item = dict(merged)
            item["source"] = item.get("source", "merged_original")
            refined.append(item)
            continue

        cursor = int(merged["start"])
        for gap_start, gap_end in split_points:
            if gap_start > cursor:
                refined.append({"start": cursor, "end": gap_start, "duration": gap_start - cursor, "source": "peak_split"})
            cursor = max(cursor, gap_end)
        if cursor < int(merged["end"]):
            refined.append({"start": cursor, "end": int(merged["end"]), "duration": int(merged["end"]) - cursor, "source": "peak_split"})

    return refined, diagnostics


def compute_interval_peak_features(interval: dict, peaks: list[dict], scores, baseline) -> dict:
    values = _as_float_array(scores)
    base = _as_float_array(baseline)
    if len(values) != len(base):
        raise ValueError("baseline length must match scores length")
    start = max(0, int(interval["start"]))
    end = min(len(values), int(interval["end"]))
    if end <= start:
        return {
            "mean_score": None,
            "max_score": None,
            "mean_residual": None,
            "max_residual": None,
            "peak_count": 0,
            "max_prominence": 0,
            "total_peak_area": 0,
            "peak_density": 0,
            "local_contrast": 0,
            "has_peak_support": False,
        }
    residual = values - base
    chunk = values[start:end]
    rchunk = residual[start:end]
    support = _interval_peaks({"start": start, "end": end}, peaks)
    prominences = [float(peak.get("prominence") or 0) for peak in support]
    areas = [float(peak.get("area_residual") or 0) for peak in support]
    return {
        "mean_score": _round(float(np.mean(chunk))),
        "max_score": _round(float(np.max(chunk))),
        "mean_residual": _round(float(np.mean(rchunk))),
        "max_residual": _round(float(np.max(rchunk))),
        "peak_count": len(support),
        "max_prominence": _round(max(prominences) if prominences else 0),
        "total_peak_area": _round(sum(areas)),
        "peak_density": _round(len(support) / max(1, end - start)),
        "local_contrast": _round(float(np.max(rchunk) - np.median(rchunk))),
        "has_peak_support": bool(support),
    }


def merge_intervals(intervals: list[dict]) -> list[dict]:
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda item: (int(item["start"]), int(item["end"])))
    merged = [dict(ordered[0])]
    for item in ordered[1:]:
        last = merged[-1]
        if int(item["start"]) <= int(last["end"]):
            last["end"] = max(int(last["end"]), int(item["end"]))
            last["duration"] = int(last["end"]) - int(last["start"])
            sources = {str(last.get("source", "unknown")), str(item.get("source", "unknown"))}
            last["source"] = "+".join(sorted(sources))
        else:
            merged.append(dict(item))
    return merged


def summarize_sources(intervals: list[dict]) -> dict:
    counts = defaultdict(int)
    for item in intervals:
        counts[str(item.get("source", "unknown"))] += 1
    return dict(sorted(counts.items()))
